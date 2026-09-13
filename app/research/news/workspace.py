"""Durable extraction, review history and immutable P2 run artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.research.news.adapters import NewsInputBatch
from app.research.news.contracts import CollectedNewsRecord, EventExtractionResult, NewsDocument
from app.research.news.evidence import package_payload, render_report
from app.research.news.features import snapshot_to_csv_rows
from app.research.news.model_extraction import ModelNewsExtraction, StructuredNewsEventExtractor
from app.research.news.normalization import NewsNormalizer


def digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _collected(document: NewsDocument) -> CollectedNewsRecord:
    return CollectedNewsRecord(
        source_name=document.source_name,
        source_document_id=document.source_document_id,
        source_ref=document.source_ref,
        version=document.version,
        title=document.title,
        body=document.body,
        published_at=document.published_at,
        collected_at=document.first_seen_at,
        updated_at=document.updated_at,
        language=document.language,
        market_tags=document.market_tags,
        metadata=document.raw_metadata,
    )


class NewsWorkspace:
    """One local SQLite workspace; every saved review is an append-only decision."""

    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "news.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS extractions (
                    cache_key TEXT PRIMARY KEY, document_id TEXT NOT NULL, payload TEXT NOT NULL,
                    reusable INTEGER NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, cache_key TEXT NOT NULL,
                    decision TEXT NOT NULL, reviewer TEXT NOT NULL, reason TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS extraction_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, cache_key TEXT NOT NULL,
                    payload TEXT NOT NULL, created_at TEXT NOT NULL);
            """)
            review_columns = {
                str(row[1]) for row in db.execute("PRAGMA table_info(reviews)").fetchall()
            }
            if "usage" not in review_columns:
                db.execute("ALTER TABLE reviews ADD COLUMN usage TEXT NOT NULL DEFAULT 'analysis'")
            if "request_id" not in review_columns:
                db.execute("ALTER TABLE reviews ADD COLUMN request_id TEXT")
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS reviews_request_id "
                "ON reviews(request_id) WHERE request_id IS NOT NULL"
            )

    def connect(self):
        return sqlite3.connect(self.database, timeout=30)

    def save_document(self, document: NewsDocument):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM documents WHERE id=?", (document.document_version_id,)).fetchone()
            if row is not None:
                old = NewsDocument.model_validate_json(row[0])
                if old.market_tags != document.market_tags:
                    raise ValueError("同一文档版本的市场标签冲突，需要新的来源版本")
                if document.first_seen_at < old.first_seen_at:
                    db.execute(
                        "UPDATE documents SET payload=? WHERE id=?",
                        (document.model_dump_json(), document.document_version_id),
                    )
            db.execute(
                "INSERT OR IGNORE INTO documents VALUES (?, ?)",
                (document.document_version_id, document.model_dump_json()),
            )

    def cache_result(self, key: str, document: NewsDocument, result: EventExtractionResult):
        self.save_document(document)
        serialized = result.model_dump_json()
        failures = [result.model_dump(mode="json"), *result.pass_results]
        reusable = not any(
            item and item["reason_code"] in {"model_unavailable", "model_response_invalid"}
            for payload in failures
            for item in [payload.get("quarantine"), *payload.get("candidate_quarantines", [])]
        )
        with self.connect() as db:
            stamp = datetime.now(UTC).isoformat()
            db.execute(
                "INSERT INTO extraction_attempts(cache_key, payload, created_at) VALUES (?, ?, ?)",
                (key, serialized, stamp),
            )
            db.execute(
                "INSERT OR REPLACE INTO extractions VALUES (?, ?, ?, ?, ?)",
                (key, document.document_version_id, serialized, int(reusable), stamp),
            )

    def cached(self, key: str) -> EventExtractionResult | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM extractions WHERE cache_key=? AND reusable=1", (key,)).fetchone()
        return EventExtractionResult.model_validate_json(row[0]) if row else None

    def review(
        self,
        key: str,
        *,
        decision: str,
        reviewer: str,
        reason: str,
        corrected: dict | None = None,
        market_timezone: str,
        usage: str = "analysis",
        expected_revision: str | None = None,
        request_id: str | None = None,
    ) -> int:
        if decision not in {"accepted", "corrected", "rejected"} or not reviewer.strip() or not reason.strip():
            raise ValueError("复核必须包含有效处置、复核人和说明")
        if usage not in {"analysis", "background_only"}:
            raise ValueError("复核用途必须是 analysis 或 background_only")
        if decision == "rejected" and usage != "analysis":
            raise ValueError("拒绝的抽取结果不能标记为背景材料")
        if (decision == "corrected") != (corrected is not None):
            raise ValueError("只有 corrected 处置需要且必须提供候选文件")
        with self.connect() as db:
            if request_id:
                duplicate = db.execute(
                    "SELECT id, cache_key, decision, usage, reviewer, reason FROM reviews WHERE request_id=?",
                    (request_id,),
                ).fetchone()
                if duplicate is not None:
                    if duplicate[1:] != (
                        key,
                        decision,
                        usage,
                        reviewer.strip(),
                        reason.strip(),
                    ):
                        raise ValueError("复核请求ID已用于另一项操作，不能复用")
                    return int(duplicate[0])
            row = db.execute(
                "SELECT d.payload, e.payload FROM extractions e JOIN documents d "
                "ON d.id=e.document_id WHERE e.cache_key=?",
                (key,),
            ).fetchone()
        if row is None:
            raise ValueError("找不到待复核抽取记录")
        document = NewsDocument.model_validate_json(row[0])
        original = EventExtractionResult.model_validate_json(row[1])
        current_revision = self.review_revision(key)
        if expected_revision is not None and expected_revision != current_revision:
            raise ValueError("复核条目已更新；请刷新窗口后再提交")
        result = original
        if corrected is not None:
            parsed = ModelNewsExtraction.model_validate(corrected)

            class ReviewGateway:
                model_name = f"human-review:{reviewer}"

                def invoke_structured(self, **kwargs):
                    return parsed

            result = StructuredNewsEventExtractor(
                ReviewGateway(), market_timezone=market_timezone, extraction_passes=1
            ).extract(document)
            # Correcting identity requires explicit split/merge adjudication, which this
            # bounded review path deliberately refuses rather than double-counting events.
            if original.events:
                identity = lambda events: sorted((e.event_type, e.asset_keys, e.region_keys) for e in events)
                if identity(original.events) != identity(result.events):
                    raise ValueError("不能在字段复核中更换事件类型或实体；需要单独的事件拆分/合并处理")
        if decision != "rejected" and (
            not result.events
            or result.quarantine
            or result.candidate_quarantines
            or any(event.event_type == "unknown" for event in result.events)
            or any(not event.evidence for event in result.events)
        ):
            raise ValueError("仍有未通过证据门禁的候选；请修正候选或拒绝，不能直接接受")
        if decision != "rejected" and usage == "analysis" and any(
            event.analysis_eligibility != "eligible" for event in result.events
        ):
            raise ValueError("事件时间或覆盖范围仍不满足精确分析条件；请修正、拒绝或仅作背景")
        if usage == "background_only":
            if not result.events or any(
                event.analysis_eligibility not in {"eligible", "needs_time_review"}
                for event in result.events
            ) or not any(event.analysis_eligibility == "needs_time_review" for event in result.events):
                raise ValueError("只有事实证据有效且至多存在时间不确定性的事件才能仅作背景")
            result = result.model_copy(
                update={
                    "events": tuple(
                        event.model_copy(update={"analysis_eligibility": "background_only"})
                        for event in result.events
                    )
                }
            )
        result = result.model_copy(
            update={
                "review_status": decision,
                "events": tuple(event.model_copy(update={"review_status": decision}) for event in result.events),
            }
        )
        with self.connect() as db:
            cursor = db.execute(
                "INSERT INTO reviews(cache_key, decision, reviewer, reason, reviewed_at, payload, usage, request_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    decision,
                    reviewer.strip(),
                    reason.strip(),
                    datetime.now(UTC).isoformat(),
                    result.model_dump_json(),
                    usage,
                    request_id,
                ),
            )
            return cursor.lastrowid

    def review_revision(self, key: str) -> str:
        """Return a stable optimistic-lock revision for one extraction and its latest review."""

        with self.connect() as db:
            row = db.execute(
                "SELECT e.payload, r.id, r.decision, r.usage, r.payload "
                "FROM extractions e LEFT JOIN reviews r ON r.id=("
                "SELECT id FROM reviews WHERE cache_key=e.cache_key ORDER BY id DESC LIMIT 1"
                ") WHERE e.cache_key=?",
                (key,),
            ).fetchone()
        if row is None:
            raise ValueError("找不到待复核抽取记录")
        return digest(
            {
                "extraction": json.loads(row[0]),
                "review_id": row[1],
                "decision": row[2],
                "usage": row[3],
                "reviewed_result": json.loads(row[4]) if row[4] else None,
            }
        )

    def review_queue(self) -> tuple[dict, ...]:
        """Read review items from SQLite; exported JSON snapshots are never accepted as input."""

        with self.connect() as db:
            rows = db.execute(
                "SELECT e.cache_key, e.document_id, e.payload, d.payload, "
                "r.id, r.decision, r.reviewer, r.reason, r.reviewed_at, r.payload, r.usage "
                "FROM extractions e JOIN documents d ON d.id=e.document_id "
                "LEFT JOIN reviews r ON r.id=("
                "SELECT id FROM reviews WHERE cache_key=e.cache_key ORDER BY id DESC LIMIT 1"
                ") ORDER BY e.created_at, e.cache_key"
            ).fetchall()
        queue = []
        for row in rows:
            queue.append(
                {
                    "cache_key": row[0],
                    "document_version_id": row[1],
                    "original_result": json.loads(row[2]),
                    "document": json.loads(row[3]),
                    "review_id": row[4],
                    "decision": row[5],
                    "reviewer": row[6],
                    "reason": row[7],
                    "reviewed_at": row[8],
                    "reviewed_result": json.loads(row[9]) if row[9] else None,
                    "usage": row[10],
                    "revision": self.review_revision(str(row[0])),
                }
            )
        return tuple(queue)

    def review_records(self) -> tuple[CollectedNewsRecord, ...]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT r.id, r.reviewed_at, r.reviewer, r.reason, d.payload, r.usage FROM reviews r "
                "JOIN extractions e ON e.cache_key=r.cache_key JOIN documents d ON d.id=e.document_id "
                "ORDER BY r.id"
            ).fetchall()
        records = []
        for review_id, stamp, reviewer, reason, payload, usage in rows:
            document = NewsDocument.model_validate_json(payload)
            record = _collected(document)
            # A review creates a distinct, explicitly labelled local revision. It only
            # becomes knowable at review time, so historical feature rows remain intact.
            records.append(
                record.model_copy(
                    update={
                        "version": 1_000_000_000 + review_id,
                        "collected_at": max(datetime.fromisoformat(stamp), document.available_at),
                        "metadata": {
                            **record.metadata,
                            "local_review_id": review_id,
                            "source_document_version_id": document.document_version_id,
                            "reviewer": reviewer,
                            "review_reason": reason,
                            "reviewed_at": stamp,
                            "review_use": usage,
                        },
                    }
                )
            )
        return tuple(records)

    def reviewed_result(self, document: NewsDocument) -> EventExtractionResult:
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM reviews WHERE id=?", (document.raw_metadata["local_review_id"],)
            ).fetchone()
        if row is None:
            raise ValueError("本地复核版本没有对应审计记录")
        result = EventExtractionResult.model_validate_json(row[0])
        events = tuple(
            event.model_copy(
                update={
                    "document_version_id": document.document_version_id,
                    "announcement_available_at": document.available_at,
                    "evidence": tuple(
                        span.model_copy(update={"document_version_id": document.document_version_id})
                        for span in event.evidence
                    ),
                    "entity_resolution": None,
                }
            )
            for event in result.events
        )
        # Rebuild all entity source references instead of leaving dangling old-version spans.
        from app.research.news.entity_resolution import resolve_entities

        rebuilt = []
        for event, old in zip(events, result.events, strict=True):
            if old.entity_resolution is not None:
                resolution = resolve_entities(
                    document,
                    event.affected_regions,
                    event.affected_assets,
                    tuple(item.value for item in old.asset_groups),
                    list(event.evidence),
                )
                event = event.model_copy(update={"entity_resolution": resolution})
            rebuilt.append(event)
        if rebuilt:
            return EventExtractionResult(
                document_version_id=document.document_version_id,
                events=tuple(rebuilt),
                review_status=result.review_status,
                supersedes_document_version_id=document.raw_metadata["source_document_version_id"],
            )
        return result.model_copy(
            update={
                "document_version_id": document.document_version_id,
                "quarantine": result.quarantine.model_copy(
                    update={"document_version_id": document.document_version_id, "review_status": result.review_status}
                )
                if result.quarantine
                else None,
                "candidate_quarantines": (),
                "supersedes_document_version_id": document.raw_metadata["source_document_version_id"],
            }
        )


class CachedNewsExtractor:
    def __init__(self, delegate, workspace: NewsWorkspace, configuration: dict):
        self.delegate = delegate
        self.workspace = workspace
        self.market_timezone = delegate.market_timezone
        self.extractor_id = delegate.extractor_id
        self.extractor_version = delegate.extractor_version
        self.configuration_hash = digest(
            {
                "configuration": configuration,
                "extractor": self.extractor_id,
                "version": self.extractor_version,
                "timezone": self.market_timezone,
            }
        )

    def extract(self, document: NewsDocument) -> EventExtractionResult:
        if "local_review_id" in document.raw_metadata:
            return self.workspace.reviewed_result(document)
        key = digest({"configuration": self.configuration_hash, "document": document.model_dump(mode="json")})
        result = self.workspace.cached(key)
        if result is None:
            result = self.delegate.extract(document)
            self.workspace.cache_result(key, document, result)
        return result


class WorkspaceNewsAdapter:
    def __init__(self, adapter, workspace: NewsWorkspace):
        self.adapter, self.workspace = adapter, workspace

    def load_batch(self):
        batch = self.adapter.load_batch()
        # A file cannot impersonate a decision in this workspace.
        if any("local_review_id" in record.metadata for record in batch.records):
            raise ValueError("输入文件不得携带保留的 local_review_id 字段")
        for record in batch.records:
            self.workspace.save_document(NewsNormalizer().normalize(record))
        with self.workspace.connect() as db:
            stored = tuple(
                _collected(NewsDocument.model_validate_json(row[0]))
                for row in db.execute("SELECT payload FROM documents ORDER BY id")
            )
        return NewsInputBatch((*stored, *self.workspace.review_records()), batch.issues)

    def load(self):
        return self.load_batch().records


def export_study(study, workspace: NewsWorkspace) -> Path:
    """Write an independent run; manifest is the completion marker and lists file hashes."""
    output = workspace.directory / "runs" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    output.mkdir(parents=True)
    payload = package_payload(study.package)
    payload["result_quality"] = asdict(study.result_quality)
    files = {
        "report.md": render_report(study.package),
        "package.json": json.dumps(payload, ensure_ascii=False, indent=2),
        "events.json": json.dumps(payload["events"], ensure_ascii=False, indent=2),
        "input-issues.json": json.dumps(payload["input_issues"], ensure_ascii=False, indent=2),
    }
    with workspace.connect() as db:
        reviews = [
            dict(
                zip(
                    ("id", "cache_key", "decision", "reviewer", "reason", "reviewed_at", "usage", "request_id"),
                    row,
                )
            )
            for row in db.execute(
                "SELECT id, cache_key, decision, reviewer, reason, reviewed_at, usage, request_id "
                "FROM reviews ORDER BY id"
            )
        ]
        queue = [
            dict(zip(("cache_key", "document_version_id", "result", "document"), row))
            for row in db.execute(
                "SELECT e.cache_key, e.document_id, e.payload, d.payload FROM extractions e "
                "JOIN documents d ON d.id=e.document_id ORDER BY e.created_at"
            )
        ]
    for item in queue:
        item["result"] = json.loads(item["result"])
        item["document"] = json.loads(item["document"])
        decisions = [review for review in reviews if review["cache_key"] == item["cache_key"]]
        item["review_status"] = decisions[-1]["decision"] if decisions else "unreviewed"
    files["review-queue.json"] = json.dumps({"extractions": queue, "decisions": reviews}, ensure_ascii=False, indent=2)
    for name, content in files.items():
        (output / name).write_text(content, encoding="utf-8")
    with (output / "features.csv").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(snapshot_to_csv_rows(study.features))
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir())}
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "status": "needs_review" if study_needs_review(study) else "complete",
                "as_of": study.as_of.isoformat(),
                "fingerprints": study.package.fingerprint(),
                "files": hashes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def study_needs_review(study) -> bool:
    return (
        not study.result_quality.passed
        or any(item.review_status == "unreviewed" for item in study.view.quarantined)
        or any(
            event.relevance == "short_term"
            and event.effective_start_at is None
            and event.review_status == "unreviewed"
            for event in study.view.events
        )
    )
