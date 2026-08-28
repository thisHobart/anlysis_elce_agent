"""Durable tool-result blobs kept outside LangGraph checkpoint snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.research.tools.contracts import ToolCall, ToolResult
from app.research.tools.executor import output_fingerprint

REFERENCE_KIND = "tool_result_ref_v1"
SAFE_STORAGE_KEY = re.compile(r"^[a-f0-9]{32}\.json$")


def _storage_key(result: ToolResult) -> str:
    """Use a short content-addressed name that stays below Windows path limits."""

    payload = f"{result.call.work_id}:{result.output_hash}".encode("ascii")
    return f"{hashlib.sha256(payload).hexdigest()[:32]}.json"


class ToolResultReference(BaseModel):
    """Small, checkpoint-safe pointer to one validated deterministic result."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["tool_result_ref_v1"] = REFERENCE_KIND
    storage_key: str = Field(pattern=r"^[a-f0-9]{32}\.json$")
    call: ToolCall
    result_key: str
    output_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    data_fingerprint: str = Field(pattern=r"^[a-f0-9]{12}$")
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: float = Field(ge=0)
    provider: str
    tool_version: str

    @classmethod
    def from_result(cls, result: ToolResult, *, storage_key: str | None = None) -> ToolResultReference:
        return cls(
            storage_key=storage_key or _storage_key(result),
            call=result.call,
            result_key=result.output.result_key,
            output_hash=result.output_hash,
            data_fingerprint=result.data_fingerprint,
            started_at=result.started_at,
            finished_at=result.finished_at,
            duration_ms=result.duration_ms,
            provider=result.provider,
            tool_version=result.tool_version,
        )


class ToolResultStore:
    """Storage boundary used by the graph while checkpoint state holds only refs."""

    def put(self, thread_id: str, result: ToolResult) -> dict[str, Any]:
        raise NotImplementedError

    def get(
        self,
        thread_id: str,
        value: dict[str, Any],
        *,
        call: ToolCall | None = None,
    ) -> ToolResult:
        raise NotImplementedError

    def bind(self, thread_id: str, value: dict[str, Any], call: ToolCall) -> dict[str, Any]:
        reference = ToolResultReference.model_validate(value)
        result = self.get(thread_id, value, call=call)
        return ToolResultReference.from_result(result, storage_key=reference.storage_key).model_dump(mode="json")

    def delete_thread(self, thread_id: str) -> None:
        return None

    def prune_thread(self, thread_id: str, storage_keys: set[str]) -> int:
        return 0

    def close(self) -> None:
        return None

    @staticmethod
    def _validate_loaded(
        result: ToolResult,
        reference: ToolResultReference,
        call: ToolCall | None,
    ) -> ToolResult:
        if result.call.work_id != reference.call.work_id:
            raise ValueError("工具结果存储中的 work_id 与引用不一致")
        if result.output.result_key != reference.result_key:
            raise ValueError("工具结果存储中的 result_key 与引用不一致")
        if result.data_fingerprint != reference.data_fingerprint:
            raise ValueError("工具结果存储中的数据指纹与引用不一致")
        if result.output_hash != reference.output_hash or output_fingerprint(result.output) != reference.output_hash:
            raise ValueError("工具结果存储中的输出哈希校验失败")
        if call is not None:
            if call.work_id != result.call.work_id or call.name != result.call.name or call.version != result.call.version:
                raise ValueError("当前函数调用与已保存工具结果不兼容")
            result = result.model_copy(update={"call": call})
        return result


class InMemoryToolResultStore(ToolResultStore):
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = RLock()

    def put(self, thread_id: str, result: ToolResult) -> dict[str, Any]:
        reference = ToolResultReference.from_result(result)
        with self._lock:
            self._items[(thread_id, reference.storage_key)] = result.model_dump(mode="json")
        return reference.model_dump(mode="json")

    def get(
        self,
        thread_id: str,
        value: dict[str, Any],
        *,
        call: ToolCall | None = None,
    ) -> ToolResult:
        if value.get("kind") != REFERENCE_KIND:
            result = ToolResult.model_validate(value)
            return result.model_copy(update={"call": call}) if call is not None else result
        reference = ToolResultReference.model_validate(value)
        with self._lock:
            payload = self._items.get((thread_id, reference.storage_key))
        if payload is None:
            raise FileNotFoundError(f"工具结果引用不存在：{reference.storage_key}")
        return self._validate_loaded(ToolResult.model_validate(payload), reference, call)

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            for key in [key for key in self._items if key[0] == thread_id]:
                self._items.pop(key, None)

    def prune_thread(self, thread_id: str, storage_keys: set[str]) -> int:
        removed = 0
        with self._lock:
            for key in [
                key
                for key in self._items
                if key[0] == thread_id and key[1] not in storage_keys
            ]:
                self._items.pop(key, None)
                removed += 1
        return removed


class FileToolResultStore(ToolResultStore):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._lock = RLock()

    @classmethod
    def beside_checkpoint(cls, checkpoint_path: str | Path) -> FileToolResultStore:
        checkpoint = Path(checkpoint_path)
        return cls(checkpoint.with_name(f"{checkpoint.stem}_tool_results"))

    @staticmethod
    def _thread_key(thread_id: str) -> str:
        return hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:24]

    def _thread_directory(self, thread_id: str) -> Path:
        return self.root / self._thread_key(thread_id)

    def put(self, thread_id: str, result: ToolResult) -> dict[str, Any]:
        reference = ToolResultReference.from_result(result)
        directory = self._thread_directory(thread_id)
        destination = directory / reference.storage_key
        with self._lock:
            directory.mkdir(parents=True, exist_ok=True)
            if not destination.is_file():
                # Different coordinator instances can briefly overlap during
                # desktop reload.  A unique temporary name keeps their atomic
                # writes from moving each other's staging file.
                temporary = destination.with_name(f"{destination.name}.{uuid4().hex}.tmp")
                temporary.write_text(
                    json.dumps(result.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, allow_nan=False),
                    encoding="utf-8",
                )
                os.replace(temporary, destination)
        return reference.model_dump(mode="json")

    def get(
        self,
        thread_id: str,
        value: dict[str, Any],
        *,
        call: ToolCall | None = None,
    ) -> ToolResult:
        if value.get("kind") != REFERENCE_KIND:
            result = ToolResult.model_validate(value)
            return result.model_copy(update={"call": call}) if call is not None else result
        reference = ToolResultReference.model_validate(value)
        if not SAFE_STORAGE_KEY.fullmatch(reference.storage_key):
            raise ValueError("invalid tool-result storage key")
        source = self._thread_directory(thread_id) / reference.storage_key
        with self._lock:
            payload = json.loads(source.read_text(encoding="utf-8"))
        return self._validate_loaded(ToolResult.model_validate(payload), reference, call)

    def delete_thread(self, thread_id: str) -> None:
        target = self._thread_directory(thread_id).resolve()
        if target.parent != self.root or not target.is_dir():
            return
        with self._lock:
            shutil.rmtree(target)

    def prune_thread(self, thread_id: str, storage_keys: set[str]) -> int:
        directory = self._thread_directory(thread_id)
        if not directory.is_dir():
            return 0
        removed = 0
        with self._lock:
            for path in directory.glob("*.json"):
                if path.name not in storage_keys:
                    path.unlink()
                    removed += 1
            try:
                directory.rmdir()
            except OSError:
                pass
        return removed
