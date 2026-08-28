"""SQLite and in-memory checkpointer construction."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

CHECKPOINT_VACUUM_THRESHOLD = 32


class CheckpointerHandle:
    """Own an optional SQLite connection for the lifetime of a coordinator."""

    def __init__(
        self,
        saver: Any,
        connection: sqlite3.Connection | None = None,
        path: Path | None = None,
    ) -> None:
        self.saver = saver
        self.connection = connection
        self.path = path
        self.last_compaction_error: str | None = None

    @classmethod
    def memory(cls) -> CheckpointerHandle:
        return cls(InMemorySaver())

    @classmethod
    def sqlite(cls, path: str | Path) -> CheckpointerHandle:
        os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:  # pragma: no cover - packaging/runtime boundary
            raise RuntimeError("langgraph-checkpoint-sqlite 尚未安装") from exc
        database = Path(path)
        database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database, check_same_thread=False, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        saver = SqliteSaver(connection)
        return cls(saver, connection, database.resolve())

    def close(self) -> None:
        if self.connection is not None:
            try:
                self.connection.execute("PRAGMA optimize")
                self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            self.connection.close()
            self.connection = None

    def compact_thread(self, thread_id: str) -> int:
        """Keep only the latest full checkpoint per namespace for one thread.

        This graph uses ordinary full-value state channels, not DeltaChannel,
        and the desktop application does not expose time travel.  Retaining the
        latest checkpoint plus its pending writes preserves interrupt/resume
        while preventing every historical super-step from accumulating forever.
        """

        if self.connection is None:
            return 0
        removed = 0
        try:
            namespaces = [
                str(row[0])
                for row in self.connection.execute(
                    "SELECT DISTINCT checkpoint_ns FROM checkpoints WHERE thread_id = ?",
                    (str(thread_id),),
                ).fetchall()
            ]
            with self.connection:
                for namespace in namespaces:
                    row = self.connection.execute(
                        """
                        SELECT checkpoint_id
                        FROM checkpoints
                        WHERE thread_id = ? AND checkpoint_ns = ?
                        ORDER BY checkpoint_id DESC
                        LIMIT 1
                        """,
                        (str(thread_id), namespace),
                    ).fetchone()
                    if row is None:
                        continue
                    latest_id = str(row[0])
                    removed += self.connection.execute(
                        """
                        DELETE FROM checkpoints
                        WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id <> ?
                        """,
                        (str(thread_id), namespace, latest_id),
                    ).rowcount
                    self.connection.execute(
                        """
                        DELETE FROM writes
                        WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id <> ?
                        """,
                        (str(thread_id), namespace, latest_id),
                    )
                    self.connection.execute(
                        """
                        UPDATE checkpoints
                        SET parent_checkpoint_id = NULL
                        WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                        """,
                        (str(thread_id), namespace, latest_id),
                    )
            if removed >= CHECKPOINT_VACUUM_THRESHOLD:
                # A one-time rebuild reclaims databases created by older
                # builds.  Smaller routine compactions reuse freed SQLite
                # pages and avoid paying VACUUM latency after every turn.
                self.connection.execute("VACUUM")
            self.last_compaction_error = None
        except sqlite3.Error as exc:
            self.last_compaction_error = f"{type(exc).__name__}: {exc}"
            return 0
        return removed

    def delete_thread(self, thread_id: str) -> None:
        """Delete one thread and reclaim a large legacy checkpoint history."""

        checkpoint_count = 0
        if self.connection is not None:
            try:
                checkpoint_count = int(
                    self.connection.execute(
                        "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                        (str(thread_id),),
                    ).fetchone()[0]
                )
            except sqlite3.Error:
                checkpoint_count = 0
        self.saver.delete_thread(thread_id)
        if self.connection is not None and checkpoint_count >= CHECKPOINT_VACUUM_THRESHOLD:
            try:
                self.connection.execute("VACUUM")
            except sqlite3.Error as exc:
                self.last_compaction_error = f"{type(exc).__name__}: {exc}"
