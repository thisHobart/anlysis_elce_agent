"""SQLite and in-memory checkpointer construction."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver


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
            self.connection.close()
            self.connection = None
