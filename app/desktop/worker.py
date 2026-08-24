"""Run blocking research work outside the Qt GUI thread."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from threading import Event
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot


class FunctionWorker(QObject):
    """Execute a callable that accepts a progress callback."""

    progress = Signal(int, str)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, operation: Callable[[Callable[[int, str], None]], Any]) -> None:
        super().__init__()
        self.operation = operation
        self._cancel_requested = Event()

    def cancel(self) -> None:
        """Request cooperative cancellation at the next progress boundary."""

        self._cancel_requested.set()

    def _report_progress(self, value: int, message: str) -> None:
        if self._cancel_requested.is_set():
            raise TaskCancelled
        self.progress.emit(value, message)

    @Slot()
    def run(self) -> None:
        try:
            result = self.operation(self._report_progress)
            if self._cancel_requested.is_set():
                raise TaskCancelled
        except TaskCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - forwarded to the GUI
            detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.failed.emit(detail)
            return
        self.completed.emit(result)


class TaskCancelled(RuntimeError):
    """Internal signal used to stop one cooperative background operation."""
