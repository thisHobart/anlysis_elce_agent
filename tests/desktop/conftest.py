"""Fixtures shared by the desktop test modules.

They are defined alongside the model doubles they build in ``test_desktop`` and
re-exported here so pytest offers them to every module in this directory.
"""

from tests.desktop.test_desktop import (  # noqa: F401 - re-exported as pytest fixtures
    desktop_study,
    model_agent,
    qt_app,
)
