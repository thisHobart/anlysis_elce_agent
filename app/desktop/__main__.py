"""Allow ``python -m app.desktop`` to launch the desktop application."""

from app.desktop.main import main

raise SystemExit(main())
