"""Entry point: `python -m mudai`."""
from __future__ import annotations

import asyncio
import sys

from PyQt6.QtWidgets import QApplication
from qasync import QEventLoop

from .config import AppConfig
from .gui.main_window import MainWindow


def main() -> int:
    cfg = AppConfig.load()
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    window = MainWindow(cfg)
    window.show()

    with loop:
        loop.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
