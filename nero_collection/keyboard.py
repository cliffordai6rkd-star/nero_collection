from __future__ import annotations

import os
import select
import sys
import termios
import tty
from types import TracebackType


class TerminalKeys:
    def __init__(self) -> None:
        self._fd: int | None = None
        self._old_settings: list[int | bytes] | None = None
        self.is_tty = sys.stdin.isatty()

    def __enter__(self) -> "TerminalKeys":
        if self.is_tty:
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def read_key(self, timeout_s: float) -> str | None:
        if not self.is_tty:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        if not ready:
            return None
        if self._fd is None:
            return None
        return os.read(self._fd, 1).decode("utf-8", errors="ignore")
