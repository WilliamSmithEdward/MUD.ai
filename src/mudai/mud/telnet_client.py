"""Async Telnet client for MUDs using telnetlib3.

Emits decoded text chunks (still containing ANSI escape sequences) to the GUI
and accepts outgoing command lines.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import telnetlib3


OnText = Callable[[str], None]
OnStatus = Callable[[str], None]


class MudClient:
    def __init__(
        self,
        host: str,
        port: int,
        on_text: OnText,
        on_status: OnStatus,
        encoding: str = "utf-8",
    ) -> None:
        self.host = host
        self.port = port
        self.encoding = encoding
        self._on_text = on_text
        self._on_status = on_status
        self._reader: Any | None = None
        self._writer: Any | None = None
        self._task: asyncio.Task[None] | None = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        if self._connected:
            return
        self._on_status(f"Connecting to {self.host}:{self.port} ...")
        try:
            reader, writer = await telnetlib3.open_connection(
                host=self.host,
                port=self.port,
                encoding=self.encoding,
                connect_minwait=0.05,
                connect_maxwait=0.5,
            )
        except Exception as e:
            self._on_status(f"Connect failed: {e}")
            raise
        self._reader = reader
        self._writer = writer
        self._connected = True
        self._on_status(f"Connected to {self.host}:{self.port}")
        self._task = asyncio.create_task(self._reader_loop())

    async def _reader_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                chunk = await self._reader.read(4096)
                if not chunk:
                    break
                self._on_text(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._on_status(f"Read error: {e}")
        finally:
            self._connected = False
            self._on_status("Disconnected")

    async def send(self, line: str) -> None:
        if not self._connected or self._writer is None:
            self._on_status("Not connected")
            return
        if not line.endswith("\n"):
            line = line + "\n"
        try:
            self._writer.write(line)
            drain = getattr(self._writer, "drain", None)
            if callable(drain):
                result = drain()
                if isinstance(result, Awaitable):
                    await result
        except Exception as e:
            self._on_status(f"Write error: {e}")

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
        self._reader = None
        self._writer = None
        self._connected = False
