"""Unix-socket JSON-lines transport for the backend contract.

Protocol (contracts/backend-api.md):
- One request per connection. The client connects, sends a submit line, and
  reads event lines on the same connection until the finish line.
- cancel() is sent on a separate control connection.
- Wire format: one JSON object per line, UTF-8, newline-terminated.
"""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Union

from .backend import (
    Backend,
    BackendEvent,
    BackendStream,
    FinishEvent,
    ImagePayload,
    SamplingParams,
    TokenEvent,
    Usage,
    event_from_dict,
)


def _submit_line(
    request_id: str,
    token_ids: Sequence[int],
    sampling: SamplingParams,
    images: Optional[Sequence[ImagePayload]],
    conversation_id: Optional[str],
) -> str:
    payload: Dict[str, Any] = {
        "op": "submit",
        "request_id": request_id,
        "token_ids": list(token_ids),
        "sampling": sampling.to_dict(),
        "images": [img.to_dict() for img in (images or [])] or None,
        "conversation_id": conversation_id,
    }
    return json.dumps(payload, ensure_ascii=False)


class SocketBackend(Backend):
    """Client side: speaks the JSON-lines protocol over a Unix socket."""

    def __init__(self, socket_path: Union[str, Path]):
        self.socket_path = str(socket_path)
        self._control_reader: Optional[asyncio.StreamReader] = None
        self._control_writer: Optional[asyncio.StreamWriter] = None

    async def _ensure_control(self) -> None:
        if self._control_writer is None or self._control_writer.is_closing():
            self._control_reader, self._control_writer = await asyncio.open_unix_connection(
                self.socket_path
            )

    async def submit(
        self,
        token_ids: Sequence[int],
        sampling: SamplingParams,
        images: Optional[Sequence[ImagePayload]] = None,
        conversation_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> BackendStream:
        request_id = request_id or uuid.uuid4().hex
        reader, writer = await asyncio.open_unix_connection(self.socket_path)
        line = _submit_line(request_id, token_ids, sampling, images, conversation_id)
        writer.write(line.encode("utf-8") + b"\n")
        await writer.drain()

        async def events() -> AsyncIterator[BackendEvent]:
            try:
                while True:
                    raw = await reader.readline()
                    if not raw:
                        raise ConnectionError(
                            f"backend closed without finish event (request {request_id})"
                        )
                    obj = json.loads(raw.decode("utf-8"))
                    yield event_from_dict(obj)
                    if obj.get("type") == "finish":
                        return
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

        return BackendStream(request_id, events())

    async def cancel(self, request_id: str) -> bool:
        await self._ensure_control()
        assert self._control_writer is not None
        payload = json.dumps({"op": "cancel", "request_id": request_id})
        self._control_writer.write(payload.encode("utf-8") + b"\n")
        await self._control_writer.drain()
        # Engine answers {"ok": bool}; tolerate engines that stay silent.
        try:
            raw = await asyncio.wait_for(self._control_reader.readline(), timeout=5.0)
            if raw:
                return bool(json.loads(raw.decode("utf-8")).get("ok"))
        except asyncio.TimeoutError:
            pass
        return False

    async def close(self) -> None:
        if self._control_writer is not None:
            self._control_writer.close()
            try:
                await self._control_writer.wait_closed()
            except Exception:
                pass
            self._control_writer = None
            self._control_reader = None


async def serve_socket(
    backend: Backend,
    socket_path: Union[str, Path],
    start_event: Optional[asyncio.Event] = None,
) -> None:
    """Server side: expose any Backend over the JSON-lines protocol.

    Runs until cancelled. Binds a Unix socket at socket_path (unlinked first).
    """
    path = Path(socket_path)
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw = await reader.readline()
            if not raw:
                return
            op = json.loads(raw.decode("utf-8"))
            if op.get("op") == "cancel":
                ok = await backend.cancel(op.get("request_id", ""))
                writer.write(
                    (json.dumps({"ok": ok}) + "\n").encode("utf-8")
                )
                await writer.drain()
                return
            if op.get("op") != "submit":
                writer.write(
                    (
                        json.dumps({"type": "finish", "finish_reason": "error",
                                    "usage": Usage().to_dict(),
                                    "error": f"unknown op {op.get('op')!r}"})
                        + "\n"
                    ).encode("utf-8")
                )
                await writer.drain()
                return

            request_id = op.get("request_id") or uuid.uuid4().hex
            sampling = SamplingParams.from_dict(op.get("sampling", {}))
            images = [ImagePayload.from_dict(d) for d in op.get("images") or []]
            token_ids = op.get("token_ids") or []
            if not token_ids:
                writer.write(
                    (
                        json.dumps({"type": "finish", "finish_reason": "error",
                                    "usage": Usage().to_dict(),
                                    "error": "token_ids must be non-empty"})
                        + "\n"
                    ).encode("utf-8")
                )
                await writer.drain()
                return

            stream = await backend.submit(
                token_ids, sampling, images, op.get("conversation_id"), request_id
            )
            async for event in stream:
                writer.write(
                    (json.dumps(event.to_dict(), ensure_ascii=False) + "\n").encode("utf-8")
                )
                await writer.drain()
        except Exception as exc:  # transport-level failure: tell the client
            try:
                writer.write(
                    (
                        json.dumps({"type": "finish", "finish_reason": "error",
                                    "usage": Usage().to_dict(), "error": str(exc)})
                        + "\n"
                    ).encode("utf-8")
                )
                await writer.drain()
            except Exception:
                pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    server = await asyncio.start_unix_server(handle, str(path))
    if start_event is not None:
        start_event.set()
    try:
        async with server:
            await server.serve_forever()
    finally:
        path.unlink(missing_ok=True)
