import os
import sys
from pathlib import Path

# Make ds41_api importable when running pytest from the repo root without
# installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Shared server factory for the round-3 endpoint tests (anthropic, responses,
# robustness). Each call starts a fresh FastAPI+uvicorn server on an
# ephemeral 127.0.0.1 port with a FakeBackend; all servers are stopped on
# fixture teardown.
# ---------------------------------------------------------------------------

import socket
import threading
import time

import pytest

import ds41_api.encoding as enc
from ds41_api.encoder import CharTokenizer, Encoder
from ds41_api.fake_backend import FakeBackend
from ds41_api.server import ServerConfig, create_app

# The real tokenizer when the checkpoint is downloaded (DS41_TOKENIZER overrides);
# otherwise a character tokenizer, which every test also supports.
WEIGHTS_TOKENIZER = Path(
    os.environ.get(
        "DS41_TOKENIZER",
        Path(__file__).resolve().parents[1] / "models/DeepSeek-V4.1-Flash-1017/tokenizer.json",
    )
)


def make_encoder() -> Encoder:
    if WEIGHTS_TOKENIZER.exists():
        try:
            return Encoder.from_file(str(WEIGHTS_TOKENIZER))
        except Exception:
            pass
    return Encoder(CharTokenizer())


class TestServer:
    """Handle to one running test server."""

    def __init__(self, port: int, backend: FakeBackend, encoder: Encoder):
        self.port = port
        self.backend = backend
        self.encoder = encoder
        self.base = f"http://127.0.0.1:{port}"

    def script(self, text: str) -> None:
        self.backend.add_script(text)

    def decoded_prompt(self, index: int = -1) -> str:
        """Decode the token ids of the backend's most recent (or indexed)
        submission back into text, for prompt-level assertions."""
        sub = self.backend.submitted[index]
        return self.encoder.decode_ids(sub["token_ids"])


@pytest.fixture()
def server_factory():
    started = []

    def _make(config: ServerConfig = None, backend_kwargs: dict = None) -> TestServer:
        import uvicorn

        encoder = make_encoder()
        backend = FakeBackend(tokenizer=encoder.tokenizer, **(backend_kwargs or {}))
        app = create_app(backend, encoder, config=config)
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        uvi = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        )
        thread = threading.Thread(target=uvi.run, daemon=True)
        thread.start()
        for _ in range(200):
            if uvi.started:
                break
            time.sleep(0.05)
        assert uvi.started, "test server did not start"
        started.append((uvi, thread))
        return TestServer(port, backend, encoder)

    yield _make

    for uvi, thread in started:
        uvi.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive(), "test server did not stop"
