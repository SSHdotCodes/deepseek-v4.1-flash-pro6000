"""Image prompt preparation (ds41_api.vision): span expansion, image sources and limits."""
import base64
import io
import time

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from ds41_api.backend import ImagePayload  # noqa: E402
from ds41_api.vision import (  # noqa: E402
    VisionConfig,
    VisionImage,
    expand_prompt,
    image_bytes,
    num_image_tokens,
    plan_image_grid,
)

CFG = VisionConfig()
IMG = CFG.image_token_id


def png(width, height, color=(200, 30, 60)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_span_expansion_matches_prepare_vl_inputs():
    ids, images = expand_prompt([1, 2, IMG, 3], [ImagePayload(data=png(640, 480))], CFG)
    n_llm_h, n_llm_w, best_h, best_w = plan_image_grid(640, 480, CFG)
    span = num_image_tokens(n_llm_h, n_llm_w)
    assert ids == [1, 2] + [IMG] * span + [3]
    (img,) = images
    assert (img.start, img.n_llm_h, img.n_llm_w) == (2, n_llm_h, n_llm_w)
    assert (img.n_vit_h, img.n_vit_w) == (best_h // 14, best_w // 14)
    assert len(img.pixels) == best_h * best_w * 3
    assert (img.n_llm_h, img.n_llm_w) == ((img.n_vit_h + 2) // 3, (img.n_vit_w + 2) // 3)


def test_two_images_in_order_and_positions():
    ids, images = expand_prompt([IMG, 5, IMG], [ImagePayload(data=png(300, 300)), ImagePayload(data=png(900, 300))], CFG)
    first = num_image_tokens(images[0].n_llm_h, images[0].n_llm_w)
    assert images[0].start == 0 and images[1].start == first + 1
    assert ids[first] == 5 and len(ids) == first + 1 + num_image_tokens(images[1].n_llm_h, images[1].n_llm_w)


def test_token_budget_and_small_images():
    for size in [(4032, 3024), (3000, 200), (200, 3000), (20, 20)]:
        n_llm_h, n_llm_w, _, _ = plan_image_grid(*size, CFG)
        assert num_image_tokens(n_llm_h, n_llm_w) <= CFG.max_n_token


def test_placeholder_count_must_match():
    with pytest.raises(ValueError, match="placeholders"):
        expand_prompt([1, IMG, IMG], [ImagePayload(data=png(64, 64))], CFG)
    with pytest.raises(ValueError, match="no vision tower"):
        expand_prompt([IMG], [ImagePayload(data=png(64, 64))], None)


def test_sources():
    raw = png(32, 32)
    url = "data:image/png;base64," + base64.b64encode(raw).decode()
    assert image_bytes(ImagePayload(url=url), CFG) == raw
    with pytest.raises(ValueError, match="data: or http"):
        image_bytes(ImagePayload(url="/etc/hostname"), CFG)
    with pytest.raises(ValueError, match="public"):
        image_bytes(ImagePayload(url="http://127.0.0.1:9/x.png"), CFG)
    no_remote = VisionConfig(allow_remote=False)
    with pytest.raises(ValueError, match="disabled"):
        image_bytes(ImagePayload(url="https://example.com/x.png"), no_remote)
    with pytest.raises(ValueError, match="cannot read image"):
        expand_prompt([IMG], [ImagePayload(data=b"not an image")], CFG)


def test_wire_round_trip():
    img = VisionImage(7, 40, 41, 14, 14, bytes(range(256)) * 3)
    assert VisionImage.from_dict(img.to_dict()) == img


class ImageTokenizer:
    """Character tokenizer that, like tokenizer.json, maps the image placeholder to image_token_id."""

    def __init__(self):
        from ds41_api.encoder import CharTokenizer
        from ds41_api.encoding import IMAGE_PLACEHOLDER

        self.chars, self.placeholder = CharTokenizer(), IMAGE_PLACEHOLDER

    def encode(self, text, add_special_tokens=False):
        ids = []
        for i, part in enumerate(text.split(self.placeholder)):
            if i:
                ids.append(IMG)
            ids += self.chars.encode(part).ids
        return type("E", (), {"ids": ids, "tokens": []})()

    def decode(self, ids, skip_special_tokens=False):
        return self.chars.decode(ids)


@pytest.mark.parametrize("endpoint", ["chat", "messages", "responses"])
def test_every_endpoint_sends_prepared_images(endpoint):
    from fastapi.testclient import TestClient

    from ds41_api.encoder import Encoder
    from ds41_api.fake_backend import FakeBackend
    from ds41_api.server import create_app

    tokenizer = ImageTokenizer()
    backend = FakeBackend(tokenizer=tokenizer)
    backend.add_script("ok") if hasattr(backend, "add_script") else None
    client = TestClient(create_app(backend, Encoder(tokenizer, VisionConfig())))
    raw = png(400, 300)
    url = "data:image/png;base64," + base64.b64encode(raw).decode()
    if endpoint == "chat":
        r = client.post("/v1/chat/completions", json={"model": "deepseek-v4.1-flash-1017", "max_tokens": 4, "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "hi"}]}]})
    elif endpoint == "messages":
        r = client.post("/v1/messages", json={"model": "deepseek-v4.1-flash-1017", "max_tokens": 4, "messages": [
            {"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                                     "data": base64.b64encode(raw).decode()}},
                                         {"type": "text", "text": "hi"}]}]})
        count = client.post("/v1/messages/count_tokens", json={"model": "deepseek-v4.1-flash-1017", "messages": [
            {"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                                     "data": base64.b64encode(raw).decode()}},
                                         {"type": "text", "text": "hi"}]}]})
        assert count.status_code == 200
    else:
        r = client.post("/v1/responses", json={"model": "deepseek-v4.1-flash-1017", "max_output_tokens": 4, "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": url}, {"type": "input_text", "text": "hi"}]}]})
    assert r.status_code == 200, r.text
    sent = backend.submitted[-1]
    (img,) = sent["images"]
    assert isinstance(img, VisionImage)
    span = num_image_tokens(img.n_llm_h, img.n_llm_w)
    assert sent["token_ids"][img.start:img.start + span] == [IMG] * span
    assert sent["token_ids"].count(IMG) == span
    if endpoint == "messages":
        assert count.json()["input_tokens"] >= span


def test_remote_hosts_must_resolve_to_public_addresses(monkeypatch):
    import socket
    import time

    from ds41_api import vision

    deadline = time.monotonic() + 5
    for literal in ["127.0.0.1", "10.1.2.3", "169.254.169.254", "::1", "fe80::1", "::ffff:127.0.0.1", "::ffff:10.0.0.1"]:
        with pytest.raises(ValueError, match="public"):
            vision._public_addresses(literal, 80, deadline)
    assert vision._public_addresses("8.8.8.8", 80, deadline) == ["8.8.8.8"]

    def mixed(host, port, family=0, type=0, *args):  # one public and one private answer
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", port)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.10", port))]

    monkeypatch.setattr(socket, "getaddrinfo", mixed)
    with pytest.raises(ValueError, match="public"):
        vision._public_addresses("images.example", 443, deadline)
    with pytest.raises(ValueError, match="too long"):
        vision._fetch("http://8.8.8.8/x.png", time.monotonic() - 1)


def test_connection_goes_only_to_the_checked_addresses(monkeypatch):
    import time

    import httpcore

    from ds41_api import vision

    dialed = []
    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp",
                        lambda self, host, port, *a, **k: dialed.append((host, port)) or "stream")
    backend = vision._pinned_backend("images.example", ["93.184.215.14"], time.monotonic() + 5)
    stream = backend.connect_tcp(host="images.example", port=443, timeout=5.0)  # httpcore passes keywords
    assert stream._stream == "stream"
    assert dialed == [("93.184.215.14", 443)]  # no second DNS lookup of the name
    with pytest.raises(httpcore.ConnectError):
        backend.connect_tcp("other.example", 443)


def test_timed_out_address_falls_through_to_the_next(monkeypatch):
    import time

    import httpcore

    from ds41_api import vision

    dialed = []

    def connect(self, host, port, timeout=None, *a):
        dialed.append(host)
        if host == "2001:db8::1":
            raise httpcore.ConnectTimeout("blackholed")
        return "stream"

    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", connect)
    backend = vision._pinned_backend("images.example", ["2001:db8::1", "93.184.215.14"], time.monotonic() + 5)
    assert backend.connect_tcp(host="images.example", port=443)._stream == "stream"
    assert dialed == ["2001:db8::1", "93.184.215.14"]


def test_deadline_bounds_endless_informational_responses(monkeypatch):
    """A server that answers 103 Early Hints forever, each within the per-read timeout, still hits the
    request deadline: every socket read is bounded by the time left (real httpcore parser, fake clock)."""
    import httpcore

    from ds41_api import vision

    clock = [1000.0]
    monkeypatch.setattr(vision.time, "monotonic", lambda: clock[0])
    reads = []

    class EarlyHints(httpcore.NetworkStream):
        def read(self, max_bytes, timeout=None):
            reads.append(timeout)
            clock[0] += 19.0  # each reply lands just inside a 20 s read timeout
            return b"HTTP/1.1 103 Early Hints\r\nLink: </x.css>\r\n\r\n"

        def write(self, buffer, timeout=None):
            pass

        def close(self):
            pass

        def get_extra_info(self, info):
            return None

    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", lambda self, *a, **k: EarlyHints())
    with pytest.raises(ValueError, match="too long"):
        vision._fetch("http://8.8.8.8/x.png", clock[0] + 60)
    assert len(reads) <= 4 and all(r <= 20 for r in reads)
    assert reads[-1] < 19  # the last read only got the time that was left


def test_deadline_bounds_every_partial_send(monkeypatch):
    """A peer that reads 16 KiB every 19 s: httpcore's own write loop would give each partial send the full
    20 s timeout; through the pinned backend every send gets only the time left (real SyncStream, fake socket)."""
    import socket

    import httpcore
    from httpcore._backends.sync import SyncStream

    from ds41_api import vision

    clock = [1000.0]
    monkeypatch.setattr(vision.time, "monotonic", lambda: clock[0])
    timeouts = []

    class SlowReader:
        def __init__(self):
            self.timeout, self.sent = None, 0

        def settimeout(self, value):
            self.timeout = value
            timeouts.append(value)

        def send(self, data):
            if self.sent:  # the first 16 KiB fit the empty send buffer; after that 16 KiB per 19 s
                if self.timeout < 19:
                    clock[0] += self.timeout
                    raise socket.timeout("timed out")
                clock[0] += 19
            n = min(len(data), 16384)
            self.sent += n
            return n

        def close(self):
            pass

    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", lambda self, *a, **k: SyncStream(SlowReader()))
    deadline = clock[0] + 60
    stream = vision._pinned_backend("8.8.8.8", ["8.8.8.8"], deadline).connect_tcp(host="8.8.8.8", port=80)
    with pytest.raises(httpcore.TimeoutException):
        stream.write(b"x" * 200_000, timeout=20)
    assert clock[0] <= deadline + 1e-9  # never past the deadline
    assert timeouts[-1] < 19  # the last send only got the time that was left


def test_urls_are_canonicalized_before_the_host_check():
    from ds41_api import vision

    c = vision._canonical_url
    assert c("http://8.8.8.8/猫.png") == ("http", "8.8.8.8", 80, "/%E7%8C%AB.png", "8.8.8.8")
    assert c("HTTPS://Bücher.example/a b.png?q=ä#frag") == (
        "https", "xn--bcher-kva.example", 443, "/a%20b.png?q=%C3%A4", "xn--bcher-kva.example")
    # IDNA 2008 (UTS #46, non-transitional), not Python's IDNA 2003 codec: these are different domains.
    assert c("https://faß.de/x.png")[1] == "xn--fa-hia.de"
    assert c("https://ς.gr/x.png")[1] == "xn--3xa.gr"
    assert c("http://[2606:4700:4700::1111]:8080/x")[1:] == ("2606:4700:4700::1111", 8080, "/x",
                                                            "[2606:4700:4700::1111]:8080")
    assert c("http://8.8.8.8/a;variant=red.png?x=1")[3] == "/a;variant=red.png?x=1"
    for bad, match in [("http://user:pw@images.example/x.png", "credentials"), ("http://images.example:0/x", "port"),
                       ("http://images.example:99999/x.png", "port"), ("http:///x.png", "no host"),
                       ("http://images.example/" + "a" * 9000, "longer"),
                       ("http://8.8.8.8/" + "\N{GRINNING FACE}" * 3000, "once encoded"),
                       ("http://[2606:4700:4700::1111%25eth0]/x", "zone"), ("http://[2606:4700:4700::1111%\u00e9]/x", "zone")]:
        with pytest.raises(ValueError, match=match):
            c(bad)


@pytest.mark.parametrize("url, target, host", [
    ("http://8.8.8.8/a;variant=red.png?x=1", b"/a;variant=red.png?x=1", b"8.8.8.8"),
    ("http://[2606:4700:4700::1111]:8080/x.png", b"/x.png", b"[2606:4700:4700::1111]:8080"),
    ("http://8.8.8.8/猫.png?x=猫", b"/%E7%8C%AB.png?x=%E7%8C%AB", b"8.8.8.8"),
])
def test_request_on_the_wire(monkeypatch, url, target, host):
    """The bytes httpcore writes: the full target (with ;parameters) and a valid Host (IPv6 in brackets)."""
    import httpcore

    from ds41_api import vision

    sent, dialed = [], []
    body = png(8, 8)

    class Server(httpcore.NetworkStream):
        def read(self, max_bytes, timeout=None):
            if not getattr(self, "done", False):
                self.done = True
                return b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n%b" % (len(body), body)
            return b""

        def write(self, buffer, timeout=None):
            sent.append(buffer)

        def close(self):
            pass

        def get_extra_info(self, info):
            return None

    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp",
                        lambda self, host, port, *a, **k: dialed.append((host, port)) or Server())
    assert vision._fetch(url, time.monotonic() + 5) == body
    request = b"".join(sent)
    assert request.startswith(b"GET " + target + b" HTTP/1.1\r\n")
    assert b"\r\nHost: " + host + b"\r\n" in request
    assert dialed[0][0] in ("8.8.8.8", "2606:4700:4700::1111")


def test_source_pixel_limit():
    from ds41_api import vision

    buffer = io.BytesIO()
    Image.new("1", (10_000, 7_000)).save(buffer, format="PNG")  # 70 MP, a few kB compressed
    assert len(buffer.getvalue()) < 1 << 20
    with pytest.raises(ValueError, match="at most"):
        vision.load_image(buffer.getvalue(), CFG)


@pytest.mark.parametrize("failure", [ValueError("image URL returned HTTP 404"), RuntimeError("unexpected")])
@pytest.mark.parametrize("endpoint", ["messages", "responses"])
def test_failed_image_frees_the_generation_slot(monkeypatch, endpoint, failure):
    """A request whose image fails (any exception) must not keep the only generation slot."""
    from fastapi.testclient import TestClient

    from ds41_api import vision
    from ds41_api.encoder import Encoder
    from ds41_api.fake_backend import FakeBackend
    from ds41_api.server import ServerConfig, create_app

    def broken(url, deadline):
        raise failure

    monkeypatch.setattr(vision, "_fetch", broken)
    tokenizer = ImageTokenizer()
    backend = FakeBackend(tokenizer=tokenizer)
    backend.add_script("ok")
    app = create_app(backend, Encoder(tokenizer, VisionConfig()), config=ServerConfig(queue_timeout=0.5))
    client = TestClient(app, raise_server_exceptions=False)
    url = "https://images.example/cat.png"
    if endpoint == "messages":
        r = client.post("/v1/messages", json={"model": "deepseek-v4.1-flash-1017", "max_tokens": 4, "messages": [
            {"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": url}},
                                         {"type": "text", "text": "hi"}]}]})
    else:
        r = client.post("/v1/responses", json={"model": "deepseek-v4.1-flash-1017", "max_output_tokens": 4, "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": url},
                                                            {"type": "input_text", "text": "hi"}]}]})
    assert r.status_code == (400 if isinstance(failure, ValueError) else 500)
    after = client.post("/v1/chat/completions", json={"model": "deepseek-v4.1-flash-1017", "max_tokens": 4,
                                                      "messages": [{"role": "user", "content": "hi"}]})
    assert after.status_code == 200, after.text


def test_malformed_anthropic_image_source_is_a_client_error():
    from fastapi.testclient import TestClient

    from ds41_api.encoder import Encoder
    from ds41_api.fake_backend import FakeBackend
    from ds41_api.server import ServerConfig, create_app

    tokenizer = ImageTokenizer()
    backend = FakeBackend(tokenizer=tokenizer)
    backend.add_script("ok")
    client = TestClient(create_app(backend, Encoder(tokenizer, VisionConfig()), config=ServerConfig(queue_timeout=0.5)),
                        raise_server_exceptions=False)
    body = {"model": "deepseek-v4.1-flash-1017", "max_tokens": 4, "messages": [
        {"role": "user", "content": [{"type": "image", "source": {"type": "base64", "data": 1}},
                                     {"type": "text", "text": "hi"}]}]}
    assert client.post("/v1/messages", json=body).status_code == 400
    assert client.post("/v1/messages/count_tokens", json=body).status_code == 400
    after = client.post("/v1/chat/completions", json={"model": "deepseek-v4.1-flash-1017", "max_tokens": 4,
                                                      "messages": [{"role": "user", "content": "hi"}]})
    assert after.status_code == 200, after.text
