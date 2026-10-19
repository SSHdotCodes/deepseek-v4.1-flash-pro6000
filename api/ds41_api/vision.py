"""Image prompts: the official inference/image_processor.py preprocessing without torch.

Each image placeholder token in the encoded prompt becomes the image's span

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

with image_token_id at every position, exactly like prepare_vl_inputs. The picture is resized and padded
with the same PIL calls as load_image, and its RGB bytes go to the engine, which normalizes them, runs the
ViT and aligner on the GPU and fills the IMAGE slots. The grid planning functions below are copied from
image_processor.py (MIT, DeepSeek) with `args` replaced by VisionConfig.
"""

import asyncio
import base64
import binascii
import concurrent.futures
import io
import ipaddress
import json
import math
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote, unquote_to_bytes, urlparse, urlsplit, urlunsplit

from .backend import ImagePayload

MAX_IMAGES = 16
MAX_IMAGE_BYTES = 32 << 20
# Decoded size limit, checked from the header before decoding: a small PNG can expand to gigabytes.
MAX_SOURCE_PIXELS = 64_000_000
# Wall-clock limits for remote images: one URL, and all of a request's images together.
REMOTE_TIMEOUT_S = 20.0
REQUEST_BUDGET_S = 60.0
# Images being decoded at once across all requests (each can take a few hundred MB while it is resized);
# a request waits up to PREPARE_WAIT_S for its turn.
MAX_PREPARING = 2
PREPARE_WAIT_S = 60.0
_PREPARING = threading.BoundedSemaphore(MAX_PREPARING)
_RESOLVER = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="ds41-image-dns")


@dataclass(frozen=True)
class VisionConfig:
    patch_size: int = 14
    downsample_ratio: int = 3
    max_n_token: int = 1024
    min_pixels: int = 544 * 544
    max_wh_ratio: Optional[float] = None
    image_token_id: int = 129264
    # Fetch http(s) image URLs (public addresses only). data: URLs and base64 always work.
    allow_remote: bool = True

    @classmethod
    def from_weights(cls, weights_dir: Path, allow_remote: bool = True) -> Optional["VisionConfig"]:
        """The checkpoint's inference/config.json, or None when it has no vision tower."""
        path = Path(weights_dir) / "inference" / "config.json"
        if not path.is_file():
            return None
        c = json.loads(path.read_text())
        if not c.get("vision_n_layers"):
            return None
        return cls(
            patch_size=c["vision_patch_size"],
            downsample_ratio=c["vision_downsample_ratio"],
            max_n_token=c["vision_max_n_token"],
            min_pixels=c["vision_min_pixels"],
            max_wh_ratio=c.get("vision_max_wh_ratio"),
            image_token_id=c["image_token_id"],
            allow_remote=allow_remote,
        )


@dataclass
class VisionImage:
    """One prepared image as the engine takes it (engine/include/ds41/engine.hpp, ImageSpan)."""

    start: int
    n_vit_h: int
    n_vit_w: int
    n_llm_h: int
    n_llm_w: int
    pixels: bytes  # RGB, n_vit_h * patch rows of n_vit_w * patch pixels

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "n_vit_h": self.n_vit_h,
            "n_vit_w": self.n_vit_w,
            "n_llm_h": self.n_llm_h,
            "n_llm_w": self.n_llm_w,
            "pixels": base64.b64encode(self.pixels).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VisionImage":
        return cls(
            start=d["start"],
            n_vit_h=d["n_vit_h"],
            n_vit_w=d["n_vit_w"],
            n_llm_h=d["n_llm_h"],
            n_llm_w=d["n_llm_w"],
            pixels=base64.b64decode(d["pixels"]),
        )


# -- grid planning: copied from image_processor.py --------------------------------------------------


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int):
    """Token grid the aligner produces from a patch grid of this pixel size."""
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio
    )


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    """Largest aspect-preserving pixel size whose token grid still fits in max_n_token."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:  # very tall: collapse to a single column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:  # very wide: collapse to a single row
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / patch_size) * patch_size, math.floor(width * beta / patch_size) * patch_size


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    """Shrink the pixel size until the image costs at most max_n_token LLM tokens."""
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(width: int, height: int, cfg: VisionConfig):
    """Resize plan for an image of the given original size; a pure function of its arguments."""
    p = cfg.patch_size
    if cfg.max_wh_ratio is not None and width > height * cfg.max_wh_ratio:
        width = height * cfg.max_wh_ratio
    if 0 < width * height < cfg.min_pixels:
        ratio = (cfg.min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(height, width, best_height, best_width, p, cfg.downsample_ratio, cfg.max_n_token)


# -- loading ---------------------------------------------------------------------------------------


def _remaining(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise ValueError("image download took too long")
    return left


def _public_addresses(host: str, port: int, deadline: float) -> List[str]:
    """The host's addresses from one lookup. Every one must be public: no loopback, private, link-local,
    reserved or IPv4-mapped private address. The connection then goes only to these addresses."""
    try:
        literal = ipaddress.ip_address(host)
        found = [str(literal)]
    except ValueError:
        lookup = _RESOLVER.submit(socket.getaddrinfo, host, port, 0, socket.SOCK_STREAM)
        try:
            infos = lookup.result(timeout=_remaining(deadline))
        except concurrent.futures.TimeoutError as exc:
            raise ValueError("image download took too long") from exc
        except OSError as exc:
            raise ValueError(f"image URL host does not resolve: {host!r}") from exc
        found = list(dict.fromkeys(info[4][0] for info in infos))
    if not found:
        raise ValueError(f"image URL host does not resolve: {host!r}")
    for text in found:
        address = ipaddress.ip_address(text)
        if address.version == 6 and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        if not address.is_global:
            raise ValueError(f"image URL host is not a public address: {host!r}")
    return found


def _canonical_url(url: str) -> Tuple[str, str, int]:
    """(ASCII URL, host, port): the hostname IDNA-encoded and the path and query percent-encoded, once, so
    the host that is checked is the host that is connected to. URLs with credentials are refused."""
    parts = urlsplit(url)
    if parts.username is not None or parts.password is not None:
        raise ValueError("image URL must not contain credentials")
    host = parts.hostname or ""
    try:
        port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise ValueError("image URL has an invalid port") from exc
    try:
        ipaddress.ip_address(host)
        netloc_host = f"[{host}]" if ":" in host else host
    except ValueError:
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise ValueError("image URL has an invalid hostname") from exc
        netloc_host = host
    if not host:
        raise ValueError("image URL has no host")
    netloc = netloc_host + (f":{parts.port}" if parts.port else "")
    path = quote(parts.path or "/", safe="/%:@!$&'()*+,;=-._~")
    query = quote(parts.query, safe="/%:@!$&'()*+,;=-._~?")
    return urlunsplit((parts.scheme.lower(), netloc, path, query, "")), host, port


def _pinned_backend(expected_host: str, addresses: List[str], deadline: float):
    """httpcore's socket backend, connecting only to the checked addresses: no second DNS lookup, so the
    host cannot resolve to a private address between the check and the connection. TLS still verifies the
    certificate for the URL's host (httpcore passes it for SNI and checking). Every connect, TLS handshake,
    read and write is bounded by the time left before the deadline, so no sequence of slow responses
    (informational 1xx replies, slow headers or chunk framing) can outlast it."""
    import httpcore

    def bounded(timeout):
        left = deadline - time.monotonic()
        if left <= 0:
            raise httpcore.ReadTimeout("image download took too long")
        return left if timeout is None else min(timeout, left)

    class DeadlineStream(httpcore.NetworkStream):
        # httpcore passes these arguments by keyword: the names follow its NetworkStream.
        def __init__(self, stream):
            self._stream = stream

        def read(self, max_bytes, timeout=None):
            return self._stream.read(max_bytes, bounded(timeout))

        def write(self, buffer, timeout=None):
            self._stream.write(buffer, bounded(timeout))

        def close(self):
            self._stream.close()

        def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            return DeadlineStream(self._stream.start_tls(ssl_context, server_hostname, bounded(timeout)))

        def get_extra_info(self, info):
            return self._stream.get_extra_info(info)

    class PinnedBackend(httpcore.SyncBackend):
        def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
            if host.strip("[]").lower() != expected_host.lower():
                raise httpcore.ConnectError(f"unexpected host {host!r}")
            error: Optional[Exception] = None
            for address in addresses:  # a refused or timed-out address falls through to the next one
                try:
                    stream = super().connect_tcp(address, port, bounded(timeout), local_address, socket_options)
                    return DeadlineStream(stream)
                except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                    error = exc
            raise error or httpcore.ConnectError("no address")

    return PinnedBackend()


def _fetch(url: str, deadline: float) -> bytes:
    """GET one public http(s) image URL: no redirects, no proxies from the environment, at most
    MAX_IMAGE_BYTES, finished before the deadline."""
    import httpcore

    url, host, port = _canonical_url(url)
    addresses = _public_addresses(host, port, deadline)
    step = min(REMOTE_TIMEOUT_S, _remaining(deadline))
    timeouts = {"connect": step, "read": step, "write": step, "pool": step}
    try:
        with httpcore.ConnectionPool(max_connections=1,
                                     network_backend=_pinned_backend(host, addresses, deadline)) as pool:
            with pool.stream("GET", url, headers={"Accept": "image/*", "User-Agent": "ds41-api"},
                             extensions={"timeout": timeouts}) as response:
                if 300 <= response.status < 400:
                    raise ValueError("image URL redirects; send the final URL or the image data")
                if response.status != 200:
                    raise ValueError(f"image URL returned HTTP {response.status}")
                data = bytearray()
                for chunk in response.iter_stream():
                    data += chunk
                    if len(data) > MAX_IMAGE_BYTES:
                        raise ValueError("image exceeds 32 MiB")
                return bytes(data)
    except httpcore.TimeoutException as exc:
        raise ValueError("image download took too long") from exc
    except (httpcore.NetworkError, httpcore.ProtocolError, httpcore.UnsupportedProtocol, ssl.SSLError, OSError,
            TypeError) as exc:
        raise ValueError(f"cannot download image URL ({type(exc).__name__})") from exc


def image_bytes(payload: ImagePayload, cfg: VisionConfig, deadline: Optional[float] = None) -> bytes:
    """Raw image file bytes from inline data, a data: URL, or (if allowed) an http(s) URL. Local paths are
    refused: the official encoder turns <image>path</image> text into image records."""
    if deadline is None:
        deadline = time.monotonic() + REMOTE_TIMEOUT_S
    if payload.data:
        data = payload.data
    elif payload.url:
        url = payload.url.strip()
        scheme = urlparse(url).scheme.lower()
        if scheme == "data":
            header, _, body = url.partition(",")
            try:
                data = base64.b64decode(body, validate=True) if ";base64" in header else unquote_to_bytes(body)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"undecodable image data URL: {exc}") from exc
        elif scheme in ("http", "https"):
            if not cfg.allow_remote:
                raise ValueError("remote image URLs are disabled; send the image as a data: URL or base64")
            data = _fetch(url, deadline)
        else:
            raise ValueError("image URL must be a data: or http(s) URL")
    else:
        raise ValueError("image without data or URL")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds 32 MiB")
    return data


def load_image(raw: bytes, cfg: VisionConfig) -> Tuple[bytes, int, int, int, int]:
    """image_processor.load_image up to the patch tensor: (RGB bytes, n_vit_h, n_vit_w, n_llm_h, n_llm_w)."""
    from PIL import Image, ImageOps

    p = cfg.patch_size
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.width * source.height > MAX_SOURCE_PIXELS:
                raise ValueError(f"image is {source.width}x{source.height}; at most {MAX_SOURCE_PIXELS:,} pixels")
            image = source.convert("RGB")
    except (OSError, Image.DecompressionBombError, ValueError) as exc:
        raise ValueError(f"cannot read image: {exc}") from exc
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(image.width, image.height, cfg)
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if cfg.max_wh_ratio is not None and image.width >= cfg.max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    pixels = image.tobytes()
    assert len(pixels) == best_height * best_width * 3
    return pixels, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def expand_prompt(
    token_ids: Sequence[int], images: Sequence[ImagePayload], cfg: Optional[VisionConfig]
) -> Tuple[List[int], List[VisionImage]]:
    """prepare_vl_inputs: expand each image placeholder token into its span. Returns the new token ids and
    the prepared images with their span start positions."""
    if cfg is None:
        raise ValueError("this model has no vision tower configured; images are not supported")
    if len(images) > MAX_IMAGES:
        raise ValueError(f"at most {MAX_IMAGES} images per request")
    placeholders = sum(1 for t in token_ids if t == cfg.image_token_id)
    if placeholders != len(images):
        raise ValueError(f"found {placeholders} image placeholders but got {len(images)} images")
    out: List[int] = []
    prepared: List[VisionImage] = []
    if not images:
        return list(token_ids), prepared
    if not _PREPARING.acquire(timeout=PREPARE_WAIT_S):
        raise ValueError("the server is busy preparing other images; try again")
    try:
        deadline = time.monotonic() + REQUEST_BUDGET_S
        pending = iter(images)
        for token in token_ids:
            if token != cfg.image_token_id:
                out.append(token)
                continue
            raw = image_bytes(next(pending), cfg, deadline)
            pixels, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(raw, cfg)
            del raw
            prepared.append(VisionImage(len(out), n_vit_h, n_vit_w, n_llm_h, n_llm_w, pixels))
            out += [cfg.image_token_id] * num_image_tokens(n_llm_h, n_llm_w)
    finally:
        _PREPARING.release()
    return out, prepared


async def expand_prompt_async(token_ids, images, cfg):
    """expand_prompt off the event loop: decoding, fetching and resizing take tens of milliseconds."""
    return await asyncio.to_thread(expand_prompt, list(token_ids), list(images), cfg)
