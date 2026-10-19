"""Image prompt preparation (ds41_api.vision): span expansion, image sources and limits."""
import base64
import io

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
