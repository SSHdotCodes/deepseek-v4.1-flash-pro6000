#!/usr/bin/env python3
"""Image prompts through the running service's API: answers, prefix reuse and the Anthropic format.

Greedy (temperature 0), thinking off. Checks:
  color      a solid color square: the answer names the color
  text       a rendered string: the answer contains it
  two        two images in one message, in order
  photo      the checkpoint's example photo (inference/examples/images), answer printed for review
  reuse      a follow-up turn on an image conversation reuses the cached prefix
  swap       same text, a different image of the same size: the cached image is not reused
  anthropic  the same color question through /v1/messages with a base64 source

  .venv/bin/python scripts/vision_e2e.py [--out results/vision-e2e.json]

The checkpoint, port and model ID default to config.local.json (or config.example.json).
"""
import argparse
import base64
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.dont_write_bytecode = True
import run  # noqa: E402


def png(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def square(color, size=512):
    return png(Image.new("RGB", (size, size), color))


def text_image(text):
    img = Image.new("RGB", (900, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 100), text, fill="black", font=ImageFont.load_default(size=72))
    return png(img)


def data_url(raw, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


class Client:
    def __init__(self, port, model):
        self.base, self.model = f"http://127.0.0.1:{port}", model

    def post(self, path, body):
        request = urllib.request.Request(self.base + path, json.dumps(body).encode(),
                                         {"content-type": "application/json", "x-api-key": "local"})
        started = time.time()
        with urllib.request.urlopen(request, timeout=900) as response:
            out = json.loads(response.read())
        out["_seconds"] = round(time.time() - started, 2)
        return out

    def chat(self, messages, max_tokens=64):
        out = self.post("/v1/chat/completions", {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                                                  "temperature": 0, "thinking": False})
        return out["choices"][0]["message"]["content"], out["usage"], out["_seconds"]


def user(text, *images):
    content = [{"type": "image_url", "image_url": {"url": data_url(i)}} for i in images]
    return {"role": "user", "content": content + [{"type": "text", "text": text}]}


def cached(usage):
    return (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


def main():
    cfg = run.configuration()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=Path(cfg["weights"]))
    ap.add_argument("--port", type=int, default=cfg["port"])
    ap.add_argument("--model", default=cfg["model_id"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    c = Client(args.port, args.model)
    results = []

    def record(name, ok, **info):
        results.append({"check": name, "ok": ok, **info})
        print(json.dumps(results[-1]), flush=True)

    answer, usage, s = c.chat([user("What is the color of this image? Answer with one word.", square((220, 20, 30)))])
    record("color", "red" in answer.lower(), answer=answer, prompt_tokens=usage["prompt_tokens"], seconds=s)

    answer, usage, s = c.chat([user("What text is written in the image? Reply with the text only.",
                                    text_image("DS41 VISION 7342"))])
    record("text", "7342" in answer and "vision" in answer.lower(), answer=answer, seconds=s)

    answer, usage, s = c.chat([user("Name the color of the first image, then of the second image, as two words.",
                                    square((20, 40, 220)), square((240, 220, 20)))])
    record("two", answer.lower().find("blue") >= 0 and answer.lower().find("yellow") > answer.lower().find("blue"),
           answer=answer, seconds=s)

    photo = next((p for p in sorted((args.weights / "inference" / "examples").rglob("*"))
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png")), None)
    if photo:
        answer, usage, s = c.chat([user("Describe this image in one sentence.", photo.read_bytes())], max_tokens=96)
        record("photo", bool(answer.strip()), image=photo.name, answer=answer, prompt_tokens=usage["prompt_tokens"],
               seconds=s)

    green = square((30, 170, 60), 640)
    first = [user("What is the color of this image? Answer with one word.", green)]
    answer1, usage1, s1 = c.chat(first)
    follow = first + [{"role": "assistant", "content": answer1},
                      {"role": "user", "content": "Is that color warm or cool? One word."}]
    answer2, usage2, s2 = c.chat(follow)
    record("reuse", cached(usage2) >= usage1["prompt_tokens"] - 8, first=answer1, follow_up=answer2,
           first_prompt_tokens=usage1["prompt_tokens"], cached_tokens=cached(usage2), seconds=[s1, s2])

    purple = square((140, 30, 190), 640)
    answer3, usage3, s3 = c.chat([user("What is the color of this image? Answer with one word.", purple)])
    record("swap", "purple" in answer3.lower() or "violet" in answer3.lower(), answer=answer3,
           cached_tokens=cached(usage3), prompt_tokens=usage3["prompt_tokens"], seconds=s3)

    out = c.post("/v1/messages", {"model": args.model, "max_tokens": 32, "temperature": 0,
                                  "messages": [{"role": "user", "content": [
                                      {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                                   "data": base64.b64encode(square((220, 20, 30))).decode()}},
                                      {"type": "text", "text": "What is the color of this image? Answer with one word."}]}]})
    text = "".join(b.get("text", "") for b in out["content"] if b.get("type") == "text")
    record("anthropic", "red" in text.lower(), answer=text, seconds=out["_seconds"])

    passed = sum(r["ok"] for r in results)
    print(f"{passed}/{len(results)} checks passed")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"schema": "ds41-vision-e2e-v1", "passed": passed, "checks": results},
                                       indent=1) + "\n")


if __name__ == "__main__":
    main()
