# from blind-counsel 8b1bf8f llm/__init__.py
"""Model access for the two roles, returning parsed JSON.

CLOUD (offline enrichment of PUBLIC law): the user's router to Google AI Studio
  (gemini-3.8-flash-high). Only public statute text is ever sent here -- that is the
  whole point of the split: the frontier model reasons over public law in advance.

DEVICE (applies that reasoning to the PRIVATE client account): runs on the laptop
  itself (Qwen3-8B, 4-bit MLX, pinned revision). Nothing private is sent anywhere --
  not to the router, and not to any hosted GPU, which would only displace trust.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

ROUTER_URL = "http://localhost:8317/v1/chat/completions"
CLOUD_MODEL = "gemini-3.8-flash-high"
DEVICE_MODEL = "mlx-community/Qwen3-8B-4bit"
DEVICE_REVISION = "545dc4251c05440727734bcd94334791f6ab0192"
_KEY_FILE = os.path.join(os.path.dirname(__file__), "..", ".router_key")


def _router_key() -> str:
    return os.environ.get("LLM_ROUTER_KEY") or open(_KEY_FILE).read().strip()


def _extract_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in output")
    return json.loads(text[start:end + 1])


def _router_chat(messages: list[dict], temperature: float = 0.0) -> str:
    body = json.dumps({"model": CLOUD_MODEL, "messages": messages,
                       "temperature": temperature}).encode()
    req = urllib.request.Request(ROUTER_URL, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_router_key()}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def cloud_chat_json(system: str, user: str) -> dict:
    """Offline enrichment call (PUBLIC inputs only). Retries once on bad JSON."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    out = ""
    for _ in range(3):
        out = _router_chat(messages)
        try:
            return _extract_json(out)
        except Exception:
            messages = messages[:2] + [
                {"role": "assistant", "content": out},
                {"role": "user", "content": "Reply with ONLY the JSON object."}]
    raise ValueError(f"router did not return JSON: {out[:200]!r}")


# ---------------------------------------------------------------- device (local)
_local: dict[str, tuple] = {}


def device_chat_json(system: str, user: str, max_tokens: int = 600) -> dict:
    """Local, greedy (deterministic) generation on the sealed device. Offline:
    loads only the pinned snapshot already on disk."""
    from huggingface_hub import snapshot_download
    from mlx_lm import generate, load
    if DEVICE_MODEL not in _local:
        path = snapshot_download(DEVICE_MODEL, revision=DEVICE_REVISION, local_files_only=True)
        _local[DEVICE_MODEL] = load(path)
    model, tok = _local[DEVICE_MODEL]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    out = ""
    for _ in range(2):
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False)
        out = generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
        try:
            return _extract_json(out)
        except Exception:
            messages = messages[:2] + [
                {"role": "assistant", "content": out},
                {"role": "user", "content": "Reply with ONLY the JSON object."}]
    raise ValueError(f"device model did not return JSON: {out[:200]!r}")
