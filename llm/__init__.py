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
_lora: dict = {}          # rank/scale the resident LoRA layers were built with


def device_model():
    """The resident model and tokenizer (pinned snapshot on disk, offline)."""
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    if DEVICE_MODEL not in _local:
        path = snapshot_download(DEVICE_MODEL, revision=DEVICE_REVISION, local_files_only=True)
        _local[DEVICE_MODEL] = load(path)
    return _local[DEVICE_MODEL]


def use_adapter(weights: dict | None, config: dict | None = None, adapter_path: str | None = None) -> float:
    """Swap the adapter in the resident model; None restores the base model. LoRA layers
    are added once; a swap only replaces their weights (lora_b = 0 is exactly the base
    model). Returns the swap time in seconds."""
    import time

    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers
    model, _ = device_model()
    if adapter_path is not None:
        from safetensors.numpy import load_file
        weights = load_file(os.path.join(adapter_path, "adapters.safetensors"))
        config = json.load(open(os.path.join(adapter_path, "adapter_config.json")))
    t = time.perf_counter()
    if not _lora:
        cfg = config or dict(num_layers=36, lora_parameters=dict(
            rank=16, scale=2.0, dropout=0.0, keys=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                                   "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]))
        linear_to_lora_layers(model, cfg["num_layers"], cfg["lora_parameters"])
        _lora.update(cfg["lora_parameters"])
        _lora["zeros"] = [(k, mx.zeros_like(v)) for k, v in tree_flatten(model.parameters())
                          if k.endswith(".lora_b")]
    if config is not None:
        lp = config["lora_parameters"]
        assert (lp["rank"], lp["scale"]) == (_lora["rank"], _lora["scale"]), "adapter shape differs from resident layers"
    if weights is None:
        model.load_weights(_lora["zeros"], strict=False)
    else:
        model.load_weights([(k, mx.array(np.asarray(v))) for k, v in weights.items()], strict=False)
    mx.eval(model.parameters())
    return time.perf_counter() - t


def device_chat_json(system: str, user: str, max_tokens: int = 600) -> dict:
    """Local, greedy (deterministic) generation on the sealed device with whatever
    adapter is resident (see use_adapter)."""
    from mlx_lm import generate
    model, tok = device_model()
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


def device_nll(input_ids: list[int], n_prompt: int) -> float:
    """Mean negative log-likelihood per target token (tokens after n_prompt)."""
    import mlx.core as mx
    model, _ = device_model()
    ids = mx.array([input_ids])
    logits = model(ids)[0, :-1].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt = ids[0, 1:]
    tok_lp = mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
    return float(-tok_lp[n_prompt - 1:].mean())
