"""PEFT LoRA adapter -> MLX adapter, and the delivery variants (FP16, INT8).

PEFT stores lora_A [r, in] and lora_B [out, r] and applies x A^T B^T * alpha/r.
mlx-lm stores lora_a [in, r] and lora_b [r, out] and applies (x a) b * scale.
So lora_a = A^T, lora_b = B^T and scale = alpha / r.

A delivery record is the serialized adapter the PIR library serves:
  fp16  safetensors of lora_a / lora_b in float16
  int8  safetensors of int8 tensors plus a float16 scale per output column
        (symmetric, per column of lora_a and of lora_b); the device dequantizes.

Run:  python -m adapters.convert <peft_dir> <out_dir>
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
from safetensors.numpy import load as st_load
from safetensors.numpy import load_file, save, save_file

KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
_PEFT = re.compile(r"^base_model\.model\.(model\.layers\.\d+\..+)\.lora_([AB])\.weight$")


def peft_to_mlx(peft: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for k, v in peft.items():
        m = _PEFT.match(k)
        if not m:
            raise ValueError(f"unexpected PEFT key {k}")
        out[f"{m.group(1)}.lora_{m.group(2).lower()}"] = np.ascontiguousarray(v.T)
    return out


def mlx_to_peft(mlx: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for k, v in mlx.items():
        base, leaf = k.rsplit(".", 1)
        out[f"base_model.model.{base}.lora_{leaf[-1].upper()}.weight"] = np.ascontiguousarray(v.T)
    return out


def mlx_config(peft_cfg: dict, num_layers: int = 36) -> dict:
    r, alpha = peft_cfg["r"], peft_cfg["lora_alpha"]
    return dict(fine_tune_type="lora", num_layers=num_layers,
                lora_parameters=dict(rank=r, scale=alpha / r, dropout=0.0, keys=KEYS))


# ------------------------------------------------------------------ delivery variants
def to_record(weights: dict[str, np.ndarray], config: dict, variant: str) -> bytes:
    meta = {"config": json.dumps(config), "variant": variant}
    if variant == "fp16":
        return save({k: v.astype(np.float16) for k, v in weights.items()}, metadata=meta)
    if variant == "int8":
        t = {}
        for k, v in weights.items():
            v = v.astype(np.float32)
            s = np.abs(v).max(axis=0) / 127.0
            s[s == 0] = 1.0
            t[k] = np.clip(np.rint(v / s), -127, 127).astype(np.int8)
            t[k + ".scale"] = s.astype(np.float16)
        return save(t, metadata=meta)
    raise ValueError(variant)


def from_record(raw: bytes) -> tuple[dict[str, np.ndarray], dict]:
    """Delivery record -> (float16 MLX weights, adapter config)."""
    n = int.from_bytes(raw[:8], "little")
    meta = json.loads(raw[8:8 + n]).get("__metadata__", {})
    t = st_load(raw)
    if meta["variant"] == "fp16":
        w = t
    else:
        w = {k: (t[k].astype(np.float32) * t[k + ".scale"].astype(np.float32)).astype(np.float16)
             for k in t if not k.endswith(".scale")}
    return w, json.loads(meta["config"])


def write_mlx_dir(weights: dict[str, np.ndarray], config: dict, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    save_file({k: v.astype(np.float16) for k, v in weights.items()}, os.path.join(out, "adapters.safetensors"))
    json.dump(config, open(os.path.join(out, "adapter_config.json"), "w"), indent=1)


def convert(peft_dir: str, out_dir: str) -> dict:
    peft = load_file(os.path.join(peft_dir, "adapter_model.safetensors"))
    cfg = json.load(open(os.path.join(peft_dir, "adapter_config.json")))
    w = peft_to_mlx({k: v.astype(np.float32) for k, v in peft.items()})
    config = mlx_config(cfg)
    write_mlx_dir(w, config, os.path.join(out_dir, "mlx"))
    sizes = {}
    for variant in ("fp16", "int8"):
        rec = to_record(w, config, variant)
        open(os.path.join(out_dir, f"adapter.{variant}"), "wb").write(rec)
        sizes[variant] = len(rec)
    n_params = sum(v.size for v in w.values())
    info = dict(params=int(n_params), tensors=len(w), sizes=sizes, config=config)
    json.dump(info, open(os.path.join(out_dir, "convert.json"), "w"), indent=1)
    return info


if __name__ == "__main__":
    print(json.dumps(convert(sys.argv[1], sys.argv[2]), indent=1))
