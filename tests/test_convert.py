import numpy as np

from adapters.convert import (from_record, mlx_config, mlx_to_peft, peft_to_mlx,
                              to_record)


def _peft(rng, r=4, d_in=12, d_out=20):
    return {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": rng.normal(size=(r, d_in)).astype(np.float32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": rng.normal(size=(d_out, r)).astype(np.float32),
    }


def test_roundtrip_keys_and_values():
    p = _peft(np.random.default_rng(0))
    back = mlx_to_peft(peft_to_mlx(p))
    assert back.keys() == p.keys()
    assert all(np.array_equal(back[k], p[k]) for k in p)


def test_same_delta_as_peft():
    rng = np.random.default_rng(1)
    p = _peft(rng)
    cfg = mlx_config({"r": 4, "lora_alpha": 8})
    m = peft_to_mlx(p)
    x = rng.normal(size=(3, 12))
    A = p["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    B = p["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    peft_delta = x @ A.T @ B.T * (8 / 4)
    pre = "model.layers.0.self_attn.q_proj."
    mlx_delta = (x @ m[pre + "lora_a"]) @ m[pre + "lora_b"] * cfg["lora_parameters"]["scale"]
    assert np.allclose(peft_delta, mlx_delta)


def test_records_decode():
    m = peft_to_mlx(_peft(np.random.default_rng(2)))
    cfg = mlx_config({"r": 4, "lora_alpha": 8})
    for variant, tol in (("fp16", 1e-3), ("int8", 2e-2)):
        w, c = from_record(to_record(m, cfg, variant))
        assert c == cfg
        for k in m:
            rel = np.abs(w[k].astype(np.float32) - m[k]).max() / np.abs(m[k]).max()
            assert rel < tol, (variant, k, rel)
