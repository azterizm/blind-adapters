"""M3 parity gate and delivery variants (laptop side).

For each adapter: convert the PEFT adapter to MLX, compute the mean NLL of the teacher
answer on the 20 dev prompts with the MLX 4-bit base, with and without the adapter, and
compare with the Modal bf16 numbers (train.py parity_modal).

Gate (pre-registered R6): |MLX adapter - Modal adapter| <= 0.05, and the adapter lowers
NLL against base on both. Also records the FP16 and INT8 delivery records: size, NLL and
dev accuracy (greedy answer equals the teacher's yes/no/unclear).

    HF_HUB_OFFLINE=1 python -m eval.parity [task ...]
"""
from __future__ import annotations

import json
import os
import sys

import llm
from adapters.convert import convert, from_record

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AD = os.path.join(ROOT, "artifacts", "adapters")
TASKS = ["uk_unfair_dismissal", "uk_redundancy_payment"]
TOL = 0.05


def _mean(x):
    return sum(x) / len(x)


def _accuracy(dev: list[dict]) -> tuple[int, list[str]]:
    from mlx_lm import generate
    model, tok = llm.device_model()
    got = []
    for ex in dev:
        out = generate(model, tok, prompt=ex["input_ids"][: ex["n_prompt"]], max_tokens=500, verbose=False)
        try:
            got.append(str(llm._extract_json(out).get("answer", "")).strip().lower())
        except Exception:
            got.append("invalid")
    return sum(g == ex["answer"] for g, ex in zip(got, dev)), got


def run(task: str, accuracy: bool = True) -> dict:
    d = os.path.join(AD, task)
    info = convert(os.path.join(d, "peft"), d)
    dev = [json.loads(l) for l in open(os.path.join(d, "data", "dev.jsonl"))]
    modal = json.load(open(os.path.join(d, "parity_modal.json")))
    assert modal["ids"] == [e["id"] + ":" + e["step"] for e in dev]
    res = dict(task=task, n_dev=len(dev), params=info["params"], sizes=info["sizes"],
               modal_bf16=dict(adapter=_mean(modal["adapter"]), base=_mean(modal["base"])))
    llm.use_adapter(None)
    base = [llm.device_nll(e["input_ids"], e["n_prompt"]) for e in dev]
    res["mlx_4bit"] = dict(base=_mean(base))
    if accuracy:
        res["mlx_4bit"]["base_accuracy"] = _accuracy(dev)[0]
    for variant in ("fp16", "int8"):
        w, cfg = from_record(open(os.path.join(d, f"adapter.{variant}"), "rb").read())
        swap = llm.use_adapter(w, cfg)
        nll = [llm.device_nll(e["input_ids"], e["n_prompt"]) for e in dev]
        res["mlx_4bit"][variant] = dict(nll=_mean(nll), swap_s=swap)
        if accuracy:
            res["mlx_4bit"][variant]["accuracy"] = _accuracy(dev)[0]
    llm.use_adapter(None)
    a_mlx, a_modal = res["mlx_4bit"]["fp16"]["nll"], res["modal_bf16"]["adapter"]
    res["gate"] = dict(
        nll_gap=abs(a_mlx - a_modal),
        within_tol=abs(a_mlx - a_modal) <= TOL,
        lowers_nll_mlx=a_mlx < res["mlx_4bit"]["base"],
        lowers_nll_modal=a_modal < res["modal_bf16"]["base"])
    res["gate"]["passed"] = all(v for k, v in res["gate"].items() if k != "nll_gap")
    return res


def main(tasks: list[str]):
    out = {}
    for t in tasks:
        r = run(t)
        out[t] = r
        m, x = r["mlx_4bit"], r["modal_bf16"]
        print(f"{t}: params {r['params']:,}; record fp16 {r['sizes']['fp16'] / 1e6:.1f} MB, int8 {r['sizes']['int8'] / 1e6:.1f} MB")
        print(f"  mean NLL  Modal bf16: base {x['base']:.4f} adapter {x['adapter']:.4f}")
        print(f"            MLX 4-bit:  base {m['base']:.4f} fp16 {m['fp16']['nll']:.4f} int8 {m['int8']['nll']:.4f}")
        print(f"  dev accuracy ({r['n_dev']}): base {m.get('base_accuracy')} fp16 {m['fp16'].get('accuracy')} "
              f"int8 {m['int8'].get('accuracy')}; swap fp16 {m['fp16']['swap_s'] * 1e3:.0f} ms int8 {m['int8']['swap_s'] * 1e3:.0f} ms")
        g = r["gate"]
        print(f"  gate: gap {g['nll_gap']:.4f} (<= {TOL}: {g['within_tol']}), lowers NLL MLX {g['lowers_nll_mlx']} "
              f"Modal {g['lowers_nll_modal']} -> {'PASS' if g['passed'] else 'FAIL'}")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    if set(tasks) == set(TASKS):
        out["_gate_passed"] = all(out[t]["gate"]["passed"] for t in TASKS)
        json.dump(out, open(os.path.join(ROOT, "results", "parity.json"), "w"), indent=1)
        print(f"PARITY GATE (R6): {'PASS' if out['_gate_passed'] else 'FAIL'}")
    return out


if __name__ == "__main__":
    main(sys.argv[1:] or TASKS)
