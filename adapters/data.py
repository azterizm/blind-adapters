"""Tokenize teacher examples with the device's own tokenizer and chat template.

Training on Modal uses Qwen/Qwen3-8B, whose tokenizer_config.json changed after the MLX
conversion the device runs. To keep training and inference token-identical, prompts are
rendered here with the pinned MLX snapshot (enable_thinking=False, as on the device)
and shipped to Modal as token ids.

Per task, writes artifacts/adapters/<task>/data/{train,dev}.jsonl with
  {"id", "step", "answer", "input_ids", "n_prompt"}
The dev split is 20 examples (stratified by step) used for the parity gate and the
FP16/INT8 dev check; it is never trained on.

Run:  python -m adapters.data
"""
from __future__ import annotations

import json
import os
import random

from device.prompts import SYS_STEP

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEACHER = os.path.join(ROOT, "artifacts", "teacher")
OUT = os.path.join(ROOT, "artifacts", "adapters")
TASKS = ["uk_unfair_dismissal", "uk_redundancy_payment"]
MAX_LEN = 3072
N_DEV = 20


def tokenizer():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    import llm
    path = snapshot_download(llm.DEVICE_MODEL, revision=llm.DEVICE_REVISION, local_files_only=True)
    return AutoTokenizer.from_pretrained(path)


def encode(tok, prompt: str, target: str) -> tuple[list[int], int]:
    messages = [{"role": "system", "content": SYS_STEP}, {"role": "user", "content": prompt}]
    p = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    p_ids = tok(p, add_special_tokens=False)["input_ids"]
    t_ids = tok(target + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return p_ids + t_ids, len(p_ids)


def build() -> dict:
    tok = tokenizer()
    report = {}
    for task in TASKS:
        ex = [json.loads(l) for l in open(os.path.join(TEACHER, task, "examples.jsonl"))]
        rng = random.Random(task)
        by_matter = sorted({e["id"] for e in ex})
        rng.shuffle(by_matter)
        # dev = whole matters, so no dev matter text is trained on; stratify by step
        dev_ids, dev = set(), []
        need = {s: N_DEV // 3 + (1 if i < N_DEV % 3 else 0)
                for i, s in enumerate(sorted({e["step"] for e in ex}))}
        for mid in by_matter:
            rows = [e for e in ex if e["id"] == mid]
            if any(need.get(r["step"], 0) > 0 for r in rows) and len(dev) < N_DEV:
                dev_ids.add(mid)
                for r in rows:
                    if need.get(r["step"], 0) > 0 and len(dev) < N_DEV:
                        dev.append(r)
                        need[r["step"]] -= 1
        train = [e for e in ex if e["id"] not in dev_ids]
        d = os.path.join(OUT, task, "data")
        os.makedirs(d, exist_ok=True)
        stats = {}
        for split, rows in (("train", train), ("dev", dev)):
            kept, long = 0, 0
            with open(os.path.join(d, f"{split}.jsonl"), "w") as f:
                for e in rows:
                    ids, n_prompt = encode(tok, e["prompt"], e["target"])
                    if len(ids) > MAX_LEN:
                        long += 1
                        continue
                    f.write(json.dumps(dict(id=e["id"], step=e["step"], answer=e["answer"],
                                            input_ids=ids, n_prompt=n_prompt)) + "\n")
                    kept += 1
            stats[split] = dict(examples=kept, dropped_over_max_len=long)
        report[task] = stats
        print(f"{task}: {stats}")
    return report


if __name__ == "__main__":
    build()
