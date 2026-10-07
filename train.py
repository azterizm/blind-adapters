"""M3: train the two adapters on Modal (QLoRA on Qwen/Qwen3-8B), then measure the
Modal side of the parity gate (bf16 base, with and without the adapter).

Inputs are public: token ids of teacher examples (artifacts/adapters/<task>/data/),
built by `python -m adapters.data`. Outputs land in artifacts/adapters/<task>/.

    modal run train.py
"""
from __future__ import annotations

import json
import os
import time

import modal

BASE = "Qwen/Qwen3-8B"
REV = "b968826d9c46dd6066d109eabc6255188de91218"   # weights unchanged since 47719a2, the MLX source
TASKS = ["uk_unfair_dismissal", "uk_redundancy_payment"]
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
HP = dict(lr=1e-4, epochs=2, grad_accum=8, warmup=0.03, max_len=3072, seed=0)
GPU = "A100-40GB"
ROOT = os.path.dirname(os.path.abspath(__file__))

app = modal.App("blind-adapters-train")
vol = modal.Volume.from_name("blind-adapters-vol", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .run_commands("python -m pip install -U pip")
         .pip_install("torch==2.14.1", "transformers==5.17.0", "tokenizers==0.23.2", "peft==0.21.2",
                      "bitsandbytes==0.50.2", "accelerate==1.15.0", "safetensors==0.8.0",
                      "huggingface_hub[hf_transfer]==1.32.0")
         .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}))


def _versions() -> dict:
    import importlib.metadata as md
    return {p: md.version(p) for p in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")}


@app.function(image=image, volumes={"/vol": vol}, timeout=3600)
def download() -> str:
    from huggingface_hub import snapshot_download
    path = snapshot_download(BASE, revision=REV)
    vol.commit()
    return path


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=6 * 3600)
def train(task: str) -> dict:
    import math
    import random

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    vol.reload()
    data = [json.loads(l) for l in open(f"/vol/data/{task}/train.jsonl")]
    torch.manual_seed(HP["seed"])
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(BASE, revision=REV, quantization_config=bnb,
                                                 dtype=torch.bfloat16, device_map={"": 0},
                                                 attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=HP["lr"], weight_decay=0.0)
    updates = math.ceil(len(data) * HP["epochs"] / HP["grad_accum"])
    warm = max(1, int(HP["warmup"] * updates))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda u: min(1.0, (u + 1) / warm) * max(0.0, (updates - u) / max(1, updates - warm)))
    rng = random.Random(HP["seed"])
    log, t0, step, run = [], time.time(), 0, 0.0
    model.train()
    for epoch in range(HP["epochs"]):
        order = list(range(len(data)))
        rng.shuffle(order)
        for i, j in enumerate(order):
            ex = data[j]
            ids = torch.tensor([ex["input_ids"]], device="cuda")
            labels = ids.clone()
            labels[:, : ex["n_prompt"]] = -100
            loss = model(input_ids=ids, labels=labels).loss
            (loss / HP["grad_accum"]).backward()
            run += loss.item()
            step += 1
            if step % HP["grad_accum"] == 0 or (epoch == HP["epochs"] - 1 and i == len(order) - 1):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if step % 50 == 0:
                log.append(dict(step=step, epoch=epoch, loss=run / 50, s=round(time.time() - t0)))
                print(f"{task} step {step}/{len(data) * HP['epochs']} loss {run / 50:.4f} {time.time() - t0:.0f}s",
                      flush=True)
                run = 0.0
    out = f"/vol/out/{task}/peft"
    model.save_pretrained(out)
    info = dict(task=task, examples=len(data), updates=updates, seconds=round(time.time() - t0),
                gpu=GPU, base=BASE, revision=REV, lora=LORA, hp=HP, versions=_versions(), log=log)
    json.dump(info, open(f"/vol/out/{task}/train.json", "w"), indent=1)
    vol.commit()
    return info


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=3600)
def parity_modal(task: str) -> dict:
    """Mean NLL per target token of the teacher answer on the dev split, bf16 base with
    and without the adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    vol.reload()
    dev = [json.loads(l) for l in open(f"/vol/data/{task}/dev.jsonl")]
    base = AutoModelForCausalLM.from_pretrained(BASE, revision=REV, dtype=torch.bfloat16,
                                                device_map={"": 0}, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, f"/vol/out/{task}/peft").eval()

    @torch.no_grad()
    def nll(ex):
        ids = torch.tensor([ex["input_ids"]], device="cuda")
        logits = model(input_ids=ids).logits[0, :-1].float()
        tgt = ids[0, 1:]
        lp = torch.log_softmax(logits, -1).gather(1, tgt[:, None])[:, 0]
        return float(-lp[ex["n_prompt"] - 1:].mean())

    with_adapter = [nll(ex) for ex in dev]
    with model.disable_adapter():
        without = [nll(ex) for ex in dev]
    return dict(task=task, adapter=with_adapter, base=without, ids=[e["id"] + ":" + e["step"] for e in dev])


@app.local_entrypoint()
def main(tasks: str = ",".join(TASKS), skip_train: bool = False):
    tasks = tasks.split(",")
    with vol.batch_upload(force=True) as b:
        for t in tasks:
            for split in ("train", "dev"):
                b.put_file(os.path.join(ROOT, "artifacts", "adapters", t, "data", f"{split}.jsonl"),
                           f"/data/{t}/{split}.jsonl")
    print("base:", download.remote())
    if not skip_train:
        for info in train.map(tasks):
            print(f"trained {info['task']}: {info['examples']} examples, {info['seconds']}s, versions {info['versions']}")
    for t in tasks:
        d = os.path.join(ROOT, "artifacts", "adapters", t, "peft")
        os.makedirs(d, exist_ok=True)
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            with open(os.path.join(d, name), "wb") as f:
                for chunk in vol.read_file(f"/out/{t}/peft/{name}"):
                    f.write(chunk)
        with open(os.path.join(ROOT, "artifacts", "adapters", t, "train.json"), "wb") as f:
            for chunk in vol.read_file(f"/out/{t}/train.json"):
                f.write(chunk)
    for r in parity_modal.map(tasks):
        json.dump(r, open(os.path.join(ROOT, "artifacts", "adapters", r["task"], "parity_modal.json"), "w"), indent=1)
        a, b = sum(r["adapter"]) / len(r["adapter"]), sum(r["base"]) / len(r["base"])
        print(f"{r['task']}: bf16 mean NLL adapter {a:.4f} base {b:.4f}")
