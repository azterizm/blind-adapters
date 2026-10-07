# from blind-counsel 8b1bf8f eval/final.py
"""Evaluation: arms B, T, A, AT (and the TEACHER reference row) on the dev set or the
held-out set, scored against eval/heldout_key.json, with the pre-registered rule.

    HF_HUB_OFFLINE=1 python -m eval.final --split dev
    HF_HUB_OFFLINE=1 python -m eval.final --split held_out --teacher     # once, after freezing

Held-out runs refuse to start if the key, the question list or the held-out matters
differ from the pre-registration commit, or if a held-out result already exists.

Routing and fact extraction run once per matter on base weights. Arms on in-domain
matters use the key's task, so routing does not affect the adapter rule; routing is
scored on its own. Out-of-scope matters run the product flow with the router's task.

Private outputs (memos, full results) go to artifacts/device_out/<split>/ and never leave
the laptop. A summary without client text goes to results/.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import time

import numpy as np

from device.reasoner import ARMS, RichCloud, SealedDevice
from eval.matters import DEV, HELD_OUT
from pir.adapters import AdapterFetcher, IntegrityError, load_library

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PREREG = "f7857640cd8ea539454c2751d23411e762774f04"
KEY = json.load(open(os.path.join(ROOT, "eval", "heldout_key.json")))
EVAL_N, EVAL_C = 2, 1024        # adapter library in the evaluation: the 2 real adapters
CALC_FIELDS = {"qualifying_service", "time_limit", "compensation_cap", "relevant_date", "payment_amount"}
VALUE_FIELDS = {"compensation_cap", "payment_amount", "relevant_date"}
META = {"task", "split", "novel_fields"}


def check_prereg():
    """The key, the question list and the held-out matters must equal the commit."""
    r = subprocess.run(["git", "diff", "--quiet", PREREG, "--", "eval/heldout_key.json", "cloud/questions.py"],
                       cwd=ROOT)
    old = subprocess.run(["git", "show", f"{PREREG}:eval/matters.py"], cwd=ROOT, capture_output=True, text=True).stdout
    held = next(ast.literal_eval(n.value) for n in ast.walk(ast.parse(old))
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "HELD_OUT")
    if r.returncode != 0 or held != HELD_OUT:
        raise SystemExit("held-out key, question list or matters differ from the pre-registration commit")


def secrets_of(narrative: str) -> list[str]:
    toks = re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*|£[\d,]+|\d{1,2} [A-Z][a-z]+ \d{4}|\d{4}", narrative)
    return sorted({t for t in toks if len(t) >= 4})


def score(key: dict, r: dict) -> list[dict]:
    """One row per scored field: expected, got, correct, and whether a wrong answer had
    nonetheless passed the grounding or computation gate."""
    steps = {s["step_id"]: s for s in r.get("steps", [])}
    novel = set(key.get("novel_fields", []))
    rows = []
    for field, exp in key.items():
        if field in META:
            continue
        s = steps.get(field)
        got = None if s is None else (s.get("value") if field in VALUE_FIELDS else s.get("answer"))
        if isinstance(exp, (int, float)) and isinstance(got, (int, float)):
            ok = abs(float(got) - exp) < 0.5
        else:
            ok = got == exp
        rows.append(dict(field=field, expected=exp, got=got, correct=ok, novel=field in novel,
                         kind="calc" if field in CALC_FIELDS else "model",
                         gated=(not ok) and s is not None and s["status"] in ("computed", "grounded"),
                         status=s["status"] if s else "missing"))
    return rows


def memo(narrative_id: str, r: dict) -> str:
    out = [f"# {narrative_id}, arm {r['arm']}: {r['task']}", f"Status: {r['verdict']}", ""]
    for s in r.get("steps", []):
        val = f" ({s['value']})" if s.get("value") not in (None, "") else ""
        out += [f"## {s['step_id']}: {s['answer'].upper()}{val} [{s['status']}]", s["reasoning"]]
        out += [f'  > "{e}"' for e in s.get("evidence", [])]
        out += [f"  authority: {', '.join(s.get('authority') or s['step_authority'])}", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "held_out"], default="dev")
    ap.add_argument("--arms", default="B,T,A,AT")
    ap.add_argument("--teacher", action="store_true", help="add the TEACHER reference row (cloud model)")
    ap.add_argument("--only", default="", help="comma-separated matter ids (dev only)")
    a = ap.parse_args()
    arms = a.arms.split(",") + (["TEACHER"] if a.teacher else [])
    out_dir = os.path.join(ROOT, "artifacts", "device_out", a.split)
    summary_path = os.path.join(ROOT, "results", f"{a.split}_results.json")
    matters = HELD_OUT if a.split == "held_out" else DEV
    if a.split == "held_out":
        check_prereg()
        if os.path.exists(summary_path):
            raise SystemExit(f"{summary_path} exists: the held-out set runs once")
    elif a.only:
        matters = {k: v for k, v in matters.items() if k in a.only.split(",")}
    os.makedirs(out_dir, exist_ok=True)
    keys = KEY[a.split]

    text_cloud = RichCloud()
    acloud, manifest, lay = load_library(EVAL_N, EVAL_C)
    t = time.perf_counter()
    H = acloud.hint()
    hint_s = time.perf_counter() - t
    lib = json.load(open(os.path.join(ROOT, "artifacts", "library", "manifest.json")))
    frozen = dict(code=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                  dirty=bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()),
                  library=lib, layout=lay, hint_compute_s=hint_s)
    print(f"adapter library N={EVAL_N} c={EVAL_C}: {lib['variant']}, rows {lay['m']}, hint {lay['hint'] / 1e6:.0f} MB "
          f"({hint_s:.0f}s), per fetch up {lay['upload'] / 1e6:.1f} MB down {lay['download'] / 1e6:.0f} MB")

    report, runs, routing = {}, {}, {}
    for mid, narrative in matters.items():
        key = keys[mid]
        fetcher = AdapterFetcher(acloud.seed, H, lay, manifest, rng_seed=int(hashlib.sha256(mid.encode()).hexdigest()[:8], 16))
        dev = SealedDevice(text_cloud, acloud, fetcher, narrative,
                           seed=int(hashlib.sha256(mid.encode()).hexdigest()[8:16], 16))
        t = time.perf_counter()
        routed = dev.select_task()
        route_s = time.perf_counter() - t
        routing[mid] = dict(expected=key["task"], got=routed, correct=routed == key["task"], seconds=route_s)
        task = key["task"] if key["task"] else routed
        facts = dev.extract_facts(task) if task else None
        print(f"\n[{a.split}] {mid}: routed {routed} (key {key['task']}) {route_s:.0f}s; run as {task}", flush=True)
        for arm in arms:
            if arm == "TEACHER" and not key["task"]:
                continue
            n0, a0 = len(text_cloud.views), len(acloud.views)
            try:
                r = dev.advise(arm, task=task, routed=False, facts_cache=facts)
                r["adapter_fetch_ok"] = True
            except IntegrityError as e:
                r = dict(arm=arm, task=task, verdict=f"REFUSE: {e}", steps=[], adapter_fetch_ok=False,
                         router_calls=0, seconds=0.0, swap_s=None, fetched={})
            tv, av = text_cloud.views[n0:], acloud.views[a0:]
            blob = b"".join(v.tobytes() for v in tv + av)
            r["leaked"] = [s for s in secrets_of(narrative) if s.encode() in blob]
            r["n_secrets"] = len(secrets_of(narrative))
            r["text_queries"] = sum(v.shape[0] for v in tv)
            r["adapter_queries"] = sum(v.shape[0] for v in av)
            rows = score(key, r) if key["task"] else []
            runs[(mid, arm)] = r
            report.setdefault(mid, dict(split=key.get("split", "dev"), arms={}))["arms"][arm] = rows
            open(os.path.join(out_dir, f"{mid}.{arm}.md"), "w").write(memo(mid, r))
            ok = sum(x["correct"] for x in rows)
            print(f"  {arm:<7} {ok}/{len(rows)} | {r['verdict'][:40]:<40} | {r['seconds']:.0f}s | swap "
                  f"{(r['swap_s'] or 0) * 1e3:.0f}ms | q {r['text_queries']}+{r['adapter_queries']} | "
                  f"leaked {len(r['leaked'])}/{r['n_secrets']} | router {r['router_calls']}", flush=True)
            for x in rows:
                if not x["correct"]:
                    print(f"     XX {x['field']:<24} expected {x['expected']!s:<10} got {x['got']!s:<10} {x['status']}")

    # ---------------------------------------------------------------- summary
    def rows_of(arm, pred):
        return [x for mid, v in report.items() if pred(mid, v) for x in v["arms"].get(arm, [])]

    def novel_right(arm):
        return sum(all(x["correct"] for x in v["arms"].get(arm, []) if x["novel"])
                   for mid, v in report.items() if v["split"] == "novel" and arm in v["arms"])

    table = {}
    for arm in arms:
        allr = rows_of(arm, lambda m, v: True)
        rs = [r for (m, ar), r in runs.items() if ar == arm]
        table[arm] = dict(
            correct=sum(x["correct"] for x in allr), fields=len(allr),
            wrong=sum(not x["correct"] for x in allr), wrong_past_gate=sum(x["gated"] for x in allr),
            refused=sum(r["verdict"].startswith("REFUSE") for r in rs),
            in_scope=sum(x["correct"] for x in rows_of(arm, lambda m, v: v["split"] == "in_scope")),
            novel_fields=sum(x["correct"] for x in rows_of(arm, lambda m, v: v["split"] == "novel") if x["novel"]),
            novel_matters=novel_right(arm),
            calc=sum(x["correct"] for x in allr if x["kind"] == "calc"),
            model=sum(x["correct"] for x in allr if x["kind"] == "model"),
            mean_seconds=float(np.mean([r["seconds"] for r in rs])) if rs else None,
            mean_swap_ms=float(np.mean([r["swap_s"] * 1e3 for r in rs if r.get("swap_s")])) if any(r.get("swap_s") for r in rs) else None,
            leaked=sum(len(r["leaked"]) for r in rs))
    print("\n" + "=" * 78)
    print(f"{'arm':<8}{'correct':>9}{'wrong':>7}{'past gate':>11}{'refused':>9}{'in-scope':>10}{'novel f':>9}"
          f"{'novel m':>9}{'secs':>7}{'swap ms':>9}{'leaked':>8}")
    for arm, v in table.items():
        print(f"{arm:<8}{v['correct']:>5}/{v['fields']:<3}{v['wrong']:>7}{v['wrong_past_gate']:>11}{v['refused']:>9}"
              f"{v['in_scope']:>10}{v['novel_fields']:>9}{v['novel_matters']:>9}"
              f"{v['mean_seconds'] or 0:>7.0f}{v['mean_swap_ms'] or 0:>9.0f}{v['leaked']:>8}")
    r_ok = sum(v["correct"] for v in routing.values())
    print(f"routing: {r_ok}/{len(routing)}  misses: {[m for m, v in routing.items() if not v['correct']]}")

    rule, verdicts = {}, {}
    if a.split == "held_out" and all(x in table for x in ("B", "T", "A")):
        B, T, A = table["B"], table["T"], table["A"]
        product = [r for (m, ar), r in runs.items() if ar != "TEACHER"]
        counts = {(r["text_queries"], r["adapter_queries"]) for r in product}
        parity = json.load(open(os.path.join(ROOT, "results", "parity.json")))
        rule = {
            f"R1 in-scope fields: A {A['in_scope']} >= T {T['in_scope']} and >= B {B['in_scope']}":
                A["in_scope"] >= T["in_scope"] and A["in_scope"] >= B["in_scope"],
            f"R2 novel matters A - B = {A['novel_matters']} - {B['novel_matters']} >= 2, novel fields A {A['novel_fields']} >= T {T['novel_fields']}":
                A["novel_matters"] - B["novel_matters"] >= 2 and A["novel_fields"] >= T["novel_fields"],
            f"R3 wrong past the gate: A {A['wrong_past_gate']} <= T {T['wrong_past_gate']}":
                A["wrong_past_gate"] <= T["wrong_past_gate"],
            f"R4 leaked {sum(len(r['leaked']) for r in product)}, router calls {sum(r['router_calls'] for r in product)}, "
            f"queries per matter {sorted(counts)} (need [(56, {EVAL_C})])":
                all(not r["leaked"] and r["router_calls"] == 0 for r in product) and counts == {(56, EVAL_C)},
            f"R5 adapter fetches byte-exact with sha256 match: {sum(r['adapter_fetch_ok'] for r in product)}/{len(product)}":
                all(r["adapter_fetch_ok"] for r in product),
            f"R6 parity gate: {'passed' if parity['_gate_passed'] else 'failed'}": parity["_gate_passed"],
        }
        verdicts = dict(adapter="PASS" if all(rule.values()) else "FAIL",
                        routing="PASS" if r_ok == len(routing) else "FAIL")
        print("\nPRE-REGISTERED RULE (held-out, commit " + PREREG[:8] + ")")
        for k, v in rule.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        print(f"ADAPTER VERDICT: {verdicts['adapter']}")
        print(f"ROUTING VERDICT: {verdicts['routing']} ({r_ok}/{len(routing)})")
    print("=" * 78)

    def clean(r):        # summary without client text
        return dict(arm=r["arm"], task=r["task"], verdict=r["verdict"], seconds=r["seconds"], swap_s=r.get("swap_s"),
                    text_queries=r["text_queries"], adapter_queries=r["adapter_queries"], leaked=len(r["leaked"]),
                    n_secrets=r["n_secrets"], router_calls=r["router_calls"], adapter_fetch_ok=r["adapter_fetch_ok"],
                    fetched=r.get("fetched", {}))
    full = dict(split=a.split, frozen=frozen, routing=routing, table=table, rule=rule, verdicts=verdicts,
                report=report, runs={f"{m}|{ar}": dict(clean(r), steps=r.get("steps", [])) for (m, ar), r in runs.items()})
    json.dump(full, open(os.path.join(out_dir, "results.json"), "w"), indent=1, default=str)
    summary = dict(full, runs={f"{m}|{ar}": clean(r) for (m, ar), r in runs.items()})
    if not a.only:
        json.dump(summary, open(summary_path, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
