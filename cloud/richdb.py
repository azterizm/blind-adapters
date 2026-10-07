# from blind-counsel 8b1bf8f cloud/richdb.py
"""Cloud-side: pack the RICH public database for PIR.

Rows hold two kinds of public entry, keyed by public signatures:
  text:<coord>                          verbatim statute (every section of ERA 1996)
  explain:/relation:/procedure:<...>    precomputed reasoning (cloud/reason.py)

Also published (public, synced to every device in full, so syncing leaks nothing):
  catalogue.json   key -> {row, n_rows, nbytes, sha256}
  tasks.json       task -> {description, entry_keys}   (what a matter of that type needs)
  meta.json        {N, L, fetch_budget}  -- every matter issues exactly fetch_budget
                   PIR queries, whatever the task, and even when the device refuses.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from cloud import reason
from cloud.precompute import CELL, render_sections

ART = reason.ART


def build():
    reasoning = reason.build()
    texts = render_sections(reason.ACT)

    blobs: list[tuple[str, bytes]] = [(f"text:{c}", t.encode("utf-8")) for c, t in sorted(texts.items())]
    blobs += [(e["key"], json.dumps(e, ensure_ascii=False, sort_keys=True).encode("utf-8")) for e in reasoning]

    catalogue, rows = {}, []
    for key, raw in blobs:
        n = max(1, -(-len(raw) // CELL))
        catalogue[key] = dict(row=len(rows), n_rows=n, nbytes=len(raw),
                              sha256=hashlib.sha256(raw).hexdigest())
        rows += [raw[i * CELL:(i + 1) * CELL].ljust(CELL, b"\x00") for i in range(n)]
    D = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(len(rows), CELL)

    tasks = {}
    for task, spec in reason.TASKS.items():
        provs = {reason.P + s for s in spec["provisions"]}
        keys = [f"procedure:{task}"]
        keys += [f"explain:{c}" for c in sorted(provs)]
        keys += [e["key"] for e in reasoning if e["kind"] == "calc" and e["coordinate"] in provs]
        keys += [e["key"] for e in reasoning
                 if e["kind"] == "relation" and (e["a"] in provs or e["b"] in provs)]
        keys += [f"text:{c}" for c in sorted(provs)]
        tasks[task] = dict(description=spec["description"], entry_keys=keys,
                           rows_needed=sum(catalogue[k]["n_rows"] for k in keys))

    budget = -(-max(t["rows_needed"] for t in tasks.values()) // 8) * 8 + 8   # round up + margin
    os.makedirs(ART, exist_ok=True)
    np.save(os.path.join(ART, "pir_db.npy"), D)
    json.dump(catalogue, open(os.path.join(ART, "catalogue.json"), "w"), indent=0)
    json.dump(tasks, open(os.path.join(ART, "tasks.json"), "w"), indent=1)
    json.dump(dict(N=len(rows), L=CELL, fetch_budget=budget, n_entries=len(catalogue),
                   n_reasoning=len(reasoning)), open(os.path.join(ART, "meta.json"), "w"), indent=1)

    kinds = {}
    for e in reasoning:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(f"rich PIR DB: {len(rows)} rows x {CELL} B ({D.nbytes/1e6:.1f} MB); "
          f"{len(texts)} statute texts + {len(reasoning)} reasoning entries {kinds}")
    for t, v in tasks.items():
        print(f"  task {t}: {len(v['entry_keys'])} entries, {v['rows_needed']} rows")
    print(f"  fixed fetch budget per matter: {budget} PIR queries")


if __name__ == "__main__":
    build()
