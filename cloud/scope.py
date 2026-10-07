"""M5a, cloud side: a public scope note per task, for the device's router.

blind-counsel routed a still-employed pregnancy matter to unfair dismissal (task
selection 7 of 8). The router saw only a one-line task description. Here the cloud
model writes, from public material only (the task description and its public question
list), what a matter must contain to be in scope and what is out of scope. The note is
public and synced in full by every device, like the task index.

Run:  python -m cloud.scope        (cached under artifacts/cloud_rich/reasoning/)
"""
from __future__ import annotations

import json
import os

from cloud.reason import ART, CACHE, SYSTEM, _cached
from llm import CLOUD_MODEL, cloud_chat_json


def scope_note(task: str, spec: dict, others: dict) -> dict:
    steps = "\n".join(f"  - {s['question']}" for s in spec["steps"])
    other = "\n".join(f"  - {t}: {v['description']}" for t, v in others.items())
    user = (f"Task: {spec['description']}\n\nThe task answers these questions:\n{steps}\n\n"
            f"The other tasks the practice covers:\n{other}\n\n"
            "Write a scope note for a triage step that reads a client's account and decides whether this "
            "task applies. Whether the facts amount to a dismissal, including a resignation claimed as a "
            "constructive dismissal, is decided inside the task, not at triage. Return JSON:\n"
            '{"in_scope": ["conditions a matter must meet, in plain words"], '
            '"out_of_scope": ["kinds of matter that look similar but are out of scope"], '
            '"one_line": "one sentence a triage step can apply"}')
    raw = cloud_chat_json(SYSTEM, user)
    return dict(key=f"scope:{task}", kind="scope", task=task,
                in_scope=[x for x in raw.get("in_scope", []) if isinstance(x, str)],
                out_of_scope=[x for x in raw.get("out_of_scope", []) if isinstance(x, str)],
                one_line=str(raw.get("one_line", "")), model=CLOUD_MODEL)


def build() -> dict:
    qs = json.load(open(os.path.join(ART, "questions.json")))
    out = {}
    for task, spec in qs.items():
        others = {t: v for t, v in qs.items() if t != task}
        out[task] = _cached(f"scope:{task}", lambda task=task, spec=spec, others=others:
                            scope_note(task, spec, others))
    json.dump(out, open(os.path.join(ART, "scope.json"), "w"), indent=1, ensure_ascii=False)
    return out


if __name__ == "__main__":
    for t, n in build().items():
        print(f"{t}: {n['one_line']}\n  in: {n['in_scope']}\n  out: {n['out_of_scope']}")
    print(f"cached under {CACHE}")
