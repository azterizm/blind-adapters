"""The fixed public question list per task. Every arm (B, T, A, A+T) answers exactly
these questions, in this order, with the same PIR-fetched statute text.

In-scope steps reuse the question text of the blind-counsel v2 procedures, so arm T
sees the same questions it was scored on. Novel steps (automatic_unfair_reason,
suitable_alternative) cover law outside the precomputed entries: ss.99, 100, 103A with
s.108(3), and s.141. Fact definitions are the public fact notes of the v2 procedures.

Pre-registered harness rules (identical for every arm):
  * precondition: a grounded NO on `dismissed` stops the matter as REFUSE-AND-FLAG;
  * payment_amount is 0 when by_reason_of_redundancy or suitable_alternative is a
    grounded NO (ss.135, 139, 141(2));
  * compensation_cap is "uncapped" when automatic_unfair_reason is a grounded YES that
    cites s100 or s103A (s124(1A)).

Run:  python -m cloud.questions        (writes artifacts/cloud_rich/questions.json)
"""
from __future__ import annotations

import json
import os

from cloud.reason import ART, FACT_FIELDS, P, STEP_SCHEMA, TASKS

NOVEL_STEPS = {
    "uk_unfair_dismissal": [
        # (after step id, step)
        ("qualifying_service", dict(
            id="automatic_unfair_reason", kind="model", calc=None, novel=True,
            authority=["s99", "s100", "s103A", "s108"],
            question="Was the reason, or the principal reason, for the dismissal one that makes it "
                     "automatically unfair under section 99 (pregnancy, childbirth, maternity and other "
                     "family reasons), section 100 (health and safety) or section 103A (protected "
                     "disclosure), so that section 108(3) disapplies the two-year qualifying period?")),
    ],
    "uk_redundancy_payment": [
        ("by_reason_of_redundancy", dict(
            id="suitable_alternative", kind="model", calc=None, novel=True,
            authority=["s141"],
            question="Does the employee keep the right to a redundancy payment under section 141? "
                     "Answer yes if no offer to renew the contract or to re-engage the employee was made, "
                     "or the offered employment was not suitable, or the employee refused it reasonably. "
                     "Answer no if the employee unreasonably refused an offer to which section 141 applies.")),
    ],
}
PRECONDITIONS = {"dismissed"}
NOT_SCORED = {"fairness"}


def _v2(task: str) -> dict:
    return json.load(open(os.path.join(ART, "reasoning", f"procedure_v2_{task}.json")))


def build() -> dict:
    out = {}
    for task, spec in TASKS.items():
        proc = _v2(task)
        by_id = {s["id"]: s for s in proc["steps"]}
        steps = []
        for sid, kind, calc, auth in STEP_SCHEMA[task]:
            steps.append(dict(id=sid, kind=kind, calc=calc, novel=False,
                              authority=[P + a for a in auth], question=by_id[sid]["question"],
                              precondition=sid in PRECONDITIONS, scored=sid not in NOT_SCORED))
            for after, novel in NOVEL_STEPS[task]:
                if after == sid:
                    steps.append(dict(novel, authority=[P + a for a in novel["authority"]],
                                      precondition=False, scored=True))
        provs = sorted({a for s in steps for a in s["authority"]})
        out[task] = dict(description=spec["description"], steps=steps,
                         facts={k: dict(type=t, note=proc["fact_notes"].get(k, ""))
                                for k, t in FACT_FIELDS.items()},
                         statute=provs)
    return out


if __name__ == "__main__":
    qs = build()
    os.makedirs(ART, exist_ok=True)
    json.dump(qs, open(os.path.join(ART, "questions.json"), "w"), indent=1, ensure_ascii=False)
    for t, v in qs.items():
        print(f"{t}: {len(v['steps'])} steps "
              f"({sum(s['novel'] for s in v['steps'])} novel), statute {[c.split('/')[-1] for c in v['statute']]}")
