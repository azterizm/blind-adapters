"""Teacher data for the adapters (Gemini via the router; public inputs only).

The teacher sees public statute text and fictional matters it wrote itself. It never
sees the held-out matters. Per adapter (= task):

  1. specs     -- a seeded, balanced list of scenario specs (issue, intended answers,
                  service length, pay, dates). Written by code, not by the teacher.
  2. matters   -- the teacher writes a fictional client note for each spec.
  3. answers   -- the teacher answers every model question of the task on that note,
                  given the statute text, in the device JSON schema, plus the facts
                  with quotes and a statute quote per authority.
  4. audit     -- an example is dropped and logged when: evidence is not verbatim from
                  the matter, an authority quote is not verbatim from that provision,
                  an authority is outside the step's allowed set, a yes/no answer has
                  no evidence, the answer disagrees with the spec's intended answer, or
                  the matter shares an 8-gram with a held-out matter.
  5. examples  -- prompt = device/prompts.step_prompt in the B/A layout (statute text,
                  no precomputed reasoning); target = the device JSON. Facts in the
                  prompt are the teacher's facts after the device's own verify_facts.

Matters stopped by a failed precondition (dismissed NO) yield only that step, as on
the device.

Run:  python -m cloud.teacher [n_matters_per_task]     (cached under artifacts/teacher/)
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import json
import os
import random
import re
import sys

from cloud.precompute import render_sections
from cloud.reason import ACT, P, verbatim
from device.prompts import step_prompt, target_json
from llm import cloud_chat_json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "artifacts", "teacher")
QUESTIONS = os.path.join(ROOT, "artifacts", "cloud_rich", "questions.json")
WORKERS = 16

SYSTEM = ("You are a senior UK employment lawyer preparing training material for junior "
          "solicitors. Use ONLY the statutory text you are given. Reply with a single JSON object.")

# ------------------------------------------------------------------ scenario specs
UD_ISSUES = [
    # (issue description, intended answers)
    ("was dismissed by the employer with notice for a stated reason such as poor performance, misconduct, "
     "attendance or a restructure", dict(dismissed="yes", automatic_unfair_reason="no")),
    ("was dismissed by the employer with immediate effect, without notice, for a stated reason",
     dict(dismissed="yes", automatic_unfair_reason="no")),
    ("resigned in response to a serious breach of contract by the employer (for example an imposed pay cut, "
     "unpaid wages, a demotion the contract does not allow, or serious bullying by a manager), so this is a "
     "constructive dismissal", dict(dismissed="yes", automatic_unfair_reason="no")),
    ("resigned although the employer did nothing in breach of contract (for example to take another job, "
     "because of a change the contract expressly allows, or for personal reasons)", dict(dismissed="no")),
    ("worked on a fixed-term contract that expired and was not renewed",
     dict(dismissed="yes", automatic_unfair_reason="no")),
    ("was dismissed because of pregnancy or maternity leave (for a woman), or because they took or asked for "
     "parental, paternity or adoption leave", dict(dismissed="yes", automatic_unfair_reason="yes")),
    ("was dismissed because they raised a genuine health and safety concern, or refused to work in what they "
     "reasonably believed was serious and imminent danger", dict(dismissed="yes", automatic_unfair_reason="yes")),
    ("was dismissed because they reported wrongdoing such as fraud, a breach of a legal obligation or a danger "
     "to health and safety to the employer or a regulator (a protected disclosure)",
     dict(dismissed="yes", automatic_unfair_reason="yes")),
    ("was dismissed soon after a grievance about a personal disagreement with a colleague, or after asking for "
     "a pay rise, with no pregnancy, family leave, health and safety or whistleblowing element",
     dict(dismissed="yes", automatic_unfair_reason="no")),
]
R_ISSUES = [
    ("was dismissed because the employer closed the business or the workplace where they worked, and was not "
     "offered any other job", dict(dismissed="yes", by_reason_of_redundancy="yes", suitable_alternative="yes")),
    ("was dismissed because the employer needed fewer employees to do their kind of work, and was not offered "
     "any other job", dict(dismissed="yes", by_reason_of_redundancy="yes", suitable_alternative="yes")),
    ("was dismissed for redundancy and, before the employment ended, was offered another job on the same or "
     "very similar terms at the same place, starting within four weeks, and refused it for no good reason",
     dict(dismissed="yes", by_reason_of_redundancy="yes", suitable_alternative="no")),
    ("was dismissed for redundancy and was offered another job that was clearly not suitable (for example much "
     "lower pay, a much more junior role, or a workplace very far away), and refused it",
     dict(dismissed="yes", by_reason_of_redundancy="yes", suitable_alternative="yes")),
    ("was dismissed for redundancy and was offered a broadly suitable job, but refused it for a strong personal "
     "reason that makes the refusal reasonable (for example caring duties that make the new hours or place "
     "impossible, or a serious health reason)",
     dict(dismissed="yes", by_reason_of_redundancy="yes", suitable_alternative="yes")),
    ("was dismissed for misconduct or poor performance while the employer still needed the same number of "
     "people for that work, and someone else was hired to do the same job",
     dict(dismissed="yes", by_reason_of_redundancy="no", suitable_alternative="yes")),
    ("resigned to take another job before the employer had given any notice of dismissal, although redundancies "
     "had been rumoured", dict(dismissed="no")),
]
ISSUES = {"uk_unfair_dismissal": UD_ISSUES, "uk_redundancy_payment": R_ISSUES}
SECTORS = ["retail", "hospitality", "logistics", "construction", "software", "healthcare", "education",
           "manufacturing", "finance", "legal services", "media", "agriculture", "charity", "transport",
           "property", "telecoms", "engineering", "public relations", "security", "cleaning services"]


def specs(task: str, n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(f"{task}-{seed}")
    out = []
    issues = ISSUES[task]
    for i in range(n):
        issue, intended = issues[i % len(issues)]
        end = dt.date(2025, 1, 1) + dt.timedelta(days=rng.randrange(640))
        months = rng.choice([4, 9, 14, 20, 23, 26, 30, 40, 60, 90, 130, 200, 260])
        start = end - dt.timedelta(days=int(months * 30.4))
        advice = end + dt.timedelta(days=rng.choice([3, 10, 20, 40, 60, 80, 95, 120]))
        out.append(dict(
            id=f"{task}-{i:04d}", task=task, issue=issue, intended=intended,
            sector=rng.choice(SECTORS), gender=rng.choice(["woman", "man"]),
            start=start.isoformat(), end=end.isoformat(), advice=advice.isoformat(),
            notice=rng.choice(["with notice", "without notice", "with notice", "with pay in lieu of notice"]),
            pay=rng.choice(["annual", "weekly"]),
            amount=rng.choice([19500, 23400, 26000, 31200, 38000, 45500, 52000, 61000, 78000, 95000]),
            age=rng.randrange(19, 64),
            claim=rng.choice(["has not presented a claim", "has not presented a claim",
                              "presented a claim to the tribunal on a stated date"]),
        ))
    return out


# ------------------------------------------------------------------ teacher calls
def write_matter(spec: dict) -> dict:
    pay = (f"salary £{spec['amount']:,} a year" if spec["pay"] == "annual"
           else f"gross weekly pay £{round(spec['amount'] / 52):,}")
    user = (
        "Write one fictional, realistic UK solicitor's note of a first meeting with a client about their "
        "employment ending. Invent names for the client and the employer.\n"
        f"- Sector: {spec['sector']}. Client: a {spec['gender']}, aged {spec['age']}.\n"
        f"- Situation: the client {spec['issue']}.\n"
        f"- Employment started about {spec['start']}; it ended about {spec['end']}, {spec['notice']} "
        "(adapt the notice details if the situation requires it). State exact dates in the form '4 August 2025'.\n"
        f"- Pay: {pay}. The client {spec['claim']}.\n"
        f"- The meeting is on {spec['advice']}; the note starts with that date.\n"
        "Write 90 to 170 words of plain prose. State the facts a lawyer needs; do not state any legal "
        "conclusion and do not name any statute. Vary your wording; do not use stock phrases.\n"
        'Return JSON {"note": "..."}.')
    out = cloud_chat_json(SYSTEM, user)
    return dict(spec=spec, note=str(out.get("note", "")).strip())


def answer_matter(task: str, note: str, qs: dict, statute: dict) -> dict:
    steps = [s for s in qs[task]["steps"] if s["kind"] == "model"]
    provs = sorted({a for s in steps for a in s["authority"]})
    law = "\n\n".join(f"[{c}]\n<<<\n{statute[c]}\n>>>" for c in provs)
    qtext = "\n".join(f'  - "{s["id"]}": {s["question"]} (allowed authority: {", ".join(s["authority"])})'
                      for s in steps)
    facts = "\n".join(f'  - "{k}" ({v["type"]}): {v["note"]}' for k, v in qs[task]["facts"].items())
    user = (
        f"STATUTE (Employment Rights Act 1996):\n{law}\n\nCLIENT ACCOUNT:\n<<<\n{note}\n>>>\n\n"
        f"Answer each question by applying ONLY the statute above to the account:\n{qtext}\n\n"
        f"Also extract these facts:\n{facts}\n\n"
        'Return JSON {"steps": {"<question id>": {"answer": "yes" | "no" | "unclear", '
        '"reasoning": "2-4 sentences applying the law to the facts", '
        '"evidence": ["sentences or phrases copied EXACTLY from the CLIENT ACCOUNT"], '
        '"authority": [{"coordinate": "uk/ukpga/1996/18/sNN", "quote": "a phrase of 5-25 words copied '
        'EXACTLY from that provision"}]}}, '
        '"facts": {"<fact>": {"value": <YYYY-MM-DD date, number, or true/false>, "quote": "copied EXACTLY '
        'from the account"} or null}}. Answer "unclear" only if the account truly does not let you decide.')
    return cloud_chat_json(SYSTEM, user)


# ------------------------------------------------------------------ audit
def ngrams(text: str, n: int = 8) -> set[tuple]:
    w = re.findall(r"[a-z0-9£]+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def audit(task: str, spec: dict, note: str, ans: dict, qs: dict, statute: dict,
          heldout_grams: set) -> tuple[list[dict], list[dict]]:
    """Returns (examples, drops). One example per model step that the device would run."""
    from device.reasoner import verify_facts
    drops, examples = [], []
    if not note:
        return [], [dict(id=spec["id"], step=None, why="empty note")]
    if ngrams(note) & heldout_grams:
        return [], [dict(id=spec["id"], step=None, why="shares an 8-gram with a held-out matter")]
    facts, _, _ = verify_facts(ans.get("facts") or {}, note)
    facts = {k: (v.isoformat() if isinstance(v, dt.date) else v) for k, v in facts.items()}
    got = ans.get("steps") or {}
    for step in qs[task]["steps"]:
        if step["kind"] != "model":
            continue
        a = got.get(step["id"])
        why = None
        if not isinstance(a, dict):
            why = "missing answer"
        else:
            answer = str(a.get("answer", "")).strip().lower()
            ev = [e for e in a.get("evidence") or [] if isinstance(e, str)]
            au = [x for x in a.get("authority") or [] if isinstance(x, dict)]
            want = spec["intended"].get(step["id"])
            if answer not in ("yes", "no", "unclear"):
                why = f"bad answer {answer!r}"
            elif want and answer != want:
                why = f"answer {answer} disagrees with intended {want}"
            elif answer != "unclear" and not ev:
                why = "no evidence"
            elif any(not verbatim(e, note) for e in ev):
                why = "evidence not verbatim from the matter"
            elif not au:
                why = "no authority"
            elif any(x.get("coordinate") not in step["authority"] for x in au):
                why = "authority outside the allowed set"
            elif any(not verbatim(x.get("quote"), statute[x["coordinate"]]) for x in au):
                why = "authority quote not verbatim from the statute"
        if why:
            drops.append(dict(id=spec["id"], step=step["id"], why=why))
            if step.get("precondition"):
                break
            continue
        coords = list(dict.fromkeys(x["coordinate"] for x in au))
        examples.append(dict(
            id=spec["id"], task=task, step=step["id"], novel=step["novel"], answer=answer,
            prompt=step_prompt(step, statute, facts, note),
            target=target_json(answer, str(a.get("reasoning", "")).strip(), ev, coords)))
        if step.get("precondition") and answer == "no":
            break                                   # the device stops here
    return examples, drops


# ------------------------------------------------------------------ build
def _retry(fn, tries: int = 4):
    """Retry router overload (HTTP 5xx, timeouts) with backoff; other errors propagate."""
    import time
    import urllib.error
    for k in range(tries):
        try:
            return fn()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if k == tries - 1 or (isinstance(e, urllib.error.HTTPError) and e.code < 500):
                raise
            time.sleep(20 * 2 ** k)


def _cached(path: str, fn):
    if os.path.exists(path):
        return json.load(open(path))
    out = _retry(fn)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    return out


def build(n: int) -> None:
    from eval.matters import HELD_OUT              # read only for the 8-gram guard
    held = set().union(*(ngrams(t) for t in HELD_OUT.values()))
    qs = json.load(open(QUESTIONS))
    texts = render_sections(ACT)
    statute = {c: texts[c] for t in qs.values() for c in t["statute"]}
    for task in qs:
        d = os.path.join(ART, task)
        sp = specs(task, n)

        def one(spec):
            try:
                m = _cached(os.path.join(d, "matters", spec["id"] + ".json"), lambda: write_matter(spec))
                a = _cached(os.path.join(d, "answers", spec["id"] + ".json"),
                            lambda: answer_matter(task, m["note"], qs, statute))
                return audit(task, spec, m["note"], a, qs, statute, held)
            except Exception as e:                  # router or JSON failure: drop, log
                return [], [dict(id=spec["id"], step=None, why=f"teacher error: {e}"[:200])]

        examples, drops = [], []
        with cf.ThreadPoolExecutor(WORKERS) as ex:
            for i, (e, dr) in enumerate(ex.map(one, sp), 1):
                examples += e
                drops += dr
                if i % 25 == 0:
                    print(f"  {task}: {i}/{len(sp)} matters, {len(examples)} examples, {len(drops)} drops", flush=True)
        with open(os.path.join(d, "examples.jsonl"), "w") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        json.dump(drops, open(os.path.join(d, "drops.json"), "w"), indent=1)
        why = {}
        for x in drops:
            k = re.sub(r"teacher error: .*", "teacher error", x["why"])
            why[k] = why.get(k, 0) + 1
        by_step = {}
        for e in examples:
            by_step.setdefault(e["step"], {}).setdefault(e["answer"], 0)
            by_step[e["step"]][e["answer"]] += 1
        print(f"{task}: {len(examples)} examples from {len(sp)} matters; drops {why}; by step {by_step}", flush=True)


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
