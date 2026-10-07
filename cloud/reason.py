# from blind-counsel 8b1bf8f cloud/reason.py
"""Offline, cloud-side: precompute PUBLIC legal reasoning for the rich PIR database.

Everything here is a function of public law only -- no client appears anywhere --
so it is safe to compute in advance with a large model and store in the cloud.

Three entry types (keys are public signatures the device can name):
  explain:<coord>      plain-language summary, the legal ELEMENTS the provision
                       imposes, thresholds -- each element tied to a verbatim quote
  relation:<a>|<b>     how two provisions interact (from an extracted link graph:
                       explicit cross-references + defined-term usage)
  calc:<coord>         numeric parameters a calculator needs, each with its quote
  procedure:<task>     the decision procedure on a fixed public step schema:
                       questions, pitfall notes, fact definitions

Central audit (done ONCE, because the entries are public): every quote the model
attaches to a claim must appear verbatim in the statute; failing claims are
dropped and counted. Prose (summary/analysis) is labelled model-written.

Run:  python -m cloud.reason        (cached under artifacts/cloud_rich/reasoning/)
"""
from __future__ import annotations

import json
import os
import re

from cloud.precompute import render_sections
from llm import CLOUD_MODEL, cloud_chat_json

ART = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "cloud_rich"))
CACHE = os.path.join(ART, "reasoning")
ACT = "uk_ukpga_1996_18"
P = "uk/ukpga/1996/18/"

TASKS = {
    "uk_unfair_dismissal": dict(
        description="Advise whether an employee has a viable ordinary unfair-dismissal claim "
                    "under the Employment Rights Act 1996 (eligibility, fairness, time limit, "
                    "compensation).",
        provisions=["s86", "s94", "s95", "s97", "s98", "s108", "s111", "s123", "s124"]),
    "uk_redundancy_payment": dict(
        description="Advise whether an employee is entitled to a statutory redundancy payment "
                    "under the Employment Rights Act 1996, and how it is calculated.",
        provisions=["s86", "s135", "s136", "s139", "s145", "s155", "s162", "s227"]),
}

# ---- public task design (Lane A). Step ids and which steps are arithmetic are fixed
# here; the cloud model writes the questions, pitfalls and fact definitions, and
# extracts the numeric parameters from the statute.
STEP_SCHEMA = {
    "uk_unfair_dismissal": [
        ("dismissed", "model", None, ["s95", "s97"]),
        ("qualifying_service", "calc", "ud_qualifying", ["s108", "s97", "s86"]),
        ("time_limit", "calc", "time_limit", ["s111", "s97"]),
        ("fairness", "model", None, ["s98"]),
        ("compensation_cap", "calc", "compensation_cap", ["s123", "s124"]),
    ],
    "uk_redundancy_payment": [
        ("dismissed", "model", None, ["s135", "s136"]),
        ("by_reason_of_redundancy", "model", None, ["s139"]),
        ("relevant_date", "calc", "relevant_date", ["s145", "s86"]),
        ("qualifying_service", "calc", "red_qualifying", ["s155", "s145", "s86"]),
        ("payment_amount", "calc", "redundancy_amount", ["s162", "s227"]),
    ],
}

FACT_FIELDS = {
    "start_date": "date", "termination_date": "date", "notice_given": "bool",
    "notice_given_date": "date", "advice_date": "date", "claim_presented_date": "date",
    "annual_pay": "gbp", "weekly_pay": "gbp", "age": "int",
}

CALC_SCHEMA = {
    "s86": {"notice_weeks_under_2y": "weeks of notice where continuous employment is under two years",
            "notice_weeks_per_year": "weeks of notice per year where employment is two years or more",
            "notice_weeks_max": "maximum weeks of notice (twelve years or more)"},
    "s108": {"qualifying_years": "years of continuous employment required"},
    "s111": {"time_limit_months": "months, beginning with the effective date of termination, to present a complaint"},
    "s124": {"cap_fixed_gbp": "the fixed cap on the compensatory award in pounds",
             "cap_weeks": "number of weeks' pay in the alternative cap"},
    "s155": {"qualifying_years": "years of continuous employment required"},
    "s162": {"rate_upper_weeks": "weeks' pay per year at the highest age band",
             "age_upper": "age from which the highest rate applies",
             "rate_middle_weeks": "weeks' pay per year at the middle age band",
             "age_lower": "age from which the middle rate applies",
             "rate_lower_weeks": "weeks' pay per year below the middle band",
             "max_years": "maximum number of years counted"},
    "s227": {"week_pay_limit_gbp": "maximum amount of a week's pay, in pounds, for a redundancy payment"},
}

# defined terms: phrase -> defining provision (links any in-scope user to it)
DEFINED_TERMS = {
    "effective date of termination": "s97",
    "relevant date": "s145",
    "by reason of redundancy": "s139",
}

SYSTEM = ("You are a senior UK employment lawyer writing a public practice manual. "
          "Use ONLY the statutory text you are given. Reply with a single JSON object.")


# ------------------------------------------------------------------ helpers
def norm(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", s).strip().strip(".,;:—- ").lower()


def verbatim(quote: str, source: str) -> bool:
    q = norm(quote or "")
    return len(q) >= 8 and q in norm(source)


def _cached(key: str, fn):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, re.sub(r"[^A-Za-z0-9]+", "_", key) + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    entry = fn()
    json.dump(entry, open(path, "w"), indent=1, ensure_ascii=False)
    return entry


# ------------------------------------------------------------------ link graph
def link_graph(texts: dict[str, str]) -> dict[tuple[str, str], list[str]]:
    """Deterministic, verifiable edges among in-scope provisions."""
    scope = {s for t in TASKS.values() for s in t["provisions"]}
    ref = re.compile(r"\bsections?\s+(\d+[A-Z]*)(?:\s+to\s+(\d+[A-Z]*))?", re.I)
    edges: dict[tuple[str, str], list[str]] = {}

    def add(a, b, why):
        if a != b and a in scope and b in scope:
            edges.setdefault(tuple(sorted((a, b), key=lambda x: int(re.sub(r"\D", "", x)))), []).append(why)

    for s in scope:
        txt = texts[s]
        for m in ref.finditer(txt):
            lo, hi = m.group(1), m.group(2)
            targets = [lo]
            if hi and lo.isdigit() and hi.isdigit():
                targets = [str(n) for n in range(int(lo), int(hi) + 1)]
            for t in targets:
                add(s, "s" + t, f"{s} cites section {t}")
        for phrase, definer in DEFINED_TERMS.items():
            if s != definer and phrase in txt.lower():
                add(s, definer, f'{s} uses "{phrase}", defined in {definer}')
    return edges


# ------------------------------------------------------------------ generators
def explain(sec: str, text: str) -> dict:
    user = (f"Provision {P}{sec} of the Employment Rights Act 1996:\n<<<\n{text}\n>>>\n\n"
            "Return JSON with keys:\n"
            '  "summary": 2-3 plain-English sentences on what it does,\n'
            '  "elements": list of the conditions/tests it imposes, each '
            '{"requirement": "...", "quote": "a short phrase copied EXACTLY from the text"},\n'
            '  "thresholds": list of time periods, amounts or numeric limits, each '
            '{"what": "...", "value": "...", "quote": "exact phrase from the text"} (may be empty).')
    raw = cloud_chat_json(SYSTEM, user)
    els = raw.get("elements", []) or []
    ths = raw.get("thresholds", []) or []
    ok_els = [e for e in els if isinstance(e, dict) and verbatim(e.get("quote"), text)]
    ok_ths = [t for t in ths if isinstance(t, dict) and verbatim(t.get("quote"), text)]
    return dict(key=f"explain:{P}{sec}", coordinate=P + sec, kind="explain",
                summary=raw.get("summary", ""), elements=ok_els, thresholds=ok_ths,
                audit=dict(elements_proposed=len(els), elements_verified=len(ok_els),
                           thresholds_proposed=len(ths), thresholds_verified=len(ok_ths),
                           prose="model-written (not quote-verifiable)"),
                model=CLOUD_MODEL)


def relation(a: str, b: str, why: list[str], ta: str, tb: str) -> dict:
    user = (f"Provision A ({P}{a}):\n<<<\n{ta}\n>>>\n\nProvision B ({P}{b}):\n<<<\n{tb}\n>>>\n\n"
            f"They are linked: {'; '.join(why)}.\n"
            "Explain for a practitioner how A and B interact. Return JSON with keys:\n"
            '  "relation": one of "precondition", "defines_term_used_by", "limits", '
            '"exception_to", "procedural", "calculates",\n'
            '  "analysis": 2-4 sentences,\n'
            '  "quote_a": a phrase copied EXACTLY from A that shows the link,\n'
            '  "quote_b": a phrase copied EXACTLY from B that shows the link.')
    raw = cloud_chat_json(SYSTEM, user)
    qa, qb = verbatim(raw.get("quote_a"), ta), verbatim(raw.get("quote_b"), tb)
    return dict(key=f"relation:{P}{a}|{P}{b}", kind="relation", a=P + a, b=P + b,
                basis=why, relation=raw.get("relation"), analysis=raw.get("analysis", ""),
                quote_a=raw.get("quote_a") if qa else None,
                quote_b=raw.get("quote_b") if qb else None,
                audit=dict(quote_a_verified=qa, quote_b_verified=qb,
                           prose="model-written (not quote-verifiable)"),
                model=CLOUD_MODEL)


def calc_params(sec: str, text: str) -> dict:
    """Extract the numeric parameters a calculator needs, each with its statute quote.
    Central audit: the quote must be verbatim AND the value must appear in it."""
    from lawtext import numbers_in
    schema = CALC_SCHEMA[sec]
    spec = "\n".join(f'  - "{k}": {v}' for k, v in schema.items())
    user = (f"Provision {P}{sec}:\n<<<\n{text}\n>>>\n\nExtract these numeric parameters:\n{spec}\n"
            'Return JSON {"params": [{"name": "...", "value": <number>, "quote": "a phrase of 5-25 '
            'words copied EXACTLY from the text that states this number together with the '
            'condition it applies to"}]}')
    raw = cloud_chat_json(SYSTEM, user)
    ok, rejected = [], []
    for p in raw.get("params", []) or []:
        try:
            p["value"] = float(str(p["value"]).replace("£", "").replace(",", ""))
            why = ("unknown parameter" if p.get("name") not in schema else
                   "quote not verbatim in statute" if not verbatim(p.get("quote"), text) else
                   "value not in quote" if p["value"] not in numbers_in(p["quote"]) else None)
        except (TypeError, ValueError, KeyError):
            why = "unparseable value"
        if why:
            rejected.append({**p, "why": why})
        else:
            ok.append(p)
    return dict(key=f"calc:{P}{sec}", kind="calc", coordinate=P + sec,
                params={p["name"]: dict(value=float(p["value"]), quote=p["quote"]) for p in ok},
                audit=dict(proposed=len(ok) + len(rejected), verified=len(ok),
                           missing=sorted(set(schema) - {p["name"] for p in ok}),
                           rejected=rejected),
                model=CLOUD_MODEL)


def procedure_v2(task: str, spec: dict, explains: dict[str, dict]) -> dict:
    """Decision procedure on the fixed public step schema. The cloud model writes each
    step's question (phrased so YES = the requirement is satisfied, judged at the advice
    date), pitfall notes, and definitions of the client facts to extract."""
    lines = []
    for s in spec["provisions"]:
        e = explains[s]
        lines.append(f"- {P}{s}: {e['summary']} Elements: "
                     + "; ".join(x["requirement"] for x in e["elements"]))
    steps_spec = "\n".join(
        f'  - "{sid}" ({"computed by a calculator" if kind == "calc" else "decided by applying the law to the facts"}); '
        f"authority {', '.join(P + a for a in auth)}"
        for sid, kind, _, auth in STEP_SCHEMA[task])
    user = (f"Task: {spec['description']}\n\nProvisions:\n" + "\n".join(lines) +
            f"\n\nThe procedure has exactly these steps, in this order:\n{steps_spec}\n\n"
            f"Client facts that will be extracted: {', '.join(FACT_FIELDS)}.\n\n"
            "Return JSON:\n"
            '{"steps": [{"id": "<step id above>", "question": "a question about the client\'s '
            'situation, phrased so that YES means the requirement is satisfied, judged at the date '
            'of the advice", "pitfalls": ["common mistakes when answering this step"]}],\n'
            ' "fact_notes": {"<fact name>": "how to identify this fact correctly, including pitfalls"}}')
    raw = cloud_chat_json(SYSTEM, user)
    by_id = {s.get("id"): s for s in raw.get("steps", []) or [] if isinstance(s, dict)}
    steps = []
    for sid, kind, calc, auth in STEP_SCHEMA[task]:
        s = by_id.get(sid, {})
        steps.append(dict(id=sid, kind=kind, calc=calc, authority=[P + a for a in auth],
                          question=s.get("question") or sid.replace("_", " "),
                          pitfalls=[p for p in s.get("pitfalls", []) if isinstance(p, str)]))
    notes = {k: v for k, v in (raw.get("fact_notes") or {}).items() if k in FACT_FIELDS}
    return dict(key=f"procedure:{task}", kind="procedure", version=2, task=task,
                description=spec["description"], provisions=sorted(P + s for s in spec["provisions"]),
                steps=steps, fact_notes=notes,
                audit=dict(steps_written=len(by_id), steps_expected=len(steps),
                           fact_notes=len(notes)),
                model=CLOUD_MODEL)


# ------------------------------------------------------------------ build
def build() -> list[dict]:
    texts_full = render_sections(ACT)
    scope = sorted({s for t in TASKS.values() for s in t["provisions"]},
                   key=lambda x: int(re.sub(r"\D", "", x)))
    texts = {s: texts_full[P + s] for s in scope}
    entries = []

    explains = {}
    for s in scope:
        e = _cached(f"explain:{s}", lambda s=s: explain(s, texts[s]))
        explains[s] = e
        entries.append(e)
        a = e["audit"]
        print(f"  explain  {s:<5} elements {a['elements_verified']}/{a['elements_proposed']} verified, "
              f"thresholds {a['thresholds_verified']}/{a['thresholds_proposed']}", flush=True)

    for (a, b), why in sorted(link_graph(texts).items()):
        r = _cached(f"relation:{a}|{b}", lambda a=a, b=b, why=why: relation(a, b, why, texts[a], texts[b]))
        entries.append(r)
        au = r["audit"]
        print(f"  relation {a}<->{b:<6} {r['relation']!s:<22} quotes A/B verified "
              f"{au['quote_a_verified']}/{au['quote_b_verified']}", flush=True)

    for s in sorted(CALC_SCHEMA, key=lambda x: int(re.sub(r"\D", "", x))):
        c = _cached(f"calc:{s}", lambda s=s: calc_params(s, texts[s]))
        entries.append(c)
        print(f"  calc     {s:<5} params {c['audit']['verified']}/{len(CALC_SCHEMA[s])} verified"
              f"{' missing ' + str(c['audit']['missing']) if c['audit']['missing'] else ''}", flush=True)

    for task, spec in TASKS.items():
        p = _cached(f"procedure_v2:{task}", lambda task=task, spec=spec: procedure_v2(task, spec, explains))
        entries.append(p)
        print(f"  procedure {task}: {p['audit']['steps_written']}/{p['audit']['steps_expected']} steps "
              f"written, {p['audit']['fact_notes']} fact notes", flush=True)
    return entries


if __name__ == "__main__":
    es = build()
    print(f"\n{len(es)} precomputed public-reasoning entries ready ({CLOUD_MODEL}).")
