# from blind-counsel 8b1bf8f device/reasoner.py
"""Device-side Tier 2: the laptop applies PIR-fetched public reasoning to the private
client account with its own local model (Qwen3-8B, MLX). Nothing private leaves.

Per matter:
  1. select   -- the local model picks the task type from the PUBLIC task index (a
                 closed choice) or "none" -> refuse-and-flag. Local only.
  2. fetch    -- ONE fixed batch of `fetch_budget` PIR queries, the same count for
                 every matter and even for a refusal (dummy rows pad the batch), so
                 the cloud cannot tell the task type or whether the device answered.
                 Fetching happens before any reasoning, so private reasoning never
                 steers what is asked of the cloud.
  3. re-verify-- sha256 of every entry; the cloud's statutory quotes and calculator
                 parameters are re-checked against the fetched statute (the device
                 does not trust the cloud's audit).
  4. facts    -- the local model extracts dates/amounts; each is kept only if its quote
                 is verbatim from the client account AND contains that date/amount.
  5. steps    -- arithmetic steps are computed by code (device/calc.py) from verified
                 facts and verified statutory parameters; judgment steps are answered by
                 the local model with the precomputed pitfall notes, then pass the
                 grounding gate (verbatim client evidence + fetched authority).

A tripwire makes any call to the cloud LLM router during a matter raise.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
import re

import numpy as np

import llm
from cloud.reason import FACT_FIELDS, P, verbatim
from device import calc
from device.egress import EgressGuard
from lawtext import amounts_in, dates_in, numbers_in
from pir.simplepir import PIRClient, PIRServer

RICH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "cloud_rich"))
GROUNDED_MIN = 0.6          # below this fraction of decided steps -> REFER


# ------------------------------------------------------------------ the cloud
class RichCloud:
    """Untrusted: holds the public rich DB, answers PIR queries, records its view."""

    def __init__(self):
        self.D = np.load(os.path.join(RICH, "pir_db.npy"))
        self.server = PIRServer(self.D, seed=2024)
        self.seen: list[np.ndarray] = []

    def hint(self):
        return self.server.A, self.server.hint()

    def answer(self, qu):
        self.seen.append(qu.copy())
        return self.server.answer(qu)


class RouterTripwire:
    """Within a matter, the cloud LLM must never be called."""

    def __enter__(self):
        self._orig = llm._router_chat
        self.calls = 0

        def trip(*a, **k):
            self.calls += 1
            raise RuntimeError("cloud LLM called during a private matter")
        llm._router_chat = trip
        return self

    def __exit__(self, *exc):
        llm._router_chat = self._orig
        return False


# ------------------------------------------------------------------ verification
def _coord(a: str) -> str | None:
    a = str(a).strip()
    if a.startswith(P):
        return a
    m = re.search(r"(\d+[A-Za-z]*)", a)
    return P + "s" + m.group(1) if m else None


def ground(out: dict, narrative: str, allowed: set[str]) -> dict:
    ans = str(out.get("answer", "")).strip().lower()
    if ans not in {"yes", "no", "unclear"}:
        ans = "unclear"
    ev = [e for e in (out.get("evidence") or []) if isinstance(e, str)]
    ev_ok = [e for e in ev if verbatim(e, narrative)]
    ev_bad = [e for e in ev if e not in ev_ok]
    au = [_coord(a) for a in (out.get("authority") or [])]
    au_ok = sorted({a for a in au if a in allowed})
    au_bad = [a for a in (out.get("authority") or []) if _coord(a) not in allowed]
    if ans == "unclear":
        status = "unclear"
    elif not ev_ok:
        status, ans = "ungrounded", "unclear"
    else:
        status = "grounded"
    return dict(answer=ans, status=status, reasoning=str(out.get("reasoning", "")),
                evidence=ev_ok, evidence_rejected=ev_bad,
                authority=au_ok, authority_rejected=au_bad)


def verify_facts(raw: dict, narrative: str) -> tuple[dict, dict, list]:
    """Keep a fact only if its quote is verbatim from the account and the value is
    literally present in that quote. Returns (facts, quotes, rejected)."""
    facts, quotes, rejected = {}, {}, []
    for name, typ in FACT_FIELDS.items():
        item = raw.get(name)
        if not isinstance(item, dict) or item.get("value") in (None, "", "null"):
            continue
        q, v = item.get("quote") or "", item["value"]
        try:
            if not verbatim(q, narrative):
                raise ValueError("quote not in account")
            if typ == "date":
                val = dt.date.fromisoformat(str(v)[:10])
                if val not in dates_in(q):
                    raise ValueError("date not in quote")
            elif typ == "gbp":
                val = float(str(v).replace("£", "").replace(",", ""))
                if not any(abs(val - a) < 0.5 for a in amounts_in(q)):
                    raise ValueError("amount not in quote")
            elif typ == "int":
                val = int(float(v))
                if float(val) not in numbers_in(q):
                    raise ValueError("number not in quote")
            else:  # bool
                val = v if isinstance(v, bool) else str(v).strip().lower() in {"true", "yes"}
        except (ValueError, TypeError) as e:
            rejected.append(dict(field=name, value=v, quote=q, why=str(e)))
            continue
        facts[name], quotes[name] = val, q
    return facts, quotes, rejected


class _Tracked(dict):
    """Records which facts a calculator actually read (for provenance)."""

    def __init__(self, *a):
        super().__init__(*a)
        self.used: set[str] = set()

    def get(self, k, default=None):
        self.used.add(k)
        return super().get(k, default)

    def __getitem__(self, k):
        self.used.add(k)
        return super().__getitem__(k)


# ------------------------------------------------------------------ the device
SYS_SELECT = ("You triage confidential legal matters on a solicitor's laptop. "
              "Reply with a single JSON object.")
SYS_FACTS = ("You extract facts from a confidential client note. Copy quotes exactly. "
             "Reply with a single JSON object.")
SYS_STEP = ("You assist a UK employment solicitor on a confidential matter. Answer ONE "
            "question by applying ONLY the law provided to the client's account. Do not "
            "assume facts that are not in the account. Reply with a single JSON object.")


class SealedDevice:
    def __init__(self, cloud: RichCloud, narrative: str, seed: int = 7):
        self.cloud, self.narrative = cloud, narrative
        # public artifacts, synced in full by every device (so syncing leaks nothing)
        self.catalogue = json.load(open(os.path.join(RICH, "catalogue.json")))
        self.tasks = json.load(open(os.path.join(RICH, "tasks.json")))
        self.budget = json.load(open(os.path.join(RICH, "meta.json")))["fetch_budget"]
        A, H = cloud.hint()
        self.pir = PIRClient(A, H, seed=seed)
        self.rng = random.Random(seed)
        words = set(re.findall(r"[A-Za-z£0-9][\w£,.'-]{4,}", narrative))
        self.guard = EgressGuard(sorted(words))

    # 1 ---------------------------------------------------------------------
    def select_task(self) -> str | None:
        menu = "\n".join(f'- "{t}": {v["description"]}' for t, v in self.tasks.items())
        out = llm.device_chat_json(SYS_SELECT, (
            f"Task types this practice covers:\n{menu}\n\nClient account:\n<<<\n{self.narrative}\n>>>\n\n"
            'Which task type is this matter? Reply {"task": "<one id from the list>"} '
            'or {"task": "none"} if none of them fits.'), max_tokens=60)
        t = str(out.get("task", "none"))
        return t if t in self.tasks else None

    # 2 + 3 ---------------------------------------------------------------------
    def fetch(self, task: str | None) -> dict:
        keys = self.tasks[task]["entry_keys"] if task else []
        wanted = [r for k in keys for r in range(self.catalogue[k]["row"],
                                                 self.catalogue[k]["row"] + self.catalogue[k]["n_rows"])]
        plan = wanted + [self.rng.randrange(self.cloud.D.shape[0]) for _ in range(self.budget - len(wanted))]
        assert len(plan) == self.budget, "matter exceeds the fixed fetch budget"
        order = list(range(len(plan)))
        self.rng.shuffle(order)                              # order carries no meaning
        got: dict[int, bytes] = {}
        for i in order:
            qu, s = self.pir.query(plan[i])
            self.guard.send("pir", qu, note="rich-db row")
            got[plan[i]] = self.pir.decode(self.cloud.answer(qu), s).tobytes()
        entries, bad_sha = {}, []
        for k in keys:
            c = self.catalogue[k]
            raw = b"".join(got[r] for r in range(c["row"], c["row"] + c["n_rows"]))[: c["nbytes"]]
            if hashlib.sha256(raw).hexdigest() != c["sha256"]:
                bad_sha.append(k)
                continue
            entries[k] = raw.decode("utf-8") if k.startswith("text:") else json.loads(raw)
        # the device re-checks the cloud's statutory quotes and parameters itself
        q_ok = q_all = 0
        params: dict[str, dict] = {}
        for k, e in entries.items():
            stat = entries.get("text:" + e["coordinate"], "") if isinstance(e, dict) and "coordinate" in e else ""
            if k.startswith("explain:"):
                for x in e["elements"] + e["thresholds"]:
                    q_all += 1
                    if verbatim(x["quote"], stat):
                        q_ok += 1
                    else:
                        x["quote"] = None
            elif k.startswith("calc:"):
                params[e["coordinate"]] = {}
                for name, p in e["params"].items():
                    q_all += 1
                    if verbatim(p["quote"], stat) and p["value"] in numbers_in(p["quote"]):
                        q_ok += 1
                        params[e["coordinate"]][name] = p
        return dict(entries=entries, bad_sha=bad_sha, quotes_rechecked=(q_ok, q_all),
                    params=params, queries=len(plan))

    # 4 ---------------------------------------------------------------------
    def extract_facts(self, proc: dict) -> tuple[dict, dict, list]:
        notes = proc.get("fact_notes", {})
        spec = "\n".join(f'  "{n}" ({t}): {notes.get(n, "")}' for n, t in FACT_FIELDS.items())
        try:
            raw = llm.device_chat_json(SYS_FACTS, (
                f"Client account:\n<<<\n{self.narrative}\n>>>\n\nExtract these facts:\n{spec}\n\n"
                'Return JSON with one key per fact: {"<fact>": {"value": <YYYY-MM-DD date, '
                'number, or true/false>, "quote": "the sentence or phrase copied EXACTLY from the '
                'account that states it"}}. Use null for a fact the account does not state.'),
                max_tokens=700)
        except ValueError:
            raw = {}
        return verify_facts(raw, self.narrative)

    # 5 ---------------------------------------------------------------------
    def _law_block(self, entries: dict, authority: list[str]) -> str:
        lines = []
        for c in authority:
            e = entries.get(f"explain:{c}")
            if not e:
                continue
            lines.append(f"[{c}] {e['summary']}")
            for x in e["elements"]:
                q = f' -- statute: "{x["quote"]}"' if x.get("quote") else ""
                lines.append(f"  - element: {x['requirement']}{q}")
        rel = [e for k, e in entries.items() if k.startswith("relation:")
               and (e["a"] in authority or e["b"] in authority)]
        if rel:
            lines.append("Related provisions:")
            lines += [f"  [{e['a'].split('/')[-1]} <-> {e['b'].split('/')[-1]}] ({e['relation']}) "
                      f"{e['analysis']}" for e in rel]
        return "\n".join(lines)

    def calc_step(self, step: dict, fetched: dict, facts: dict, quotes: dict) -> dict:
        tracked = _Tracked(facts)
        base = dict(step_id=step["id"], question=step["question"], kind="calc",
                    step_authority=step["authority"], authority=step["authority"],
                    evidence_rejected=[], authority_rejected=[])
        try:
            ans, value, why = calc.run(step["calc"], tracked, fetched["params"])
        except calc.Missing as m:
            return {**base, "answer": "unclear", "value": None, "status": "unclear",
                    "reasoning": f"Not computed: missing verified input ({m}).", "evidence": []}
        used = sorted(k for k in tracked.used if k in quotes)
        return {**base, "answer": ans, "value": value, "status": "computed", "reasoning": why,
                "evidence": [quotes[k] for k in used]}

    def model_step(self, step: dict, entries: dict, facts: dict) -> dict:
        allowed = set(step["authority"])
        for k, e in entries.items():
            if k.startswith("relation:") and (e["a"] in allowed or e["b"] in allowed):
                allowed |= {e["a"], e["b"]}
        pit = "\n".join(f"  - {p}" for p in step.get("pitfalls", []))
        known = "\n".join(f"  - {k}: {v}" for k, v in facts.items())
        user = (f"QUESTION: {step['question']}\n\n"
                f"LAW (precomputed public analysis, verified against the statute):\n"
                f"{self._law_block(entries, step['authority'])}\n\n"
                f"PITFALLS to avoid:\n{pit}\n\n"
                f"Verified facts (already checked against the account):\n{known}\n\n"
                f"CLIENT ACCOUNT (confidential):\n<<<\n{self.narrative}\n>>>\n\n"
                'Return JSON: {"answer": "yes" | "no" | "unclear", '
                '"reasoning": "2-4 sentences applying the law to the facts", '
                '"evidence": ["sentences or phrases copied EXACTLY from the CLIENT ACCOUNT"], '
                '"authority": ["provision coordinates you relied on, e.g. uk/ukpga/1996/18/s95"]}')
        try:
            out = llm.device_chat_json(SYS_STEP, user, max_tokens=500)
        except ValueError:
            out = {"answer": "unclear", "reasoning": "model returned no valid JSON"}
        g = ground(out, self.narrative, allowed)
        g.update(step_id=step["id"], question=step["question"], kind="model", value=None,
                 step_authority=step["authority"])
        return g

    def advise(self) -> dict:
        with RouterTripwire() as tw:
            task = self.select_task()
            fetched = self.fetch(task)
            if task is None:
                return dict(task=None, verdict="REFUSE-AND-FLAG: outside covered task types",
                            steps=[], fetched=fetched, router_calls=tw.calls, memo=None)
            proc = fetched["entries"].get(f"procedure:{task}")
            if proc is None:
                return dict(task=task, verdict="REFUSE: procedure failed integrity check",
                            steps=[], fetched=fetched, router_calls=tw.calls, memo=None)
            facts, quotes, rejected = self.extract_facts(proc)
            steps = [self.calc_step(s, fetched, facts, quotes) if s["kind"] == "calc"
                     else self.model_step(s, fetched["entries"], facts)
                     for s in proc["steps"]]
            decided = sum(s["status"] in ("grounded", "computed") for s in steps) / max(len(steps), 1)
            verdict = ("ANSWERED (every step computed or grounded)" if decided == 1 else
                       "ANSWERED WITH FLAGS" if decided >= GROUNDED_MIN else
                       "REFER: could not decide enough steps")
            memo = render_memo(proc, steps, fetched, facts, quotes, rejected, verdict)
            return dict(task=task, verdict=verdict, steps=steps, fetched=fetched, facts=facts,
                        facts_rejected=rejected, router_calls=tw.calls, memo=memo)


def render_memo(proc, steps, fetched, facts, quotes, rejected, verdict) -> str:
    E = fetched["entries"]
    out = [f"# Advice note: {proc['description']}",
           f"Drafted on-device ({llm.DEVICE_MODEL}). Status: {verdict}.", "",
           "## Verified client facts"]
    out += [f'- {k}: {v}  ("{quotes[k]}")' for k, v in facts.items()]
    if rejected:
        out.append(f"- ({len(rejected)} extracted fact(s) rejected: not literally in the account)")
    out.append("")
    for i, s in enumerate(steps, 1):
        out.append(f"## {i}. {s['question']}")
        val = f" — {s['value']}" if s.get("value") not in (None, "") else ""
        tag = "computed by code" if s["kind"] == "calc" else "on-device model"
        out.append(f"**Answer: {s['answer'].upper()}{val}** [{s['status']}; {tag}]")
        out.append(s["reasoning"])
        if s["evidence"]:
            out += [f'  > "{e}"' for e in s["evidence"]]
        if s.get("evidence_rejected"):
            out.append(f"  ! rejected {len(s['evidence_rejected'])} quote(s) not found in the account")
        out.append("Law (statute fetched by PIR, sha256-verified on device):")
        for c in (s["authority"] or s["step_authority"]):
            sha = hashlib.sha256(E.get("text:" + c, "").encode()).hexdigest()[:12]
            out.append(f"  - {c} (sha256 {sha})")
            for p in fetched["params"].get(c, {}).values():
                out.append(f'      "{p["quote"]}"')
        out.append("")
    return "\n".join(out)
