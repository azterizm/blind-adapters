# from blind-counsel 8b1bf8f device/reasoner.py
"""Device side: the laptop applies public law to the private client account with its own
model (Qwen3-8B, MLX). Nothing private leaves.

Per matter:
  1. select   -- the base model picks the task from the PUBLIC task index and scope
                 notes (a closed choice) or "none" -> refuse-and-flag. Local only.
  2. fetch    -- one fixed batch of `fetch_budget` text PIR queries (statute text and
                 precomputed reasoning), then exactly c adapter PIR queries for one
                 adapter. The same counts for every matter, including a refusal (dummy
                 rows and a dummy adapter pad the schedule). Fetching happens before any
                 reasoning, so private reasoning never steers what is asked of the cloud.
  3. verify   -- sha256 of every text entry and of the adapter against the public
                 manifest; the cloud's statutory quotes and calculator parameters are
                 re-checked against the fetched statute.
  4. facts    -- the base model extracts dates and amounts; each is kept only if its
                 quote is verbatim from the account and contains that value.
  5. steps    -- the fixed public question list. Calculator steps are code
                 (device/calc.py). Model steps run on the resident model, with or
                 without the adapter and the precomputed analysis depending on the arm,
                 and pass the grounding gate. A grounded NO on a precondition stops the
                 matter as REFUSE-AND-FLAG.

Arms: B (base, statute), T (base, statute + precomputed analysis), A (adapter,
statute), AT (adapter, statute + precomputed analysis), TEACHER (cloud model, statute;
fictional matters only, outside the tripwire, as an upper bound).

A tripwire makes any call to the cloud LLM router during a matter raise.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
import re
import time

import numpy as np

import llm
from adapters.convert import from_record
from cloud.reason import FACT_FIELDS, P, verbatim
from device import calc
from device.egress import EgressGuard
from device.prompts import SYS_STEP, step_prompt
from lawtext import amounts_in, dates_in, numbers_in
from pir.adapters import AdapterFetcher
from pir.simplepir import PIRClient, PIRServer, cells_for, pack, plaintext_bits, unpack

RICH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "cloud_rich"))
GROUNDED_MIN = 0.6          # below this fraction of decided steps -> REFER
ARMS = {"B": dict(adapter=False, analysis=False), "T": dict(adapter=False, analysis=True),
        "A": dict(adapter=True, analysis=False), "AT": dict(adapter=True, analysis=True),
        "TEACHER": dict(adapter=False, analysis=False, cloud=True)}
UNCAPPED = {P + "s100", P + "s103A"}     # s124(1A)


# ------------------------------------------------------------------ the cloud
class RichCloud:
    """Untrusted: holds the public rich DB (statute text and precomputed reasoning),
    answers PIR queries, records its view."""

    def __init__(self, seed: int = 2024):
        raw = np.load(os.path.join(RICH, "pir_db.npy"))            # [710, 2048] bytes
        self.b = plaintext_bits(raw.shape[0])
        self.row_bytes = raw.shape[1]
        L = cells_for(self.row_bytes, self.b)
        self.server = PIRServer(np.stack([pack(r.tobytes(), self.b, L) for r in raw]), self.b, seed=seed)
        self.seed = seed
        self._H = None

    @property
    def m(self) -> int:
        return self.server.m

    def hint(self):
        if self._H is None:
            self._H = self.server.hint()
        return self._H

    def answer(self, Qu):
        return self.server.answer(Qu)

    @property
    def views(self):
        return self.server.views


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


class SealedDevice:
    """One matter on the laptop. Public artifacts (catalogue, task index, question list,
    scope notes, adapter manifest) are synced in full by every device."""

    def __init__(self, text_cloud: RichCloud, adapter_cloud, adapter_fetcher: AdapterFetcher,
                 narrative: str, seed: int = 7):
        self.cloud, self.acloud, self.afetch = text_cloud, adapter_cloud, adapter_fetcher
        self.narrative = narrative
        self.catalogue = json.load(open(os.path.join(RICH, "catalogue.json")))
        self.tasks = json.load(open(os.path.join(RICH, "tasks.json")))
        self.questions = json.load(open(os.path.join(RICH, "questions.json")))
        self.scope = json.load(open(os.path.join(RICH, "scope.json")))
        self.budget = json.load(open(os.path.join(RICH, "meta.json")))["fetch_budget"]
        self.pir = PIRClient(text_cloud.seed, text_cloud.hint(), text_cloud.m, text_cloud.b, rng_seed=seed)
        self.rng = random.Random(seed)
        words = set(re.findall(r"[A-Za-z£0-9][\w£,.'-]{4,}", narrative))
        self.guard = EgressGuard(sorted(words))

    # 1 ---------------------------------------------------------------------
    def select_task(self) -> str | None:
        llm.use_adapter(None)
        menu = []
        for t, v in self.tasks.items():
            s = self.scope.get(t, {})
            menu.append(f'- "{t}": {v["description"]}\n  Applies when: {s.get("one_line", "")}\n'
                        + "".join(f"    + {x}\n" for x in s.get("in_scope", []))
                        + "  Does not apply to:\n" + "".join(f"    - {x}\n" for x in s.get("out_of_scope", [])))
        out = llm.device_chat_json(SYS_SELECT, (
            "Task types this practice covers:\n" + "\n".join(menu) +
            f"\nClient account:\n<<<\n{self.narrative}\n>>>\n\n"
            'Which task type is this matter? Reply {"task": "<one id from the list>"} '
            'or {"task": "none"} if none of them fits.'), max_tokens=60)
        t = str(out.get("task", "none"))
        return t if t in self.tasks else None

    # 2 + 3 ---------------------------------------------------------------------
    def _text_keys(self, task: str | None) -> list[str]:
        if not task:
            return []
        keys = list(self.tasks[task]["entry_keys"])
        keys += [f"text:{c}" for c in self.questions[task]["statute"] if f"text:{c}" not in keys]
        return keys

    def fetch(self, task: str | None) -> dict:
        """The fixed schedule: `budget` text queries in one batch, then c adapter queries."""
        keys = self._text_keys(task)
        wanted = [r for k in keys for r in range(self.catalogue[k]["row"],
                                                 self.catalogue[k]["row"] + self.catalogue[k]["n_rows"])]
        plan = wanted + [self.rng.randrange(self.cloud.m) for _ in range(self.budget - len(wanted))]
        assert len(plan) == self.budget, "matter exceeds the fixed fetch budget"
        self.rng.shuffle(plan)                               # order carries no meaning
        t0 = time.perf_counter()
        Qu, S = self.pir.query(plan)
        self.guard.send("pir", Qu, note="rich-db rows")
        rows = self.pir.decode(self.cloud.answer(Qu), S)
        got = {r: unpack(rows[j], self.cloud.b, self.cloud.row_bytes) for j, r in enumerate(plan)}
        text_s = time.perf_counter() - t0
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
        # adapter: exactly c queries for one adapter (a dummy one when refusing)
        names = [e["name"] for e in self.afetch.manifest]
        idx = names.index(task) if task in names else self.rng.randrange(len(names))
        t0 = time.perf_counter()
        adapter_raw = self.afetch.fetch(idx, self.acloud, guard=self.guard)   # raises on sha mismatch
        adapter_s = time.perf_counter() - t0
        lay = self.afetch.lay
        return dict(entries=entries, bad_sha=bad_sha, quotes_rechecked=(q_ok, q_all), params=params,
                    text_queries=len(plan), adapter_queries=lay["c"], adapter_index=idx,
                    adapter=adapter_raw if task in names else None,
                    adapter_sha_ok=True, text_s=text_s, adapter_s=adapter_s,
                    bytes=dict(text_up=Qu.nbytes, text_down=4 * len(plan) * self.pir.H.shape[1],
                               adapter_up=lay["upload"], adapter_down=lay["download"]))

    # 4 ---------------------------------------------------------------------
    def extract_facts(self, task: str) -> tuple[dict, dict, list]:
        llm.use_adapter(None)
        fdef = self.questions[task]["facts"]
        spec = "\n".join(f'  "{n}" ({v["type"]}): {v["note"]}' for n, v in fdef.items())
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
    @staticmethod
    def law_block(entries: dict, authority: list[str]) -> str:
        """The precomputed public analysis for a step (arms T and AT)."""
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
        return "\n".join(lines) if lines else "(no precomputed analysis for these provisions)"

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

    def model_step(self, step: dict, fetched: dict, facts: dict, arm: dict, proc: dict | None) -> dict:
        entries = fetched["entries"]
        statute = {c: entries[f"text:{c}"] for c in step["authority"] if f"text:{c}" in entries}
        analysis = None
        s = dict(step)
        if arm["analysis"]:
            analysis = self.law_block(entries, step["authority"])
            s["pitfalls"] = next((x["pitfalls"] for x in (proc or {}).get("steps", []) if x["id"] == step["id"]), [])
        shown = {k: (v.isoformat() if isinstance(v, dt.date) else v) for k, v in facts.items()}
        user = step_prompt(s, statute, shown, self.narrative, analysis)
        t0 = time.perf_counter()
        try:
            if arm.get("cloud"):
                from cloud.teacher import _retry
                out = _retry(lambda: llm.cloud_chat_json(SYS_STEP, user))
            else:
                out = llm.device_chat_json(SYS_STEP, user, max_tokens=500)
        except ValueError:
            out = {"answer": "unclear", "reasoning": "model returned no valid JSON"}
        g = ground(out, self.narrative, set(step["authority"]))
        g.update(step_id=step["id"], question=step["question"], kind="model", value=None,
                 step_authority=step["authority"], seconds=time.perf_counter() - t0)
        return g

    def run_steps(self, task: str, fetched: dict, facts: dict, quotes: dict, arm_name: str) -> tuple[list, str]:
        arm = ARMS[arm_name]
        proc = fetched["entries"].get(f"procedure:{task}")
        steps, by_id = [], {}
        for step in self.questions[task]["steps"]:
            if step["kind"] == "calc":
                s = self.calc_step(step, fetched, facts, quotes)
                # pre-registered derived rules (cloud/questions.py)
                if step["id"] == "payment_amount" and s["status"] == "computed" and any(
                        by_id.get(k, {}).get("answer") == "no" and by_id[k]["status"] == "grounded"
                        for k in ("by_reason_of_redundancy", "suitable_alternative")):
                    s.update(answer="no", value=0, reasoning="No payment: the dismissal is not by reason of "
                             "redundancy, or the employee unreasonably refused an offer covered by s141(2).")
                if step["id"] == "compensation_cap" and s["status"] == "computed":
                    au = by_id.get("automatic_unfair_reason", {})
                    if au.get("answer") == "yes" and au.get("status") == "grounded" and UNCAPPED & set(au["authority"]):
                        s.update(value="uncapped", reasoning="s124(1A): no cap where the dismissal is "
                                 "automatically unfair under s100 or s103A.")
            else:
                s = self.model_step(step, fetched, facts, arm, proc)
            steps.append(s)
            by_id[step["id"]] = s
            if step.get("precondition") and s["answer"] == "no" and s["status"] == "grounded":
                return steps, f"REFUSE-AND-FLAG: precondition '{step['id']}' failed"
        scored = [s for s, q in zip(steps, self.questions[task]["steps"]) if q.get("scored", True)]
        decided = sum(s["status"] in ("grounded", "computed") for s in scored) / max(len(scored), 1)
        verdict = ("ANSWERED (every step computed or grounded)" if decided == 1 else
                   "ANSWERED WITH FLAGS" if decided >= GROUNDED_MIN else
                   "REFER: could not decide enough steps")
        return steps, verdict

    def advise(self, arm_name: str, task: str | None = None, routed: bool = True,
               facts_cache: tuple | None = None) -> dict:
        """One matter in one arm. `task` overrides routing (the evaluation scores arms with
        the key's task); `facts_cache` reuses the shared base-model fact stage."""
        arm = ARMS[arm_name]
        t_start = time.perf_counter()
        ctx = RouterTripwire() if not arm.get("cloud") else _NoTrip()
        with ctx as tw:
            if routed:
                task = self.select_task()
            fetched = self.fetch(task)
            res = dict(arm=arm_name, task=task, fetched={k: v for k, v in fetched.items()
                                                         if k not in ("entries", "adapter")})
            if task is None:
                return dict(res, verdict="REFUSE-AND-FLAG: outside covered task types", steps=[],
                            router_calls=tw.calls, seconds=time.perf_counter() - t_start, swap_s=None)
            swap = None
            facts, quotes, rejected = facts_cache or self.extract_facts(task)
            if arm["adapter"]:
                w, cfg = from_record(fetched["adapter"])
                swap = llm.use_adapter(w, cfg)
            else:
                llm.use_adapter(None)
            steps, verdict = self.run_steps(task, fetched, facts, quotes, arm_name)
            llm.use_adapter(None)
            return dict(res, verdict=verdict, steps=steps, facts={k: str(v) for k, v in facts.items()},
                        facts_rejected=rejected, router_calls=tw.calls, swap_s=swap,
                        seconds=time.perf_counter() - t_start)


class _NoTrip:
    calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
