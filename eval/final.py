# from blind-counsel 8b1bf8f eval/final.py
"""Final round: score the rich-DB prototype against a key written BEFORE this ran
(eval/final_round_key.json), on 8 held-out matters the fixes were never tuned on,
plus the 2 round-1 matters as a dev set.

    HF_HUB_OFFLINE=1 python -m eval.final

Private outputs (memos, results) go to artifacts/device_out/final/ and never leave
the laptop.
"""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np

from device.reasoner import RichCloud, SealedDevice

HERE = os.path.dirname(__file__)
OUT = os.path.abspath(os.path.join(HERE, "..", "artifacts", "device_out", "final"))
KEY = json.load(open(os.path.join(HERE, "final_round_key.json")))
CALC_FIELDS = {"qualifying_service", "time_limit", "compensation_cap", "relevant_date", "payment_amount"}
MODEL_FIELDS = {"dismissed", "by_reason_of_redundancy"}

HELD_OUT = {
    "UD-A": "Note of client meeting, 9 September 2026. Client: Marcus Bell. Employer: Corvid Software Ltd. "
            "Mr Bell started work as a QA engineer on 2 March 2024. On 10 July 2026 he was given four weeks' "
            "notice of dismissal for alleged poor performance, and his employment ended when that notice "
            "expired on 7 August 2026. He had received no warnings and no performance review before the "
            "dismissal. His salary was £41,600 a year. He has not yet presented a claim to an employment tribunal.",
    "UD-B": "Note of client meeting, 15 June 2026. Client: Leah Fontaine. Employer: Brightwater Hotels Ltd. "
            "Ms Fontaine started work as a front-of-house manager on 1 June 2024. On 28 May 2026 the general "
            "manager told her in a meeting that she was dismissed with immediate effect after a guest complaint, "
            "and she was not given any notice or pay in lieu of notice. Her salary was £34,320 a year. "
            "She has not presented a claim.",
    "UD-C": "Note of client meeting, 20 April 2026. Client: Owen Hartley. Employer: Pinecrest Builders Ltd. "
            "Mr Hartley started work as a site supervisor on 5 October 2024. He was dismissed with immediate "
            "effect on 31 March 2026 after a dispute with a customer. His salary was £39,000 a year. "
            "He has not presented a claim.",
    "UD-D": "Note of client meeting, 5 May 2026. Client: Farah Siddiqui. Employer: Northgate Logistics Ltd. "
            "Ms Siddiqui worked as a transport planner from 14 January 2019. She was dismissed with immediate "
            "effect on 2 February 2026 for alleged gross misconduct. Her salary was £45,500 a year. She has not "
            "contacted ACAS and has not presented a claim.",
    "UD-E": "Note of client meeting, 1 July 2026. Client: Jonathan Price. Employer: Ardent Capital LLP. "
            "Mr Price was head of risk from 3 September 2018. He was dismissed with immediate effect on 12 June "
            "2026; the firm said his role was being restructured, but he believes this was a pretext and wants "
            "to bring an unfair dismissal claim. His salary was £182,000 a year. He has not presented a claim.",
    "R-A": "Note of client meeting, 14 August 2026. Client: Grace Mensah, aged 30. Employer: Lumen Retail Ltd. "
           "Ms Mensah worked as a store supervisor from 1 July 2020. On 3 July 2026 she was given written notice "
           "that her store would close and that her job would end on 31 July 2026; her employment ended on "
           "31 July 2026. No alternative job was offered. Her gross weekly pay was £500. She wants to know "
           "whether she is owed a statutory redundancy payment and how much.",
    "R-B": "Note of client meeting, 10 September 2026. Client: Daniel Reyes, aged 27. Employer: Quayside Print "
           "Ltd. Mr Reyes worked as a machine operator from 4 January 2025. On 7 August 2026 the company told "
           "him the print room was closing and gave him one week's notice of dismissal for redundancy, ending "
           "on 14 August 2026. His gross weekly pay was £520. He wants to know whether he is entitled to a "
           "statutory redundancy payment.",
    "OOS": "Note of client meeting, 22 September 2026. Client: Hannah Osei. Employer: Redfern Accountancy LLP. "
           "Ms Osei is still employed as a senior associate. Since she told her manager in June 2026 that she "
           "was pregnant, she has been removed from two client accounts and was passed over for a promotion "
           "that went to a less experienced colleague. She has not resigned and has not been dismissed. She "
           "wants to know what claim she might have.",
}

DEV = {
    "dismissal": "Note of client meeting, 15 May 2026. Client: Adaeze Okafor. Employer: Meridian Freight plc, "
                 "Daventry depot. Ms Okafor started work as a dispatch supervisor on 4 August 2023 under a "
                 "written contract of employment. On 12 March 2026 her manager, Dave Pritchard, telephoned her "
                 "and told her she was dismissed with immediate effect for gross misconduct, namely three days' "
                 "unauthorised absence from 16 to 18 February 2026. She says she emailed Mr Pritchard on 15 "
                 "February explaining that her daughter had been admitted to hospital, and she has kept a copy "
                 "of that email. No investigation meeting or disciplinary hearing took place and she was not "
                 "offered an appeal. She received a short dismissal letter dated 13 March 2026. Her gross pay "
                 "was £48,500 a year (about £932 a week). She has been unable to find new work. She has not yet "
                 "contacted ACAS or presented a claim to an employment tribunal.",
    "redundancy": "Note of client meeting, 2 June 2026. Client: Tomasz Wisniewski, aged 44. Employer: Halcyon "
                  "Print Ltd. He has worked as a press operator at the Leicester site since 1 March 2017. On 30 "
                  "April 2026 he received written notice that the Leicester site would close because the company "
                  "is moving all printing to its Glasgow site, and that his employment would end on 29 May 2026. "
                  "He was not offered a role in Glasgow or anywhere else. His gross weekly pay was £610. The "
                  "employer has told him he will receive a 'goodwill payment' but has said nothing about "
                  "statutory redundancy pay.",
}


def secrets_of(narrative: str) -> list[str]:
    toks = re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*|£[\d,]+|\d{1,2} [A-Z][a-z]+ \d{4}|\d{4}", narrative)
    return sorted({t for t in toks if len(t) >= 4})


def run(narrative: str) -> dict:
    cloud = RichCloud()
    t = time.time()
    r = SealedDevice(cloud, narrative).advise()
    r["seconds"] = time.time() - t
    blob = b"".join(q.astype(np.int64).tobytes() for q in cloud.seen)
    r["leaked"] = [s for s in secrets_of(narrative) if s.encode() in blob]
    r["n_secrets"] = len(secrets_of(narrative))
    r["queries"] = len(cloud.seen)
    return r


def score(key: dict, r: dict) -> list[dict]:
    """One row per scored field: expected, got, correct, and whether a wrong answer had
    nonetheless passed the grounding/computation gate."""
    rows = [dict(field="task", expected=key["task"], got=r["task"], correct=r["task"] == key["task"],
                 kind="task", gated=False)]
    steps = {s["step_id"]: s for s in r["steps"]}
    for field, exp in key.items():
        if field == "task":
            continue
        s = steps.get(field)
        if s is None:
            got = None
        elif field in ("compensation_cap", "payment_amount"):
            got = s.get("value")
        elif field == "relevant_date":
            got = s.get("value")
        else:
            got = s.get("answer")
        if isinstance(exp, (int, float)) and got is not None:
            ok = abs(float(got) - exp) < 0.5
        else:
            ok = got == exp
        rows.append(dict(field=field, expected=exp, got=got, correct=ok,
                         kind="calc" if field in CALC_FIELDS else "model",
                         gated=(not ok) and s is not None and s["status"] in ("computed", "grounded"),
                         status=s["status"] if s else "missing"))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    report, runs = {}, {}
    for split, matters in (("held_out", HELD_OUT), ("dev", DEV)):
        for mid, narrative in matters.items():
            r = run(narrative)
            rows = score(KEY[split][mid], r)
            runs[mid], report[mid] = r, dict(split=split, rows=rows)
            if r.get("memo"):
                open(os.path.join(OUT, f"{mid}.md"), "w").write(r["memo"])
            fair = next((s for s in r["steps"] if s["step_id"] == "fairness"), None)
            print(f"\n[{split}] {mid}: task={r['task']} | {r['verdict']} | {r['seconds']:.0f}s | "
                  f"queries {r['queries']} | leaked {len(r['leaked'])}/{r['n_secrets']} | router {r['router_calls']}")
            if r.get("facts_rejected"):
                print(f"   facts rejected: {[x['field'] + ':' + x['why'] for x in r['facts_rejected']]}")
            for row in rows:
                mark = "ok " if row["correct"] else "XX "
                print(f"   {mark}{row['field']:<24} expected {row['expected']!s:<12} got {row['got']!s:<12} "
                      f"{row.get('status', '')}")
            if fair:
                print(f"   (not scored) fairness: {fair['answer']} [{fair['status']}]")

    held = [row for m, v in report.items() if v["split"] == "held_out" for row in v["rows"]]
    def frac(rows):
        return (sum(r["correct"] for r in rows), len(rows))
    t_ok, t_n = frac([r for r in held if r["kind"] == "task"])
    c_ok, c_n = frac([r for r in held if r["kind"] == "calc"])
    m_ok, m_n = frac([r for r in held if r["kind"] == "model"])
    gated_wrong = [r for r in held if r["gated"]]
    counts = {runs[m]["queries"] for m in runs}
    secure = (len(counts) == 1 and all(not runs[m]["leaked"] and runs[m]["router_calls"] == 0 for m in runs))
    rule = {
        f"task selection {t_ok}/{t_n} (need {t_n}/{t_n})": t_ok == t_n,
        f"calculator fields {c_ok}/{c_n} (need 100%)": c_ok == c_n,
        f"model-judged fields {m_ok}/{m_n} (need >= 90%)": m_n == 0 or m_ok / m_n >= 0.9,
        f"wrong answers that passed the gate: {len(gated_wrong)} (need 0)": not gated_wrong,
        f"security: queries {sorted(counts)} per matter, 0 leaked, 0 router calls": secure,
    }
    dev = [row for m, v in report.items() if v["split"] == "dev" for row in v["rows"]]
    d_ok, d_n = frac(dev)
    print("\n" + "=" * 72)
    print("PRE-REGISTERED RULE (held-out matters)")
    for k, v in rule.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  dev set (round-1 matters, not part of the rule): {d_ok}/{d_n} fields correct")
    print(f"VERDICT: {'PASS' if all(rule.values()) else 'FAIL'}")
    print("=" * 72)
    json.dump(dict(report=report, rule={k: v for k, v in rule.items()}), open(os.path.join(OUT, "results.json"), "w"),
              indent=1, default=str)


if __name__ == "__main__":
    main()
