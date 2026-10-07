"""The device's model-step prompt, shared by the device and the teacher-data builder so
training prompts match inference prompts exactly.

Every arm gets the question, the PIR-fetched statute text of the step's authority, the
verified facts and the client account. Arms T and A+T also get the precomputed public
analysis and pitfall notes; arms B and A do not. Training prompts use the B/A layout.
"""
from __future__ import annotations

import json

SYS_STEP = ("You assist a UK employment solicitor on a confidential matter. Answer ONE "
            "question by applying ONLY the law provided to the client's account. Do not "
            "assume facts that are not in the account. Reply with a single JSON object.")

ANSWER_SPEC = ('Return JSON: {"answer": "yes" | "no" | "unclear", '
               '"reasoning": "2-4 sentences applying the law to the facts", '
               '"evidence": ["sentences or phrases copied EXACTLY from the CLIENT ACCOUNT"], '
               '"authority": ["provision coordinates you relied on, e.g. uk/ukpga/1996/18/s95"]}')


def statute_block(statute: dict[str, str], authority: list[str]) -> str:
    return "\n".join(f"[{c}]\n<<<\n{statute[c]}\n>>>" for c in authority if c in statute)


def step_prompt(step: dict, statute: dict[str, str], facts: dict, narrative: str,
                analysis: str | None = None) -> str:
    """`analysis` is the precomputed law block (arms T and A+T) or None (arms B and A)."""
    parts = [f"QUESTION: {step['question']}",
             f"STATUTE (fetched by PIR, sha256-verified on the device):\n{statute_block(statute, step['authority'])}"]
    if analysis is not None:
        pit = "\n".join(f"  - {p}" for p in step.get("pitfalls", []))
        parts.append(f"PRECOMPUTED ANALYSIS (public, verified against the statute):\n{analysis}")
        parts.append(f"PITFALLS to avoid:\n{pit}")
    known = "\n".join(f"  - {k}: {v}" for k, v in facts.items())
    parts.append(f"Verified facts (already checked against the account):\n{known}")
    parts.append(f"CLIENT ACCOUNT (confidential):\n<<<\n{narrative}\n>>>")
    parts.append(ANSWER_SPEC)
    return "\n\n".join(parts)


def target_json(answer: str, reasoning: str, evidence: list[str], authority: list[str]) -> str:
    """The training target, in the same key order the device asks for."""
    return json.dumps(dict(answer=answer, reasoning=reasoning, evidence=evidence,
                           authority=authority), ensure_ascii=False)
