# blind-adapters

Confidential legal AI on an untrusted cloud without trusted hardware (TEE): reasoning adapters fetched by private information retrieval.

Research proof of concept. Not legal advice. All client matters in this repository are fictional. Do not use it with real client data.

Builds on [blind-counsel](https://github.com/azterizm/blind-counsel) (tag poc-v1).

## Results

1. The adapter layer passed its pre-registered rule. On 12 held-out matters the adapter matched the precomputed text layer on in-scope questions (22 of 22) and beat the base model on novel questions by 2 of 6 matters. It made 1 wrong answer that passed the grounding gate, against 3 for the text layer.
2. Privacy held. 0 secrets reached the cloud. Every matter issued 56 text and 1024 adapter queries, including refusals. 48 of 48 adapter fetches decoded byte-exact and matched the public manifest.
3. Routing failed its own rule, 10 of 12. Two redundancy matters were routed to unfair dismissal. Both out-of-scope matters were refused.
4. The INT8 adapter is 46.6 MB. One fetch downloads 186 MB. PIR beats downloading the whole library only from 16 adapters upward. At 1,024 adapters one fetch takes minutes of server time with this layout.

Full details in [REPORT.md](REPORT.md).

## Product goal

The goal is a layered system. This repository builds and measures only the adapter layer.

| Need | Layer | Status |
|---|---|---|
| Facts and statute text | Text database fetched by PIR | Built in blind-counsel |
| Reasoning | 8B student LoRA adapters fetched by PIR | Built here |
| Prompts neither layer solves | User choice: FHE in the cloud, or a closed frontier model sent the legal coordinates with user consent | Not built |

A router that decides between the factual and reasoning layers is not built.

## How it works

1. The cloud model writes fictional matters and gold answers from public statute text. Every quote is checked. Matters that share an 8-gram with a held-out matter are dropped.
2. Modal trains one LoRA adapter per task on Qwen3-8B. The adapter is converted to MLX for the laptop.
3. The cloud serves the adapters from a PIR library with a public manifest of sha256 hashes.
4. For each matter, the laptop sends a fixed batch of 56 text queries and c adapter queries. The cloud cannot tell which rows or which adapter were read.
5. The laptop verifies the adapter hash and swaps it into the resident model.
6. The local model answers the fixed question list on the client account. Code does the date and amount calculations. Every answer needs verbatim evidence from the account.

## Layout

```
adapters/     PEFT to MLX conversion, delivery records, training data
cloud/        statute DB, precomputed reasoning, question list, scope notes, teacher data
device/       laptop side: reasoner, prompts, calculators, egress guard
eval/         matters, pre-registered key, parity gate, evaluation
llm/          router client, local model with adapter swap
pir/          SimplePIR, adapter library, Modal benchmarks
results/      curated evidence
tests/        pytest
train.py      Modal training
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
git clone https://github.com/azterizm/legal-rag-router ../legal-rag-router
export LEGAL_RAG_ROUTER=../legal-rag-router
```

Requires Apple Silicon for the local model (MLX) and a Modal account for training and the PIR server.

## Run

```bash
.venv/bin/python -m pytest
.venv/bin/python -m cloud.richdb
.venv/bin/python -m cloud.questions
.venv/bin/python -m cloud.scope
.venv/bin/python -m cloud.teacher 330
HF_HUB_OFFLINE=1 .venv/bin/python -m adapters.data
.venv/bin/modal run train.py
HF_HUB_OFFLINE=1 .venv/bin/python -m eval.parity
.venv/bin/python -m pir.adapters --library int8
.venv/bin/modal run pir/modal_app.py::sweep
.venv/bin/python -m pir.adapters --client-bench
.venv/bin/python -m pir.adapters --sweep
.venv/bin/modal run pir/modal_app.py::network
HF_HUB_OFFLINE=1 .venv/bin/python -m eval.final --split dev
HF_HUB_OFFLINE=1 .venv/bin/python -m eval.final --split held_out --teacher
```

The teacher and scope notes use a local router at localhost:8317 to gemini-3.8-flash-high. It receives public statute text and fictional matters only.

## Data and licences

- Code: AGPL-3.0.
- UK legislation: Crown copyright, Open Government Licence v3.0, source legislation.gov.uk.
- Models: Qwen3-8B, Apache-2.0.
- SimplePIR reference implementation (benchmark only, not included): github.com/ahenzinger/simplepir, MIT.

## Citation and security

See [CITATION.cff](CITATION.cff) and [SECURITY.md](SECURITY.md).
