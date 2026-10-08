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

## Goal

A law firm should get strong AI help on a client matter without the client data leaving its own device. The cloud is not trusted. This includes the operator, staff, other tenants and chip vendors, so no TEE is used. The cloud sees public law only.

## Layers

Each layer covers one kind of need. The cloud sees no client data and, through PIR, not which item was fetched. This repository builds and measures only the adapter layer.

| Need | Layer | Cloud | Device | Repository |
|---|---|---|---|---|
| Facts and statute text | Text database by PIR | Precomputes reasoning over public law | Fetches, verifies quotes, applies to the account | [blind-counsel](https://github.com/azterizm/blind-counsel) |
| Reasoning on new fact patterns | LoRA adapters by PIR | Distils a large model into 8B adapters on public and synthetic data | Fetches, verifies the hash, swaps into the local model | blind-adapters |
| Prompts neither layer solves | FHE in the cloud, or a frontier model sent the legal coordinates with user consent | Computes on encrypted data, or answers on coordinates only | Asks the user | Not built |

A router that decides between the factual and reasoning layers is not built.

## Why combine the layers

No single layer covers legal work. Text covers what the cloud anticipated and gives exact citations. Adapters handle fact patterns the text did not anticipate. Escalation covers the rest at a higher cost and only with consent.

On 12 held-out matters, the base model scored 35 of 50 with 4 wrong answers past the gate. Text scored 40 with 3. The adapter scored 42 with 1. Adapter plus text also scored 42 with 1.

The layers share one set of rules, so they can be stacked without widening what the cloud sees. Code does every date and amount. Every answer needs verbatim evidence from the account. The grounding gate turns unsupported answers into unclear. The egress guard blocks client data from leaving the device.

The cloud does the expensive work once for all clients: precomputing reasoning and distilling a large model into adapters. Each matter costs about one minute on the laptop and a fixed batch of PIR queries.

## Open problems

- Routing failed in both repositories: 7 of 8 in blind-counsel, 10 of 12 here. The router must pick the right task or refuse before any layer helps.
- Adapter PIR is costly at scale. Below 16 adapters, downloading the whole library is cheaper. At 1,024 adapters one fetch takes 4 to 27 minutes of server time with the tested layout.
- FHE costs 4.6 to 11.8 hours per 500 tokens for a 7B model. The consented frontier route reveals the legal coordinates.
- Samples are small. The precomputed reasoning and adapters have had no legal review.

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
