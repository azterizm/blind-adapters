# blind-adapters

Adversarial research on confidential legal AI without trusted hardware (TEE): reasoning adapters fetched by private information retrieval.

Research proof of concept. Not legal advice. All client matters in this repository are fictional. Do not use it with real client data.

Part of a two-repository study. Builds on [blind-counsel](https://github.com/azterizm/blind-counsel) (tag poc-v1).

## What this is

I set out to remove the TEE from confidential legal AI. The test was strict. I treated the cloud operator, staff, other tenants and chip vendors as untrusted. [blind-counsel](https://github.com/azterizm/blind-counsel) showed that standard cloud RAG leaked 16 private tokens from one matter where the blind-cloud design leaked 0. This repository adds reasoning adapters fetched by PIR. Together they record what worked, what failed and what it costs. The result shows where a TEE is needed.

## Findings

1. The adapter layer passed its pre-registered rule. On 12 held-out matters the adapter matched the precomputed text layer on in-scope questions (22 of 22) and beat the base model on novel questions by 2 of 6 matters. It made 1 wrong answer that passed the grounding gate, against 3 for the text layer.
2. Privacy held. 0 secrets reached the cloud. Every matter issued 56 text and 1024 adapter queries, including refusals. 48 of 48 adapter fetches decoded byte-exact and matched the public manifest.
3. Routing failed its own rule, 10 of 12. Two redundancy matters were routed to unfair dismissal. Both out-of-scope matters were refused.
4. The INT8 adapter is 46.6 MB. One fetch downloads 186 MB. PIR beats downloading the whole library only from 16 adapters upward. At 1,024 adapters one fetch takes minutes of server time with this layout.

Full details in [REPORT.md](REPORT.md).

## Limits

Without a TEE the cloud cannot compute on client data. That removes most of what cloud AI is used for.

- All work on the matter runs on the firm's device. Here that is an 8B model on a laptop at about one minute per matter.
- Only task types prepared in advance are covered. A new task needs new precomputed reasoning or a new adapter. Open-ended prompts are refused.
- Routing to the right task failed in both repositories: 7 of 8 in blind-counsel, 10 of 12 here.
- An open model run in the cloud on client data without hardware protection is not safe. Keyed obfuscation leaked 100%.
- FHE takes 4.6 to 11.8 hours per 500 tokens for a 7B model.
- Fetching adapters by PIR from a library of 1,024 takes 4 to 27 minutes of server time per fetch.
- Samples are small. Nothing has had legal review.

## Where TEEs fit

A TEE runs the model on client data inside isolated, encrypted hardware memory and proves what is running through attestation. [Prem Enclave](https://www.premai.io/prem-enclave) does this on standard GPU clusters with Intel TDX, AMD SEV and NVIDIA confidential computing, on-premise or in the customer's own cloud account. It removes the limits above.

| Need | This research | TEE such as Prem Enclave |
|---|---|---|
| Compute on client data in the cloud | No | Yes |
| Open-ended prompts and new tasks | No, fixed task types only | Yes |
| Model size | Bounded by the device, 8B tested | Large open models |
| Trust in chip vendors | None | Required |

The trade is trust in the chip vendors and their attestation. [Prem's security model](https://docs.prem.io/security-model) states this. It lists hardware side channels and metadata such as timing and payload size as outside its protection.

Use a TEE such as Prem Enclave when the work needs cloud compute on client data. That covers most production legal and healthcare work, including drafting, open-ended analysis and large models.

The design here fits one narrow case. The tasks are fixed and run over public law, the firm will not trust any chip vendor, and laptop compute is enough.

## Controls for any backend

These run on the firm side, with a TEE or without one.

| Repository | What it does | Status |
|---|---|---|
| [legal-rag-router](https://github.com/azterizm/legal-rag-router) | Binds a query to the cited statute provision and refuses invented law. No model, no network. 0.14 ms median, 2.10 ms p99 on Apple M4. | Measured on 2,112 queries |
| [legal-rag-mask](https://github.com/azterizm/legal-rag-mask) | Replaces names, parties and deal terms with consistent pseudonyms before text leaves the firm. | Prototype, not yet evaluated on real contracts |
| [legal-rag-verifier](https://github.com/azterizm/legal-rag-verifier) | Checks each sentence against the statute it was given while the model generates, and rolls back a failed sentence. | Proof of concept |

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
