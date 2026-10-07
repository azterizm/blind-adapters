# blind-adapters masterplan

## Context

blind-counsel (github.com/azterizm/blind-counsel, main at 8b1bf8f, tag poc-v1) showed the text layer. The cloud precomputes reasoning over public law, the laptop fetches it by PIR, and Qwen3-8B applies it to the client account on the laptop. Results were 0 secrets to the cloud and 30 of 30 held-out conclusions correct. The pre-registered verdict was FAIL on one gate: the local task router sent 1 of 8 matters to the wrong task. It was out of scope and was stopped at REFER with no advice. The text layer itself had no failures. Routing sits upstream of both layers, so adapters do not fix it and the old verdict stands.

The product goal is a layered system. Factual needs go to the text PIR database. Reasoning needs go to 8B student LoRA adapters fetched by PIR. Prompts that neither layer solves get a user choice: FHE in the cloud, or a closed SOTA model sent the legal coordinates with user consent. This PoC builds and measures only the adapter layer. The docs disclose the full goal.

The new repo is ~/Code/blind-adapters, built from scratch in a new session. Code copied from blind-counsel carries a provenance header.

## Corrections to the pasted design

These are checked numbers for Qwen3-8B (hidden 4096, 36 layers, MLP 12288, 8 KV heads). The PoC measures each claim instead of assuming it.

| Pasted claim | Check |
|---|---|
| Rank-16 all-linear adapter is 25 to 35 MB at FP16 | 43.7M params. 87 MB at FP16, about 23 MB at 4-bit. |
| SimplePIR response is 1.2 to 1.5x the payload | 4x at q=2^32, p=256. Lower p means more expansion. |
| Online query 16 to 64 KB after a small hint | Hint is n x record length x 4 bytes. For an 8 MB record in one row that is 32 GB at n=1024. Splitting the adapter into c chunks divides the hint by c and multiplies queries and server work by c. |
| 0.53 s server scan for 8 GB | Holds for one query at 15 GB/s. A chunked fetch runs c queries as one batched product. |
| LRU cache gives 0 ms for repeat domains | It leaks if fetches depend on cache hits. The fetch schedule stays fixed whatever the cache holds. |
| LoRA generalizes and text does not | This is hypothesis H1, tested below. |
| Hot swap under 20 ms | Measured in MLX. |

The missing baseline is download-all. The client downloads the whole library, with perfect privacy and no server work. It wins whenever N x S fits on the device; at PoC scale, 2 adapters is about 50 to 180 MB. PIR matters only for large libraries, so the report gives the crossover N.

## Hypotheses (pre-registered at M1)

An arm is one configuration of the laptop model, run on the same matters with the same questions. All arms get the same statute text by PIR, the same fact checks, calculators and grounding gate.

| Arm | Model on laptop | Extra input | Purpose |
|---|---|---|---|
| B | Qwen3-8B, no adapter | None | Baseline. What the student does alone. |
| T | Qwen3-8B, no adapter | Precomputed reasoning by text PIR | The blind-counsel text layer. |
| A | Qwen3-8B plus adapter fetched by PIR | None | The new adapter layer. |
| A+T | Qwen3-8B plus adapter | Precomputed reasoning by text PIR | Both layers together, as in the product goal. |
| Teacher | Gemini, run on the fictional matters | None | Upper bound. Never sees client data in the product. |

- H1, quality. On held-out matters, adapter arm A is at least as good as text arm T on in-scope questions. On novel in-domain questions, A beats base B and is at least as good as T.
- H2, safety. Wrong answers that pass the grounding gate in A are no more than in T.
- H3, delivery. Adapter PIR decodes byte-exact and the sha256 matches the manifest. The cloud's view is independent of the adapter index. Cost is reported against download-all and text PIR.
- H4, privacy. 0 secrets reach the cloud, and the fetch schedule is fixed per matter.

## Roles and where they run

| Role | Runs on | Sees |
|---|---|---|
| Teacher, training data | Laptop script calling the Gemini router (gemini-3.8-flash-high, localhost:8317) | Public statute, fictional matters |
| Adapter training | Modal GPU | Public synthetic data |
| PIR server, adapter library | Modal | PIR queries only |
| Device: PIR client, verify, adapter swap, Qwen3-8B inference, harness, eval | Laptop | Client account |

The device role stays on the laptop. Moving it to Modal is trust displacement, which was settled in blind-counsel. Modal takes every cloud role. Laptop inference is proven: 135 s per matter in the final round.

## Carry over from blind-counsel

Copy, do not import. Add a header `# from blind-counsel 8b1bf8f <path>`.

| From | Use |
|---|---|
| pir/simplepir.py | Base for adapter PIR. Change to n=1024, Gaussian sigma 6.4 per the SimplePIR paper, p chosen by its correctness bound per row count, chunked layout, batched queries, blocked uint matmul. Keep server_view_independent_of_index. |
| cloud/precompute.py, config.py | Statute DB for all of ERA 1996 from legal-rag-router (read-only, LEGAL_RAG_ROUTER env). |
| cloud/richdb.py, cloud/reason.py, results/precomputed/*.json | Text arm T, unchanged entries. |
| device/reasoner.py | verify_facts, ground, RouterTripwire, select_task, SealedDevice. Add adapter fetch and swap. |
| device/calc.py (with 8b1bf8f fix), lawtext.py | Calculators for all arms. |
| device/egress.py | Egress guard. |
| llm/__init__.py | Router client, MLX device client. Add an adapter_path argument and swap support. Pinned base mlx-community/Qwen3-8B-4bit rev 545dc425. |
| eval/final.py, eval/final_round_key.json | Matter and key format, dev matters. The old held-out matters become dev data. |
| .router_key | Copy the local file. Git-ignored, never committed. |
| LICENSE (AGPL-3.0), CITATION.cff, SECURITY.md | Templates. Voice rules apply. |

## Scope

The PoC builds:
- 2 real adapters, uk_unfair_dismissal and uk_redundancy_payment. These are the same tasks as T, so the comparison is fair.
- Filler adapters of the same size, random bytes, used only to scale the PIR library to N = 2, 16, 128, 1024.

The PoC does not build the factual-or-reasoning router, FHE, the consented SOTA route, hierarchical PIR, the LRU cache or other domains. The docs disclose them as the goal.

## Milestones

### M0. Repo setup
- git init ~/Code/blind-adapters. Add a venv, requirements (mlx-lm, numpy, modal, pytest) and .gitignore (artifacts/, .router_key, .venv).
- Copy the files above and save this plan as MASTERPLAN.md.
- Build the statute DB.
- Smoke tests: simplepir correctness, egress block, calc.

### M1. Pre-registration, before any training data
- Write 12 held-out fictional matters:
  - 4 in-scope, like the old set;
  - 6 novel in-domain, outside the precomputed entries: constructive dismissal s95(1)(c), automatic unfair reasons with no qualifying period (s99, s100, s103A with s108(3)), and refusal of suitable alternative employment s141;
  - 2 out-of-scope.
- Write a fixed public question list per task, including the novel questions. All arms get the same questions and the same PIR-fetched statute text.
- Write the key with the expected answer per field. Commit the key and the rule, then record the commit hash.
- Draft rule, finalized with the user here:

| Rule | Condition |
|---|---|
| 1 | A correct is at least T correct on in-scope fields. |
| 2 | A minus B is at least 2 of 6 novel matters, and A is at least T on novel fields. |
| 3 | Wrong answers passing the gate: A is at most T. |
| 4 | Leakage is 0. |
| 5 | Every adapter fetch decodes byte-exact and the sha256 matches. |
| 6 | The M3 parity gate passed. |

### M2. Teacher data (Gemini router, public inputs only)
- The teacher generates fictional matters across in-scope and novel issues per adapter, and gold answers in the device JSON schema (answer, reasoning, evidence quotes, authority).
- Audit every example:
  - evidence must be verbatim from the matter (verbatim from cloud/reason.py);
  - authority quotes must be verbatim from the statute;
  - failures are dropped and logged.
- Contamination guard:
  - the teacher never sees held-out text;
  - drop any example that shares an 8-gram with a held-out matter.
- Training prompts use exactly the device's model_step layout with statute text and without precomputed reasoning, so A is distinct from T.
- Target 600 to 1000 examples per adapter, cached under artifacts/teacher/.

### M3. Training on Modal
- Run a Modal app with PEFT QLoRA on Qwen/Qwen3-8B, the same source as the MLX conversion.
  - LoRA: rank 16, alpha 32, all 7 linear projections;
  - training: lr 1e-4, 2 epochs, seq 2048, chat template with enable_thinking=False;
  - infrastructure: one A100 or L40S, and a Modal volume for public data and outputs.
- Convert the PEFT adapter to MLX adapter format (key mapping, scale alpha/r, adapter_config.json).
- Parity gate on 20 dev prompts:
  - mean NLL of the teacher answer on MLX 4-bit plus adapter is within 0.05 of Modal bf16 plus adapter;
  - the adapter lowers NLL against base on both.
- Produce delivery variants at FP16 and INT8, and record size and dev accuracy.
- Expected cost is under $10. List and delete the Modal apps and volumes at the end.

### M4. Adapter PIR
- Library: each record is the serialized adapter padded to the max size S, plus fillers up to N.
- The public manifest holds index, name, size and sha256, and every client downloads it in full. The device rejects a sha256 mismatch.
- Chunked SimplePIR: c chunks per adapter, N x c rows, and c queries batched into one product. Sweep c to get hint, upload, download, server time and client time.
- Throughput is measured two ways:
  - own numpy code on Modal, labelled unoptimized;
  - the reference SimplePIR and DoublePIR Go implementation (github.com/ahenzinger/simplepir) on Modal, on the same DB sizes.
- Network figures are measured laptop to Modal: hint download, query upload, answer download and wall time.
- Baselines: download-all (N x S bytes), and text PIR from blind-counsel (710 rows of 2048 bytes, 56 queries).
- Privacy: run the view-independence test, and check that every matter uses exactly c adapter queries plus 56 text queries, out-of-scope included.

### M5a. Routing fix (from the blind-counsel report)
- A failed precondition, such as dismissed = NO, becomes a hard refuse-and-flag instead of REFER.
- The cloud precomputes a public scope note per task for the router.
- Routing is scored as its own line, on the 2 out-of-scope held-out matters plus the 10 in-scope matters. A routing miss does not count against the adapter rule, and adapter results do not count for routing.

### M5. Device integration (laptop)
- Flow per matter:
  1. select_task, with the router tripwire;
  2. text PIR fetch, then adapter PIR fetch;
  3. sha256 verify;
  4. load or swap the adapter into the resident MLX model, timing the swap;
  5. run the question list.
- Arms B, T, A and A+T share the harness: fact verification, calculators, grounding gate, greedy decoding. Only the weights and the precomputed context differ.
- A teacher reference row runs Gemini on the fictional matters as an upper bound. The docs state that the product never sends client data to the teacher.

### M6. Evaluation
- Run all arms on the dev set. Freeze. Run held-out once and apply the rule.
- Report per arm: correct, wrong, wrong past the gate, refused, latency, swap time, fetch bytes and leakage.

### M7. Docs
- README, REPORT, SECURITY and CITATION in the user's voice.
- REPORT discloses the product goal and its layers, and cites FHE costs from blind-counsel (7B at 4.6 to 11.8 h per 500 tokens).
- SECURITY covers the adapter as attack surface:
  - a cloud-authored adapter can be backdoored;
  - mitigations: a public manifest of hashes, every client gets the same bytes, the gate and calculators bound wrong answers, and the egress guard blocks exfiltration;
  - backdoor detection is not solved.
- Publish only on the user's call.

## Verification
- `pytest` checks:
  - adapter PIR recovers byte-exact at every (N, c) at small scale;
  - the view is independent of the index;
  - a sha256 mismatch is rejected;
  - the egress guard blocks;
  - calculators;
  - the PEFT to MLX converter round trip.
- `python -m pir.adapters --sweep` prints the cost table against download-all and text PIR.
- `modal run train.py` produces adapters, and `python -m eval.parity` applies the parity gate.
- `HF_HUB_OFFLINE=1 python -m eval.final` runs the 4 arms and prints the verdict against the committed key.
