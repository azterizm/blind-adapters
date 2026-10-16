# Index

Map of this repository, its evidence and the related repositories. Research proof of concept. All client matters are fictional.

## Documents

| File | Contents |
|---|---|
| [README.md](README.md) | Findings, limits, where TEEs fit, setup and run commands |
| [REPORT.md](REPORT.md) | Full write-up: design, teacher data, training, adapter PIR, routing fix, results |
| [MASTERPLAN.md](MASTERPLAN.md) | The plan written before the build, with the milestones M0 to M7 |
| [results/README.md](results/README.md) | What each evidence file is |
| [SECURITY.md](SECURITY.md) | Threat model, the adapter as attack surface, reporting |
| [CITATION.cff](CITATION.cff) | Citation |
| [LICENSE](LICENSE) | AGPL-3.0 |

## Findings to evidence

| Finding | Report section | Code | Evidence |
|---|---|---|---|
| Pre-registered rule and key, committed before training data | [Pre-registration](REPORT.md#pre-registration) | eval/matters.py, eval/heldout_key.json | [results/preregistration.txt](results/preregistration.txt) |
| Fixed question list for every arm | [Question list](REPORT.md#question-list) | cloud/questions.py | [results/questions.json](results/questions.json) |
| Teacher data and its audit | [Teacher data](REPORT.md#teacher-data) | cloud/teacher.py, adapters/data.py | [results/teacher_audit.json](results/teacher_audit.json) |
| Training on Modal and the parity gate | [Training and parity](REPORT.md#training-and-parity) | train.py, adapters/convert.py, eval/parity.py | [results/train_uk_unfair_dismissal.json](results/train_uk_unfair_dismissal.json), [results/train_uk_redundancy_payment.json](results/train_uk_redundancy_payment.json), [results/parity.json](results/parity.json) |
| Adapter PIR cost against download-all | [Adapter PIR](REPORT.md#adapter-pir), [Download-all](REPORT.md#download-all) | pir/adapters.py, pir/modal_app.py, pir/goref/ | [results/pir_sweep.json](results/pir_sweep.json) |
| Laptop to Modal fetch, byte-exact | [Network](REPORT.md#network-laptop-to-modal) | pir/modal_app.py | [results/pir_network_N16_c1024.json](results/pir_network_N16_c1024.json), [results/adapter_manifest.json](results/adapter_manifest.json) |
| Routing fix and scope notes | [Routing fix](REPORT.md#routing-fix) | cloud/scope.py, device/reasoner.py | [results/precomputed/scope.json](results/precomputed/scope.json) |
| Dev set, all arms | [Dev set](REPORT.md#dev-set) | eval/final.py | [results/dev_results.json](results/dev_results.json), [results/dev/](results/dev/) |
| Adapter rule PASS, routing FAIL 10 of 12 | [Verdict](REPORT.md#verdict), [Held-out results](REPORT.md#held-out-results) | eval/final.py | [results/held_out_results.json](results/held_out_results.json), [results/held_out/run.log](results/held_out/run.log) |
| Per-matter notes for each arm | [Held-out results](REPORT.md#held-out-results) | device/reasoner.py | [results/held_out/memos/](results/held_out/memos/) |
| No-TEE limit and where a TEE is needed | [Scope and position](REPORT.md#scope-and-position) | | README.md, Where TEEs fit |

Memo names are matter id, then arm. IS is in scope, NV is novel, OS is out of scope. Arms are B base, T text, A adapter, AT adapter plus text, TEACHER the upper bound.

## Code

| Path | Role | Contents |
|---|---|---|
| cloud/precompute.py | Cloud | Public statute PIR database |
| cloud/reason.py, cloud/richdb.py | Cloud | Precomputed reasoning and the text PIR database, from blind-counsel |
| cloud/questions.py | Cloud | Public question list per task |
| cloud/scope.py | Cloud | Public scope note per task, for the router |
| cloud/teacher.py | Cloud | Teacher matters and gold answers, public inputs only |
| adapters/data.py | Cloud | Tokenize teacher examples with the device template |
| train.py | Cloud, Modal | QLoRA training on Qwen3-8B |
| adapters/convert.py | Cloud | PEFT to MLX conversion, FP16 and INT8 variants |
| pir/simplepir.py | Both | SimplePIR over Z_q |
| pir/adapters.py | Both | Adapter library over chunked SimplePIR, client benchmark, cost sweep |
| pir/modal_app.py | Cloud, Modal | PIR server benchmarks and the network fetch |
| pir/goref/ | Cloud, Modal | Timing of the reference SimplePIR Go implementation |
| device/reasoner.py | Laptop | Task selection, fetch, verify, adapter swap, fact checks, grounding gate |
| device/prompts.py | Laptop | Model-step prompt, shared with the teacher data builder |
| device/calc.py | Laptop | Date and amount calculators |
| device/egress.py | Laptop | Taint-tracked egress guard |
| llm/__init__.py | Both | Router client, MLX client with adapter swap |
| eval/matters.py | | Fictional dev and held-out matters |
| eval/heldout_key.json, eval/final_round_key.json | | Held-out key and the blind-counsel key used for dev |
| eval/parity.py | | Parity gate on the laptop |
| eval/final.py | | Runs the arms and applies the rule |
| tests/ | | PIR, adapter PIR, converter, calculator and egress tests |
| lawtext.py, config.py | | Legal text parsing, path to legal-rag-router |

## Local artifacts

artifacts/ is git-ignored. It holds adapters, the library and private outputs that the scripts regenerate.

| Path | Written by |
|---|---|
| artifacts/cloud/, artifacts/cloud_rich/ | cloud.precompute, cloud.richdb, cloud.questions, cloud.scope |
| artifacts/teacher/ | cloud.teacher |
| artifacts/adapters/ | adapters.data, train.py, adapters.convert |
| artifacts/library/ | pir.adapters --library |
| artifacts/device_out/ | eval.final. Private outputs. Curated copies are in results/. |
| artifacts/*.log | Console logs of each run. Curated copies are in results/. |

## Related repositories

| Repository | Relation |
|---|---|
| [blind-counsel](https://github.com/azterizm/blind-counsel) | Prior work. Text layer, blind cloud, keyed obfuscation attacks. Source of the copied code, precomputed reasoning and dev matters. |
| [legal-rag-router](https://github.com/azterizm/legal-rag-router) | Dependency. UK legislation and the statute router. Set LEGAL_RAG_ROUTER. |
| [legal-rag-mask](https://github.com/azterizm/legal-rag-mask) | Control for any backend. Pseudonymises client text. |
| [legal-rag-verifier](https://github.com/azterizm/legal-rag-verifier) | Control for any backend. Checks generated sentences against statute. |

## Versions

| Ref | State |
|---|---|
| tag poc-v1 | Code and evidence behind the scores in REPORT.md |
| main | poc-v1 plus documentation and this index |
