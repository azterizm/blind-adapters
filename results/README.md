# Results

Curated evidence behind REPORT.md. All client matters are fictional.

| Path | Content |
|---|---|
| held_out/memos/ | One note per held-out matter and arm (B, T, A, AT, TEACHER): answer, status, reasoning, verbatim evidence and authority per step |
| held_out/full_results.json | Every step of every arm, with verified and rejected facts |
| held_out/run.log | Console output of the held-out run, including the rule and verdicts |
| held_out/run_attempt1_aborted.log | The first held-out attempt, stopped by a session restart before any matter finished |
| held_out_results.json | Held-out summary: scores per field, routing, rule, verdicts, query counts |
| dev/ and dev_results.json | The same for the dev set (blind-counsel final-round matters) |
| preregistration.txt | The pre-registration commit |
| parity.json | M3 parity gate: NLL on Modal bf16 and MLX 4-bit, dev accuracy of FP16 and INT8 |
| train_*.json | Training runs: examples, time, hyperparameters, package versions, loss log |
| teacher_audit.json | Teacher data: examples kept, drops by reason, answers per step |
| adapter_manifest.json | Public adapter manifest: name, size, sha256 |
| pir_sweep.json | Adapter PIR costs: numpy and reference Go server time, laptop client time |
| pir_network_N16_c1024.json | Laptop to Modal fetch of both adapters |
| questions.json | The public question list |
| precomputed/ | blind-counsel precomputed reasoning and the scope notes |

Memo file names are matter id, then arm. A memo shows only the steps the arm ran. A matter stopped by a failed precondition ends at that step.
