# Reasoning adapters fetched by private information retrieval

This report tests whether a law firm can add task reasoning to a laptop model by fetching a LoRA adapter from an untrusted cloud, without the cloud learning which adapter was fetched.

It builds on blind-counsel (github.com/azterizm/blind-counsel, tag poc-v1). There, the cloud precomputed reasoning over public law as text, the laptop fetched it by PIR, and Qwen3-8B applied it to the client account on the laptop. 0 secrets reached the cloud and 30 of 30 held-out conclusions were correct. The pre-registered verdict was FAIL on task routing, 7 of 8. The text layer had no failures.

Research proof of concept. Not legal advice. All client matters are fictional.

---

## Verdict

The adapter layer passes. All six pre-registered rules held on the 12 held-out matters.

Routing fails its own rule, 10 of 12. Two redundancy matters were routed to unfair dismissal. Both out-of-scope matters were refused.

| Rule | Result |
|---|---|
| R1 in-scope fields | A 22/22, T 22/22, B 22/22. PASS |
| R2 novel matters | A 5/6, B 3/6, difference 2. Novel fields A 5, T 4. PASS |
| R3 wrong answers past the gate | A 1, T 3. PASS |
| R4 privacy | 0 secrets reached the cloud. 0 cloud model calls during matters. Every matter in every arm issued 56 text and 1024 adapter queries. PASS |
| R5 delivery | 48 of 48 adapter fetches byte-exact with sha256 match. PASS |
| R6 parity | Passed. PASS |

The adapter matched the precomputed text layer on in-scope questions and made fewer wrong answers that passed the gate. On novel matters it beat the base model by 2 of 6 and the text layer by 1 of 6. These are small margins on a small set. Routing, which sits upstream of both layers, failed again.

---

## Product goal

The goal is a layered system. This PoC builds and measures only the adapter layer.

| Need | Layer | Status |
|---|---|---|
| Facts and statute text | Text database fetched by PIR | Built in blind-counsel |
| Reasoning | 8B student LoRA adapters fetched by PIR | Built here |
| Prompts neither layer solves | User choice: FHE in the cloud, or a closed frontier model sent the legal coordinates with user consent | Not built |

Not built here: the router that decides between the factual and reasoning layers, FHE, the consented frontier route, hierarchical PIR, an adapter cache, and other legal domains.

FHE remains far from practical for this use. blind-counsel cites Llama-2-7B under FHE at 33 to 85 s per token on 8 GPUs, which is 4.6 to 11.8 h per 500 tokens. The laptop arms here take 63 to 82 s per matter.

---

## Design

### Roles

| Role | Runs on | Sees |
|---|---|---|
| Teacher, training data | Laptop script calling gemini-3.8-flash-high through a local router | Public statute, fictional matters it wrote |
| Adapter training | Modal, one A100-40GB per adapter | Public synthetic data |
| PIR server, adapter library | Modal | PIR queries only |
| Device: PIR client, verification, adapter swap, Qwen3-8B, evaluation | Laptop (Apple M4, 16 GB) | Client account |

The device role stays on the laptop. Moving it to a hosted GPU only displaces trust.

### Arms

Every arm answers the same public question list on the same matters, with the same PIR-fetched statute text, fact checks, calculators and grounding gate. Routing and fact extraction run once per matter on the base weights and are shared by all arms. Arms differ only on the model-judged steps.

| Arm | Model on laptop | Extra input |
|---|---|---|
| B | Qwen3-8B 4-bit, no adapter | None |
| T | Qwen3-8B 4-bit, no adapter | Precomputed reasoning by text PIR (blind-counsel) |
| A | Qwen3-8B 4-bit plus adapter fetched by PIR | None |
| AT | Qwen3-8B 4-bit plus adapter | Precomputed reasoning |
| Teacher | gemini-3.8-flash-high | None. Upper bound on fictional matters. The product never sends client data to it. |

### Question list

Two tasks, one adapter each: uk_unfair_dismissal and uk_redundancy_payment. The in-scope steps reuse the blind-counsel questions. Two novel steps cover law outside the precomputed entries.

| Task | Model steps | Calculator steps |
|---|---|---|
| Unfair dismissal | dismissed (precondition), automatic_unfair_reason (novel: ss.99, 100, 103A with s.108(3)), fairness (not scored) | qualifying_service, time_limit, compensation_cap |
| Redundancy payment | dismissed (precondition), by_reason_of_redundancy, suitable_alternative (novel: s.141) | relevant_date, qualifying_service, payment_amount |

A grounded NO on dismissed stops the matter as REFUSE-AND-FLAG. The payment is 0 when the dismissal is not by reason of redundancy or s.141 removes the right. The compensatory cap is lifted when the automatic reason is s.100 or s.103A (s.124(1A)). These rules were fixed before any run.

### Pre-registration

The held-out matters, the key and the rule were committed at f7857640 before any teacher data existed. The evaluation refuses to score held-out matters if these files differ from that commit, and runs the held-out set once.

| Rule | Condition |
|---|---|
| R1 | In-scope fields (4 matters, 22 fields): A correct is at least T correct and at least B correct. |
| R2 | Novel matters (6): A right minus B right is at least 2, and A is at least T on novel fields. |
| R3 | Wrong answers that passed the gate, all 10 in-domain matters: A is at most T. |
| R4 | 0 secrets reach the cloud in any arm, and every matter issues exactly 56 text and c adapter queries. |
| R5 | Every adapter fetch decodes byte-exact and its sha256 matches the manifest. |
| R6 | M3 parity gate passed. |

Routing has its own verdict: 12 of 12 held-out matters routed to the key's task, or to none for the 2 out-of-scope matters. A misrouted matter later stopped by a precondition still counts as a miss.

The 12 held-out matters are 4 in-scope, 6 novel (constructive dismissal yes and no, automatic unfair dismissal under s.99 and s.103A, s.141 refusal unreasonable and reasonable) and 2 out of scope (a wage deduction while employed, a tenancy deposit). The 10 blind-counsel final-round matters are the dev set.

---

## Teacher data

The teacher wrote 330 fictional matters per task from seeded specs, then answered every model question in the device JSON format with a statute quote per authority. The teacher never saw a held-out matter. An example was dropped when evidence was not verbatim from the matter, a statute quote was not verbatim, an authority was outside the step, the answer disagreed with the spec, or the matter shared an 8-gram with a held-out matter.

| Task | Examples kept | 8-gram drops | Answer disagreed with spec | Statute quote not verbatim |
|---|---|---|---|---|
| Unfair dismissal | 870 | 2 | 39 | 1 |
| Redundancy payment | 874 | 2 | 10 | 6 |

Router overload (HTTP 503) failed 77 matters on the first pass. They were retried, not dropped. Training prompts use the device's own model-step layout with statute text and without precomputed reasoning. They were tokenized with the device's MLX tokenizer and chat template, because the Qwen/Qwen3-8B tokenizer config changed after the MLX conversion.

## Training and parity

QLoRA on Qwen/Qwen3-8B (weights unchanged since the commit the MLX conversion used): rank 16, alpha 32, all 7 linear projections, lr 1e-4, 2 epochs, sequence 3072. The plan said 2048. The novel step prompt is about 2,900 tokens. 20 dev examples per task were held back for the parity gate.

| Adapter | Train examples | A100 time | Params |
|---|---|---|---|
| Unfair dismissal | 847 | 42 min | 43,646,976 |
| Redundancy payment | 851 | 27 min | 43,646,976 |

Mean NLL of the teacher answer on the 20 dev prompts:

| Adapter | Modal bf16 base | Modal bf16 adapter | MLX 4-bit base | MLX 4-bit adapter | Gap |
|---|---|---|---|---|---|
| Unfair dismissal | 1.004 | 0.223 | 0.971 | 0.233 | 0.010 |
| Redundancy payment | 1.119 | 0.256 | 1.095 | 0.272 | 0.016 |

The gate needs a gap of 0.05 or less and a lower NLL with the adapter on both sides. It passed for both adapters.

| Delivery record | Size | Dev accuracy UD | Dev accuracy redundancy |
|---|---|---|---|
| Base model | | 17/20 | 19/20 |
| FP16 | 87.3 MB | 17/20 | 20/20 |
| INT8 | 46.6 MB | 17/20 | 20/20 |

INT8 matched FP16 and is the delivered record. Swapping an adapter into the resident model took 9 to 30 ms in the evaluation, with one outlier of 225 ms.

Total Modal cost for training, the PIR benchmarks and the network test was about $6.6.

---

## Adapter PIR

The library holds every adapter padded to the largest record, S = 46.6 MB. Random filler records scale it to N. Each record is split into c chunks. A fetch is exactly c queries, sent as one batch and answered as one matrix product. The public manifest lists index, name, size and sha256, and the device rejects a mismatch.

Parameters follow the SimplePIR paper: n 1024, q 2^32, rounded Gaussian noise with sigma 6.4. The plaintext modulus comes from the correctness bound per row count and is capped at 2^8 so the cloud stores one byte per cell.

| N | c | Rows | Hint MB | Upload MB | Download MB | Server numpy s | Server Go s | Client s | Download-all MB |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 16 | 32 | 11,921 | 0.0 | 186 | 1.1 | 0.3 | - | 93 |
| 2 | 64 | 128 | 2,980 | 0.0 | 186 | 1.0 | 0.8 | 2.2 | 93 |
| 2 | 256 | 512 | 745 | 0.5 | 186 | 2.3 | 3.8 | 1.1 | 93 |
| 2 | 1024 | 2,048 | 186 | 8 | 186 | 4.2 | 9.3 | 1.0 | 93 |
| 16 | 16 | 256 | 11,921 | 0.0 | 186 | 9.4 | 3.3 | - | 745 |
| 16 | 64 | 1,024 | 2,980 | 0.3 | 186 | 4.4 | 8.2 | 2.0 | 745 |
| 16 | 256 | 4,096 | 745 | 4 | 186 | 11.2 | 16.5 | 1.1 | 745 |
| 16 | 1024 | 16,384 | 186 | 67 | 186 | 30.4 | 60.3 | 1.3 | 745 |
| 128 | 16 | 2,048 | 11,921 | 0.1 | 186 | 29.8 | 9.2 | - | 5,960 |
| 128 | 64 | 8,192 | 2,980 | 2 | 186 | 34.0 | 51.7 | 2.1 | 5,960 |
| 128 | 256 | 32,768 | 745 | 34 | 186 | 89.0 | 122 | 1.4 | 5,960 |
| 128 | 1024 | 131,072 | 186 | 537 | 186 | 251 | 432 | 4.4 | 5,960 |
| 1024 | 16 | 16,384 | 11,921 | 1 | 186 | 245 | - | - | 47,684 |
| 1024 | 64 | 65,536 | 2,980 | 17 | 186 | 237 | - | 2.3 | 47,684 |
| 1024 | 256 | 262,144 | 745 | 268 | 186 | 617 | - | 3.3 | 47,684 |
| 1024 | 1024 | 1,048,576 | 186 | 4,295 | 186 | 1,600 | - | - | 47,684 |

Times are for one fetch of one adapter. A dash means not measured: client time where the hint exceeds 3.1 GB or the upload exceeds 1.5 GB on the 16 GB laptop, and Go at N = 1024.

The server cost grows with N times c times S, because each of the c queries scans the whole library. At N = 1024 one fetch takes 4 to 27 minutes of a 48-core server in numpy. Small c keeps server work and upload low but makes the hint large (11.9 GB at c = 16). Large c keeps the hint small but makes the upload large (4.3 GB at N = 1024, c = 1024). No setting is cheap at N = 1024 with this layout.

The numpy server is this repository's code on a 48-core Modal container, labelled unoptimized. The Go column is the reference SimplePIR implementation (ahenzinger/simplepir, e9020b0) on a single thread, one full query times c. Its in-memory format needs 4 bytes per cell before packing, so N = 1024 (190 GB) was not run. Client time is the laptop building c queries and decoding c answers. At c = 16 the single-thread reference code is 3 times faster than numpy on 48 cores. At large c numpy is faster, because the batch becomes a real matrix product while the reference code answers queries one by one.

Text PIR from blind-counsel, for comparison: 710 rows of 2048 bytes, 56 queries per matter, hint 6.7 MB, upload 0.2 MB, download 0.4 MB.

### Network, laptop to Modal

N = 16, c = 1024, the two trained adapters plus 14 fillers, over a home connection.

| Step | Bytes | Time |
|---|---|---|
| Hint download, once per library version | 186 MB | 101 s |
| Fetch, unfair dismissal | 67 MB up, 186 MB down | 154 s wall, 27 s server |
| Fetch, redundancy payment | 67 MB up, 186 MB down | 156 s wall, 26 s server |

Both fetches decoded byte-exact and matched the manifest.

### Download-all

Downloading the whole library gives perfect privacy and needs no server work. It wins whenever the library fits the device. One PIR fetch plus the hint moves fewer bytes than download-all from N = 16 at c = 1024 (745 MB), N = 32 at c = 256, N = 128 at c = 64 and N = 512 at c = 16. At the PoC size of 2 adapters, download-all is 93 MB and is the better choice. PIR matters only for large libraries.

### Checks on the planning numbers

| Planning claim | Measured |
|---|---|
| Rank-16 all-linear adapter is 87 MB at FP16 | 87.3 MB FP16, 46.6 MB INT8 with scales |
| Response is 4x the payload at p = 2^8 | 186 MB downloaded for a 46.6 MB record |
| Hint is n x row length x 4 bytes | 186 MB at c = 1024, 11.9 GB at c = 16 |
| Fetch schedule fixed whatever the cache holds | No cache built; every matter issued 56 + 1024 queries |
| Hot swap under 20 ms | Median 20 ms on MLX, 9 to 30 ms, one outlier of 225 ms |

---

## Routing fix

blind-counsel routed a still-employed pregnancy matter to unfair dismissal. Two changes were made. The cloud writes a public scope note per task from the task description and question list, and the router reads it. A grounded NO on a precondition such as dismissed is a hard REFUSE-AND-FLAG instead of REFER. On the dev set routing was 10 of 10, including the matter blind-counsel misrouted.

---

## Dev set

The 10 blind-counsel final-round matters. Not part of the rule.

| Arm | Correct | Wrong past the gate | Mean s per matter |
|---|---|---|---|
| B | 44/48 | 0 | 65 |
| T | 43/48 | 2 | 82 |
| A | 47/48 | 0 | 63 |
| AT | 47/48 | 0 | 82 |

Most base-model misses were on automatic_unfair_reason. The base model answered NO without quoting the account, so the gate turned it into unclear. T answered YES wrongly on two matters and both passed the gate. The one miss shared by every arm was a calculator field on UD-A. The base model wrote the notice date as 2926-07-10, fact verification rejected it, and the calculator refused instead of guessing.

---

## Held-out results

| Arm | Correct of 50 | In-scope of 22 | Novel matters of 6 | Wrong past the gate | Mean s per matter | Mean swap ms |
|---|---|---|---|---|---|---|
| B | 35 | 22 | 3 | 4 | 51 | |
| T | 40 | 22 | 4 | 3 | 70 | |
| A | 42 | 22 | 5 | 1 | 58 | 39 |
| AT | 42 | 22 | 5 | 1 | 74 | 12 |
| Teacher | 43 | 22 | 6 | 0 | 52 | |

Mean s includes the 2 out-of-scope refusals for the local arms. The A swap mean includes one 225 ms outlier. The median is 20 ms.

Novel matters, novel field per arm (Y right, N wrong):

| Matter | Issue | Expected | B | T | A | AT | Teacher |
|---|---|---|---|---|---|---|---|
| NV-CD1 | Constructive dismissal after demotion and accusation | dismissed YES | N | Y | Y | Y | Y |
| NV-CD2 | Resignation after a change the contract allows | dismissed NO | N | N | N | N | Y |
| NV-AU1 | Dismissal after pregnancy, under 2 years | automatic YES | Y | Y | Y | Y | Y |
| NV-AU2 | Dismissal after a protected disclosure, under 2 years | automatic YES | Y | Y | Y | Y | Y |
| NV-SA1 | Same job offered on the same terms, refused without reason | keeps right NO | N | N | Y | Y | Y |
| NV-SA2 | Junior job 300 miles away at half pay, refused | keeps right YES | Y | Y | Y | Y | Y |

The adapter fixed NV-SA1, which neither B nor T answered correctly. B and T both granted a redundancy payment of £12,480 that s.141 removes, and both answers passed the gate. B also refused NV-CD1 as not dismissed, so a valid constructive dismissal claim got no advice.

NV-CD2 failed in every local arm. Each said a resignation after a contractually permitted shift change was a constructive dismissal and quoted the account. The adapter had 31 training examples with dismissed NO and did not learn this case. This is the one wrong answer past the gate for A.

7 of A's 8 wrong fields are calculator fields shared by every arm, the teacher row included. On NV-AU1 and NV-AU2 the base model quoted the dismissal sentence without its date. On NV-SA2 it wrote the start date one day early. Fact verification rejected each value and the calculators returned unclear instead of guessing. The fact stage runs on base weights and is not part of the adapter comparison.

The first held-out attempt was stopped by a session restart before any matter finished. No output was produced or read. The run reported here is the first complete one. Its log is kept with the aborted one in the private artifacts.

Routing misrouted IS-R1 and NV-SA2, both redundancy matters, to unfair dismissal. In the product flow both would have received unfair dismissal advice instead of a refusal. The scope notes fixed the blind-counsel miss and both out-of-scope matters here, but did not separate the two in-scope tasks.

---

## Limits

- The novel set is 6 matters. A difference of 2 matters is a small effect.
- The teacher wrote the training matters. Held-out matters were written by the author in a similar register. The 8-gram guard prevents copied text, not similar style.
- The evaluation's PIR server runs in the same process as the device at N = 2. The laptop-to-Modal path was measured separately at N = 16.
- Calculator fields depend on the shared base-model fact stage, so a fact error costs every arm the same field.
- The adapter is an attack surface. See SECURITY.md.
- PIR parameters and code are research grade.

## Reproduce

Commands are in README.md. Curated evidence is in results/.
