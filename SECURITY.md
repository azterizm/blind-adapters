# Security

This is a research proof of concept. Do not use it with real client data.

## Threat model

- The cloud is untrusted. This includes the operator, staff, other tenants, physical access and chip vendors. No TEE is used.
- The firm's laptop is the only trusted device. Securing it is the firm's responsibility.
- The cloud model receives public statute text and fictional matters it wrote itself. It never receives client data.
- Modal runs every cloud role: adapter training on public synthetic data and the PIR server. It sees PIR queries only.

## The adapter is an attack surface

The cloud authors the adapters. A cloud-authored adapter can be backdoored. It could, for example, give a wrong answer on a trigger phrase or try to make the device leak data.

Mitigations in this PoC:

- The adapter manifest is public. It lists the sha256 of every adapter. The device rejects any adapter whose hash does not match.
- Every client fetches the same bytes for the same adapter. A targeted adapter for one client would need a different hash, which other clients and auditors would see.
- The grounding gate and the calculators bound wrong answers. Every model answer needs verbatim evidence from the client account. Dates and amounts are computed by code, not by the model.
- The egress guard blocks exfiltration. The device sends only PIR queries, and the guard declassifies a query only if it looks like an LWE ciphertext. A backdoored adapter cannot add a network channel.

Not solved:

- Backdoor detection. Nothing in this PoC detects a backdoor in the adapter weights. A backdoor that produces wrong but grounded answers is limited only by the gate and the calculators.
- A public manifest helps only if someone audits it. The PoC does not include an audit process.

## Other known limits

- The egress guard is tested on the channels in the evaluation only. It is not audited for production.
- PIR parameters follow the SimplePIR paper (n 1024, q 2^32, sigma 6.4). The implementation is research grade and not reviewed for production. The noise is a rounded Gaussian, not an exact discrete Gaussian.
- The PIR hides which adapter is fetched. It does not hide that a fetch happened or its timing. Every matter issues the same number of queries, including refusals.

## Reporting

Report issues through GitHub private vulnerability reporting on the Security tab, or to abdullah@memonsystems.com. Do not open public issues for vulnerabilities.
