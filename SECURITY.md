# Security

This is a research proof of concept. Do not use it with real client data.

## Threat model

- The cloud is untrusted. This includes the operator, staff, other tenants, physical access and chip vendors. No TEE is used.
- The firm's laptop is the only trusted device. Securing it is the firm's responsibility.
- The cloud model receives public statute text only.

## Attack code

The code in attacks/ targets the keyed weight shift in keyed/. That scheme is part of this project. The attacks evaluate it.

## Known limits

- The egress guard is tested on the channels in the evaluation only. It is not audited for production.
- PIR parameters are research grade and not reviewed for production.
- Final round verdict: FAIL, task routing 7 of 8.

## Reporting

Report issues through GitHub private vulnerability reporting on the Security tab, or to abdullah@memonsystems.com. Do not open public issues for vulnerabilities.
