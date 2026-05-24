# Reviewer Notes for v2.3

v2.3 does not weaken the v2.2 runtime/INT8 evidence gates. It adds domain-positioning controls so that the project can move from road-damage pilot evidence toward medical-lesion-oriented evaluation without overclaiming.

## What changed from v2.2

1. README and docs now frame road-damage segmentation as a motivating pilot domain and prior evidence.
2. Medical lesion segmentation is described as the follow-up evaluation target, not as a claim of direct transfer.
3. `domain_context`, `claim_scope`, `clinical_claim_allowed`, and `clinical_claim_status` are added to generated matrices and reports.
4. `clinical_claim_allowed` is forced to `false` in package outputs.
5. `docs/domain_bridge_road_to_medical.md` documents the allowed bridge and forbidden claims.
6. `report.md` now includes Scope and Claim Boundary, Road-Damage Pilot Motivation, Medical Lesion Extension Rationale, Runtime Evidence Limitations, Low-Precision Claim Policy, and Clinical Claim Disclaimer sections.
7. `small_region_*` remains the primary generic terminology; `small_lesion_*` is retained only as a medical-extension alias.

## What did not change

- requested provider and executed provider remain separated.
- ONNX Runtime profile audit remains integrated.
- unique-node counts and node-event counts remain separate.
- fallback ratio is derived from profile evidence when available.
- backend and hardware claim gates remain explicit.
- INT8 graph audit remains integrated into the result matrix.
- empty-GT boundary samples remain excluded from boundary means.
- p90/p95/p99 latency fields remain part of the schema.
- OpenVINO actual device evidence remains required for strong NPU claims.
- missing baseline still stops evaluation when configured.
- NOT_RUN / FAILED / MISSING_BACKEND / SUCCESS style status separation remains supported by the compatibility log and report warnings.

## Remaining honest limits

- No medical dataset is bundled.
- No medical-domain validation is performed by this package.
- Road-damage pilot results do not imply lesion segmentation results.
- Hardware or INT8 compatibility does not imply medical safety.
- TPU/XLA evidence remains backend-specific and is not parsed as deeply as ONNX Runtime provider profiles.
