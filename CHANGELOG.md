# Changelog

## 0.2.7 - Report polish and evidence-path clarity

- Replaced weak prose-derived warning labels with stable warning codes such as `MISSING_COMPATIBILITY_ROWS` and `TOTAL_MISSING_PREDICTIONS`.
- Added grouped warning presentation through `reporting.group_related_warnings` while keeping the raw warning log available.
- Documented that relative evidence paths in `compatibility.csv` are resolved from the compatibility log directory.
- Split wide runtime-evidence report sections into provider placement, fallback, and device evidence tables.
- Kept runtime, fallback, INT8, boundary, and claim-boundary gates unchanged.

## 0.2.6 - Presentation cleanup and warning deduplication

- Deduplicated condition-level warnings by warning code plus model/runtime/precision.
- Kept sample-level warnings distinct.
- Clarified `actual_compiled_device` as a device label, not a provider name.
- Reorganized `report.md` into narrower sections for review and PDF/Word conversion.
- Kept `small_lesion_*` aliases in raw CSV files while hiding them from road-damage Markdown reports by default.

## 0.2.5 - README and review-facing cleanup

- Reframed the README around the road-damage pilot and the planned medical-lesion evaluation target.
- Preserved the claim boundary: no direct transfer claim and no clinical claim.
- Retained audit-output, provider/device, report-alias, and mixed-domain report fixes.

## 0.2.4 - Audit output, device evidence, and report presentation

- Created parent directories for nested `audit-profile --out` and `audit-onnx --out` paths.
- Corrected TensorRT scaffold rows so provider and device fields are not mixed.
- Added `reporting.show_medical_alias_columns` with default `false`.
- Added a Domain Context Distribution section to generated reports.

## 0.2.3 - Road-damage pilot to medical-lesion evaluation bridge

- Added domain and claim metadata to matrices and reports.
- Forced `clinical_claim_allowed=false` in generated outputs.
- Added road-damage motivation, medical-extension rationale, runtime-evidence limitation, low-precision claim policy, and clinical-claim disclaimer sections.
- Standardized `small_region_*` as the primary generic terminology, with `small_lesion_*` retained as a future medical-domain alias.

## 0.2.2 - Runtime and INT8 evidence integration

- Integrated INT8 audit evidence into normalized compatibility and summary matrices.
- Added safer provider-count fields and requested-provider fractions.
- Made missing FP32 baseline handling a hard gate when configured.
- Clarified TPU/XLA backend-specific profiling requirements.
- Added small-region metric aliases for road-damage and visual-anomaly use.

## 0.2.1 / 0.2.0 lineage

- Added ONNX Runtime provider profile audit.
- Added INT8 graph audit.
- Added boundary empty-GT exclusion.
- Added p90/p95/p99 latency fields.
- Added OpenVINO actual-device evidence fields.
