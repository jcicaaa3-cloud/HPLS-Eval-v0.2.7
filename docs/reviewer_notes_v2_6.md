# Reviewer Notes v2.6

This revision handles final presentation and review-friction issues.

## What changed

- Warning de-duplication now uses `warning_code + model_id + runtime_id + precision` when a condition identity is present. This keeps one actionable warning per condition/gate instead of repeating similar hardware, runtime, and fallback warnings.
- `actual_compiled_device` is documented only as a device label (`CPU`, `GPU`, `CUDA-GPU`, `NPU`, `TPU`). Execution-provider names stay in `requested_provider`.
- `report.md` no longer prints one very wide runtime matrix. It is split into claim gate, provider audit, device/backend evidence, latency, INT8 graph audit, segmentation summary, and retention/HPLS tables.
- CSV outputs remain full-fidelity artifacts. For advisor-facing PDF/Word/slides, prefer the split Markdown report unless raw CSV alias columns are being discussed deliberately.

## Preserved boundaries

Road-damage results remain pilot evidence. Medical-lesion terminology remains a future evaluation target and does not imply clinical validation, diagnostic readiness, or direct transfer from road damage to medicine.
