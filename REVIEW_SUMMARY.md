# Review Summary

HPLS-Eval v0.2.7 focuses on two review risks.

1. Runtime claims can be wrong when a requested accelerator silently falls back to another provider.
2. Domain claims can be overstated when road-damage pilot evidence is presented as medical-lesion validation.

This version keeps the evidence gates added in the earlier 0.2.x releases and improves how those gates appear in the generated report.

## Retained evidence gates

- Requested provider is stored separately from observed provider evidence.
- Provider unique-node counts and node-event counts are reported separately.
- Fallback evidence remains visible in result matrices.
- Strict INT8 graph audit fields are integrated into condition summaries.
- Empty GT boundary masks do not inflate boundary means.
- p90, p95, and p99 latency fields are preserved.
- OpenVINO/NPU actual-device evidence is tracked.
- Missing FP32 baseline can stop evaluation when configured.
- Road-damage Markdown reports hide `small_lesion_*` aliases by default.
- Mixed-domain matrices include a Domain Context Distribution section.

## v0.2.7 report changes

- Condition-level warnings are deduplicated by warning code plus model/runtime/precision.
- Warning summaries use stable codes instead of first-word prose labels.
- Related runtime evidence warnings can be grouped by condition.
- `actual_compiled_device` is documented as a device label only.
- `report.md` separates claim gates, provider placement, fallback, device evidence, latency, INT8 audit, segmentation, and summary tables.
- Relative evidence paths in `compatibility.csv` are documented as relative to the compatibility log directory.

## Limitation

This package is not medical validation. Medical-lesion terminology is kept for a planned evaluation target only. Real medical data, a domain-specific protocol, and separate validation are required before making any medical-domain conclusion.
