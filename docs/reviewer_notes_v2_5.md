# Reviewer Notes for v2.5

v2.5 is a narrative-polish pass. The README no longer repeats the same road-damage-to-medical bridge in the opening paragraph and the research-position section. The intended reading is now:

1. Road damage is the pilot domain.
2. Medical lesion segmentation is the follow-up target.
3. The connection is evaluation-risk similarity, not model or performance transfer.
4. Clinical claims remain forbidden.

The technical audit controls from v2.2/v2.4 remain unchanged.

## Review-facing fixes retained

- Nested `--out` directories are created for audit JSON outputs.
- TensorRT examples separate provider from device (`requested_provider=TensorRTExecutionProvider`, `actual_compiled_device=GPU`).
- Road-damage Markdown reports hide `small_lesion_*` columns by default.
- Mixed-domain reports include `Domain Context Distribution`.

## Still not claimed

- Road-damage performance does not imply medical-lesion performance.
- Medical-lesion terminology does not imply diagnostic readiness.
- Accelerator or INT8 compatibility does not imply medical safety.
