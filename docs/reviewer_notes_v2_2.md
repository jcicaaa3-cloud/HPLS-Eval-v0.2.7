# Reviewer Notes for v2.2

This file records the design decisions made after critical reviews of the v2 package line.

## What was tightened

1. Backend execution evidence separates requested provider from executed provider. The evaluator writes `runtime_evidence_matrix.csv` before interpreting segmentation quality.
2. Provider profile parsing reports unique graph-node counts and raw node-event counts separately.
3. Safer explicit provider columns were added: `unique_node_provider_counts`, `node_event_provider_counts`, `requested_provider_unique_node_fraction`, and `requested_provider_event_fraction`.
4. Backend claims have explicit gates: `backend_claim_status`, `backend_claim_allowed`, and `hardware_claim_allowed`.
5. Tail latency aliases are normalized to `latency_ms_p50`, `latency_ms_p90`, `latency_ms_p95`, `latency_ms_p99`, `latency_ms_min`, and `latency_ms_max`.
6. INT8 graph audit is now integrated into the evaluation matrix through `int8_audit_path`.
7. INT8 graph audit uses Conv-like coverage and quantized Conv weight-path evidence, not just the existence of INT8 tensors.
8. Boundary metrics exclude empty GT boundaries, while false predicted boundaries on empty GT masks are recorded as `boundary_fp_empty_gt` and `boundary_fp_empty_gt_count`.
9. OpenVINO/NPU rows get `device_claim_status`, and NPU rows whose actual device resolves to CPU/GPU are warned.
10. TPU/XLA rows are marked as requiring backend-specific evidence; ORT provider profile fields are not treated as sufficient for TPU verification.
11. The public wording was moved from the older domain wording to road-damage / visual-anomaly segmentation for LiteRaceSegNet-style work.
12. `reporting.fail_on_missing_baseline` now actually stops evaluation when the configured FP32 baseline has no valid rows.

## Remaining honest limits

- ONNX Runtime profiling is only as good as the profile produced by the installed runtime. If the runtime does not emit provider metadata, the row is not treated as strong evidence.
- Provider-node fractions do not prove electrical power use, kernel-level tensor-core use, or accelerator occupancy. They only prove runtime-level node placement.
- INT8 graph structure does not prove INT8 kernel execution. The profile/backend log still needs to support the runtime claim.
- OpenVINO `AUTO` requires actual-device evidence. Without it, the result remains weak even if latency looks good.
- TPU/XLA evidence is backend-specific and currently not parsed as deeply as ONNX Runtime EP profile JSON.
