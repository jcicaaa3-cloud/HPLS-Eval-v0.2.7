# Provider Execution Audit

The central audit question is not “which backend was requested?” but “which backend executed the nodes?”

## Why this matters

A session can be created with `TensorRTExecutionProvider`, `CUDAExecutionProvider`, or `OpenVINOExecutionProvider` and still execute part of the graph on CPU. This can happen because of unsupported operators, dynamic shapes, graph partitioning, precision mismatch, build failure, or failed fusion.

For a hardware comparison, reporting only the requested provider is not enough.

## Required v2.2 outputs

HPLS-Eval v2.2 carries these fields into `compatibility_normalized.csv`, `runtime_evidence_matrix.csv`, `condition_summary.csv`, and `report.md` when `provider_profile_path` is available:

| Field | Meaning |
|---|---|
| `unique_node_provider_counts` | Unique graph-node counts by actual execution provider. This is the main denominator for backend claims. |
| `node_event_provider_counts` | Raw node-event counts. Useful when a profile contains many inference repetitions. |
| `requested_provider_unique_node_fraction` | Fraction of unique profiled graph nodes executed by the requested provider. |
| `requested_provider_event_fraction` | Same idea, but using raw node-event counts. |
| `fallback_unique_node_counts` | Unique graph-node counts executed by non-requested providers. |
| `fallback_event_counts` | Raw node-event fallback counts. |
| `fallback_ratio_effective` | Measured node-level fallback ratio if profile exists; otherwise manually logged fallback ratio. |
| `fallback_ratio_source` | `provider_profile_node_fraction` or `compatibility_log`. |
| `backend_claim_status` | Anti-overclaim status such as `profile_verified_requested_provider`, `partial_fallback_below_threshold`, or `log_only_no_provider_profile`. |
| `backend_claim_allowed` | Boolean gate for saying the requested backend mostly executed the graph. |
| `hardware_claim_allowed` | Boolean gate that also considers actual-device evidence. |
| `runtime_evidence_grade` | A/B/C/D evidence grade for runtime claims. |

Backward-compatible aliases such as `provider_node_counts`, `provider_event_counts`, and `requested_provider_fraction` are still emitted, but papers should prefer the explicit names above.

## Why unique nodes and raw events are both reported

ONNX Runtime profiles can include repeated node events when a runner performs many inference iterations before calling `sess.end_profiling()`. Raw event counts are still useful, but they can make “node count” look inflated. v2.2 therefore reports:

- unique graph-node provider counts for the main claim;
- raw node-event provider counts for debugging and sanity checks.

## Interpretation

- `requested_provider_unique_node_fraction >= 0.95`: strong evidence that the run mostly executed on the requested provider.
- `0 < requested_provider_unique_node_fraction < 0.95`: partial execution; report fallback clearly.
- `requested_provider_unique_node_fraction == 0`: the requested provider did not execute profiled nodes.
- missing profile: log-only evidence; do not use for a strong hardware claim.
- `hardware_claim_allowed = false`: do not phrase the row as a verified TensorRT/CUDA/NPU/TPU result.

## Suggested wording for papers

Use this style:

> The TensorRT FP16 row is reported with provider-profile evidence. ONNX Runtime profiling showed 128/132 unique graph nodes on TensorRTExecutionProvider and 4/132 on CPUExecutionProvider, corresponding to a requested-provider unique-node fraction of 0.970 and measured fallback ratio of 0.030.

Avoid this style:

> The model was evaluated on TensorRT.

The second sentence hides whether TensorRT actually executed the graph.
