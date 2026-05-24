# HPLS-Eval Protocol

## 0. Research scope and domain bridge

HPLS-Eval is an evidence-first evaluation harness for small irregular segmentation under low-precision and runtime-dependent execution conditions. The package keeps road-damage segmentation as a motivating pilot domain and prior evidence. The follow-up research direction is narrowed toward medical-lesion-oriented evaluation, but the framework does not claim diagnostic readiness or clinical deployment suitability.

Road-damage pilot observations motivate the evaluation axes: boundary degradation, small-region miss, sparse-mask behavior, low-contrast confusion, low-precision sensitivity, runtime fallback, and tail-latency risk. These issues can also be relevant to medical lesion segmentation, but road-damage performance does not imply medical lesion performance.

## 1. Fixed evaluation axes

HPLS-Eval must be configured before looking at condition-specific results.

- Model axis: lightweight segmentation model IDs.
- Domain context axis: `road_damage_pilot`, `generic_visual_anomaly`, or `medical_lesion_target`.
- Dataset axis: fixed manifest and split.
- Runtime axis: processor-oriented runtime path, not only hardware name.
- Precision axis: FP32, FP16, INT8, or explicitly declared subset.
- Claim scope: `pilot_only`, `evaluation_framework`, or `medical_extension_planned`.

`clinical_claim_allowed` is forced to `false` by the package output.

## 2. Baseline condition

The baseline condition is defined by:

```yaml
baseline:
  runtime_id: onnxruntime_cpu
  precision: FP32
```

Retention metrics are computed against this fixed condition for the same model. If `reporting.fail_on_missing_baseline: true`, evaluation stops when the baseline condition has no valid prediction rows. This prevents a result table from silently reporting NaN retention values as if the matrix were complete.

## 3. Segmentation quality metrics

The framework records Dice, IoU, recall, boundary recall, boundary IoU, and small-region recall. Boundary radius and small-region threshold must be fixed before evaluation.

Boundary metrics follow a strict empty-GT policy: if a sample has no GT boundary, boundary recall and boundary IoU are `NaN` for that sample and are excluded from boundary means. The aggregate still reports `empty_gt_boundary_count`, `boundary_fp_empty_gt_count`, and `n_boundary_valid`.

## 4. Terminology and alias policy

`small_region_*` is the primary generic terminology used for road-damage and visual-anomaly experiments. `small_lesion_*` is retained as a medical-domain alias for future lesion segmentation experiments. These aliases must not be interpreted as evidence of clinical validity.

When writing a road-damage pilot report, prefer `small_region_recall` and `small_region_retention_srs`. Generated Markdown reports hide `small_lesion_*` columns by default through `reporting.show_medical_alias_columns: false`; CSV outputs keep aliases for compatibility. Use the medical aliases in reports only after medical-domain data and a domain-specific protocol are connected.

## 5. Runtime evidence

Latency and compatibility evidence must be measured externally under a documented runtime setup. HPLS-Eval does not invent latency. If latency is missing, RTG and runtime retention remain missing.

The preferred latency fields are:

- `latency_ms_p50` for RTG
- `latency_ms_p90`, `latency_ms_p95`, `latency_ms_p99` for tail-latency reporting
- `latency_ms_min`, `latency_ms_max` for spread/outlier reporting
- `compile_ms` for build overhead, kept separate from steady-state inference latency

## 6. Provider execution evidence

The requested runtime provider is not enough. For accelerated ONNX Runtime paths, attach provider profiling evidence where possible.

Minimum provider evidence:

- `requested_provider`
- `provider_profile_path`
- `unique_node_provider_counts`
- `node_event_provider_counts`
- `requested_provider_unique_node_fraction`
- `requested_provider_event_fraction`
- `fallback_unique_node_counts`
- `fallback_ratio_effective`

Backward-compatible names such as `provider_node_counts`, `provider_event_counts`, and `requested_provider_fraction` are still emitted, but papers should prefer the explicit unique-node and node-event names.

If this evidence is absent, the result is treated as log-only compatibility evidence. It may be useful during development, but it should not support a strong hardware-comparison claim.

## 7. INT8 graph evidence integrated into the matrix

INT8 should be reported as separate layers of evidence:

1. graph conversion/export success
2. INT8 graph audit result
3. runtime build success
4. provider execution profile
5. segmentation quality retention

For matrix integration, put the audit output path into `compatibility.csv`:

```csv
int8_audit_path
compatibility/audits/model_runtime_int8_audit.json
```

`evaluate` reads this JSON and carries these fields into `runtime_evidence_matrix.csv` and `condition_summary.csv`:

- `strict_int8_graph_pass`
- `strict_int8_graph_reason`
- `conv_total`
- `qlinear_conv_count`
- `qdq_wrapped_conv_count`
- `quantized_conv_like_count`
- `conv_quantized_fraction`
- `fp32_conv_fraction_estimated`
- `int8_graph_claim_allowed`

A graph with a few INT8 tensors but mostly FP32 Conv operations must not be described as a strong INT8 deployment result.

## 8. TPU / XLA evidence boundary

TensorRT, CUDA, and some OpenVINO ONNX Runtime paths can be audited with ORT provider profile JSON. TPU/XLA rows are different. ORT provider-profile fields alone are not sufficient for TPU execution verification.

TPU/XLA paths require backend-specific profiling evidence, such as XLA/XProf traces or framework-specific metrics reports. Record these with `xla_profile_path` or `backend_profile_path`, and keep the row claim-bounded unless the evidence is inspected.

## 9. Compatibility evidence

Precision Compatibility Score is computed from:

- conversion success
- operator support
- fallback avoidance
- build/compile success
- precision support, optionally adjusted by INT8 graph-audit evidence

Fallback is not an implementation footnote. It is part of deployment stability.

## 10. Reporting principle

Always report HPLS Score together with component metrics. HPLS Score must not be used as a single absolute ranking without BRS, SRS/small-region retention, RTG, PCS, provider audit fields, INT8 graph audit fields, tail latency, and missing-evidence logs.

## 11. Non-clinical medical-extension rule

Medical-lesion terminology may appear as a future evaluation target. It must not be used as evidence of diagnostic readiness. Medical-domain claims require medical data, domain-specific review, and separate validation outside this package. Generated reports include a clinical claim disclaimer by default.
