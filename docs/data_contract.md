# Data Contract

## Dataset manifest

Required columns:

```csv
sample_id,gt_mask,split
```

- `sample_id`: unique sample key. This key is used to resolve prediction masks.
- `gt_mask`: path to a ground-truth binary or grayscale mask.
- `split`: optional but recommended (`train`, `val`, `test`, `external`, etc.).

## Prediction layout

Prediction layout is not hard-coded. It is controlled by:

```yaml
paths:
  prediction_pattern: "{prediction_root}/{model_id}/{runtime_id}/{precision}/{sample_id}.png"
```

Supported placeholders:

- `{prediction_root}`
- `{model_id}`
- `{runtime_id}`
- `{precision}`
- `{sample_id}`

## Mask format

The default implementation expects image-readable masks. Grayscale values above `mask.positive_threshold` are treated as positive small-region / anomaly / lesion-candidate pixels depending on the declared `domain_context`.

For probability maps, export deterministic thresholded masks or set the threshold in the config and keep it fixed across conditions.

## Domain and claim metadata

The config should include conservative domain/claim metadata:

```yaml
project:
  domain_context: road_damage_pilot        # or generic_visual_anomaly / medical_lesion_target
  claim_scope: medical_extension_planned   # or pilot_only / evaluation_framework
  clinical_claim_allowed: false
```

The evaluator writes these fields into the generated matrices. `clinical_claim_allowed` is forced to `false` because this package is a non-clinical research prototype. Optional `run_status` values such as `NOT_RUN`, `FAILED`, `MISSING_BACKEND`, and `SUCCESS` are surfaced as `run_status_normalized`.

## Compatibility log

The compatibility log is one row per `model_id`, `runtime_id`, and `precision` condition.

Required minimum columns:

```csv
model_id,runtime_id,precision,conversion_success,operator_support,fallback_ratio,build_success,precision_support
```

Recommended v2.7 evidence columns:

```csv
domain_context,claim_scope,clinical_claim_allowed,run_status,int8_audit_path,latency_ms_p50,latency_ms_p90,latency_ms_p95,latency_ms_p99,latency_ms_min,latency_ms_max,compile_ms,unsupported_ops,requested_provider,provider_profile_path,requested_device,available_devices,actual_compiled_device,performance_hint,xla_profile_path,backend_profile_path,backend_evidence_notes,notes
```

Path fields are resolved relative to the compatibility CSV directory.

`actual_compiled_device` should contain the resolved device label (`CPU`, `GPU`, `CUDA-GPU`, `NPU`, `TPU`, etc.), not an execution-provider name. Keep provider names such as `TensorRTExecutionProvider` in `requested_provider`.

## Evidence path resolution

Relative paths in evidence columns such as `provider_profile_path`, `int8_audit_path`, `xla_profile_path`, and `backend_profile_path` are resolved relative to the directory containing `compatibility.csv`, not the project root.

Example: if the compatibility log is `compatibility/compatibility.csv`, use `provider_profile_path=profiles/trt_profile.json` for `compatibility/profiles/trt_profile.json`. Do not use `provider_profile_path=compatibility/profiles/trt_profile.json` unless the compatibility log itself is somewhere else.

compatibility.csv 안의 상대 경로는 프로젝트 루트가 아니라 compatibility.csv가 위치한 폴더를 기준으로 해석된다.

## Alias policy

`small_region_*` is the primary generic terminology used for road-damage and visual-anomaly experiments. `small_lesion_*` is retained as a medical-domain alias for future lesion segmentation experiments. These aliases must not be interpreted as evidence of clinical validity. CSV outputs keep both families for compatibility; Markdown reports hide `small_lesion_*` by default through `reporting.show_medical_alias_columns: false`.

Presentation note: raw CSV files intentionally retain backward-compatible aliases such as `small_lesion_*`. For road-damage pilot presentations, prefer the generated `report.md`, which hides medical aliases by default and splits wide matrices into claim, provider, latency, INT8, and segmentation tables.
