# Runtime and Compatibility Logging Guide

HPLS-Eval separates segmentation output evaluation from runtime measurement. This is intentional: runtime logs must come from actual target runtimes.

## Required compatibility fields

```csv
model_id,runtime_id,precision,domain_context,claim_scope,clinical_claim_allowed,run_status,conversion_success,operator_support,fallback_ratio,build_success,precision_support,int8_audit_path,latency_ms,latency_ms_p50,latency_ms_p90,latency_ms_p95,latency_ms_p99,latency_ms_min,latency_ms_max,compile_ms,unsupported_ops,requested_provider,provider_profile_path,requested_device,available_devices,actual_compiled_device,performance_hint,xla_profile_path,backend_profile_path,backend_evidence_notes,notes
```

## Domain and claim fields

- `domain_context`: `road_damage_pilot`, `generic_visual_anomaly`, or `medical_lesion_target`. Road damage remains the pilot domain; medical lesion segmentation is the future evaluation target.
- `claim_scope`: `pilot_only`, `evaluation_framework`, or `medical_extension_planned`.
- `clinical_claim_allowed`: keep `false`. The evaluator forces generated outputs to `false` because the package is a non-clinical research prototype.
- `run_status`: optional condition status such as `NOT_RUN`, `FAILED`, `MISSING_BACKEND`, or `SUCCESS`; it is surfaced as `run_status_normalized`.

## Field definitions

- `conversion_success`: 1 if model export/conversion succeeded; otherwise 0.
- `operator_support`: 0 to 1 score. Use 1 if all required ops are supported by the intended execution provider.
- `fallback_ratio`: 0 to 1. Use only when provider profile evidence is unavailable. If `provider_profile_path` is present, v2.2 derives `fallback_ratio_effective` from actual provider node counts.
- `build_success`: 1 if runtime build/compile/session initialization completed; otherwise 0.
- `precision_support`: 1 if the declared precision actually executed as intended; otherwise 0.
- `int8_audit_path`: JSON from `hpls-eval audit-onnx --out ...`; required for strong INT8 graph-coverage claims.
- `latency_ms`: mean latency or representative latency if p50 is not available.
- `latency_ms_p50`: preferred latency field for RTG. Backward-compatible aliases such as `latency_p50_ms` are accepted.
- `latency_ms_p90`, `latency_ms_p95`, `latency_ms_p99`: tail latency evidence.
- `latency_ms_min`, `latency_ms_max`: spread/outlier evidence.
- `compile_ms`: build/compile/session creation overhead.
- `unsupported_ops`: count of unsupported operators.
- `requested_provider`: requested execution provider, e.g. `TensorRTExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider`.
- `provider_profile_path`: ONNX Runtime profiling JSON produced after `sess.end_profiling()`.
- `requested_device`: device requested by runtime, e.g. `AUTO`, `GPU`, `NPU`.
- `available_devices`: semicolon-separated list observed at runtime.
- `actual_compiled_device`: actual device label selected after compilation, e.g., `CPU`, `GPU`, `CUDA-GPU`, `NPU`, or `TPU`. Do not put execution-provider names here. This is mandatory evidence for OpenVINO/NPU claims.
- `performance_hint`: OpenVINO performance hint or equivalent runtime policy.
- `xla_profile_path`, `backend_profile_path`: backend-specific evidence for TPU/XLA or non-ORT runtimes.
- `backend_evidence_notes`: short note describing how backend-specific evidence was produced.
- `notes`: evidence notes, runtime version, EP/provider info, calibration info.


## Evidence path resolution

Relative evidence paths in `compatibility.csv` are resolved relative to the directory containing `compatibility.csv`, not the project root.

```text
compatibility/compatibility.csv + provider_profile_path=profiles/trt_profile.json
=> compatibility/profiles/trt_profile.json
```

Do not write `provider_profile_path=compatibility/profiles/trt_profile.json` inside `compatibility/compatibility.csv`, because that will be resolved as `compatibility/compatibility/profiles/trt_profile.json`.

compatibility.csv 안의 상대 경로는 프로젝트 루트가 아니라 compatibility.csv가 위치한 폴더를 기준으로 해석된다.

## ONNX Runtime provider profiling rule

For TensorRT/CUDA/OpenVINO Execution Provider comparisons, do not stop at `InferenceSession(..., providers=[...])`. That only proves what was requested. It does not prove that graph nodes executed on that provider.

A minimum ONNX Runtime profile runner should follow this shape:

```python
import onnxruntime as ort

so = ort.SessionOptions()
so.enable_profiling = True
sess = ort.InferenceSession("model.onnx", sess_options=so, providers=["TensorRTExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])

# warmup and measured inference loop here

profile_path = sess.end_profiling()
print(profile_path)
```

Then audit it:

```bash
hpls-eval audit-profile \
  --profile path/from/end_profiling.json \
  --requested-provider TensorRTExecutionProvider
```

The important output fields are:

- `unique_node_provider_counts`
- `node_event_provider_counts`
- `requested_provider_unique_node_fraction`
- `requested_provider_event_fraction`
- `fallback_unique_node_counts`
- `fallback_event_counts`
- `fallback_ratio_measured`
- `backend_claim_status`
- `backend_claim_allowed`
- `hardware_claim_allowed`

## INT8 audit integration rule

Run:

```bash
hpls-eval audit-onnx --model exports/model_int8.onnx --out compatibility/audits/model_int8_audit.json
```

Then put the JSON path in `compatibility.csv` as `int8_audit_path`. `evaluate` will load it into the final matrices. Strong INT8 claims require both graph-audit evidence and runtime execution evidence.

## Evidence quality rule

Do not write `1` for compatibility because a model "should" work. Write `1` only when the conversion/runtime path was actually tested.

A runtime row with no `provider_profile_path` may still be useful during development, but it is reported as `C_log_only_no_provider_profile` and should not support a strong hardware claim.

## OpenVINO/NPU rule

When using OpenVINO with `AUTO`, record all of the following:

- `requested_device`: for example `AUTO`
- `available_devices`: for example `CPU;GPU;NPU`
- `actual_compiled_device`: the compiled device label actually selected by OpenVINO, not the provider name
- `performance_hint`: if used

Do not report an NPU result if the compiled model actually ran on CPU.

## TPU/XLA rule

TPU/XLA paths require separate backend-specific profiling evidence. ONNX Runtime provider profile fields are not sufficient for TPU execution verification. Record `xla_profile_path` or `backend_profile_path`, and describe the evidence in `backend_evidence_notes`.


## Domain bridge note

Road-damage rows may be reported as pilot evidence. Future medical-lesion rows must use medical-domain data and domain-specific review. Do not infer medical lesion performance from road-damage performance, and do not treat INT8 or hardware compatibility as evidence of medical safety.
