# HPLS-Eval v0.2.7

**Road-damage pilot evidence → medical-lesion-oriented follow-up**

Evidence-first evaluation for small, irregular segmentation masks across CPU, GPU, NPU, and TPU-style runtime paths.

HPLS-Eval is an evaluation harness, not a model repository. It is designed to compare how a lightweight segmentation model behaves when the runtime and numeric precision change. The intended matrix is simple:

```text
same task / same model family
        × CPU, GPU, NPU, TPU runtime paths
        × FP32, FP16, INT8 precision modes
        × accuracy, boundary quality, small-region retention, latency, fallback evidence
```

The road-damage experiment is treated as pilot evidence because it exposed practical issues around thin boundaries, sparse masks, small objects, threshold sensitivity, and backend fallback. Medical lesion segmentation is a planned follow-up evaluation target, not because road-damage performance transfers to medicine, but because the same evaluation risks can reappear around small irregular masks. This repository does not claim medical validity, diagnostic performance, or deployment safety.

## What this repository does

- Computes segmentation metrics for small and irregular masks.
- Separates FP32 baseline performance from FP16/INT8 retention.
- Tracks latency with p90, p95, p99, min, and max fields.
- Records whether an accelerated backend was actually used or silently fell back.
- Integrates INT8 graph-audit evidence instead of trusting a filename or conversion label.
- Produces CSV matrices and a reviewer-readable Markdown report.

## What it does not do

- It does not include patient data.
- It does not provide clinical validation.
- It does not create synthetic results to fill missing experiments.
- It does not assume that road-damage performance transfers to medical data.
- It does not treat a requested provider as proof of actual execution.

## Evaluation scope

| Axis | Meaning |
|---|---|
| Model | A lightweight segmentation model supplied by the user |
| Runtime path | CPU, GPU, NPU, or TPU-style execution path |
| Precision | FP32, FP16, INT8 |
| Baseline | Configured FP32 CPU row by default |
| Outputs | Metrics, latency summaries, runtime evidence, INT8 audit fields, report tables |
| Claim policy | Non-clinical; missing evidence remains missing evidence |

Default runtime IDs are defined in `configs/hpls_eval.yaml`:

```yaml
runtime_paths:
  - onnxruntime_cpu
  - tensorrt_gpu
  - openvino_npu
  - xla_tpu

precisions: [FP32, FP16, INT8]
```

## Metrics and evidence

HPLS-Eval reports conventional segmentation metrics and deployment-oriented evidence in the same run.

| Group | Examples |
|---|---|
| Segmentation | Dice, IoU |
| Boundary | Boundary recall, boundary IoU, empty-GT boundary counts |
| Small region | Small-region recall and retention aliases |
| Runtime | p90/p95/p99 latency, run status, conversion/build status |
| Provider audit | Requested provider, actual provider fractions, node counts, fallback counts |
| INT8 audit | QLinearConv / ConvInteger / QDQ-wrapped Conv counts, strict INT8 graph pass |
| Summary | Performance retention, BRS, SRS, RTG, PCS, HPLS score |

Boundary means exclude samples with empty GT boundaries. The excluded cases are still counted so that the report does not hide empty-mask behavior.

## Install

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

Optional extras:

```bash
pip install -e '.[onnx]'       # ONNX graph audit
pip install -e '.[ort]'        # ONNX Runtime profile support
pip install -e '.[dev]'        # tests and lint tools
```

## Quick start

Create a project scaffold:

```bash
hpls-eval scaffold --out ./my_hpls_project
```

Validate the configured inputs:

```bash
hpls-eval validate --config configs/hpls_eval.yaml
```

Run evaluation:

```bash
hpls-eval evaluate --config configs/hpls_eval.yaml --out results/run_001
```

Run tests:

```bash
python -m pytest -q
```

## Input contract

The evaluator expects three categories of input.

| Input | Purpose |
|---|---|
| Dataset manifest | Connects sample IDs to ground-truth masks and domain context |
| Prediction files | Model outputs by `model_id/runtime_id/precision/sample_id` |
| Compatibility log | Runtime, precision, latency, provider, fallback, and audit evidence |

Recommended compatibility columns:

```csv
model_id,runtime_id,precision,domain_context,claim_scope,clinical_claim_allowed,run_status,conversion_success,operator_support,fallback_ratio,build_success,precision_support,int8_audit_path,latency_ms,latency_ms_p50,latency_ms_p90,latency_ms_p95,latency_ms_p99,latency_ms_min,latency_ms_max,compile_ms,unsupported_ops,requested_provider,provider_profile_path,requested_device,available_devices,actual_compiled_device,performance_hint,xla_profile_path,backend_profile_path,backend_evidence_notes,notes
```

Important distinction:

```text
requested_provider     = TensorRTExecutionProvider / CUDAExecutionProvider / OpenVINOExecutionProvider / CPUExecutionProvider
actual_compiled_device = CPU / GPU / CUDA-GPU / NPU / TPU
```

`actual_compiled_device` is a device label, not an execution-provider name.

Relative evidence paths in compatibility.csv are resolved relative to the directory containing compatibility.csv, not the project root.

compatibility.csv 안의 상대 경로는 프로젝트 루트가 아니라 compatibility.csv가 위치한 폴더를 기준으로 해석된다.

```text
compatibility/compatibility.csv + provider_profile_path=profiles/trt_profile.json
=> compatibility/profiles/trt_profile.json
```

Do not write `compatibility/profiles/trt_profile.json` inside a file that already lives in `compatibility/`, unless that duplicated path is intentional.

## Provider execution audit

A requested accelerator is not enough to support an accelerator claim. Attach provider-profile evidence when possible.

```bash
python scripts/ort_profile_runner.py \
  --model exports/model.onnx \
  --input-npz sample_input.npz \
  --providers TensorRTExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider \
  --out-json compatibility/profiles/model_trt_fp16_measurement.json

hpls-eval audit-profile \
  --profile compatibility/profiles/model_trt_fp16_profile.json \
  --requested-provider TensorRTExecutionProvider \
  --out results/audits/model_trt_fp16_provider_audit.json
```

For TPU/XLA rows, ONNX Runtime provider fields are not enough. Use backend-specific evidence such as XLA or XProf traces and record the path in the compatibility log.

## INT8 graph audit

INT8 should be supported by graph evidence, not by the model filename.

```bash
hpls-eval audit-onnx \
  --model exports/model_int8.onnx \
  --strict-conv-threshold 0.80 \
  --max-fp32-conv-fraction 0.20 \
  --out compatibility/audits/model_int8_audit.json
```

The audit output is merged into `condition_summary.csv` and `runtime_evidence_matrix.csv`.

## Main outputs

```text
results/run_001/
  config_snapshot.yaml
  environment.json
  input_hashes.csv
  validation_warnings.log
  per_sample_metrics.csv
  aggregate_metrics.csv
  compatibility_normalized.csv
  runtime_evidence_matrix.csv
  condition_summary.csv
  report.md
```

`report.md` is intended for review, presentation, and document conversion. Raw CSV files keep wider schemas, including compatibility aliases that may be useful for future medical-domain experiments. Avoid raw CSV screenshots in advisor slides unless the alias policy or full schema is the topic.

## v0.2.7 changes

- Warning summaries now use stable warning codes instead of weak prose fragments.
- Related runtime, fallback, device, and INT8 warnings can be grouped by condition.
- Provider placement, fallback, and device evidence are separated into narrower report tables.
- Relative evidence-path resolution is documented explicitly.
- Existing runtime, INT8, boundary, and claim-boundary gates remain unchanged.

## Claim boundary

This repository is a non-clinical research prototype. Medical-lesion terminology marks a future evaluation target only. Medical-domain conclusions require medical data, a domain-specific protocol, reviewer oversight, and separate validation.

The default configuration keeps:

```yaml
clinical_claim_allowed: false
reporting:
  show_medical_alias_columns: false
```

Road-damage pilot results may motivate the evaluation design. They must not be presented as evidence of medical performance.
