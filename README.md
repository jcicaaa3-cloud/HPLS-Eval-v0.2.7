# HPLS-Eval

[日本語](README_JA.md) · [한국어](README_KO.md)

An evaluation harness for segmentation models whose masks are small and irregular. It measures how the same model's results change when the runtime (CPU, GPU, NPU, TPU) and the numeric precision (FP32, FP16, INT8) change.

This is not a model, and the repository contains no experiment results. You supply ground-truth masks, prediction files for each runtime and precision, and a log of how each run went. The tool computes the metrics and writes the tables and a report.

I started it while working on [LiteRaceSegNet](https://github.com/jcicaaa3-cloud/LiteRaceSegNet-FINAL-), a road-damage segmentation model. That work ran into thin boundaries, sparse masks, small objects, threshold sensitivity, and backend fallback. This tool measures them separately.

Version 0.2.7.

## What it checks

| Group | Contents |
| --- | --- |
| Segmentation | Dice, IoU |
| Boundary | Boundary recall and boundary IoU. Samples with an empty ground-truth boundary are left out of the mean but still counted. |
| Small regions | Recall on small regions |
| Retention | FP16 and INT8 results relative to the FP32 baseline (FP32 on CPU by default) |
| Latency | p50, p90, p95, p99, min, max |
| Provider audit | Which execution provider actually ran the nodes, compared with the one requested (from an ONNX Runtime profile) |
| INT8 audit | Counts quantized convolution nodes in the ONNX graph, so a file named `int8` is not taken at its word |

Summary scores (BRS, SRS, RTG, PCS, HPLS score) are defined in [docs/protocol.md](docs/protocol.md). Latency is never estimated: if it was not measured, the fields stay empty.

## Install

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .

pip install -e '.[onnx]'       # ONNX graph audit
pip install -e '.[ort]'        # ONNX Runtime profiling
pip install -e '.[dev]'        # tests
```

## Usage

```bash
hpls-eval scaffold --out ./my_project                      # empty templates
hpls-eval validate --config configs/hpls_eval.yaml         # check inputs
hpls-eval evaluate --config configs/hpls_eval.yaml --out results/run_001
python -m pytest -q
```

Audits:

```bash
python scripts/ort_profile_runner.py \
  --model exports/model.onnx --input-npz sample_input.npz \
  --providers TensorRTExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider \
  --out-json compatibility/profiles/model_trt_fp16_measurement.json

hpls-eval audit-profile \
  --profile compatibility/profiles/model_trt_fp16_profile.json \
  --requested-provider TensorRTExecutionProvider \
  --out results/audits/model_trt_fp16_provider_audit.json

hpls-eval audit-onnx \
  --model exports/model_int8.onnx \
  --strict-conv-threshold 0.80 --max-fp32-conv-fraction 0.20 \
  --out compatibility/audits/model_int8_audit.json
```

## Inputs

| Input | Purpose |
| --- | --- |
| Dataset manifest | Maps sample IDs to ground-truth masks |
| Prediction files | Model outputs under `model_id/runtime_id/precision/sample_id` |
| Compatibility log (`compatibility.csv`) | Run status, latency, requested provider, fallback, and paths to audit files |

Column details are in [docs/data_contract.md](docs/data_contract.md). Relative paths inside `compatibility.csv` are resolved from the folder that contains `compatibility.csv`, not from the project root.

## Outputs

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

## Scope

This is a non-clinical research prototype. Medical lesion segmentation appears in the docs as a possible later target because it has the same kind of small, irregular masks. There is no patient data here, no clinical validation, and road-damage results say nothing about medical performance. The default config keeps `clinical_claim_allowed: false`.

## Docs

- [protocol.md](docs/protocol.md): metrics and scores
- [data_contract.md](docs/data_contract.md): input formats
- [runtime_logging_guide.md](docs/runtime_logging_guide.md): how to record runtime results
- [provider_execution_audit.md](docs/provider_execution_audit.md), [int8_graph_audit.md](docs/int8_graph_audit.md): the two audits
- [claim_boundary.md](docs/claim_boundary.md), [domain_bridge_road_to_medical.md](docs/domain_bridge_road_to_medical.md): scope
- [security_notes.md](docs/security_notes.md): loading checkpoints safely

## License

MIT
