from __future__ import annotations

from pathlib import Path


def scaffold_project(out_dir: str | Path) -> Path:
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(exist_ok=True)
    (out / "manifests").mkdir(exist_ok=True)
    (out / "compatibility").mkdir(exist_ok=True)
    (out / "compatibility" / "profiles").mkdir(exist_ok=True)
    (out / "compatibility" / "audits").mkdir(exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    (out / "ground_truth").mkdir(exist_ok=True)
    (out / "results").mkdir(exist_ok=True)

    (out / "manifests" / "dataset_manifest.csv").write_text(
        "sample_id,gt_mask,split\n"
        "# case_0001,../ground_truth/case_0001.png,test\n",
        encoding="utf-8",
    )
    (out / "compatibility" / "compatibility.csv").write_text(
        "model_id,runtime_id,precision,domain_context,claim_scope,clinical_claim_allowed,run_status,conversion_success,operator_support,fallback_ratio,build_success,precision_support,"
        "int8_audit_path,latency_ms,latency_ms_p50,latency_ms_p90,latency_ms_p95,latency_ms_p99,latency_ms_min,latency_ms_max,"
        "compile_ms,unsupported_ops,requested_provider,provider_profile_path,requested_device,available_devices,actual_compiled_device,performance_hint,"
        "xla_profile_path,backend_profile_path,backend_evidence_notes,notes\n"
        "# your_model,onnxruntime_cpu,FP32,road_damage_pilot,pilot_only,false,SUCCESS,1,1,0,1,1,,,12.1,13.4,15.0,18.2,11.5,22.0,0,0,CPUExecutionProvider,,CPU,CPU,CPU,,,,,replace with measured pilot evidence; no medical or clinical claim\n"
        "# your_model,tensorrt_gpu,INT8,road_damage_pilot,pilot_only,false,SUCCESS,1,0.93,,1,1,audits/your_model_int8_audit.json,,4.8,5.5,6.4,9.2,4.4,13.7,812,2,TensorRTExecutionProvider,profiles/your_model_trt_int8.json,GPU,CPU;CUDA;TensorRT,GPU,latency,,,,fallback_ratio is derived from provider_profile_path when available\n",
        encoding="utf-8",
    )
    (out / "configs" / "hpls_eval.yaml").write_text(
        "project:\n"
        "  name: HPLS-Eval Applied Pilot Run\n"
        "  description: Evidence-first runtime and low-precision evaluation for small irregular segmentation\n"
        "  domain_context: road_damage_pilot\n"
        "  claim_scope: medical_extension_planned\n"
        "  clinical_claim_allowed: false\n"
        "  notes: Road-damage is a pilot domain; medical lesion segmentation is a future evaluation target, not a clinical claim.\n"
        "paths:\n"
        "  dataset_manifest: ../manifests/dataset_manifest.csv\n"
        "  prediction_root: ../predictions\n"
        "  prediction_pattern: \"{prediction_root}/{model_id}/{runtime_id}/{precision}/{sample_id}.png\"\n"
        "  compatibility_log: ../compatibility/compatibility.csv\n"
        "baseline:\n"
        "  runtime_id: onnxruntime_cpu\n"
        "  precision: FP32\n"
        "models:\n"
        "  - model_id: your_model\n"
        "    display_name: Your actual lightweight segmentation model\n"
        "runtime_paths:\n"
        "  - runtime_id: onnxruntime_cpu\n"
        "    processor_class: CPU\n"
        "    runtime_name: ONNX Runtime CPU\n"
        "precisions: [FP32, FP16, INT8]\n"
        "mask:\n"
        "  positive_threshold: 127\n"
        "metrics:\n"
        "  eps: 1.0e-8\n"
        "  main_performance_metric: dice\n"
        "  boundary:\n"
        "    radius: 2\n"
        "    metric: boundary_recall\n"
        "  small_region:\n"
        "    area_threshold_px: 512\n"
        "    connectivity: 8\n"
        "    aggregation: pixel_micro\n"
        "pcs_weights:\n"
        "  conversion_success: 0.25\n"
        "  operator_support: 0.20\n"
        "  fallback_avoidance: 0.20\n"
        "  build_success: 0.20\n"
        "  precision_support: 0.15\n"
        "hpls_weights:\n"
        "  performance_retention: 0.25\n"
        "  boundary_retention: 0.25\n"
        "  small_region_retention: 0.20\n"
        "  runtime_retention: 0.15\n"
        "  precision_compatibility: 0.15\n"
        "runtime_audit:\n"
        "  require_provider_profile_for_accelerators: false\n"
        "  requested_provider_fraction_warning_threshold: 0.95\n"
        "  require_p99_latency: true\n"
        "int8_audit:\n"
        "  strict_conv_threshold: 0.80\n"
        "  max_fp32_conv_fraction: 0.20\n"
        "  require_for_int8: false\n"
        "  affect_pcs: true\n"
        "reporting:\n"
        "  fail_on_missing_prediction: false\n"
        "  fail_on_missing_baseline: true\n"
        "  write_markdown_report: true\n"
        "  include_per_sample_table_in_report: false\n"
        "  show_medical_alias_columns: false\n"
        "  group_related_warnings: true\n",
        encoding="utf-8",
    )
    (out / "README_NEXT_STEPS.md").write_text(
        "# Next steps\n\n"
        "1. Put ground-truth masks in `ground_truth/` or point the manifest to existing masks.\n"
        "2. Export predictions into `predictions/{model_id}/{runtime_id}/{precision}/{sample_id}.png`.\n"
        "3. Fill `compatibility/compatibility.csv` with measured conversion/runtime evidence.\n"
        "4. For TensorRT/CUDA/OpenVINO, attach ONNX Runtime profile JSON. Relative evidence paths inside `compatibility.csv` are resolved relative to the `compatibility.csv` directory, so use `profiles/run.json`, not `compatibility/profiles/run.json`, when the log lives in `compatibility/`.\n"
        "5. For INT8 rows, run `hpls-eval audit-onnx --model exports/model_int8.onnx --out compatibility/audits/model_int8_audit.json` and put that path in `int8_audit_path`.\n"
        "6. For TPU/XLA rows, attach backend-specific evidence such as `xla_profile_path`.\n"
        "7. Run `hpls-eval validate --config configs/hpls_eval.yaml` and inspect provider/device/INT8 warnings.\n"
        "8. Run `hpls-eval evaluate --config configs/hpls_eval.yaml --out results/run_001`; inspect `runtime_evidence_matrix.csv` before making hardware claims.\n"
        "9. Keep `clinical_claim_allowed: false`; medical-lesion wording is a future evaluation target, not diagnostic validation.\n",
        encoding="utf-8",
    )
    return out
