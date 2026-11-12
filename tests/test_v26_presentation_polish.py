from pathlib import Path

import pandas as pd

from hpls_eval.report import write_report
from hpls_eval.warning_utils import dedupe_warnings, warning_identity


def test_condition_level_warnings_are_deduped_without_hiding_distinct_codes_or_samples():
    warnings = [
        "HARDWARE_CLAIM_NOT_ALLOWED model=m runtime=trt precision=FP16: backend_status=a",
        "HARDWARE_CLAIM_NOT_ALLOWED model=m runtime=trt precision=FP16: backend_status=b",
        "RUNTIME_EVIDENCE_WEAK model=m runtime=trt precision=FP16: backend_status=a",
        "PROVIDER_FALLBACK_DETECTED model=m runtime=trt precision=FP16: fallback=a",
        "PROVIDER_FALLBACK_DETECTED model=m runtime=trt precision=FP16: fallback=b",
        "HARDWARE_CLAIM_NOT_ALLOWED model=m runtime=cuda precision=FP16: separate condition",
        "EVAL_ERROR model=m runtime=trt precision=FP16 sample=s1: first",
        "EVAL_ERROR model=m runtime=trt precision=FP16 sample=s2: second",
    ]
    out = dedupe_warnings(warnings)
    assert len(out) == 6
    assert warnings[0] in out and warnings[1] not in out
    assert warnings[2] in out  # different warning code for the same condition is kept
    assert warnings[3] in out and warnings[4] not in out
    assert warnings[5] in out  # different condition is kept
    assert warnings[6] in out and warnings[7] in out  # sample-level errors remain distinct
    assert warning_identity(warnings[0]) == warning_identity(warnings[1])


def test_runtime_logging_guide_forbids_provider_names_in_actual_device():
    text = Path("docs/runtime_logging_guide.md").read_text(encoding="utf-8")
    assert "actual device label selected after compilation" in text
    assert "Do not put execution-provider names here" in text
    assert "actual device or provider used after compilation" not in text


def test_report_is_split_and_hides_medical_alias_tables_for_road_domain(tmp_path):
    report_path = tmp_path / "report.md"
    cfg = {
        "project": {"domain_context": "road_damage_pilot", "claim_scope": "medical_extension_planned"},
        "reporting": {"show_medical_alias_columns": False},
    }
    runtime_evidence = pd.DataFrame([
        {
            "domain_context": "road_damage_pilot",
            "claim_scope": "medical_extension_planned",
            "clinical_claim_allowed": False,
            "clinical_claim_status": "clinical_claim_forbidden_nonclinical_research_prototype",
            "model_id": "m",
            "runtime_id": "tensorrt_gpu",
            "precision": "FP16",
            "run_status_normalized": "SUCCESS",
            "requested_provider": "TensorRTExecutionProvider",
            "requested_provider_unique_node_fraction": 0.5,
            "requested_provider_event_fraction": 0.5,
            "fallback_ratio_effective": 0.5,
            "unique_node_provider_counts": '{"TensorRTExecutionProvider":1,"CPUExecutionProvider":1}',
            "node_event_provider_counts": '{"TensorRTExecutionProvider":3,"CPUExecutionProvider":3}',
            "latency_ms_p90": 8,
            "latency_ms_p95": 9,
            "latency_ms_p99": 10,
            "backend_claim_allowed": False,
            "hardware_claim_allowed": False,
            "runtime_evidence_grade": "weak",
            "backend_claim_status": "partial_fallback_below_threshold",
            "actual_compiled_device": "GPU",
            "device_claim_status": "target_device_confirmed",
        }
    ])
    condition_summary = pd.DataFrame([
        {
            "domain_context": "road_damage_pilot",
            "model_id": "m",
            "runtime_id": "tensorrt_gpu",
            "precision": "FP16",
            "run_status_normalized": "SUCCESS",
            "n_ok": 1,
            "dice": 0.9,
            "iou": 0.8,
            "boundary_recall": 0.7,
            "boundary_iou": 0.6,
            "small_region_recall": 0.5,
            "small_lesion_recall": 0.5,
            "performance_retention": 0.9,
            "boundary_retention_brs": 0.7,
            "small_region_retention_srs": 0.5,
            "hpls_score": 0.75,
        }
    ])
    warnings = [
        "HARDWARE_CLAIM_NOT_ALLOWED model=m runtime=tensorrt_gpu precision=FP16: first",
        "HARDWARE_CLAIM_NOT_ALLOWED model=m runtime=tensorrt_gpu precision=FP16: second",
    ]
    write_report(report_path, cfg, {}, warnings, condition_summary, pd.DataFrame(), pd.DataFrame(), runtime_evidence)
    text = report_path.read_text(encoding="utf-8")
    for heading in [
        "### Claim gate table",
        "### Provider placement table",
        "### Fallback table",
        "### Device evidence table",
        "### Latency table",
        "### INT8 graph audit table",
        "### Segmentation summary table",
        "### Retention and HPLS table",
    ]:
        assert heading in text
    assert "Warnings are deduplicated by `warning_code + model/runtime/precision`" in text
    assert "### Grouped runtime evidence issues" in text
    assert "Runtime evidence issue: model=m, runtime=tensorrt_gpu, precision=FP16" in text
    assert text.count("HARDWARE_CLAIM_NOT_ALLOWED") >= 1
    assert "small_region_recall" in text
    assert "small_lesion_recall" not in text


def test_raw_csv_alias_presentation_note_is_documented():
    data_contract = Path("docs/data_contract.md").read_text(encoding="utf-8")
    assert "Presentation note: raw CSV files intentionally retain backward-compatible aliases" in data_contract
