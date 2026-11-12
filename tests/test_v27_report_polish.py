from pathlib import Path

import pandas as pd

from hpls_eval.report import _warning_code_summary, write_report
from hpls_eval.warning_utils import extract_warning_code


def test_warning_code_summary_uses_research_grade_codes_for_prose_warnings():
    warnings = [
        "Missing compatibility rows: 2",
        "Total missing prediction files: 4",
        "Missing GT masks: 1",
        "HARDWARE_CLAIM_NOT_ALLOWED model=m runtime=trt precision=INT8: nope",
    ]
    assert extract_warning_code(warnings[0]) == "MISSING_COMPATIBILITY_ROWS"
    assert extract_warning_code(warnings[1]) == "TOTAL_MISSING_PREDICTIONS"
    summary = _warning_code_summary(warnings)
    codes = set(summary["warning_code"].tolist())
    assert "MISSING_COMPATIBILITY_ROWS" in codes
    assert "TOTAL_MISSING_PREDICTIONS" in codes
    assert "MISSING_GT_MASKS" in codes
    assert "Missing" not in codes
    assert "Total" not in codes


def test_report_groups_related_runtime_warnings_and_keeps_other_warnings(tmp_path):
    report_path = tmp_path / "report.md"
    cfg = {
        "project": {"domain_context": "road_damage_pilot", "claim_scope": "medical_extension_planned"},
        "reporting": {"group_related_warnings": True, "show_medical_alias_columns": False},
    }
    condition_summary = pd.DataFrame([
        {"model_id": "m", "runtime_id": "tensorrt_gpu", "precision": "INT8", "domain_context": "road_damage_pilot"}
    ])
    warnings = [
        "HARDWARE_CLAIM_NOT_ALLOWED model=m runtime=tensorrt_gpu precision=INT8: backend_status=partial_fallback_below_threshold",
        "PROVIDER_FALLBACK_DETECTED model=m runtime=tensorrt_gpu precision=INT8: requested_provider_fraction=0.5000",
        "REQUESTED_PROVIDER_FRACTION_LOW model=m runtime=tensorrt_gpu precision=INT8: requested_provider_fraction=0.5000 < threshold=0.9500",
        "Missing compatibility rows: 2",
    ]

    write_report(report_path, cfg, {}, warnings, condition_summary, pd.DataFrame(), pd.DataFrame(), runtime_evidence=pd.DataFrame())
    text = report_path.read_text(encoding="utf-8")
    assert "### Grouped runtime evidence issues" in text
    assert "Runtime evidence issue: model=m, runtime=tensorrt_gpu, precision=INT8" in text
    assert "- HARDWARE_CLAIM_NOT_ALLOWED: backend_status=partial_fallback_below_threshold" in text
    assert "- PROVIDER_FALLBACK_DETECTED: requested_provider_fraction=0.5000" in text
    assert "- REQUESTED_PROVIDER_FRACTION_LOW: requested_provider_fraction=0.5000 < threshold=0.9500" in text
    assert "### Other warnings" in text
    assert "Missing compatibility rows: 2" in text
    assert "MISSING_COMPATIBILITY_ROWS" in text


def test_report_uses_split_provider_fallback_device_tables(tmp_path):
    report_path = tmp_path / "report.md"
    cfg = {
        "project": {"domain_context": "road_damage_pilot", "claim_scope": "medical_extension_planned"},
        "reporting": {"group_related_warnings": True},
    }
    row = {
        "model_id": "m",
        "runtime_id": "tensorrt_gpu",
        "precision": "FP16",
        "domain_context": "road_damage_pilot",
        "requested_provider": "TensorRTExecutionProvider",
        "requested_provider_unique_node_fraction": 0.5,
        "fallback_ratio_effective": 0.5,
        "actual_compiled_device": "GPU",
    }
    df = pd.DataFrame([row])
    write_report(report_path, cfg, {}, [], df, pd.DataFrame(), pd.DataFrame(), runtime_evidence=df)
    text = report_path.read_text(encoding="utf-8")
    assert "### Provider placement table" in text
    assert "### Fallback table" in text
    assert "### Device evidence table" in text
    assert "Provider fallback summary table" not in text
    assert "Device and backend-specific evidence table" not in text


def test_relative_evidence_path_rule_is_documented():
    guide = Path("docs/runtime_logging_guide.md").read_text(encoding="utf-8")
    contract = Path("docs/data_contract.md").read_text(encoding="utf-8")
    assert "Relative evidence paths in `compatibility.csv` are resolved relative to the directory containing `compatibility.csv`, not the project root." in guide
    assert "resolved relative to the directory containing `compatibility.csv`, not the project root" in contract
