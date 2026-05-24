import json
from pathlib import Path
import pandas as pd

from hpls_eval.cli import main
from hpls_eval.compatibility import normalize_compatibility
from hpls_eval.report import write_report


def test_audit_profile_out_creates_parent_dirs(tmp_path):
    profile = [
        {"cat": "Node", "name": "Conv_0", "args": {"op_name": "Conv", "provider": "TensorRTExecutionProvider"}},
    ]
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    out_path = tmp_path / "nested" / "audits" / "profile_audit.json"

    status = main([
        "audit-profile",
        "--profile", str(profile_path),
        "--requested-provider", "TensorRTExecutionProvider",
        "--out", str(out_path),
    ])

    assert status == 0
    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["requested_provider"] == "TensorRTExecutionProvider"


def test_tensorrt_provider_and_gpu_device_fields_do_not_mismatch(tmp_path):
    profile = [
        {"cat": "Node", "name": "Conv_0_kernel_time", "args": {"node_index": 0, "op_name": "Conv", "provider": "TensorRTExecutionProvider"}},
    ]
    profile_path = tmp_path / "trt_profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    df = pd.DataFrame([
        {
            "model_id": "m",
            "runtime_id": "tensorrt_gpu",
            "precision": "FP16",
            "conversion_success": 1,
            "operator_support": 1,
            "fallback_ratio": "",
            "build_success": 1,
            "precision_support": 1,
            "latency_ms_p50": 4,
            "latency_ms_p99": 7,
            "requested_provider": "TensorRTExecutionProvider",
            "provider_profile_path": "trt_profile.json",
            "requested_device": "GPU",
            "actual_compiled_device": "GPU",
        }
    ])

    out = normalize_compatibility(df, None, evidence_base_dir=tmp_path)
    row = out.iloc[0]
    assert row["requested_provider_canonical"] == "TensorRTExecutionProvider"
    assert row["device_claim_status"] == "target_device_confirmed"
    assert bool(row["hardware_claim_allowed"]) is True


def test_road_report_hides_small_lesion_alias_columns_by_default(tmp_path):
    report_path = tmp_path / "report.md"
    cfg = {
        "project": {"domain_context": "road_damage_pilot", "claim_scope": "medical_extension_planned"},
        "reporting": {"show_medical_alias_columns": False},
    }
    aggregate = pd.DataFrame([
        {
            "model_id": "m",
            "runtime_id": "onnxruntime_cpu",
            "precision": "FP32",
            "domain_context": "road_damage_pilot",
            "small_region_recall": 0.8,
            "small_lesion_recall": 0.8,
        }
    ])
    condition_summary = pd.DataFrame([
        {
            "model_id": "m",
            "runtime_id": "onnxruntime_cpu",
            "precision": "FP32",
            "domain_context": "road_damage_pilot",
            "small_region_recall": 0.8,
            "small_lesion_recall": 0.8,
        }
    ])

    write_report(report_path, cfg, {}, [], condition_summary, aggregate, pd.DataFrame(), runtime_evidence=pd.DataFrame())
    text = report_path.read_text(encoding="utf-8")
    assert "small_region_recall" in text
    assert "small_lesion_recall" not in text


def test_report_includes_domain_context_distribution_for_mixed_rows(tmp_path):
    report_path = tmp_path / "report.md"
    cfg = {
        "project": {"domain_context": "road_damage_pilot", "claim_scope": "medical_extension_planned"},
        "reporting": {},
    }
    condition_summary = pd.DataFrame([
        {"model_id": "m1", "runtime_id": "cpu", "precision": "FP32", "domain_context": "road_damage_pilot"},
        {"model_id": "m2", "runtime_id": "cpu", "precision": "FP32", "domain_context": "medical_lesion_target"},
        {"model_id": "m3", "runtime_id": "cpu", "precision": "FP32", "domain_context": "generic_visual_anomaly"},
    ])

    write_report(report_path, cfg, {}, [], condition_summary, pd.DataFrame(), pd.DataFrame(), runtime_evidence=pd.DataFrame())
    text = report_path.read_text(encoding="utf-8")
    assert "## Domain Context Distribution" in text
    assert "road_damage_pilot" in text
    assert "medical_lesion_target" in text
    assert "generic_visual_anomaly" in text
    assert "project-level domain_context" in text


def test_readme_states_domain_bridge_once_without_repeating_position_section():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "## Research position" not in text
    assert text.count("Road-damage pilot evidence") == 1
    assert "Road-damage pilot evidence → medical-lesion-oriented follow-up" in text
    assert "not because road-damage performance transfers to medicine" in text
