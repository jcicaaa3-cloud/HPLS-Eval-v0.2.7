import pandas as pd
import numpy as np

from hpls_eval.compatibility import normalize_compatibility
from hpls_eval.evaluator import build_runtime_evidence_matrix, build_condition_summary
from hpls_eval.metrics import compute_mask_metrics
from hpls_eval.report import write_report


def _minimal_compat_df():
    return pd.DataFrame([
        {
            "model_id": "m",
            "runtime_id": "onnxruntime_cpu",
            "precision": "FP32",
            "conversion_success": 1,
            "operator_support": 1,
            "fallback_ratio": 0,
            "build_success": 1,
            "precision_support": 1,
            "latency_ms_p50": 10,
            "latency_ms_p99": 12,
            "requested_provider": "CPUExecutionProvider",
        }
    ])


def test_domain_context_defaults_and_clinical_claim_false():
    out = normalize_compatibility(_minimal_compat_df(), None)
    row = out.iloc[0]
    assert row["domain_context"] in {"generic_visual_anomaly", "road_damage_pilot"}
    assert row["claim_scope"] == "evaluation_framework"
    assert bool(row["clinical_claim_allowed"]) is False
    assert row["clinical_claim_status"] == "clinical_claim_forbidden_nonclinical_research_prototype"


def test_medical_target_metadata_does_not_enable_clinical_claims():
    out = normalize_compatibility(
        _minimal_compat_df(),
        None,
        domain_claim={
            "domain_context": "medical_lesion_target",
            "claim_scope": "medical_extension_planned",
            "clinical_claim_allowed": True,  # intentionally unsafe user input
        },
    )
    row = out.iloc[0]
    assert row["domain_context"] == "medical_lesion_target"
    assert row["claim_scope"] == "medical_extension_planned"
    assert bool(row["clinical_claim_allowed"]) is False


def test_run_status_values_are_preserved_and_normalized():
    df = _minimal_compat_df()
    df.loc[0, "run_status"] = "not_run"
    out = normalize_compatibility(df, None)
    assert out.iloc[0]["run_status_normalized"] == "NOT_RUN"


def test_domain_metadata_flows_to_runtime_and_condition_matrices():
    cfg = {
        "project": {
            "domain_context": "road_damage_pilot",
            "claim_scope": "medical_extension_planned",
            "clinical_claim_allowed": False,
        },
        "baseline": {"runtime_id": "onnxruntime_cpu", "precision": "FP32"},
        "metrics": {"main_performance_metric": "dice", "eps": 1e-8},
        "hpls_weights": {
            "performance_retention": 0.25,
            "boundary_retention": 0.25,
            "small_region_retention": 0.20,
            "runtime_retention": 0.15,
            "precision_compatibility": 0.15,
        },
    }
    compat = normalize_compatibility(_minimal_compat_df(), None, domain_claim=cfg["project"])
    runtime = build_runtime_evidence_matrix(compat)
    assert runtime.iloc[0]["domain_context"] == "road_damage_pilot"
    assert bool(runtime.iloc[0]["clinical_claim_allowed"]) is False

    aggregate = pd.DataFrame([
        {
            "model_id": "m",
            "runtime_id": "onnxruntime_cpu",
            "precision": "FP32",
            "n_ok": 1,
            "dice": 1.0,
            "iou": 1.0,
            "boundary_recall": 1.0,
            "boundary_iou": 1.0,
            "small_region_recall": 1.0,
            "small_lesion_recall": 1.0,
        }
    ])
    summary = build_condition_summary(cfg, aggregate, compat)
    row = summary.iloc[0]
    assert row["domain_context"] == "road_damage_pilot"
    assert row["claim_scope"] == "medical_extension_planned"
    assert bool(row["clinical_claim_allowed"]) is False


def test_small_lesion_alias_does_not_conflict_with_small_region():
    pred = np.zeros((10, 10), dtype=bool)
    gt = np.zeros((10, 10), dtype=bool)
    gt[1:3, 1:3] = True
    pred[1:3, 1:3] = True
    vals = compute_mask_metrics(pred, gt, {"small_region": {"area_threshold_px": 4}, "boundary": {"radius": 1}})
    assert vals["small_region_recall"] == vals["small_lesion_recall"]
    assert vals["small_region_gt_components"] == vals["small_lesion_gt_components"]
    assert vals["small_region_gt_px"] == vals["small_lesion_gt_px"]


def test_report_contains_clinical_disclaimer_without_validation_claim(tmp_path):
    report_path = tmp_path / "report.md"
    cfg = {
        "project": {
            "name": "HPLS-Eval Medical Extension Smoke Test",
            "domain_context": "medical_lesion_target",
            "claim_scope": "medical_extension_planned",
            "clinical_claim_allowed": True,  # should still be forced false in report metadata
        },
        "reporting": {},
    }
    write_report(
        report_path,
        cfg,
        validation_summary={"num_samples": 0},
        warnings=[],
        condition_summary=pd.DataFrame(),
        aggregate=pd.DataFrame(),
        per_sample=pd.DataFrame(),
        runtime_evidence=pd.DataFrame(),
    )
    text = report_path.read_text(encoding="utf-8")
    assert "This report does not provide clinical validation." in text
    assert "clinical_claim_allowed**: `False`" in text
    forbidden_claims = [
        "provides clinical validation",
        "clinically validated",
        "medical diagnosis ready",
        "diagnosis-ready",
        "deployable medical AI",
        "lesion model proven robust",
    ]
    lowered = text.lower()
    for phrase in forbidden_claims:
        assert phrase not in lowered
