import json

import pandas as pd

from hpls_eval.audit import parse_ort_profile
from hpls_eval.compatibility import normalize_compatibility


def test_parse_ort_profile_counts_requested_provider_and_fallback(tmp_path):
    profile = [
        {"cat": "Node", "name": "Conv_0", "args": {"op_name": "Conv", "provider": "TensorRTExecutionProvider"}},
        {"cat": "Node", "name": "Relu_1", "args": {"op_name": "Relu", "provider": "TensorRTExecutionProvider"}},
        {"cat": "Node", "name": "Resize_2", "args": {"op_name": "Resize", "provider": "CPUExecutionProvider"}},
        {"cat": "Session", "name": "model_run", "args": {}},
    ]
    path = tmp_path / "ort_profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")

    audit = parse_ort_profile(path, requested_provider="TensorRTExecutionProvider")
    assert audit["total_profiled_node_count"] == 3
    assert audit["provider_node_counts"]["TensorRTExecutionProvider"] == 2
    assert audit["fallback_node_counts"]["CPUExecutionProvider"] == 1
    assert abs(audit["requested_provider_fraction"] - (2 / 3)) < 1e-9
    assert abs(audit["fallback_ratio_measured"] - (1 / 3)) < 1e-9


def test_normalize_compatibility_prefers_profile_fallback_ratio(tmp_path):
    profile = [
        {"cat": "Node", "name": "Conv_0", "args": {"op_name": "Conv", "provider": "CUDAExecutionProvider"}},
        {"cat": "Node", "name": "Resize_1", "args": {"op_name": "Resize", "provider": "CPUExecutionProvider"}},
    ]
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    df = pd.DataFrame([
        {
            "model_id": "m",
            "runtime_id": "cuda_gpu",
            "precision": "FP16",
            "conversion_success": 1,
            "operator_support": 1,
            "fallback_ratio": 0.0,  # intentionally overoptimistic manual value
            "build_success": 1,
            "precision_support": 1,
            "latency_p50_ms": 10,
            "latency_p99_ms": 13,
            "requested_provider": "CUDAExecutionProvider",
            "provider_profile_path": "profile.json",
        }
    ])
    out = normalize_compatibility(df, {
        "conversion_success": 0.2,
        "operator_support": 0.2,
        "fallback_avoidance": 0.2,
        "build_success": 0.2,
        "precision_support": 0.2,
    }, evidence_base_dir=tmp_path)
    row = out.iloc[0]
    assert row["fallback_ratio_source"] == "provider_profile_node_fraction"
    assert abs(row["fallback_ratio_effective"] - 0.5) < 1e-9
    assert abs(row["requested_provider_fraction"] - 0.5) < 1e-9



def test_parse_ort_profile_unique_nodes_do_not_inflate_repeated_runs(tmp_path):
    profile = []
    for _ in range(3):
        profile.extend([
            {"cat": "Node", "name": "Conv_0_kernel_time", "args": {"node_index": 0, "op_name": "Conv", "provider": "CUDAExecutionProvider"}},
            {"cat": "Node", "name": "Resize_1_kernel_time", "args": {"node_index": 1, "op_name": "Resize", "provider": "CPUExecutionProvider"}},
        ])
    path = tmp_path / "repeated_profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")

    audit = parse_ort_profile(path, requested_provider="CUDAExecutionProvider")
    assert audit["total_profiled_node_count"] == 2
    assert audit["total_profiled_node_event_count"] == 6
    assert audit["provider_node_counts"]["CUDAExecutionProvider"] == 1
    assert audit["provider_event_counts"]["CUDAExecutionProvider"] == 3
    assert abs(audit["requested_provider_fraction"] - 0.5) < 1e-9
    assert abs(audit["requested_provider_event_fraction"] - 0.5) < 1e-9


def test_normalize_compatibility_exposes_backend_claim_status_and_latency_alias(tmp_path):
    profile = [
        {"cat": "Node", "name": "Conv_0_kernel_time", "args": {"node_index": 0, "op_name": "Conv", "provider": "TensorRTExecutionProvider"}},
        {"cat": "Node", "name": "Resize_1_kernel_time", "args": {"node_index": 1, "op_name": "Resize", "provider": "CPUExecutionProvider"}},
    ]
    profile_path = tmp_path / "profile.json"
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
            "latency_p50_ms": 10,
            "latency_p99_ms": 14,
            "requested_provider": "TensorRTExecutionProvider",
            "provider_profile_path": "profile.json",
            "requested_device": "GPU",
            "actual_compiled_device": "GPU",
        }
    ])
    out = normalize_compatibility(df, None, evidence_base_dir=tmp_path, runtime_audit={"requested_provider_fraction_warning_threshold": 0.95})
    row = out.iloc[0]
    assert row["latency_ms_p99"] == 14
    assert row["backend_claim_status"] == "partial_fallback_below_threshold"
    assert bool(row["backend_claim_allowed"]) is False
    assert bool(row["hardware_claim_allowed"]) is False


def test_normalize_compatibility_loads_int8_audit_json(tmp_path):
    audit = {
        "strict_int8_graph_pass": True,
        "strict_int8_graph_reason": "passes_predeclared_conv_coverage_thresholds",
        "conv_total": 10,
        "qlinear_conv_count": 0,
        "qdq_wrapped_conv_count": 10,
        "quantized_conv_like_count": 10,
        "conv_quantized_fraction": 1.0,
        "fp32_conv_fraction_estimated": 0.0,
    }
    audit_path = tmp_path / "int8_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    df = pd.DataFrame([
        {
            "model_id": "m",
            "runtime_id": "cuda_gpu",
            "precision": "INT8",
            "conversion_success": 1,
            "operator_support": 1,
            "fallback_ratio": 0,
            "build_success": 1,
            "precision_support": 1,
            "int8_audit_path": "int8_audit.json",
            "latency_ms_p50": 8,
            "latency_ms_p99": 10,
            "requested_provider": "CUDAExecutionProvider",
        }
    ])
    out = normalize_compatibility(df, None, evidence_base_dir=tmp_path, int8_audit={"affect_pcs": True})
    row = out.iloc[0]
    assert row["int8_audit_status"] == "int8_audit_json_loaded"
    assert bool(row["strict_int8_graph_pass"]) is True
    assert row["conv_total"] == 10
    assert row["quantized_conv_like_count"] == 10
    assert row["fp32_conv_fraction_estimated"] == 0.0
    assert bool(row["int8_graph_claim_allowed"]) is True


def test_normalize_compatibility_failed_int8_audit_blocks_precision_support(tmp_path):
    audit = {
        "strict_int8_graph_pass": False,
        "strict_int8_graph_reason": "conv_quantized_fraction_below_threshold",
        "conv_total": 10,
        "quantized_conv_like_count": 2,
        "conv_quantized_fraction": 0.2,
        "fp32_conv_fraction_estimated": 0.8,
    }
    audit_path = tmp_path / "bad_int8_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    df = pd.DataFrame([
        {
            "model_id": "m",
            "runtime_id": "cuda_gpu",
            "precision": "INT8",
            "conversion_success": 1,
            "operator_support": 1,
            "fallback_ratio": 0,
            "build_success": 1,
            "precision_support": 1,
            "int8_audit_path": "bad_int8_audit.json",
            "latency_ms_p50": 8,
            "latency_ms_p99": 10,
            "requested_provider": "CUDAExecutionProvider",
        }
    ])
    out = normalize_compatibility(df, None, evidence_base_dir=tmp_path, int8_audit={"affect_pcs": True})
    row = out.iloc[0]
    assert bool(row["strict_int8_graph_pass"]) is False
    assert row["precision_support_effective_norm"] == 0.0
    assert row["pcs"] < 1.0
