from __future__ import annotations

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

from .audit import audit_onnx_int8_graph, canonical_provider_name, parse_ort_profile, safe_json_dumps
from .config import normalize_weights, project_claim_defaults


DEFAULT_PCS_WEIGHTS = {
    "conversion_success": 0.25,
    "operator_support": 0.20,
    "fallback_avoidance": 0.20,
    "build_success": 0.20,
    "precision_support": 0.15,
}


_ACCELERATOR_TOKENS = ("cuda", "tensorrt", "trt", "openvino", "npu", "tpu", "xla", "gpu", "rocm", "directml", "dml")


def _clip01(value) -> float:
    if pd.isna(value):
        return float("nan")
    try:
        return float(min(1.0, max(0.0, float(value))))
    except Exception:
        return float("nan")


def _resolve_evidence_path(base_dir: Path | None, value) -> Path | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    if base_dir is None:
        return p.resolve()
    return (base_dir / p).resolve()


def _series_with_aliases(out: pd.DataFrame, aliases: list[str]) -> pd.Series:
    result = pd.Series([np.nan] * len(out), index=out.index, dtype="float64")
    for col in aliases:
        if col in out.columns:
            result = result.fillna(pd.to_numeric(out[col], errors="coerce"))
    return result


def _looks_accelerated(runtime_id: object = "", requested_provider: object = "") -> bool:
    text = f"{runtime_id} {requested_provider}".lower()
    return any(tok in text for tok in _ACCELERATOR_TOKENS)


def _numeric(row: pd.Series, key: str) -> float:
    return pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]


def _audit_provider_row(row: pd.Series, evidence_base_dir: Path | None) -> dict[str, object]:
    profile_path = _resolve_evidence_path(evidence_base_dir, row.get("provider_profile_path"))
    requested_provider = row.get("requested_provider")
    if requested_provider is None or pd.isna(requested_provider) or str(requested_provider).strip() == "":
        requested_provider = row.get("target_provider")
    requested_provider_canonical = canonical_provider_name(str(requested_provider)) if requested_provider is not None and not pd.isna(requested_provider) else None

    common_empty = {
        "provider_profile_resolved_path": "",
        "provider_node_counts": "",
        "unique_node_provider_counts": "",
        "provider_event_counts": "",
        "provider_node_event_counts": "",
        "node_event_provider_counts": "",
        "fallback_node_counts": "",
        "fallback_unique_node_counts": "",
        "fallback_event_counts": "",
        "op_provider_counts": "",
        "profile_event_filter": "",
        "total_profiled_node_count": np.nan,
        "total_unique_profiled_node_count": np.nan,
        "total_profiled_node_event_count": np.nan,
        "unassigned_node_event_count": np.nan,
        "requested_provider_node_count": np.nan,
        "requested_provider_unique_node_count": np.nan,
        "requested_provider_event_count": np.nan,
        "requested_provider_fraction": np.nan,
        "requested_provider_unique_node_fraction": np.nan,
        "requested_provider_event_fraction": np.nan,
        "fallback_node_count": np.nan,
        "fallback_event_count": np.nan,
        "fallback_ratio_measured": np.nan,
        "fallback_ratio_event_measured": np.nan,
        "mixed_provider_node_count": np.nan,
    }
    if profile_path is None:
        return {"provider_audit_status": "profile_not_provided", "requested_provider_canonical": requested_provider_canonical or "", **common_empty}
    if not profile_path.exists():
        return {"provider_audit_status": "profile_path_missing", "requested_provider_canonical": requested_provider_canonical or "", **common_empty, "provider_profile_resolved_path": str(profile_path)}
    try:
        audit = parse_ort_profile(profile_path, requested_provider=requested_provider_canonical)
        total_nodes = audit.get("total_profiled_node_count")
        status = "profile_verified" if total_nodes and int(total_nodes) > 0 else "profile_has_no_node_events"
        return {
            "provider_audit_status": status,
            "requested_provider_canonical": audit.get("requested_provider", requested_provider_canonical or ""),
            "provider_profile_resolved_path": str(profile_path),
            "provider_node_counts": safe_json_dumps(audit.get("provider_node_counts", {})),
            "unique_node_provider_counts": safe_json_dumps(audit.get("unique_node_provider_counts", audit.get("provider_node_counts", {}))),
            "provider_event_counts": safe_json_dumps(audit.get("provider_event_counts", {})),
            "provider_node_event_counts": safe_json_dumps(audit.get("provider_node_event_counts", audit.get("provider_event_counts", {}))),
            "node_event_provider_counts": safe_json_dumps(audit.get("node_event_provider_counts", audit.get("provider_event_counts", {}))),
            "fallback_node_counts": safe_json_dumps(audit.get("fallback_node_counts", {})),
            "fallback_unique_node_counts": safe_json_dumps(audit.get("fallback_unique_node_counts", audit.get("fallback_node_counts", {}))),
            "fallback_event_counts": safe_json_dumps(audit.get("fallback_event_counts", {})),
            "op_provider_counts": safe_json_dumps(audit.get("op_provider_counts", {})),
            "profile_event_filter": audit.get("profile_event_filter", ""),
            "total_profiled_node_count": audit.get("total_profiled_node_count"),
            "total_unique_profiled_node_count": audit.get("total_unique_profiled_node_count", audit.get("total_profiled_node_count")),
            "total_profiled_node_event_count": audit.get("total_profiled_node_event_count"),
            "unassigned_node_event_count": audit.get("unassigned_node_event_count"),
            "requested_provider_node_count": audit.get("requested_provider_node_count"),
            "requested_provider_unique_node_count": audit.get("requested_provider_unique_node_count", audit.get("requested_provider_node_count")),
            "requested_provider_event_count": audit.get("requested_provider_event_count"),
            "requested_provider_fraction": audit.get("requested_provider_fraction"),
            "requested_provider_unique_node_fraction": audit.get("requested_provider_unique_node_fraction", audit.get("requested_provider_fraction")),
            "requested_provider_event_fraction": audit.get("requested_provider_event_fraction"),
            "fallback_node_count": audit.get("fallback_node_count"),
            "fallback_event_count": audit.get("fallback_event_count"),
            "fallback_ratio_measured": audit.get("fallback_ratio_measured"),
            "fallback_ratio_event_measured": audit.get("fallback_ratio_event_measured"),
            "mixed_provider_node_count": audit.get("mixed_provider_node_count"),
        }
    except Exception as e:
        return {"provider_audit_status": f"profile_parse_error:{type(e).__name__}", "requested_provider_canonical": requested_provider_canonical or "", **common_empty, "provider_profile_resolved_path": str(profile_path)}



def _audit_int8_row(row: pd.Series, evidence_base_dir: Path | None, int8_cfg: dict | None = None) -> dict[str, object]:
    """Load precomputed INT8 graph-audit evidence for an INT8 condition.

    `int8_audit_path` normally points to JSON produced by:
        hpls-eval audit-onnx --model model_int8.onnx --out audit.json

    For convenience, if the path ends with `.onnx`, the evaluator attempts to
    run the audit directly. This keeps the matrix integrated while preserving a
    clean no-fake-results policy.
    """
    precision = str(row.get("precision", "")).upper()
    empty = {
        "int8_audit_status": "not_applicable_for_non_int8" if precision != "INT8" else "int8_audit_path_not_provided",
        "int8_audit_resolved_path": "",
        "strict_int8_graph_pass": np.nan,
        "strict_int8_graph_reason": "",
        "conv_total": np.nan,
        "qlinear_conv_count": np.nan,
        "conv_integer_count": np.nan,
        "qoperator_conv_count": np.nan,
        "conv_like_total": np.nan,
        "qdq_wrapped_conv_count": np.nan,
        "qdq_conv_with_quantized_weight_count": np.nan,
        "quantized_conv_like_count": np.nan,
        "conv_quantized_fraction": np.nan,
        "fp32_conv_count_estimated": np.nan,
        "fp32_conv_fraction_estimated": np.nan,
        "int8_graph_claim_allowed": np.nan if precision != "INT8" else False,
    }
    if precision != "INT8":
        return empty

    path_value = row.get("int8_audit_path")
    audit_path = _resolve_evidence_path(evidence_base_dir, path_value)
    if audit_path is None:
        return empty
    if not audit_path.exists():
        out = dict(empty)
        out.update({"int8_audit_status": "int8_audit_path_missing", "int8_audit_resolved_path": str(audit_path)})
        return out

    try:
        if audit_path.suffix.lower() == ".onnx":
            cfg = int8_cfg or {}
            audit = audit_onnx_int8_graph(
                audit_path,
                strict_conv_threshold=float(cfg.get("strict_conv_threshold", 0.80)),
                max_fp32_conv_fraction=float(cfg.get("max_fp32_conv_fraction", 0.20)),
            )
            status = "int8_onnx_audit_computed"
        else:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            status = "int8_audit_json_loaded"
        strict_pass = bool(audit.get("strict_int8_graph_pass", False))
        out = dict(empty)
        out.update({
            "int8_audit_status": status,
            "int8_audit_resolved_path": str(audit_path),
            "strict_int8_graph_pass": strict_pass,
            "strict_int8_graph_reason": audit.get("strict_int8_graph_reason", ""),
            "conv_total": audit.get("conv_total", np.nan),
            "qlinear_conv_count": audit.get("qlinear_conv_count", np.nan),
            "conv_integer_count": audit.get("conv_integer_count", np.nan),
            "qoperator_conv_count": audit.get("qoperator_conv_count", np.nan),
            "conv_like_total": audit.get("conv_like_total", np.nan),
            "qdq_wrapped_conv_count": audit.get("qdq_wrapped_conv_count", np.nan),
            "qdq_conv_with_quantized_weight_count": audit.get("qdq_conv_with_quantized_weight_count", np.nan),
            "quantized_conv_like_count": audit.get("quantized_conv_like_count", np.nan),
            "conv_quantized_fraction": audit.get("conv_quantized_fraction", np.nan),
            "fp32_conv_count_estimated": audit.get("fp32_conv_count_estimated", np.nan),
            "fp32_conv_fraction_estimated": audit.get("fp32_conv_fraction_estimated", np.nan),
            "int8_graph_claim_allowed": strict_pass,
        })
        return out
    except Exception as e:
        out = dict(empty)
        out.update({
            "int8_audit_status": f"int8_audit_parse_error:{type(e).__name__}",
            "int8_audit_resolved_path": str(audit_path),
            "int8_graph_claim_allowed": False,
        })
        return out


def _backend_specific_profile_status(row: pd.Series, evidence_base_dir: Path | None) -> str:
    runtime_id = str(row.get("runtime_id", "")).lower()
    provider = str(row.get("requested_provider_canonical", row.get("requested_provider", ""))).lower()
    is_xla_tpu = any(tok in f"{runtime_id} {provider}" for tok in ["xla", "tpu"])
    if not is_xla_tpu:
        return "not_applicable"
    for key in ("xla_profile_path", "backend_profile_path"):
        p = _resolve_evidence_path(evidence_base_dir, row.get(key))
        if p is not None and p.exists():
            return f"{key}_provided_unparsed"
        if p is not None and not p.exists():
            return f"{key}_missing"
    return "xla_tpu_backend_profile_not_provided"

def _fallback_severity(ratio: float) -> str:
    if pd.isna(ratio):
        return "unknown"
    r = float(ratio)
    if r <= 0:
        return "none"
    if r <= 0.05:
        return "minor_le_5pct"
    if r <= 0.20:
        return "moderate_le_20pct"
    return "high_gt_20pct"


def _backend_claim_status(row: pd.Series, threshold: float) -> str:
    status = str(row.get("provider_audit_status", ""))
    requested = str(row.get("requested_provider_canonical") or row.get("requested_provider") or "").strip()
    frac = _numeric(row, "requested_provider_fraction")
    accelerated = _looks_accelerated(row.get("runtime_id", ""), requested)

    if not requested and accelerated:
        return "no_requested_provider_recorded"
    if status == "profile_verified":
        if pd.isna(frac):
            return "profile_verified_requested_fraction_missing"
        if frac <= 0:
            return "requested_provider_not_observed"
        if frac < threshold:
            return "partial_fallback_below_threshold"
        return "profile_verified_requested_provider"
    if status == "profile_not_provided":
        return "log_only_no_provider_profile" if accelerated else "profile_not_required_for_nonaccelerated_row"
    if status == "profile_path_missing":
        return "provider_profile_path_missing"
    if status == "profile_has_no_node_events":
        return "provider_profile_has_no_node_events"
    if status.startswith("profile_parse_error"):
        return "provider_profile_parse_error"
    return "provider_audit_unknown"


def _device_claim_status(row: pd.Series) -> str:
    runtime_id = str(row.get("runtime_id", ""))
    requested = str(row.get("requested_device", "") if not pd.isna(row.get("requested_device", "")) else "").strip()
    actual = str(row.get("actual_compiled_device", "") if not pd.isna(row.get("actual_compiled_device", "")) else "").strip()
    provider = str(row.get("requested_provider_canonical", "") or row.get("requested_provider", ""))
    looks_openvino_or_npu = "openvino" in runtime_id.lower() or "npu" in runtime_id.lower() or "OpenVINO" in provider

    if not requested and not looks_openvino_or_npu:
        return "device_not_applicable_or_not_recorded"
    if requested.upper() == "AUTO" and not actual:
        return "auto_requested_actual_device_missing"
    if requested.upper() == "AUTO" and actual:
        return f"auto_resolved_to_{actual}"
    if requested and not actual:
        return "requested_device_recorded_actual_missing"
    if requested and actual:
        req_tokens = set(re.findall(r"[A-Za-z0-9]+", requested.upper()))
        actual_tokens = set(re.findall(r"[A-Za-z0-9]+", actual.upper()))
        actual_upper = actual.upper()
        if "NPU" in req_tokens and "NPU" not in actual_tokens:
            return "target_device_mismatch_requested_npu"
        if "GPU" in req_tokens and "CUDA" in actual_upper:
            return "target_device_confirmed"
        if "GPU" in req_tokens and "GPU" not in actual_tokens and "CUDA" not in actual_upper:
            return "target_device_mismatch_requested_gpu"
        if req_tokens and (req_tokens & actual_tokens):
            return "target_device_confirmed"
        return "actual_device_recorded_but_not_textually_matched"
    return "device_audit_unknown"


def _runtime_evidence_grade(row: pd.Series, threshold: float) -> str:
    """Human-readable evidence grade for runtime claims."""
    backend_status = str(row.get("backend_claim_status", ""))
    p99 = _numeric(row, "latency_ms_p99")
    device_status = str(row.get("device_claim_status", ""))
    device_mismatch = "mismatch" in device_status or device_status.endswith("missing")
    if backend_status == "profile_verified_requested_provider" and pd.notna(p99) and not device_mismatch:
        return "A_profile_verified_tail_latency_device_ok"
    if backend_status == "profile_verified_requested_provider":
        return "B_profile_verified_but_tail_or_device_evidence_incomplete"
    if backend_status in {"partial_fallback_below_threshold", "requested_provider_not_observed"}:
        return "B_minus_profile_verified_fallback_visible"
    backend_specific = str(row.get("backend_specific_profile_status", ""))
    if backend_status == "log_only_no_provider_profile":
        if backend_specific.endswith("_provided_unparsed"):
            return "C_plus_backend_specific_profile_provided_unparsed"
        return "C_log_only_no_provider_profile"
    if backend_status == "profile_not_required_for_nonaccelerated_row":
        return "CPU_or_nonaccelerated_log_row"
    return "D_runtime_evidence_unusable_or_missing"


def _condition_run_status(row: pd.Series) -> str:
    """Normalize optional condition-level execution status.

    Users may log NOT_RUN / FAILED / MISSING_BACKEND / SUCCESS explicitly in
    compatibility.csv. If absent, derive a conservative status from conversion,
    build, precision, and backend-evidence fields without hiding weak evidence.
    """
    for key in ("run_status", "condition_status", "execution_status", "status"):
        value = row.get(key)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).strip().upper()
    conv = _clip01(row.get("conversion_success_norm", row.get("conversion_success")))
    build = _clip01(row.get("build_success_norm", row.get("build_success")))
    precision = _clip01(row.get("precision_support_norm", row.get("precision_support")))
    if pd.notna(conv) and conv <= 0:
        return "FAILED"
    if pd.notna(build) and build <= 0:
        return "FAILED"
    if pd.notna(precision) and precision <= 0:
        return "FAILED"
    backend_status = str(row.get("backend_claim_status", ""))
    if backend_status in {"provider_profile_path_missing", "no_requested_provider_recorded"}:
        return "MISSING_BACKEND"
    if backend_status in {"provider_profile_parse_error", "provider_profile_has_no_node_events"}:
        return "FAILED"
    return "SUCCESS"


def _bool_backend_claim_allowed(row: pd.Series, threshold: float) -> bool:
    accelerated = _looks_accelerated(row.get("runtime_id", ""), row.get("requested_provider_canonical", row.get("requested_provider", "")))
    status = str(row.get("provider_audit_status", ""))
    frac = _numeric(row, "requested_provider_fraction")
    if not accelerated and status == "profile_not_provided":
        return True
    return bool(status == "profile_verified" and pd.notna(frac) and float(frac) >= threshold)


def _bool_hardware_claim_allowed(row: pd.Series, threshold: float) -> bool:
    accelerated = _looks_accelerated(row.get("runtime_id", ""), row.get("requested_provider_canonical", row.get("requested_provider", "")))
    if not accelerated:
        return True
    if not _bool_backend_claim_allowed(row, threshold):
        return False
    device_status = str(row.get("device_claim_status", ""))
    if "mismatch" in device_status or device_status.endswith("missing"):
        return False
    return True


def normalize_compatibility(
    df: pd.DataFrame,
    weights: dict[str, float] | None,
    *,
    evidence_base_dir: str | Path | None = None,
    runtime_audit: dict | None = None,
    int8_audit: dict | None = None,
    domain_claim: dict | None = None,
) -> pd.DataFrame:
    if df.empty:
        out_empty = df.copy()
        claims = project_claim_defaults(domain_claim or {})
        for key in ["domain_context", "claim_scope", "clinical_claim_allowed", "clinical_claim_status"]:
            if key not in out_empty.columns:
                out_empty[key] = []
        return out_empty
    w = normalize_weights(weights or DEFAULT_PCS_WEIGHTS, name="pcs_weights")
    out = df.copy()
    base_dir = Path(evidence_base_dir).expanduser().resolve() if evidence_base_dir else None
    audit_cfg = runtime_audit or {}
    int8_cfg = int8_audit or {}
    threshold = float(audit_cfg.get("requested_provider_fraction_warning_threshold", 0.95))
    claims = project_claim_defaults(domain_claim or {})

    for key in ["domain_context", "claim_scope"]:
        if key not in out.columns:
            out[key] = claims[key]
        else:
            blank = out[key].isna() | (out[key].astype(str).str.strip() == "")
            out.loc[blank, key] = claims[key]
    # Hard gate for this non-clinical research prototype. Even when medical
    # lesion terminology is used as a future target, no clinical claim is
    # allowed by this package output.
    out["clinical_claim_allowed"] = False
    out["clinical_claim_status"] = claims["clinical_claim_status"]

    for col in [
        "requested_provider",
        "target_provider",
        "provider_profile_path",
        "requested_device",
        "available_devices",
        "actual_compiled_device",
        "performance_hint",
        "runtime_id",
        "int8_audit_path",
        "xla_profile_path",
        "backend_profile_path",
        "backend_evidence_notes",
        "run_status",
        "condition_status",
        "execution_status",
        "status",
    ]:
        if col not in out.columns:
            out[col] = ""

    provider_audits = [_audit_provider_row(row, base_dir) for _, row in out.iterrows()]
    provider_df = pd.DataFrame(provider_audits, index=out.index)
    out = pd.concat([out, provider_df], axis=1)

    int8_audits = [_audit_int8_row(row, base_dir, int8_cfg=int8_cfg) for _, row in out.iterrows()]
    int8_df = pd.DataFrame(int8_audits, index=out.index)
    out = pd.concat([out, int8_df], axis=1)
    out["backend_specific_profile_status"] = out.apply(lambda row: _backend_specific_profile_status(row, base_dir), axis=1)

    for col in ["conversion_success", "operator_support", "fallback_ratio", "build_success", "precision_support"]:
        if col not in out:
            out[col] = np.nan
    out["conversion_success_norm"] = out["conversion_success"].map(_clip01)
    out["operator_support_norm"] = out["operator_support"].map(_clip01)
    out["build_success_norm"] = out["build_success"].map(_clip01)
    out["precision_support_norm"] = out["precision_support"].map(_clip01)
    affect_pcs = bool(int8_cfg.get("affect_pcs", True))
    require_int8_audit = bool(int8_cfg.get("require_for_int8", False))

    def _precision_support_effective(row: pd.Series) -> float:
        base = row.get("precision_support_norm", np.nan)
        if str(row.get("precision", "")).upper() != "INT8" or not affect_pcs:
            return base
        audit_allowed = row.get("int8_graph_claim_allowed")
        audit_status = str(row.get("int8_audit_status", ""))
        if pd.notna(audit_allowed) and bool(audit_allowed) is True:
            return base
        if audit_status in {"int8_audit_path_not_provided", "int8_audit_path_missing"} and not require_int8_audit:
            return base
        if audit_status in {"int8_audit_path_not_provided", "int8_audit_path_missing"} and require_int8_audit:
            return float("nan")
        return 0.0

    out["precision_support_effective_norm"] = out.apply(_precision_support_effective, axis=1)

    logged_fallback = pd.to_numeric(out["fallback_ratio"], errors="coerce")
    measured_fallback = pd.to_numeric(out["fallback_ratio_measured"], errors="coerce")
    out["fallback_ratio_effective"] = measured_fallback.fillna(logged_fallback)
    out["fallback_ratio_source"] = np.where(measured_fallback.notna(), "provider_profile_node_fraction", "compatibility_log")
    out["fallback_avoidance_norm"] = out["fallback_ratio_effective"].map(
        lambda x: 1.0 - _clip01(x) if not pd.isna(_clip01(x)) else float("nan")
    )
    out["fallback_severity"] = out["fallback_ratio_effective"].map(_fallback_severity)

    def row_pcs(row) -> float:
        terms = {
            "conversion_success": row["conversion_success_norm"],
            "operator_support": row["operator_support_norm"],
            "fallback_avoidance": row["fallback_avoidance_norm"],
            "build_success": row["build_success_norm"],
            "precision_support": row["precision_support_effective_norm"],
        }
        if any(pd.isna(v) for v in terms.values()):
            return float("nan")
        return float(sum(w[k] * terms[k] for k in w))

    out["pcs"] = out.apply(row_pcs, axis=1)

    # Preserve and normalize latency fields. v2-deep accepts both latency_p99_ms
    # and latency_ms_p99, but outputs the latency_ms_* form requested in the
    # review notes.
    out["latency_effective_ms"] = _series_with_aliases(out, ["latency_ms_p50", "latency_p50_ms", "latency_ms", "latency_mean_ms"])
    out["latency_ms_p50"] = _series_with_aliases(out, ["latency_ms_p50", "latency_p50_ms"])
    out["latency_ms_p90"] = _series_with_aliases(out, ["latency_ms_p90", "latency_p90_ms"])
    out["latency_ms_p95"] = _series_with_aliases(out, ["latency_ms_p95", "latency_p95_ms"])
    out["latency_ms_p99"] = _series_with_aliases(out, ["latency_ms_p99", "latency_p99_ms"])
    out["latency_ms_min"] = _series_with_aliases(out, ["latency_ms_min", "latency_min_ms"])
    out["latency_ms_max"] = _series_with_aliases(out, ["latency_ms_max", "latency_max_ms"])
    out["efficiency_inv_latency"] = out["latency_effective_ms"].map(lambda x: 1.0 / float(x) if pd.notna(x) and float(x) > 0 else float("nan"))

    out["requested_provider_fraction_threshold"] = threshold
    out["backend_claim_status"] = out.apply(lambda row: _backend_claim_status(row, threshold), axis=1)
    out["backend_claim_allowed"] = out.apply(lambda row: _bool_backend_claim_allowed(row, threshold), axis=1)
    out["device_claim_status"] = out.apply(_device_claim_status, axis=1)
    out["hardware_claim_allowed"] = out.apply(lambda row: _bool_hardware_claim_allowed(row, threshold), axis=1)
    out["runtime_evidence_grade"] = out.apply(lambda row: _runtime_evidence_grade(row, threshold), axis=1)
    out["run_status_normalized"] = out.apply(_condition_run_status, axis=1)
    return out


def lookup_condition(df: pd.DataFrame, model_id: str, runtime_id: str, precision: str) -> pd.Series | None:
    if df.empty:
        return None
    mask = (
        (df["model_id"].astype(str) == str(model_id))
        & (df["runtime_id"].astype(str) == str(runtime_id))
        & (df["precision"].astype(str) == str(precision))
    )
    sub = df[mask]
    if sub.empty:
        return None
    return sub.iloc[0]
