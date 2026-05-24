from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .compatibility import normalize_compatibility, lookup_condition
from .config import resolve_from_config, ensure_project_claim_defaults, project_claim_defaults
from .io import (
    read_dataset_manifest, read_compatibility_log, resolve_gt_path, resolve_prediction_path,
    condition_matrix, write_json, environment_snapshot, collect_input_hashes,
)
from .masks import load_mask
from .metrics import compute_mask_metrics
from .report import write_report
from .validation import validate_inputs
from .warning_utils import dedupe_warnings


def _safe_div(num: float, den: float, eps: float = 1e-8) -> float:
    if pd.isna(num) or pd.isna(den):
        return float("nan")
    return float(num / (den + eps))


def _runtime_transfer_gap(e_ref: float, e: float, eps: float = 1e-8) -> float:
    if pd.isna(e_ref) or pd.isna(e):
        return float("nan")
    return float(abs(e_ref - e) / (e_ref + eps))


def _runtime_retention_from_rtg(rtg: float) -> float:
    if pd.isna(rtg):
        return float("nan")
    return float(1.0 / (1.0 + rtg))


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    return dedupe_warnings(items)


def _runtime_evidence_warnings(compat_norm: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if compat_norm is None or compat_norm.empty:
        return warnings
    for _, row in compat_norm.iterrows():
        model_id = str(row.get("model_id", ""))
        runtime_id = str(row.get("runtime_id", ""))
        precision = str(row.get("precision", ""))
        prefix = f"model={model_id} runtime={runtime_id} precision={precision}"
        backend_status = str(row.get("backend_claim_status", ""))
        grade = str(row.get("runtime_evidence_grade", ""))
        if backend_status in {
            "partial_fallback_below_threshold",
            "requested_provider_not_observed",
            "provider_profile_has_no_node_events",
            "provider_profile_path_missing",
            "provider_profile_parse_error",
            "log_only_no_provider_profile",
            "no_requested_provider_recorded",
        }:
            warnings.append(f"RUNTIME_EVIDENCE_WEAK {prefix}: backend_claim_status={backend_status}, runtime_evidence_grade={grade}")
        hardware_allowed = row.get("hardware_claim_allowed")
        if pd.notna(hardware_allowed) and bool(hardware_allowed) is False:
            warnings.append(f"HARDWARE_CLAIM_NOT_ALLOWED {prefix}: profile/device evidence is insufficient for a strong hardware claim")
        frac = pd.to_numeric(pd.Series([row.get("requested_provider_fraction")]), errors="coerce").iloc[0]
        if pd.notna(frac) and frac < float(row.get("requested_provider_fraction_threshold", 0.95)):
            warnings.append(f"PROVIDER_FALLBACK_DETECTED {prefix}: requested_provider_fraction={frac:.4f}, fallback_node_counts={row.get('fallback_node_counts', '')}")
        device_status = str(row.get("device_claim_status", ""))
        if "mismatch" in device_status or device_status.endswith("missing"):
            warnings.append(f"DEVICE_EVIDENCE_WEAK {prefix}: device_claim_status={device_status}")
        if str(row.get("precision", "")).upper() == "INT8":
            int8_status = str(row.get("int8_audit_status", ""))
            int8_allowed = row.get("int8_graph_claim_allowed")
            if int8_status not in {"int8_audit_json_loaded", "int8_onnx_audit_computed"}:
                warnings.append(f"INT8_AUDIT_NOT_INTEGRATED {prefix}: int8_audit_status={int8_status}")
            elif pd.notna(int8_allowed) and bool(int8_allowed) is False:
                warnings.append(f"INT8_GRAPH_CLAIM_NOT_ALLOWED {prefix}: strict_int8_graph_pass=false, reason={row.get('strict_int8_graph_reason', '')}")
        backend_specific = str(row.get("backend_specific_profile_status", ""))
        if backend_specific in {"xla_tpu_backend_profile_not_provided"} or backend_specific.endswith("_missing"):
            warnings.append(f"TPU_XLA_BACKEND_EVIDENCE_WEAK {prefix}: backend_specific_profile_status={backend_specific}")
    return warnings



def _baseline_missing_messages(cfg: dict[str, Any], aggregate: pd.DataFrame) -> list[str]:
    messages: list[str] = []
    baseline = cfg.get("baseline", {})
    baseline_runtime = str(baseline.get("runtime_id"))
    baseline_precision = str(baseline.get("precision", "FP32"))
    models = [m["model_id"] if isinstance(m, dict) else str(m) for m in cfg.get("models", [])]
    for model_id in models:
        if aggregate is None or aggregate.empty:
            messages.append(f"BASELINE_MISSING model={model_id} runtime={baseline_runtime} precision={baseline_precision}: aggregate table is empty")
            continue
        mask = (
            (aggregate["model_id"].astype(str) == str(model_id))
            & (aggregate["runtime_id"].astype(str) == baseline_runtime)
            & (aggregate["precision"].astype(str) == baseline_precision)
        )
        row = aggregate[mask]
        if row.empty:
            messages.append(f"BASELINE_MISSING model={model_id} runtime={baseline_runtime} precision={baseline_precision}: no baseline row")
            continue
        n_ok = pd.to_numeric(row.iloc[0].get("n_ok"), errors="coerce")
        if pd.isna(n_ok) or int(n_ok) <= 0:
            messages.append(f"BASELINE_MISSING model={model_id} runtime={baseline_runtime} precision={baseline_precision}: n_ok={row.iloc[0].get('n_ok')}")
    return messages

def evaluate(cfg: dict[str, Any], config_path: Path, out_dir: str | Path) -> dict[str, Any]:
    ensure_project_claim_defaults(cfg)
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    warnings, validation_summary = validate_inputs(cfg, config_path)
    (out / "validation_warnings.log").write_text("\n".join(warnings) + ("\n" if warnings else ""), encoding="utf-8")
    with (out / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    write_json(out / "environment.json", environment_snapshot())

    paths = cfg.get("paths", {})
    manifest_path = resolve_from_config(config_path, paths.get("dataset_manifest"))
    compat_path = resolve_from_config(config_path, paths.get("compatibility_log"))
    samples = read_dataset_manifest(manifest_path)
    conditions = condition_matrix(cfg)
    metrics_cfg = cfg.get("metrics", {})
    mask_cfg = cfg.get("mask", {})
    threshold = float(mask_cfg.get("positive_threshold", 127))
    fail_missing_pred = bool(cfg.get("reporting", {}).get("fail_on_missing_prediction", False))

    compat_raw = read_compatibility_log(compat_path)
    compat_norm = normalize_compatibility(
        compat_raw,
        cfg.get("pcs_weights", {}),
        evidence_base_dir=compat_path.parent if compat_path else config_path.parent,
        runtime_audit=cfg.get("runtime_audit", {}),
        int8_audit=cfg.get("int8_audit", {}),
        domain_claim=cfg.get("project", {}),
    )
    compat_norm.to_csv(out / "compatibility_normalized.csv", index=False)
    runtime_evidence = build_runtime_evidence_matrix(compat_norm)
    runtime_evidence.to_csv(out / "runtime_evidence_matrix.csv", index=False)
    warnings.extend(_runtime_evidence_warnings(compat_norm))
    warnings = _dedupe_preserve_order(warnings)

    per_sample_rows = []
    hash_paths = [manifest_path]
    if compat_path and compat_path.exists():
        hash_paths.append(compat_path)

    for cond in tqdm(conditions, desc="Evaluating conditions", unit="condition"):
        model_id = cond["model_id"]
        runtime_id = cond["runtime_id"]
        precision = cond["precision"]
        for _, sample in samples.iterrows():
            sample_id = str(sample["sample_id"])
            gt_path = resolve_gt_path(config_path, sample["gt_mask"])
            pred_path = resolve_prediction_path(cfg, config_path, model_id=model_id, runtime_id=runtime_id, precision=precision, sample_id=sample_id)
            base = {
                "model_id": model_id,
                "runtime_id": runtime_id,
                "precision": precision,
                "sample_id": sample_id,
                "split": sample.get("split", ""),
                "gt_mask": str(gt_path),
                "prediction_mask": str(pred_path),
                "status": "ok",
                "error": "",
            }
            try:
                if not pred_path.exists():
                    msg = f"Prediction not found: {pred_path}"
                    if fail_missing_pred:
                        raise FileNotFoundError(msg)
                    row = dict(base)
                    row.update({"status": "missing_prediction", "error": msg})
                    per_sample_rows.append(row)
                    continue
                gt = load_mask(gt_path, positive_threshold=threshold)
                pred = load_mask(pred_path, positive_threshold=threshold)
                hash_paths.extend([gt_path, pred_path])
                vals = compute_mask_metrics(pred, gt, metrics_cfg)
                row = dict(base)
                row.update(vals)
                per_sample_rows.append(row)
            except Exception as e:
                row = dict(base)
                row.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
                per_sample_rows.append(row)
                warnings.append(f"EVAL_ERROR model={model_id} runtime={runtime_id} precision={precision} sample={sample_id}: {e}")

    base_cols = [
        "model_id", "runtime_id", "precision", "sample_id", "split",
        "gt_mask", "prediction_mask", "status", "error",
    ]
    metric_cols = [
        "dice", "iou", "recall", "boundary_recall", "boundary_iou", "small_region_recall", "small_lesion_recall",
        "gt_boundary_px", "pred_boundary_px", "empty_gt_boundary", "boundary_fp_empty_gt",
        "has_small_region", "small_region_gt_components", "small_region_gt_px",
        "has_small_lesion", "small_lesion_gt_components", "small_lesion_gt_px",
    ]
    per_sample = pd.DataFrame(per_sample_rows)
    if per_sample.empty:
        per_sample = pd.DataFrame(columns=base_cols + metric_cols)
    per_sample.to_csv(out / "per_sample_metrics.csv", index=False)

    ok = per_sample[per_sample["status"] == "ok"].copy() if "status" in per_sample.columns else pd.DataFrame()
    if ok.empty:
        aggregate = pd.DataFrame(columns=["model_id", "runtime_id", "precision"] + metric_cols + ["n_ok"])
    else:
        aggregate = ok.groupby(["model_id", "runtime_id", "precision"], dropna=False)[metric_cols].mean(numeric_only=True).reset_index()
        grouped = ok.groupby(["model_id", "runtime_id", "precision"], dropna=False)
        n_ok = grouped.size().reset_index(name="n_ok")
        boundary_valid = grouped["boundary_iou"].apply(lambda x: int(x.notna().sum())).reset_index(name="n_boundary_valid")
        empty_boundary = grouped["empty_gt_boundary"].sum(min_count=1).reset_index(name="empty_gt_boundary_count")
        boundary_fp_empty = grouped["boundary_fp_empty_gt"].sum(min_count=1).reset_index(name="boundary_fp_empty_gt_count")
        small_valid = grouped["small_region_recall"].apply(lambda x: int(x.notna().sum())).reset_index(name="n_small_region_valid")
        small_components = grouped["small_region_gt_components"].sum(min_count=1).reset_index(name="small_region_gt_components_total")
        small_px = grouped["small_region_gt_px"].sum(min_count=1).reset_index(name="small_region_gt_px_total")
        for extra in [n_ok, boundary_valid, empty_boundary, boundary_fp_empty, small_valid, small_components, small_px]:
            aggregate = aggregate.merge(extra, on=["model_id", "runtime_id", "precision"], how="left")
    aggregate.to_csv(out / "aggregate_metrics.csv", index=False)

    baseline_missing = _baseline_missing_messages(cfg, aggregate)
    if baseline_missing:
        warnings.extend(baseline_missing)
        fail_on_missing_baseline = bool(cfg.get("reporting", {}).get("fail_on_missing_baseline", True))
        if fail_on_missing_baseline:
            warnings = _dedupe_preserve_order(warnings)
            (out / "validation_warnings.log").write_text("\n".join(warnings) + "\n", encoding="utf-8")
            raise RuntimeError("Missing required FP32 baseline condition. " + " | ".join(baseline_missing))

    condition_summary = build_condition_summary(cfg, aggregate, compat_norm)
    condition_summary.to_csv(out / "condition_summary.csv", index=False)

    warnings = _dedupe_preserve_order(warnings)
    hash_df = collect_input_hashes(hash_paths)
    hash_df.to_csv(out / "input_hashes.csv", index=False)
    (out / "validation_warnings.log").write_text("\n".join(warnings) + ("\n" if warnings else ""), encoding="utf-8")

    if bool(cfg.get("reporting", {}).get("write_markdown_report", True)):
        write_report(out / "report.md", cfg, validation_summary, warnings, condition_summary, aggregate, per_sample, runtime_evidence)

    return {
        "out_dir": str(out),
        "warnings": warnings,
        "validation_summary": validation_summary,
        "num_per_sample_rows": int(len(per_sample)),
        "num_condition_rows": int(len(condition_summary)),
        "num_runtime_evidence_rows": int(len(runtime_evidence)),
    }


def build_runtime_evidence_matrix(compat_norm: pd.DataFrame) -> pd.DataFrame:
    if compat_norm is None or compat_norm.empty:
        return pd.DataFrame()
    cols = [
        "domain_context", "claim_scope", "clinical_claim_allowed", "clinical_claim_status",
        "model_id", "runtime_id", "precision", "run_status", "run_status_normalized", "requested_provider", "requested_provider_canonical",
        "provider_audit_status", "backend_claim_status", "backend_claim_allowed", "hardware_claim_allowed",
        "runtime_evidence_grade", "requested_provider_fraction", "requested_provider_unique_node_fraction", "requested_provider_event_fraction",
        "requested_provider_fraction_threshold", "fallback_ratio_effective", "fallback_ratio_source", "fallback_severity",
        "provider_node_counts", "unique_node_provider_counts", "provider_node_event_counts", "node_event_provider_counts",
        "provider_event_counts", "fallback_node_counts", "fallback_unique_node_counts", "fallback_event_counts",
        "total_profiled_node_count", "total_unique_profiled_node_count", "total_profiled_node_event_count", "profile_event_filter",
        "int8_audit_path", "int8_audit_status", "strict_int8_graph_pass", "strict_int8_graph_reason",
        "conv_total", "qlinear_conv_count", "qdq_wrapped_conv_count", "quantized_conv_like_count",
        "conv_quantized_fraction", "fp32_conv_fraction_estimated", "int8_graph_claim_allowed",
        "latency_effective_ms", "latency_ms_p50", "latency_ms_p90", "latency_ms_p95", "latency_ms_p99",
        "latency_ms_min", "latency_ms_max", "compile_ms", "requested_device", "available_devices",
        "actual_compiled_device", "device_claim_status", "backend_specific_profile_status", "xla_profile_path", "backend_profile_path",
        "performance_hint", "unsupported_ops", "notes",
    ]
    existing = [c for c in cols if c in compat_norm.columns]
    return compat_norm[existing].copy()


def build_condition_summary(cfg: dict[str, Any], aggregate: pd.DataFrame, compat_norm: pd.DataFrame) -> pd.DataFrame:
    claims = project_claim_defaults(cfg)
    baseline = cfg.get("baseline", {})
    baseline_runtime = str(baseline.get("runtime_id"))
    baseline_precision = str(baseline.get("precision", "FP32"))
    metrics_cfg = cfg.get("metrics", {})
    eps = float(metrics_cfg.get("eps", 1e-8))
    main_metric = str(metrics_cfg.get("main_performance_metric", "dice"))
    hpls_weights = cfg.get("hpls_weights", {})
    if not hpls_weights:
        hpls_weights = {
            "performance_retention": 0.25,
            "boundary_retention": 0.25,
            "small_region_retention": 0.20,
            "runtime_retention": 0.15,
            "precision_compatibility": 0.15,
        }
    total_w = sum(float(v) for v in hpls_weights.values())
    hpls_weights = {k: float(v) / total_w for k, v in hpls_weights.items()}

    rows = []
    if aggregate.empty:
        return pd.DataFrame()
    for _, row in aggregate.iterrows():
        model_id = str(row["model_id"])
        runtime_id = str(row["runtime_id"])
        precision = str(row["precision"])
        base_mask = (
            (aggregate["model_id"].astype(str) == model_id)
            & (aggregate["runtime_id"].astype(str) == baseline_runtime)
            & (aggregate["precision"].astype(str) == baseline_precision)
        )
        base = aggregate[base_mask]
        base_row = base.iloc[0] if not base.empty else None

        compat_row = lookup_condition(compat_norm, model_id, runtime_id, precision)
        compat_base = lookup_condition(compat_norm, model_id, baseline_runtime, baseline_precision)

        perf_ret = _safe_div(row.get(main_metric, np.nan), base_row.get(main_metric, np.nan) if base_row is not None else np.nan, eps=eps)
        brs = _safe_div(row.get("boundary_recall", np.nan), base_row.get("boundary_recall", np.nan) if base_row is not None else np.nan, eps=eps)
        srs = _safe_div(row.get("small_region_recall", row.get("small_lesion_recall", np.nan)), base_row.get("small_region_recall", base_row.get("small_lesion_recall", np.nan)) if base_row is not None else np.nan, eps=eps)

        e = compat_row.get("efficiency_inv_latency", np.nan) if compat_row is not None else np.nan
        e_ref = compat_base.get("efficiency_inv_latency", np.nan) if compat_base is not None else np.nan
        rtg = _runtime_transfer_gap(e_ref, e, eps=eps)
        runtime_retention = _runtime_retention_from_rtg(rtg)
        pcs = compat_row.get("pcs", np.nan) if compat_row is not None else np.nan

        parts = {
            "performance_retention": perf_ret,
            "boundary_retention_brs": brs,
            "small_region_retention_srs": srs,
            "small_damage_region_retention_srs": srs,
            "small_lesion_retention_srs": srs,
            "runtime_transfer_gap_rtg": rtg,
            "runtime_retention": runtime_retention,
            "precision_compatibility_pcs": pcs,
        }
        hpls_terms = {
            "performance_retention": perf_ret,
            "boundary_retention": brs,
            "small_region_retention": srs,
            "small_damage_region_retention": srs,
            "small_lesion_retention": srs,
            "runtime_retention": runtime_retention,
            "precision_compatibility": pcs,
        }
        if any(pd.isna(v) for v in hpls_terms.values()):
            hpls_score = float("nan")
        else:
            hpls_score = float(sum(hpls_weights[k] * hpls_terms[k] for k in hpls_weights))

        out = {
            "domain_context": claims["domain_context"],
            "claim_scope": claims["claim_scope"],
            "clinical_claim_allowed": claims["clinical_claim_allowed"],
            "clinical_claim_status": claims["clinical_claim_status"],
            "model_id": model_id,
            "runtime_id": runtime_id,
            "precision": precision,
            "baseline_runtime_id": baseline_runtime,
            "baseline_precision": baseline_precision,
            "n_ok": row.get("n_ok", np.nan),
            "n_boundary_valid": row.get("n_boundary_valid", np.nan),
            "empty_gt_boundary_count": row.get("empty_gt_boundary_count", np.nan),
            "boundary_fp_empty_gt_count": row.get("boundary_fp_empty_gt_count", np.nan),
            "n_small_region_valid": row.get("n_small_region_valid", np.nan),
            "small_region_gt_components_total": row.get("small_region_gt_components_total", np.nan),
            "small_region_gt_px_total": row.get("small_region_gt_px_total", np.nan),
            "n_small_lesion_valid": row.get("n_small_region_valid", row.get("n_small_lesion_valid", np.nan)),
            "small_lesion_gt_components_total": row.get("small_region_gt_components_total", row.get("small_lesion_gt_components_total", np.nan)),
            "small_lesion_gt_px_total": row.get("small_region_gt_px_total", row.get("small_lesion_gt_px_total", np.nan)),
            main_metric: row.get(main_metric, np.nan),
            "dice": row.get("dice", np.nan),
            "iou": row.get("iou", np.nan),
            "boundary_recall": row.get("boundary_recall", np.nan),
            "boundary_iou": row.get("boundary_iou", np.nan),
            "small_region_recall": row.get("small_region_recall", row.get("small_lesion_recall", np.nan)),
            "small_lesion_recall": row.get("small_lesion_recall", row.get("small_region_recall", np.nan)),
            **parts,
            "hpls_score": hpls_score,
        }
        if compat_row is not None:
            compat_cols = [
                "latency_effective_ms", "latency_ms_p50", "latency_ms_p90", "latency_ms_p95", "latency_ms_p99",
                "domain_context", "claim_scope", "clinical_claim_allowed", "clinical_claim_status",
                "run_status", "run_status_normalized", "latency_ms_min", "latency_ms_max", "compile_ms", "pcs",
                "fallback_ratio", "fallback_ratio_effective", "fallback_ratio_source", "fallback_severity",
                "provider_audit_status", "runtime_evidence_grade", "backend_claim_status", "backend_claim_allowed",
                "hardware_claim_allowed", "requested_provider", "requested_provider_canonical",
                "provider_node_counts", "unique_node_provider_counts", "provider_node_event_counts", "node_event_provider_counts",
                "provider_event_counts", "requested_provider_fraction", "requested_provider_unique_node_fraction", "requested_provider_event_fraction",
                "fallback_node_counts", "fallback_unique_node_counts", "fallback_event_counts", "fallback_node_count", "fallback_event_count",
                "total_profiled_node_count", "total_unique_profiled_node_count", "total_profiled_node_event_count", "profile_event_filter",
                "int8_audit_path", "int8_audit_status", "strict_int8_graph_pass", "strict_int8_graph_reason",
                "conv_total", "qlinear_conv_count", "qdq_wrapped_conv_count", "quantized_conv_like_count",
                "conv_quantized_fraction", "fp32_conv_fraction_estimated", "int8_graph_claim_allowed",
                "requested_device", "available_devices", "actual_compiled_device", "device_claim_status", "backend_specific_profile_status",
                "xla_profile_path", "backend_profile_path", "performance_hint",
                "unsupported_ops", "notes",
            ]
            for col in compat_cols:
                if col in compat_row:
                    out[col] = compat_row[col]
        rows.append(out)
    return pd.DataFrame(rows)
