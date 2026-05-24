from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

from .compatibility import normalize_compatibility
from .warning_utils import dedupe_warnings
from .config import resolve_from_config, project_claim_defaults
from .io import read_dataset_manifest, read_compatibility_log, resolve_gt_path, resolve_prediction_path, condition_matrix


def _looks_accelerated_runtime(runtime_id: str, provider: str = "") -> bool:
    token = f"{runtime_id} {provider}".lower()
    return any(x in token for x in ["cuda", "tensorrt", "trt", "openvino", "npu", "tpu", "xla", "gpu", "rocm", "directml", "dml"])


def _row_prefix(row: pd.Series) -> str:
    return f"model={row.get('model_id', '')} runtime={row.get('runtime_id', '')} precision={row.get('precision', '')}"


def _has_p99(row: pd.Series, columns: set[str]) -> bool:
    for col in ("latency_ms_p99", "latency_p99_ms"):
        if col in columns and pd.notna(row.get(col)) and str(row.get(col)).strip() != "":
            return True
    return False


def _compat_evidence_warnings(cfg: dict[str, Any], compat_df: pd.DataFrame, compat_base_dir: Path | None) -> list[str]:
    warnings: list[str] = []
    if compat_df.empty:
        return warnings

    runtime_audit = cfg.get("runtime_audit", {}) or {}
    threshold = float(runtime_audit.get("requested_provider_fraction_warning_threshold", 0.95))
    require_profile = bool(runtime_audit.get("require_provider_profile_for_accelerators", False))
    require_p99 = bool(runtime_audit.get("require_p99_latency", True))

    columns = set(compat_df.columns)
    for _, row in compat_df.iterrows():
        prefix = _row_prefix(row)
        runtime_id = str(row.get("runtime_id", ""))
        requested_provider = str(row.get("requested_provider", row.get("target_provider", "")))
        accelerated = _looks_accelerated_runtime(runtime_id, requested_provider)
        if accelerated:
            profile = row.get("provider_profile_path", "")
            if pd.isna(profile) or str(profile).strip() == "":
                level = "ERROR" if require_profile else "WARN"
                warnings.append(f"PROVIDER_PROFILE_MISSING_{level} {prefix}: accelerated runtime claim is log-only; requested provider fraction cannot be verified")
            else:
                p = Path(str(profile)).expanduser()
                if not p.is_absolute() and compat_base_dir is not None:
                    p = compat_base_dir / p
                if not p.exists():
                    warnings.append(f"PROVIDER_PROFILE_NOT_FOUND {prefix}: {p}")
            if not requested_provider.strip():
                warnings.append(f"REQUESTED_PROVIDER_MISSING {prefix}: accelerated runtime row needs requested_provider")
        if require_p99 and not _has_p99(row, columns):
            warnings.append(f"P99_LATENCY_MISSING {prefix}: tail latency evidence is incomplete")
        requested_device = str(row.get("requested_device", "") if not pd.isna(row.get("requested_device", "")) else "")
        actual_device = str(row.get("actual_compiled_device", "") if not pd.isna(row.get("actual_compiled_device", "")) else "")
        if requested_device.upper() == "AUTO" and not actual_device.strip():
            warnings.append(f"OPENVINO_AUTO_DEVICE_UNRESOLVED {prefix}: requested_device=AUTO but actual_compiled_device is not recorded")
        if "openvino" in runtime_id.lower() or "npu" in runtime_id.lower():
            if not actual_device.strip():
                warnings.append(f"ACTUAL_DEVICE_MISSING {prefix}: OpenVINO/NPU result needs actual_compiled_device")
            if "npu" in runtime_id.lower() and actual_device.strip() and "NPU" not in actual_device.upper():
                warnings.append(f"NPU_DEVICE_MISMATCH {prefix}: runtime_id implies NPU but actual_compiled_device={actual_device}")
        if str(row.get("precision", "")).upper() == "INT8":
            int8_path = row.get("int8_audit_path", "")
            require_int8 = bool((cfg.get("int8_audit", {}) or {}).get("require_for_int8", False))
            if pd.isna(int8_path) or str(int8_path).strip() == "":
                level = "ERROR" if require_int8 else "WARN"
                warnings.append(f"INT8_AUDIT_PATH_MISSING_{level} {prefix}: INT8 graph audit is not integrated into the evaluation matrix")
        if "tpu" in runtime_id.lower() or "xla" in runtime_id.lower():
            xla_path = row.get("xla_profile_path", row.get("backend_profile_path", ""))
            if pd.isna(xla_path) or str(xla_path).strip() == "":
                warnings.append(f"TPU_XLA_BACKEND_PROFILE_MISSING {prefix}: ORT provider profile is not sufficient for TPU/XLA verification; attach backend-specific profiling evidence")

    # Parse provided profiles during validation so a bad backend claim is caught
    # before report generation. This is the key anti-overclaim gate.
    try:
        compat_norm = normalize_compatibility(
            compat_df,
            cfg.get("pcs_weights", {}),
            evidence_base_dir=compat_base_dir,
            runtime_audit=runtime_audit,
            int8_audit=cfg.get("int8_audit", {}),
            domain_claim=cfg.get("project", {}),
        )
    except Exception as e:
        warnings.append(f"COMPATIBILITY_NORMALIZATION_FAILED {type(e).__name__}: {e}")
        return warnings

    for _, row in compat_norm.iterrows():
        prefix = _row_prefix(row)
        backend_status = str(row.get("backend_claim_status", ""))
        grade = str(row.get("runtime_evidence_grade", ""))
        accelerated = _looks_accelerated_runtime(str(row.get("runtime_id", "")), str(row.get("requested_provider_canonical", row.get("requested_provider", ""))))
        hardware_allowed = row.get("hardware_claim_allowed")
        if accelerated and pd.notna(hardware_allowed) and bool(hardware_allowed) is False:
            warnings.append(f"HARDWARE_CLAIM_NOT_ALLOWED {prefix}: backend_status={backend_status}, evidence_grade={grade}")
        frac = pd.to_numeric(pd.Series([row.get("requested_provider_fraction")]), errors="coerce").iloc[0]
        if pd.notna(frac) and frac < threshold:
            warnings.append(f"REQUESTED_PROVIDER_FRACTION_LOW {prefix}: requested_provider_fraction={frac:.4f} < threshold={threshold:.4f}; fallback_node_counts={row.get('fallback_node_counts', '')}")
        status = str(row.get("provider_audit_status", ""))
        if status not in {"profile_verified", "profile_not_provided"}:
            warnings.append(f"PROVIDER_PROFILE_UNUSABLE {prefix}: provider_audit_status={status}")
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


def validate_inputs(cfg: dict[str, Any], config_path: Path) -> tuple[list[str], dict[str, Any]]:
    warnings: list[str] = []
    paths = cfg.get("paths", {})
    manifest_path = resolve_from_config(config_path, paths.get("dataset_manifest"))
    compat_path = resolve_from_config(config_path, paths.get("compatibility_log"))
    if manifest_path is None:
        raise ValueError("paths.dataset_manifest is required")
    samples = read_dataset_manifest(manifest_path)
    conditions = condition_matrix(cfg)

    missing_gt = []
    for _, row in samples.iterrows():
        gt = resolve_gt_path(config_path, row["gt_mask"])
        if not gt.exists():
            missing_gt.append(str(gt))
    if missing_gt:
        warnings.append(f"MISSING_GT_MASKS: count={len(missing_gt)}")
        warnings.extend(f"  GT_NOT_FOUND {p}" for p in missing_gt[:50])

    missing_predictions = 0
    for cond in conditions:
        for _, row in samples.iterrows():
            pred_path = resolve_prediction_path(cfg, config_path, sample_id=row["sample_id"], **cond)
            if not pred_path.exists():
                missing_predictions += 1
                if missing_predictions <= 50:
                    warnings.append(f"PRED_NOT_FOUND {pred_path}")
    if missing_predictions:
        warnings.append(f"TOTAL_MISSING_PREDICTIONS: count={missing_predictions}")

    compat_df = read_compatibility_log(compat_path)
    warnings.extend(_compat_evidence_warnings(cfg, compat_df, compat_path.parent if compat_path else None))
    expected = {(c["model_id"], c["runtime_id"], c["precision"]) for c in conditions}
    observed = set()
    if not compat_df.empty:
        for _, row in compat_df.iterrows():
            observed.add((str(row["model_id"]), str(row["runtime_id"]), str(row["precision"])))
    missing_compat = expected - observed
    if missing_compat:
        warnings.append(f"MISSING_COMPATIBILITY_ROWS: count={len(missing_compat)}")
        for m, r, p in sorted(missing_compat)[:50]:
            warnings.append(f"  COMPAT_NOT_FOUND model={m} runtime={r} precision={p}")

    baseline = cfg.get("baseline", {})
    baseline_runtime = baseline.get("runtime_id")
    baseline_precision = baseline.get("precision", "FP32")
    if baseline_runtime is None:
        raise ValueError("baseline.runtime_id is required")
    for model in [m["model_id"] if isinstance(m, dict) else str(m) for m in cfg.get("models", [])]:
        if (model, baseline_runtime, baseline_precision) not in expected:
            warnings.append(f"BASELINE_CONDITION_NOT_PRESENT model={model} runtime={baseline_runtime} precision={baseline_precision}: baseline condition is not present in the configured matrix")

    summary = {
        "manifest_path": str(manifest_path),
        "compatibility_path": str(compat_path) if compat_path else None,
        "num_samples": int(len(samples)),
        "num_conditions": int(len(conditions)),
        "num_missing_predictions": int(missing_predictions),
        "num_missing_compatibility_rows": int(len(missing_compat)),
        **project_claim_defaults(cfg),
    }
    return dedupe_warnings(warnings), summary
