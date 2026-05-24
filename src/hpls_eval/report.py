from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

from .config import project_claim_defaults
from .warning_utils import (
    condition_key_from_warning,
    dedupe_warnings,
    extract_warning_code,
    is_runtime_evidence_warning,
)


def _md_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or df.empty:
        return "_No rows available._"
    shown = df.head(max_rows).copy()
    return shown.to_markdown(index=False)


def _value_counts_md(df: pd.DataFrame, col: str) -> str:
    if df is None or df.empty or col not in df.columns:
        return "_No rows available._"
    counts = df[col].astype(str).value_counts(dropna=False).reset_index()
    counts.columns = [col, "count"]
    return _md_table(counts, max_rows=30)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _show_medical_alias_columns(cfg: dict[str, Any], df: pd.DataFrame | None = None) -> bool:
    value = cfg.get("reporting", {}).get("show_medical_alias_columns", False)
    if isinstance(value, str) and value.strip().lower() == "auto":
        if df is None or df.empty or "domain_context" not in df.columns:
            return False
        contexts = {str(x).strip() for x in df["domain_context"].dropna().unique()}
        return bool(contexts) and contexts <= {"medical_lesion_target"}
    return _as_bool(value, default=False)


def _filter_medical_alias_columns(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if _show_medical_alias_columns(cfg, df):
        return df
    keep = [c for c in df.columns if "small_lesion" not in str(c)]
    return df[keep].copy()


def _select_existing(df: pd.DataFrame | None, cols: list[str], cfg: dict[str, Any]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    existing: list[str] = []
    for c in cols:
        if c in df.columns and c not in existing:
            existing.append(c)
    if not existing:
        return pd.DataFrame()
    return _filter_medical_alias_columns(df[existing].copy(), cfg)




def _int8_only(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "precision" not in df.columns:
        return df.copy()
    return df[df["precision"].astype(str).str.upper() == "INT8"].copy()


def _append_table_section(
    lines: list[str],
    title: str,
    df: pd.DataFrame | None,
    cols: list[str],
    cfg: dict[str, Any],
    *,
    max_rows: int = 80,
    note: str | None = None,
) -> None:
    lines.append(f"### {title}")
    lines.append("")
    if note:
        lines.append(note)
        lines.append("")
    lines.append(_md_table(_select_existing(df, cols, cfg), max_rows=max_rows))
    lines.append("")


def _domain_context_distribution(condition_summary: pd.DataFrame, runtime_evidence: pd.DataFrame | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source, df in (("condition_summary", condition_summary), ("runtime_evidence_matrix", runtime_evidence)):
        if df is None or df.empty or "domain_context" not in df.columns:
            continue
        counts = df["domain_context"].fillna("UNKNOWN").astype(str).value_counts(dropna=False)
        for domain_context, count in counts.items():
            rows.append({"source": source, "domain_context": domain_context, "row_count": int(count)})
    return pd.DataFrame(rows)


def _warning_code_summary(warnings: list[str]) -> pd.DataFrame:
    rows: dict[str, int] = {}
    for warning in warnings:
        code = extract_warning_code(warning)
        rows[code] = rows.get(code, 0) + 1
    return pd.DataFrame([{"warning_code": k, "count": v} for k, v in sorted(rows.items())])


def _group_related_warnings(warnings: list[str]) -> tuple[dict[tuple[str, str, str], list[str]], list[str]]:
    grouped: dict[tuple[str, str, str], list[str]] = {}
    other: list[str] = []
    for warning in warnings:
        key = condition_key_from_warning(warning)
        if key is not None and is_runtime_evidence_warning(warning):
            grouped.setdefault(key, []).append(warning)
        else:
            other.append(warning)
    return grouped, other


def _warning_group_heading(key: tuple[str, str, str]) -> str:
    model, runtime, precision = key
    return f"Runtime evidence issue: model={model}, runtime={runtime}, precision={precision}"


def _warning_detail(warning: str) -> str:
    code = extract_warning_code(warning)
    text = str(warning).strip()
    # Runtime evidence warnings usually have the form
    # CODE model=m runtime=r precision=p: human-readable detail. In grouped
    # reports the condition is already in the heading, so keep only the detail.
    if ":" in text:
        detail = text.split(":", 1)[1].strip()
    elif text.startswith(code):
        detail = text[len(code):].strip()
    else:
        detail = text
    return f"{code}: {detail}" if detail else code


CLAIM_GATE_COLS = [
    "domain_context", "claim_scope", "clinical_claim_allowed", "clinical_claim_status",
    "model_id", "runtime_id", "precision", "run_status", "run_status_normalized",
    "backend_claim_status", "backend_claim_allowed", "hardware_claim_allowed",
    "runtime_evidence_grade", "device_claim_status", "int8_graph_claim_allowed",
]

PROVIDER_PLACEMENT_COLS = [
    "model_id", "runtime_id", "precision", "requested_provider", "requested_provider_canonical",
    "provider_audit_status", "requested_provider_unique_node_fraction", "requested_provider_event_fraction",
    "unique_node_provider_counts", "node_event_provider_counts",
    "total_unique_profiled_node_count", "total_profiled_node_event_count", "profile_event_filter",
]

FALLBACK_COLS = [
    "model_id", "runtime_id", "precision", "fallback_ratio_effective", "fallback_ratio_source",
    "fallback_severity", "fallback_unique_node_counts", "fallback_event_counts",
    "fallback_node_count", "fallback_event_count", "backend_claim_status", "backend_claim_allowed",
    "hardware_claim_allowed", "runtime_evidence_grade",
]

DEVICE_EVIDENCE_COLS = [
    "model_id", "runtime_id", "precision", "requested_device", "available_devices",
    "actual_compiled_device", "device_claim_status", "performance_hint",
    "backend_specific_profile_status", "xla_profile_path", "backend_profile_path",
]

# Backward-compatible alias for callers that imported the old constant. Reports use
# the narrower provider/fallback/device sections above.
PROVIDER_AUDIT_COLS = PROVIDER_PLACEMENT_COLS + FALLBACK_COLS
DEVICE_AUDIT_COLS = DEVICE_EVIDENCE_COLS

LATENCY_COLS = [
    "model_id", "runtime_id", "precision", "latency_effective_ms", "latency_ms_p50",
    "latency_ms_p90", "latency_ms_p95", "latency_ms_p99", "latency_ms_min",
    "latency_ms_max", "compile_ms", "runtime_transfer_gap_rtg", "runtime_retention",
]

INT8_AUDIT_COLS = [
    "model_id", "runtime_id", "precision", "int8_audit_path", "int8_audit_status",
    "strict_int8_graph_pass", "strict_int8_graph_reason", "conv_total",
    "qlinear_conv_count", "conv_integer_count", "qoperator_conv_count", "qdq_wrapped_conv_count",
    "quantized_conv_like_count", "conv_quantized_fraction", "fp32_conv_count_estimated",
    "fp32_conv_fraction_estimated", "int8_graph_claim_allowed",
]

SEGMENTATION_COLS = [
    "model_id", "runtime_id", "precision", "n_ok", "dice", "iou", "recall",
    "boundary_recall", "boundary_iou", "n_boundary_valid", "empty_gt_boundary_count",
    "boundary_fp_empty_gt_count", "small_region_recall", "n_small_region_valid",
    "small_region_gt_components_total", "small_region_gt_px_total", "performance_retention",
    "boundary_retention_brs", "small_region_retention_srs", "precision_compatibility_pcs",
    "hpls_score",
]

RETENTION_COLS = [
    "model_id", "runtime_id", "precision", "performance_retention",
    "boundary_retention_brs", "small_region_retention_srs",
    "small_damage_region_retention_srs", "runtime_transfer_gap_rtg",
    "runtime_retention", "precision_compatibility_pcs", "hpls_score",
]

AGGREGATE_COLS = [
    "model_id", "runtime_id", "precision", "n_ok", "dice", "iou", "recall",
    "boundary_recall", "boundary_iou", "n_boundary_valid", "empty_gt_boundary_count",
    "boundary_fp_empty_gt_count", "small_region_recall", "n_small_region_valid",
    "small_region_gt_components_total", "small_region_gt_px_total",
]


def write_report(
    path: Path,
    cfg: dict[str, Any],
    validation_summary: dict[str, Any],
    warnings: list[str],
    condition_summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    per_sample: pd.DataFrame,
    runtime_evidence: pd.DataFrame | None = None,
) -> None:
    project = cfg.get("project", {})
    claims = project_claim_defaults(cfg)
    title = project.get("name", "HPLS-Eval")
    show_med_alias = _show_medical_alias_columns(cfg, condition_summary)
    group_related = _as_bool(cfg.get("reporting", {}).get("group_related_warnings", True), default=True)
    display_warnings = dedupe_warnings(warnings)

    lines: list[str] = []
    lines.append(f"# {title} Evaluation Report")
    lines.append("")
    lines.append("Generated from the configured manifest, prediction pattern, compatibility log, and optional runtime evidence. Missing measurements remain missing; the report does not fabricate experiment results.")
    lines.append("")
    lines.append("## Scope and Claim Boundary")
    lines.append("")
    lines.append(f"- **project-level domain_context**: `{claims['domain_context']}`")
    lines.append(f"- **project-level claim_scope**: `{claims['claim_scope']}`")
    lines.append(f"- **clinical_claim_allowed**: `{claims['clinical_claim_allowed']}`")
    lines.append(f"- **clinical_claim_status**: `{claims['clinical_claim_status']}`")
    lines.append(f"- **show_medical_alias_columns**: `{show_med_alias}`")
    lines.append(f"- **group_related_warnings**: `{group_related}`")
    lines.append("")
    lines.append("The report is limited to deployment-oriented evaluation: segmentation retention, boundary/small-region behavior, low-precision graph evidence, runtime/provider evidence, fallback visibility, and latency stability. It is not a clinical validation report and does not establish diagnostic readiness.")
    lines.append("")
    lines.append("| Allowed interpretation | Not allowed interpretation |")
    lines.append("|---|---|")
    lines.append("| Road-damage segmentation can be used as a motivating pilot domain. | Road-damage performance implies medical-lesion performance. |")
    lines.append("| Shared evaluation problems can motivate a medical-lesion-oriented follow-up. | A road-damage model supports medical conclusions. |")
    lines.append("| Runtime and low-precision instability should be measured before deployment-oriented claims. | INT8, TensorRT, CUDA, NPU, or TPU compatibility implies medical safety. |")
    lines.append("| Evaluation concepts may be reused after medical-domain validation. | Medical deployment is validated without medical data and domain-specific review. |")
    lines.append("")
    lines.append("## Domain Context Distribution")
    lines.append("")
    lines.append("Project metadata is a default label. Row-level `domain_context` values are counted separately so mixed-domain matrices are visible.")
    lines.append("")
    lines.append(_md_table(_domain_context_distribution(condition_summary, runtime_evidence), max_rows=30))
    lines.append("")
    lines.append("## Road-Damage Pilot Motivation")
    lines.append("")
    lines.append("Road damage remains the pilot case because it exposes the evaluation failure modes this harness is built to measure: boundary erosion, small-region misses, sparse masks, low-contrast confusion, low-precision sensitivity, fallback, and tail latency. It is not presented as a clinical target.")
    lines.append("")
    lines.append("도로 손상 segmentation은 최종 의료 응용을 직접 주장하기 위한 대상이 아니라, 작은 불규칙 영역과 경계 손실 문제가 실제로 관찰된 선행 도메인으로 사용한다.")
    lines.append("")
    lines.append("## Medical Lesion Extension Rationale")
    lines.append("")
    lines.append("The follow-up scope narrows toward medical-lesion-oriented evaluation because lesion masks can also be small, sparse, low-contrast, boundary-sensitive, and annotation-uncertain. This is a rationale for evaluation design, not evidence that a road-damage model transfers to medical data.")
    lines.append("")
    lines.append("## Terminology Display Policy")
    lines.append("")
    lines.append("CSV artifacts retain `small_lesion_*` aliases for future medical-extension compatibility. Markdown reports hide those aliases by default and prioritize `small_region_*` terminology for road-damage or generic visual-anomaly rows. For presentations or PDF/Word conversion, prefer the split Markdown report tables instead of raw CSV screenshots when the active domain is `road_damage_pilot`.")
    lines.append("")
    lines.append("## Runtime Evidence Limitations")
    lines.append("")
    lines.append("A requested backend is not proof of executed backend placement. Inspect provider fractions, unique-node counts, node-event counts, fallback counts, device evidence, and backend-specific profiles before making runtime or hardware claims. TPU/XLA evidence requires backend-specific profiling; ORT provider-profile fields alone are not sufficient for TPU execution verification.")
    lines.append("")
    lines.append("## Low-Precision Claim Policy")
    lines.append("")
    lines.append("INT8 conversion success, INT8 graph coverage, provider execution, and segmentation quality retention are separate evidence layers. A strong INT8 statement requires integrated graph-audit evidence and runtime execution evidence; it still must not be interpreted as clinical safety.")
    lines.append("")
    lines.append("## Clinical Claim Disclaimer")
    lines.append("")
    lines.append("This report does not provide clinical validation. Medical-lesion terminology, when present, refers to a future evaluation target and not to diagnostic readiness. Hardware or INT8 compatibility results must not be interpreted as evidence of medical safety or clinical deployment suitability.")
    lines.append("")

    lines.append("## Validation summary")
    lines.append("")
    for k, v in validation_summary.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")

    lines.append("## Runtime evidence audit")
    lines.append("")
    lines.append("This section is split into smaller tables. The full schema remains in `runtime_evidence_matrix.csv`.")
    lines.append("")
    if runtime_evidence is not None and not runtime_evidence.empty:
        lines.append("### Evidence grades")
        lines.append("")
        lines.append(_value_counts_md(runtime_evidence, "runtime_evidence_grade"))
        lines.append("")
        lines.append("### Backend claim status")
        lines.append("")
        lines.append(_value_counts_md(runtime_evidence, "backend_claim_status"))
        lines.append("")
        _append_table_section(lines, "Claim gate table", runtime_evidence, CLAIM_GATE_COLS, cfg)
        _append_table_section(lines, "Provider placement table", runtime_evidence, PROVIDER_PLACEMENT_COLS, cfg)
        _append_table_section(lines, "Fallback table", runtime_evidence, FALLBACK_COLS, cfg)
        _append_table_section(lines, "Device evidence table", runtime_evidence, DEVICE_EVIDENCE_COLS, cfg)
        _append_table_section(lines, "Latency table", runtime_evidence, LATENCY_COLS, cfg)
        _append_table_section(lines, "INT8 graph audit table", _int8_only(runtime_evidence), INT8_AUDIT_COLS, cfg)
    else:
        lines.append("_No runtime evidence rows available._")
        lines.append("")

    lines.append("## Condition summary")
    lines.append("")
    lines.append("The full condition matrix is written to `condition_summary.csv`. The report separates it into readable reviewer-facing tables.")
    lines.append("")
    _append_table_section(lines, "Claim gate table", condition_summary, CLAIM_GATE_COLS, cfg)
    _append_table_section(lines, "Segmentation summary table", condition_summary, SEGMENTATION_COLS, cfg)
    _append_table_section(lines, "Retention and HPLS table", condition_summary, RETENTION_COLS, cfg)
    _append_table_section(lines, "Latency and runtime-transfer table", condition_summary, LATENCY_COLS, cfg)
    _append_table_section(lines, "Provider placement table", condition_summary, PROVIDER_PLACEMENT_COLS, cfg)
    _append_table_section(lines, "Fallback table", condition_summary, FALLBACK_COLS, cfg)
    _append_table_section(lines, "Device evidence table", condition_summary, DEVICE_EVIDENCE_COLS, cfg)
    _append_table_section(lines, "INT8 graph summary table", _int8_only(condition_summary), INT8_AUDIT_COLS, cfg)

    lines.append("## Aggregate segmentation metrics")
    lines.append("")
    lines.append(_md_table(_select_existing(aggregate, AGGREGATE_COLS, cfg), max_rows=100))
    lines.append("")

    lines.append("## Warnings and missing evidence")
    lines.append("")
    lines.append("Warnings are deduplicated by `warning_code + model/runtime/precision` when a condition identity is present. Sample-level warnings remain distinct. Warning-code summaries normalize prose warnings such as `Missing compatibility rows` into stable all-caps codes.")
    lines.append("")
    if display_warnings:
        lines.append("### Warning code summary")
        lines.append("")
        lines.append(_md_table(_warning_code_summary(display_warnings), max_rows=50))
        lines.append("")
        if group_related:
            grouped, other = _group_related_warnings(display_warnings)
            lines.append("### Grouped runtime evidence issues")
            lines.append("")
            if grouped:
                for key in sorted(grouped):
                    lines.append(f"#### {_warning_group_heading(key)}")
                    lines.append("")
                    for w in grouped[key]:
                        lines.append(f"- {_warning_detail(w)}")
                    lines.append("")
            else:
                lines.append("_No grouped runtime evidence issues._")
                lines.append("")
            lines.append("### Other warnings")
            lines.append("")
            if other:
                for w in other[:200]:
                    lines.append(f"- {w}")
                if len(other) > 200:
                    lines.append(f"- ... {len(other) - 200} more non-grouped warnings omitted from report; see validation_warnings.log")
            else:
                lines.append("_No other warnings._")
            lines.append("")
            lines.append("The full deduplicated warning list remains available in `validation_warnings.log`.")
        else:
            lines.append("### Deduplicated warning list")
            lines.append("")
            for w in display_warnings[:300]:
                lines.append(f"- {w}")
            if len(display_warnings) > 300:
                lines.append(f"- ... {len(display_warnings) - 300} more warnings omitted from report; see validation_warnings.log")
    else:
        lines.append("_No warnings._")
    lines.append("")

    if cfg.get("reporting", {}).get("include_per_sample_table_in_report", False):
        lines.append("## Per-sample metrics")
        lines.append("")
        lines.append(_md_table(_filter_medical_alias_columns(per_sample, cfg), max_rows=200))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
