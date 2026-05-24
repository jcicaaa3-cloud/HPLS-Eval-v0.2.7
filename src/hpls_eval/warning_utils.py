from __future__ import annotations

import re

_FIELD_RE = re.compile(r"\b(?P<key>model|runtime|precision|sample)=([^\s:]+)")
_NON_CODE_PREFIXES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^Missing compatibility rows\b", re.IGNORECASE), "MISSING_COMPATIBILITY_ROWS"),
    (re.compile(r"^Total missing prediction files\b", re.IGNORECASE), "TOTAL_MISSING_PREDICTIONS"),
    (re.compile(r"^Missing GT masks\b", re.IGNORECASE), "MISSING_GT_MASKS"),
    (re.compile(r"^Baseline condition not present in matrix\b", re.IGNORECASE), "BASELINE_CONDITION_NOT_PRESENT"),
]
_RUNTIME_EVIDENCE_CODES = {
    "HARDWARE_CLAIM_NOT_ALLOWED",
    "RUNTIME_EVIDENCE_WEAK",
    "PROVIDER_FALLBACK_DETECTED",
    "REQUESTED_PROVIDER_FRACTION_LOW",
    "PROVIDER_PROFILE_MISSING_WARN",
    "PROVIDER_PROFILE_MISSING_ERROR",
    "PROVIDER_PROFILE_NOT_FOUND",
    "REQUESTED_PROVIDER_MISSING",
    "PROVIDER_PROFILE_UNUSABLE",
    "DEVICE_EVIDENCE_WEAK",
    "ACTUAL_DEVICE_MISSING",
    "OPENVINO_AUTO_DEVICE_UNRESOLVED",
    "NPU_DEVICE_MISMATCH",
    "TPU_XLA_BACKEND_PROFILE_MISSING",
    "TPU_XLA_BACKEND_EVIDENCE_WEAK",
    "P99_LATENCY_MISSING",
    "INT8_AUDIT_PATH_MISSING_WARN",
    "INT8_AUDIT_PATH_MISSING_ERROR",
    "INT8_AUDIT_NOT_INTEGRATED",
    "INT8_GRAPH_CLAIM_NOT_ALLOWED",
}


def extract_warning_code(message: str) -> str:
    """Return a reviewer-friendly warning code.

    Some human-readable validation warnings historically began with prose such as
    ``Missing compatibility rows`` or ``Total missing prediction files``. Splitting
    by the first word made the summary table show weak code names like ``Missing``
    and ``Total``. This function normalizes those prose prefixes into stable
    all-caps codes while preserving already-coded warnings.
    """
    text = str(message).strip()
    if not text:
        return "UNKNOWN"
    for pattern, code in _NON_CODE_PREFIXES:
        if pattern.search(text):
            return code
    first = text.split(None, 1)[0].strip()
    first = first.rstrip(":")
    if first.startswith("#"):
        return "UNKNOWN"
    return first or "UNKNOWN"


def warning_fields(message: str) -> dict[str, str]:
    """Extract model/runtime/precision/sample fields from a warning message."""
    text = str(message)
    return {m.group("key"): m.group(2) for m in _FIELD_RE.finditer(text)}


def warning_identity(message: str) -> tuple[str, ...]:
    """Return a stable identity for reviewer-facing warning de-duplication.

    Many runtime-evidence warnings are emitted from both validation and
    evaluation. Exact-string de-duplication misses those because the prose can be
    slightly different. When a condition identity is present, the useful review
    unit is warning_code + model_id + runtime_id + precision. If a sample id is
    present, include it so per-sample errors remain distinct. Messages without a
    condition identity fall back to normalized warning code + exact text.
    """
    text = str(message).strip()
    if not text:
        return ("literal", "")
    code = extract_warning_code(text)
    fields = warning_fields(text)
    if {"model", "runtime", "precision"}.issubset(fields):
        key = ("condition", code, fields["model"], fields["runtime"], fields["precision"])
        if "sample" in fields:
            return (*key, fields["sample"])
        return key
    return ("literal", code, text)


def is_runtime_evidence_warning(message: str) -> bool:
    return extract_warning_code(message) in _RUNTIME_EVIDENCE_CODES


def condition_key_from_warning(message: str) -> tuple[str, str, str] | None:
    fields = warning_fields(message)
    if {"model", "runtime", "precision"}.issubset(fields):
        return fields["model"], fields["runtime"], fields["precision"]
    return None


def dedupe_warnings(warnings: list[str]) -> list[str]:
    """Deduplicate warnings while preserving first-occurrence order."""
    seen: set[tuple[str, ...]] = set()
    out: list[str] = []
    for warning in warnings:
        key = warning_identity(warning)
        if key in seen:
            continue
        seen.add(key)
        out.append(warning)
    return out


def dedupe_warnings_by_code_and_condition(messages: list[str]) -> list[str]:
    """Backward-compatible alias used by validation/CLI modules."""
    return dedupe_warnings(messages)
