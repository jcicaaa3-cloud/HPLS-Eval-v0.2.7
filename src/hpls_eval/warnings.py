from __future__ import annotations

# Compatibility shim for older imports. Keep the public helper names stable.
from .warning_utils import (
    condition_key_from_warning,
    dedupe_warnings,
    dedupe_warnings_by_code_and_condition,
    extract_warning_code,
    is_runtime_evidence_warning,
    warning_fields,
    warning_identity,
)

__all__ = [
    "condition_key_from_warning",
    "dedupe_warnings",
    "dedupe_warnings_by_code_and_condition",
    "extract_warning_code",
    "is_runtime_evidence_warning",
    "warning_fields",
    "warning_identity",
]
