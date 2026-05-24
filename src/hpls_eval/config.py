from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


DEFAULT_DOMAIN_CONTEXT = "generic_visual_anomaly"
DEFAULT_CLAIM_SCOPE = "evaluation_framework"


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    return default


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    ensure_project_claim_defaults(cfg)
    return cfg, config_path


def project_claim_defaults(cfg_or_project: dict[str, Any] | None) -> dict[str, Any]:
    """Return conservative domain/claim metadata for matrix/report outputs.

    The framework is intentionally non-clinical. `clinical_claim_allowed` is
    forced to False even if a user-supplied config tries to set it otherwise;
    future clinical validation would require separate medical data, protocol
    review, and domain-specific evidence outside this prototype.
    """
    cfg_or_project = cfg_or_project or {}
    if "project" in cfg_or_project and isinstance(cfg_or_project.get("project"), dict):
        project = cfg_or_project.get("project", {}) or {}
    else:
        project = cfg_or_project
    domain_context = str(project.get("domain_context") or DEFAULT_DOMAIN_CONTEXT).strip() or DEFAULT_DOMAIN_CONTEXT
    claim_scope = str(project.get("claim_scope") or DEFAULT_CLAIM_SCOPE).strip() or DEFAULT_CLAIM_SCOPE
    # Hard gate: this prototype does not allow clinical claims.
    clinical_claim_allowed = False
    requested_clinical_claim = _as_bool(project.get("clinical_claim_allowed"), default=False)
    return {
        "domain_context": domain_context,
        "claim_scope": claim_scope,
        "clinical_claim_allowed": clinical_claim_allowed,
        "clinical_claim_requested_in_config": requested_clinical_claim,
        "clinical_claim_status": "clinical_claim_forbidden_nonclinical_research_prototype",
    }


def ensure_project_claim_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    project = cfg.setdefault("project", {})
    claims = project_claim_defaults(project)
    project.setdefault("domain_context", claims["domain_context"])
    project.setdefault("claim_scope", claims["claim_scope"])
    # Always write a conservative value into config snapshots.
    project["clinical_claim_allowed"] = False
    project.setdefault("clinical_claim_status", claims["clinical_claim_status"])
    return cfg


def resolve_from_config(config_path: Path, maybe_path: str | Path | None) -> Path | None:
    if maybe_path is None or str(maybe_path).strip() == "":
        return None
    p = Path(str(maybe_path)).expanduser()
    if p.is_absolute():
        return p
    return (config_path.parent / p).resolve()


def normalize_weights(weights: dict[str, float], *, name: str, strict: bool = False) -> dict[str, float]:
    clean = {k: float(v) for k, v in weights.items()}
    total = sum(clean.values())
    if total <= 0:
        raise ValueError(f"{name} weights must sum to a positive value.")
    if strict and abs(total - 1.0) > 1e-6:
        raise ValueError(f"{name} weights must sum to 1.0; got {total}.")
    return {k: v / total for k, v in clean.items()}
