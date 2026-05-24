from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import hashlib
import json
import os
import platform
import sys
import pandas as pd

from .config import resolve_from_config


REQUIRED_SAMPLE_COLUMNS = {"sample_id", "gt_mask"}
REQUIRED_COMPAT_COLUMNS = {
    "model_id", "runtime_id", "precision", "conversion_success", "operator_support",
    "fallback_ratio", "build_success", "precision_support",
}


def file_sha256(path: str | Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_dataset_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    df = pd.read_csv(path, comment="#")
    missing = REQUIRED_SAMPLE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset manifest missing columns: {sorted(missing)}")
    df = df.dropna(subset=["sample_id", "gt_mask"]).copy()
    df["sample_id"] = df["sample_id"].astype(str)
    return df


def read_compatibility_log(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=sorted(REQUIRED_COMPAT_COLUMNS))
    df = pd.read_csv(path, comment="#")
    missing = REQUIRED_COMPAT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Compatibility log missing columns: {sorted(missing)}")
    return df


def resolve_gt_path(config_path: Path, gt_path: str | Path) -> Path:
    p = Path(str(gt_path)).expanduser()
    if p.is_absolute():
        return p
    return (config_path.parent / p).resolve()


def resolve_prediction_path(cfg: dict[str, Any], config_path: Path, *, model_id: str, runtime_id: str, precision: str, sample_id: str) -> Path:
    paths = cfg.get("paths", {})
    prediction_root = resolve_from_config(config_path, paths.get("prediction_root"))
    if prediction_root is None:
        raise ValueError("paths.prediction_root is required")
    pattern = paths.get("prediction_pattern")
    if not pattern:
        raise ValueError("paths.prediction_pattern is required")
    filled = str(pattern).format(
        prediction_root=str(prediction_root),
        model_id=model_id,
        runtime_id=runtime_id,
        precision=precision,
        sample_id=sample_id,
    )
    return Path(filled).expanduser().resolve()


def condition_matrix(cfg: dict[str, Any]) -> list[dict[str, str]]:
    models = [m["model_id"] if isinstance(m, dict) else str(m) for m in cfg.get("models", [])]
    runtimes = [r["runtime_id"] if isinstance(r, dict) else str(r) for r in cfg.get("runtime_paths", [])]
    precisions = [str(p) for p in cfg.get("precisions", [])]
    if not models or not runtimes or not precisions:
        raise ValueError("Config must define non-empty models, runtime_paths, and precisions.")
    return [
        {"model_id": m, "runtime_id": r, "precision": p}
        for m in models for r in runtimes for p in precisions
    ]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }


def collect_input_hashes(paths: list[Path]) -> pd.DataFrame:
    rows = []
    seen = set()
    for p in paths:
        p = Path(p).resolve()
        if p in seen or not p.exists() or not p.is_file():
            continue
        seen.add(p)
        rows.append({"path": str(p), "sha256": file_sha256(p), "size_bytes": p.stat().st_size})
    return pd.DataFrame(rows)
