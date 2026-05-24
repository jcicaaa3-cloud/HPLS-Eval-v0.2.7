#!/usr/bin/env python3
"""Minimal ONNX Runtime runner that records both latency and provider profile.

Usage:
  python scripts/ort_profile_runner.py \
    --model exports/model.onnx \
    --input-npz sample_input.npz \
    --providers TensorRTExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider \
    --out-json compatibility/profiles/model_trt_fp16_measurement.json

The NPZ file should contain one array per model input name. This script is not
used by the evaluator directly; it is a measurement helper so provider evidence
is produced the same way across runs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-npz", required=True, help="NPZ with keys matching ONNX model input names")
    parser.add_argument("--providers", required=True, help="Comma-separated ONNX Runtime providers, in priority order")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        raise SystemExit("onnxruntime is required for this runner: pip install onnxruntime or onnxruntime-gpu") from e

    model_path = Path(args.model).expanduser().resolve()
    input_npz = Path(args.input_npz).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    arrays = dict(np.load(input_npz))
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]

    so = ort.SessionOptions()
    so.enable_profiling = True
    sess = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)

    required_inputs = [i.name for i in sess.get_inputs()]
    missing = [name for name in required_inputs if name not in arrays]
    if missing:
        raise SystemExit(f"NPZ is missing model inputs: {missing}")
    feed = {name: arrays[name] for name in required_inputs}

    for _ in range(args.warmup):
        sess.run(None, feed)

    latencies_ms: list[float] = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    profile_path = sess.end_profiling()

    result = {
        "model": str(model_path),
        "input_npz": str(input_npz),
        "requested_providers": providers,
        "actual_session_providers": sess.get_providers(),
        "warmup": args.warmup,
        "runs": args.runs,
        "latency_ms_mean": float(np.mean(latencies_ms)),
        "latency_ms_p50": percentile(latencies_ms, 50),
        "latency_ms_p90": percentile(latencies_ms, 90),
        "latency_ms_p95": percentile(latencies_ms, 95),
        "latency_ms_p99": percentile(latencies_ms, 99),
        "latency_ms_min": float(np.min(latencies_ms)),
        "latency_ms_max": float(np.max(latencies_ms)),
        "provider_profile_path": str(Path(profile_path).expanduser().resolve()),
    }
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
