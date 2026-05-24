import pandas as pd

from hpls_eval.evaluator import _baseline_missing_messages


def test_baseline_missing_messages_when_n_ok_zero():
    cfg = {
        "models": [{"model_id": "m"}],
        "baseline": {"runtime_id": "onnxruntime_cpu", "precision": "FP32"},
    }
    aggregate = pd.DataFrame([
        {"model_id": "m", "runtime_id": "onnxruntime_cpu", "precision": "FP32", "n_ok": 0}
    ])
    msgs = _baseline_missing_messages(cfg, aggregate)
    assert msgs
    assert "BASELINE_MISSING" in msgs[0]


def test_baseline_missing_messages_empty_when_baseline_ok():
    cfg = {
        "models": [{"model_id": "m"}],
        "baseline": {"runtime_id": "onnxruntime_cpu", "precision": "FP32"},
    }
    aggregate = pd.DataFrame([
        {"model_id": "m", "runtime_id": "onnxruntime_cpu", "precision": "FP32", "n_ok": 3}
    ])
    assert _baseline_missing_messages(cfg, aggregate) == []
