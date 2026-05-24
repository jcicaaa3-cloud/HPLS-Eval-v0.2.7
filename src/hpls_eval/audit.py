from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import json
import math
import re


_PROVIDER_CANONICAL_PATTERNS: list[tuple[str, str]] = [
    ("tensorrt", "TensorRTExecutionProvider"),
    ("trt", "TensorRTExecutionProvider"),
    ("cuda", "CUDAExecutionProvider"),
    ("openvino", "OpenVINOExecutionProvider"),
    ("cpu", "CPUExecutionProvider"),
    ("rocm", "ROCMExecutionProvider"),
    ("directml", "DmlExecutionProvider"),
    ("dml", "DmlExecutionProvider"),
    ("xla", "XLAExecutionProvider"),
    ("tpu", "TPUExecutionProvider"),
]


def _read_json(path: str | Path) -> Any:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_events(obj: Any) -> list[dict[str, Any]]:
    """Return a list of profiler events from common ORT/Chrome trace layouts."""
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("traceEvents", "events"):
            if isinstance(obj.get(key), list):
                return [x for x in obj[key] if isinstance(x, dict)]
    return []


def canonical_provider_name(value: str | None) -> str | None:
    """Normalize provider spellings so requested/executed comparisons are fair.

    ONNX Runtime profile JSONs are not perfectly uniform across releases and
    EPs. This helper maps common spellings to the standard execution-provider
    name without hiding the raw profile: raw strings are still visible in the
    original JSON, while audit outputs use canonical names for comparison.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = re.sub(r"[^a-z0-9]+", "", text.lower())
    for token, canonical in _PROVIDER_CANONICAL_PATTERNS:
        if token in lowered:
            return canonical
    return text


def _provider_from_event(event: dict[str, Any]) -> str | None:
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    for key in (
        "provider",
        "execution_provider",
        "executionProvider",
        "execution_provider_name",
        "ep",
        "provider_type",
    ):
        val = args.get(key) or event.get(key)
        canonical = canonical_provider_name(str(val)) if val is not None else None
        if canonical:
            return canonical
    return None


def _is_node_event_candidate(event: dict[str, Any]) -> bool:
    cat = str(event.get("cat", "")).lower()
    name = str(event.get("name", "")).lower()
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    if cat == "node":
        return True
    # ONNX Runtime usually uses cat=Node, but some profiling wrappers put node
    # metadata only inside args. Exclude obvious session/model events.
    if "op_name" in args or "node_index" in args or "node_name" in args:
        return "session" not in name and "model" not in name
    return False


def _is_kernel_time_event(event: dict[str, Any]) -> bool:
    name = str(event.get("name", "")).lower()
    return "kernel_time" in name or name.endswith("_kernel") or name.endswith("/kernel")


def _node_key(event: dict[str, Any]) -> str:
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    if args.get("node_index") is not None:
        return f"idx:{args.get('node_index')}"
    if args.get("node_name"):
        return f"name:{args.get('node_name')}"
    name = str(event.get("name", "UNKNOWN_NODE"))
    # ORT node names often end with _kernel_time. Strip timing suffixes so the
    # same node is counted once even if profiling emits multiple timing records.
    cleaned = re.sub(r"(_kernel_time|_fence_before|_fence_after|_kernel)$", "", name)
    return f"name:{cleaned}"


def _op_name(event: dict[str, Any]) -> str:
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    return str(args.get("op_name") or args.get("op") or args.get("operator") or "UNKNOWN")


def parse_ort_profile(profile_path: str | Path, requested_provider: str | None = None) -> dict[str, Any]:
    """Parse ONNX Runtime profiling JSON and count actual execution providers.

    The important distinction is requested provider vs. provider actually used
    by executed nodes. This function does not infer hardware execution from
    Session providers alone. It reports only what profiler node/kernel events
    recorded.

    Two count families are returned:
    - provider_node_counts: unique graph-node counts by provider. This is the
      default denominator for requested_provider_fraction.
    - provider_event_counts: raw profiled node-event counts, useful when a
      profile includes many repeated inference runs.
    """
    p = Path(profile_path).expanduser().resolve()
    obj = _read_json(p)
    events = _as_events(obj)
    node_candidates = [ev for ev in events if _is_node_event_candidate(ev)]
    kernel_events = [ev for ev in node_candidates if _is_kernel_time_event(ev)]
    used_events = kernel_events if kernel_events else node_candidates
    event_filter = "kernel_time_only" if kernel_events else "all_node_events"

    provider_event_counts: Counter[str] = Counter()
    op_provider_event_counts: dict[str, Counter[str]] = defaultdict(Counter)
    node_provider_votes: dict[str, Counter[str]] = defaultdict(Counter)
    node_op_name: dict[str, str] = {}
    unassigned_node_events = 0

    for ev in used_events:
        provider = _provider_from_event(ev)
        if provider is None:
            unassigned_node_events += 1
            provider = "UNKNOWN_PROVIDER"
        op = _op_name(ev)
        key = _node_key(ev)
        provider_event_counts[provider] += 1
        op_provider_event_counts[op][provider] += 1
        node_provider_votes[key][provider] += 1
        node_op_name.setdefault(key, op)

    provider_node_counts: Counter[str] = Counter()
    op_provider_node_counts: dict[str, Counter[str]] = defaultdict(Counter)
    mixed_provider_node_keys: list[str] = []
    for key, votes in node_provider_votes.items():
        if len(votes) > 1:
            mixed_provider_node_keys.append(key)
        # Assign each unique node to the provider that executed it most often in
        # the profile. If a node really changes provider between runs, the mixed
        # key list exposes that anomaly.
        provider = votes.most_common(1)[0][0]
        provider_node_counts[provider] += 1
        op_provider_node_counts[node_op_name.get(key, "UNKNOWN")][provider] += 1

    requested = canonical_provider_name(str(requested_provider)) if requested_provider else None
    total_unique_nodes = int(sum(provider_node_counts.values()))
    total_node_events = int(sum(provider_event_counts.values()))

    requested_node_count = None
    requested_event_count = None
    requested_fraction = None
    requested_event_fraction = None
    fallback_counts: dict[str, int] = {}
    fallback_event_counts: dict[str, int] = {}
    fallback_node_count = None
    fallback_event_count = None
    fallback_ratio = None
    fallback_event_ratio = None

    if requested:
        requested_node_count = int(provider_node_counts.get(requested, 0))
        requested_event_count = int(provider_event_counts.get(requested, 0))
        if total_unique_nodes > 0:
            requested_fraction = requested_node_count / float(total_unique_nodes)
        if total_node_events > 0:
            requested_event_fraction = requested_event_count / float(total_node_events)
        fallback_counts = {k: int(v) for k, v in provider_node_counts.items() if k != requested}
        fallback_event_counts = {k: int(v) for k, v in provider_event_counts.items() if k != requested}
        fallback_node_count = int(sum(fallback_counts.values()))
        fallback_event_count = int(sum(fallback_event_counts.values()))
        if total_unique_nodes > 0:
            fallback_ratio = fallback_node_count / float(total_unique_nodes)
        if total_node_events > 0:
            fallback_event_ratio = fallback_event_count / float(total_node_events)

    return {
        "profile_path": str(p),
        "profile_event_filter": event_filter,
        "total_trace_event_count": int(len(events)),
        "total_node_event_candidate_count": int(len(node_candidates)),
        "total_profiled_node_count": int(total_unique_nodes),  # backward-compatible alias for total_unique_profiled_node_count
        "total_unique_profiled_node_count": int(total_unique_nodes),
        "total_profiled_node_event_count": int(total_node_events),
        "unassigned_node_event_count": int(unassigned_node_events),
        "provider_node_counts": dict(sorted((k, int(v)) for k, v in provider_node_counts.items())),  # backward-compatible alias for unique_node_provider_counts
        "unique_node_provider_counts": dict(sorted((k, int(v)) for k, v in provider_node_counts.items())),
        "provider_event_counts": dict(sorted((k, int(v)) for k, v in provider_event_counts.items())),  # backward-compatible alias for node_event_provider_counts
        "provider_node_event_counts": dict(sorted((k, int(v)) for k, v in provider_event_counts.items())),
        "node_event_provider_counts": dict(sorted((k, int(v)) for k, v in provider_event_counts.items())),
        "requested_provider": requested or "",
        "requested_provider_raw": str(requested_provider) if requested_provider else "",
        "requested_provider_node_count": requested_node_count,
        "requested_provider_unique_node_count": requested_node_count,
        "requested_provider_event_count": requested_event_count,
        "requested_provider_fraction": requested_fraction,  # backward-compatible alias for requested_provider_unique_node_fraction
        "requested_provider_unique_node_fraction": requested_fraction,
        "requested_provider_event_fraction": requested_event_fraction,
        "fallback_node_counts": dict(sorted(fallback_counts.items())),  # backward-compatible alias for fallback_unique_node_counts
        "fallback_unique_node_counts": dict(sorted(fallback_counts.items())),
        "fallback_event_counts": dict(sorted(fallback_event_counts.items())),
        "fallback_node_count": fallback_node_count,
        "fallback_event_count": fallback_event_count,
        "fallback_ratio_measured": fallback_ratio,
        "fallback_ratio_event_measured": fallback_event_ratio,
        "mixed_provider_node_count": int(len(mixed_provider_node_keys)),
        "mixed_provider_node_keys_sample": mixed_provider_node_keys[:50],
        "op_provider_counts": {op: dict(sorted((k, int(v)) for k, v in c.items())) for op, c in sorted(op_provider_node_counts.items())},
        "op_provider_event_counts": {op: dict(sorted((k, int(v)) for k, v in c.items())) for op, c in sorted(op_provider_event_counts.items())},
    }


def _dtype_name(tensor: Any) -> str:
    # ONNX TensorProto numeric constants. Avoid importing TensorProto in normal
    # paths so this module can be imported without onnx installed.
    mapping = {
        1: "FLOAT",
        2: "UINT8",
        3: "INT8",
        4: "UINT16",
        5: "INT16",
        6: "INT32",
        7: "INT64",
        9: "BOOL",
        10: "FLOAT16",
        11: "DOUBLE",
        12: "UINT32",
        13: "UINT64",
        16: "BFLOAT16",
    }
    return mapping.get(int(getattr(tensor, "data_type", -1)), str(getattr(tensor, "data_type", "UNKNOWN")))


def audit_onnx_int8_graph(
    model_path: str | Path,
    *,
    strict_conv_threshold: float = 0.80,
    max_fp32_conv_fraction: float = 0.20,
) -> dict[str, Any]:
    """Audit whether an ONNX graph is actually quantized around Conv nodes.

    This distinguishes weak graph evidence from Conv-level quantization
    coverage. The strict pass is based on Conv-like compute coverage, not merely
    the presence of a few INT8 tensors.
    """
    try:
        import onnx  # type: ignore
    except Exception as e:  # pragma: no cover - depends on optional package
        raise RuntimeError("ONNX graph audit requires the optional package `onnx`. Install with: pip install onnx") from e

    p = Path(model_path).expanduser().resolve()
    model = onnx.load(str(p))
    graph = model.graph
    nodes = list(graph.node)

    node_type_counts = Counter(n.op_type for n in nodes)
    initializer_dtype_counts = Counter(_dtype_name(t) for t in graph.initializer)
    initializer_dtype_by_name = {t.name: _dtype_name(t) for t in graph.initializer}
    int_initializer_count = int(initializer_dtype_counts.get("INT8", 0) + initializer_dtype_counts.get("UINT8", 0))
    fp32_initializer_count = int(initializer_dtype_counts.get("FLOAT", 0))

    conv_nodes = [n for n in nodes if n.op_type == "Conv"]
    conv_total = len(conv_nodes)
    qlinear_conv_count = int(node_type_counts.get("QLinearConv", 0))
    conv_integer_count = int(node_type_counts.get("ConvInteger", 0))
    qoperator_conv_count = qlinear_conv_count + conv_integer_count
    conv_like_total = conv_total + qoperator_conv_count

    producer: dict[str, Any] = {}
    consumers: dict[str, list[Any]] = defaultdict(list)
    for n in nodes:
        for out in n.output:
            producer[out] = n
        for inp in n.input:
            consumers[inp].append(n)

    def producer_op(value_name: str) -> str | None:
        n = producer.get(value_name)
        return str(n.op_type) if n is not None else None

    def is_dq_from_int_initializer(value_name: str) -> bool:
        n = producer.get(value_name)
        if n is None or n.op_type != "DequantizeLinear" or not n.input:
            return False
        source = n.input[0]
        return initializer_dtype_by_name.get(source) in {"INT8", "UINT8"}

    qdq_wrapped_conv_count = 0
    qdq_conv_with_quantized_weight_count = 0
    qdq_conv_node_names: list[str] = []
    qdq_weight_conv_node_names: list[str] = []
    fp32_conv_node_names: list[str] = []
    conv_audit_rows: list[dict[str, Any]] = []

    for idx, n in enumerate(conv_nodes):
        name = n.name or f"Conv#{idx}"
        has_dq_input = any((inp in producer and producer[inp].op_type == "DequantizeLinear") for inp in n.input)
        # Weight is usually input[1]. If it is DequantizeLinear(INT8/UINT8 initializer),
        # the heavy Conv weight path is quantized structurally.
        has_quantized_weight = len(n.input) >= 2 and is_dq_from_int_initializer(n.input[1])
        has_q_output = any(c.op_type == "QuantizeLinear" for out in n.output for c in consumers.get(out, []))
        if has_dq_input and has_q_output:
            qdq_wrapped_conv_count += 1
            qdq_conv_node_names.append(name)
        if has_quantized_weight and has_dq_input:
            qdq_conv_with_quantized_weight_count += 1
            qdq_weight_conv_node_names.append(name)
        else:
            fp32_conv_node_names.append(name)
        conv_audit_rows.append({
            "node_name": name,
            "has_dq_input": bool(has_dq_input),
            "has_quantized_weight": bool(has_quantized_weight),
            "has_q_output": bool(has_q_output),
            "input_producer_ops": [producer_op(inp) for inp in n.input],
        })

    # A QOperator Conv is already quantized Conv-like compute. A QDQ Conv only
    # counts as quantized coverage when its Conv weight path is demonstrably
    # dequantized from INT8/UINT8 initializer.
    quantized_conv_like_count = qoperator_conv_count + qdq_conv_with_quantized_weight_count
    conv_quantized_fraction = float("nan")
    fp32_conv_fraction = float("nan")
    fp32_conv_count_estimated = 0
    if conv_like_total > 0:
        conv_quantized_fraction = min(1.0, quantized_conv_like_count / float(conv_like_total))
        fp32_conv_count_estimated = max(0, conv_like_total - min(conv_like_total, quantized_conv_like_count))
        fp32_conv_fraction = fp32_conv_count_estimated / float(conv_like_total)

    has_qdq = bool(node_type_counts.get("QuantizeLinear", 0) or node_type_counts.get("DequantizeLinear", 0))
    has_qoperator = bool(qoperator_conv_count or any(str(k).startswith("QLinear") for k in node_type_counts))
    has_int_initializers = int_initializer_count > 0

    if conv_like_total == 0:
        strict_pass = False
        strict_reason = "no_conv_like_nodes_to_audit"
    elif quantized_conv_like_count <= 0:
        strict_pass = False
        strict_reason = "no_quantized_conv_like_nodes"
    elif conv_quantized_fraction < float(strict_conv_threshold):
        strict_pass = False
        strict_reason = "conv_quantized_fraction_below_threshold"
    elif fp32_conv_fraction > float(max_fp32_conv_fraction):
        strict_pass = False
        strict_reason = "estimated_fp32_conv_fraction_above_threshold"
    else:
        strict_pass = True
        strict_reason = "passes_predeclared_conv_coverage_thresholds"

    return {
        "model_path": str(p),
        "node_type_counts": dict(sorted((k, int(v)) for k, v in node_type_counts.items())),
        "initializer_dtype_counts": dict(sorted((k, int(v)) for k, v in initializer_dtype_counts.items())),
        "int8_uint8_initializer_count": int_initializer_count,
        "fp32_initializer_count": fp32_initializer_count,
        "has_int8_or_uint8_initializer": bool(has_int_initializers),
        "has_qdq_nodes": has_qdq,
        "has_qoperator_nodes": has_qoperator,
        "conv_total": int(conv_total),
        "qlinear_conv_count": int(qlinear_conv_count),
        "conv_integer_count": int(conv_integer_count),
        "qoperator_conv_count": int(qoperator_conv_count),
        "conv_like_total": int(conv_like_total),
        "qdq_wrapped_conv_count": int(qdq_wrapped_conv_count),
        "qdq_conv_with_quantized_weight_count": int(qdq_conv_with_quantized_weight_count),
        "quantized_conv_like_count": int(quantized_conv_like_count),
        "conv_quantized_fraction": conv_quantized_fraction,
        "fp32_conv_count_estimated": int(fp32_conv_count_estimated),
        "fp32_conv_fraction_estimated": fp32_conv_fraction,
        "strict_int8_graph_pass": bool(strict_pass),
        "strict_int8_graph_reason": strict_reason,
        "strict_conv_threshold": float(strict_conv_threshold),
        "max_fp32_conv_fraction": float(max_fp32_conv_fraction),
        "fp32_conv_node_names_sample": fp32_conv_node_names[:50],
        "qdq_conv_node_names_sample": qdq_conv_node_names[:50],
        "qdq_weight_conv_node_names_sample": qdq_weight_conv_node_names[:50],
        "conv_audit_sample": conv_audit_rows[:50],
    }


def safe_json_dumps(obj: Any) -> str:
    def default(x: Any) -> Any:
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return str(x)

    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=default)
