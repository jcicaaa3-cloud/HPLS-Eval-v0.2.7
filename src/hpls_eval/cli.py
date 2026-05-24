from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .evaluator import evaluate
from .scaffold import scaffold_project
from .validation import validate_inputs
from .audit import audit_onnx_int8_graph, parse_ort_profile
from .warning_utils import dedupe_warnings


def cmd_validate(args: argparse.Namespace) -> int:
    cfg, config_path = load_config(args.config)
    warnings, summary = validate_inputs(cfg, config_path)
    warnings = dedupe_warnings(warnings)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if warnings:
        print("\nWarnings:")
        for w in warnings[:200]:
            print(f"- {w}")
        if len(warnings) > 200:
            print(f"- ... {len(warnings) - 200} more")
    return 0 if not args.strict or not warnings else 2


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg, config_path = load_config(args.config)
    try:
        result = evaluate(cfg, config_path, args.out)
    except RuntimeError as e:
        print(json.dumps({"error": str(e), "out_dir": str(Path(args.out).expanduser().resolve())}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_scaffold(args: argparse.Namespace) -> int:
    out = scaffold_project(args.out)
    print(f"Created HPLS-Eval project scaffold at: {out}")
    return 0


def _write_or_print_json(data: dict, out_path: str | None) -> None:
    if out_path:
        out = Path(out_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def cmd_audit_profile(args: argparse.Namespace) -> int:
    result = parse_ort_profile(args.profile, requested_provider=args.requested_provider)
    _write_or_print_json(result, args.out)
    return 0


def cmd_audit_onnx(args: argparse.Namespace) -> int:
    result = audit_onnx_int8_graph(
        args.model,
        strict_conv_threshold=args.strict_conv_threshold,
        max_fp32_conv_fraction=args.max_fp32_conv_fraction,
    )
    _write_or_print_json(result, args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hpls-eval", description="HPLS-Eval research-grade evaluation framework")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="Validate manifest, condition matrix, predictions, and compatibility logs")
    v.add_argument("--config", required=True, help="Path to HPLS-Eval YAML config")
    v.add_argument("--strict", action="store_true", help="Exit with non-zero status if warnings exist")
    v.set_defaults(func=cmd_validate)

    e = sub.add_parser("evaluate", help="Run HPLS-Eval metrics and write result artifacts")
    e.add_argument("--config", required=True, help="Path to HPLS-Eval YAML config")
    e.add_argument("--out", required=True, help="Output directory for evaluation artifacts")
    e.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("scaffold", help="Create empty project templates without fake results")
    s.add_argument("--out", required=True, help="Output directory")
    s.set_defaults(func=cmd_scaffold)

    ap = sub.add_parser("audit-profile", help="Audit ONNX Runtime profile JSON for actual provider execution and CPU fallback")
    ap.add_argument("--profile", required=True, help="ONNX Runtime profiling JSON path")
    ap.add_argument("--requested-provider", required=True, help="Provider that was requested, e.g. TensorRTExecutionProvider")
    ap.add_argument("--out", help="Optional JSON output path")
    ap.set_defaults(func=cmd_audit_profile)

    ao = sub.add_parser("audit-onnx", help="Audit ONNX quantization structure, especially INT8/QDQ coverage around Conv nodes")
    ao.add_argument("--model", required=True, help="ONNX model path")
    ao.add_argument("--strict-conv-threshold", type=float, default=0.80, help="Required quantized Conv-like fraction for strict pass")
    ao.add_argument("--max-fp32-conv-fraction", type=float, default=0.20, help="Maximum tolerated estimated FP32 Conv fraction")
    ao.add_argument("--out", help="Optional JSON output path")
    ao.set_defaults(func=cmd_audit_onnx)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
