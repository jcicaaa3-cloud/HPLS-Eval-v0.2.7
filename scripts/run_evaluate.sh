#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/hpls_eval.yaml}
OUT=${2:-results/run_001}
hpls-eval evaluate --config "$CONFIG" --out "$OUT"
