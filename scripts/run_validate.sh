#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/hpls_eval.yaml}
hpls-eval validate --config "$CONFIG"
