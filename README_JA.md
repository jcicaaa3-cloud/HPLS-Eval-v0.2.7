# HPLS-Eval

[English](README.md) · [한국어](README_KO.md)

小さくて不規則なマスクを扱うセグメンテーションモデルのための評価ツールです。同じモデルでも、ランタイム（CPU、GPU、NPU、TPU）と数値精度（FP32、FP16、INT8）を変えると結果がどう変わるかを測ります。

モデルそのものではなく、実験結果も入っていません。正解マスク、ランタイム・精度ごとの予測ファイル、各実行の記録を用意すると、指標を計算して表とレポートを出力します。

作り始めたきっかけは、道路損傷セグメンテーションモデル[LiteRaceSegNet](https://github.com/jcicaaa3-cloud/LiteRaceSegNet-FINAL-)の作業でした。細い境界、疎なマスク、小さい対象、しきい値への敏感さ、バックエンドのフォールバックといった問題に当たったので、それらを別々に測るために作りました。

バージョン 0.2.7。

## 確認する項目

| 分類 | 内容 |
| --- | --- |
| セグメンテーション | Dice、IoU |
| 境界 | 境界のrecallとIoU。正解の境界が空のサンプルは平均から外しますが、件数は数えます。 |
| 小領域 | 小さい領域のrecall |
| 維持率 | FP32のベースライン（既定ではCPUのFP32）に対するFP16・INT8の結果 |
| レイテンシ | p50、p90、p95、p99、最小、最大 |
| プロバイダー監査 | 指定した実行プロバイダーではなく、実際にノードを実行したプロバイダーを確認します（ONNX Runtimeのプロファイルから）。 |
| INT8監査 | ONNXグラフ内の量子化された畳み込みノードを数えます。ファイル名が`int8`でも、それだけでは信用しません。 |

集約スコア（BRS、SRS、RTG、PCS、HPLSスコア）の定義は[docs/protocol.md](docs/protocol.md)にあります。レイテンシは推定しません。測っていなければ空欄のままです。

## インストール

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .

pip install -e '.[onnx]'       # ONNXグラフ監査
pip install -e '.[ort]'        # ONNX Runtimeのプロファイル
pip install -e '.[dev]'        # テスト
```

## 使い方

```bash
hpls-eval scaffold --out ./my_project                      # 空のテンプレートを作成
hpls-eval validate --config configs/hpls_eval.yaml         # 入力を確認
hpls-eval evaluate --config configs/hpls_eval.yaml --out results/run_001
python -m pytest -q
```

監査：

```bash
python scripts/ort_profile_runner.py \
  --model exports/model.onnx --input-npz sample_input.npz \
  --providers TensorRTExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider \
  --out-json compatibility/profiles/model_trt_fp16_measurement.json

hpls-eval audit-profile \
  --profile compatibility/profiles/model_trt_fp16_profile.json \
  --requested-provider TensorRTExecutionProvider \
  --out results/audits/model_trt_fp16_provider_audit.json

hpls-eval audit-onnx \
  --model exports/model_int8.onnx \
  --strict-conv-threshold 0.80 --max-fp32-conv-fraction 0.20 \
  --out compatibility/audits/model_int8_audit.json
```

## 入力

| 入力 | 役割 |
| --- | --- |
| データセットのマニフェスト | サンプルIDと正解マスクの対応 |
| 予測ファイル | `model_id/runtime_id/precision/sample_id`の形で置いたモデル出力 |
| 互換性ログ（`compatibility.csv`） | 実行状態、レイテンシ、指定したプロバイダー、フォールバック、監査ファイルへのパス |

列の詳細は[docs/data_contract.md](docs/data_contract.md)にあります。`compatibility.csv`内の相対パスは、プロジェクトのルートではなく`compatibility.csv`があるフォルダを基準に解決します。

## 出力

```text
results/run_001/
  config_snapshot.yaml
  environment.json
  input_hashes.csv
  validation_warnings.log
  per_sample_metrics.csv
  aggregate_metrics.csv
  compatibility_normalized.csv
  runtime_evidence_matrix.csv
  condition_summary.csv
  report.md
```

## 範囲

臨床用途ではない研究用のプロトタイプです。ドキュメントに医療病変のセグメンテーションが出てくるのは、同じように小さく不規則なマスクを扱うため、今後の対象候補として挙げているからです。患者データも臨床での検証もなく、道路損傷の結果から医療での性能は何も言えません。既定の設定は`clinical_claim_allowed: false`です。

## ドキュメント

- [protocol.md](docs/protocol.md)：指標とスコア
- [data_contract.md](docs/data_contract.md)：入力形式
- [runtime_logging_guide.md](docs/runtime_logging_guide.md)：ランタイム結果の記録方法
- [provider_execution_audit.md](docs/provider_execution_audit.md)、[int8_graph_audit.md](docs/int8_graph_audit.md)：2つの監査
- [claim_boundary.md](docs/claim_boundary.md)、[domain_bridge_road_to_medical.md](docs/domain_bridge_road_to_medical.md)：範囲
- [security_notes.md](docs/security_notes.md)：チェックポイントの安全な読み込み

## ライセンス

MIT
