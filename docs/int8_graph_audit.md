# INT8 Graph Audit

HPLS-Eval v2.2 treats INT8 conversion as a multi-level evidence problem.

## Weak evidence

The following are weak by themselves:

- INT8 or UINT8 initializer exists somewhere.
- `QuantizeLinear` or `DequantizeLinear` exists somewhere.
- the exporter printed “quantization complete.”

These can be true even when most heavy Conv computation remains effectively FP32.

## Stronger evidence

The `hpls-eval audit-onnx` command checks:

- total regular `Conv` count;
- `QLinearConv` count;
- `ConvInteger` count;
- total Conv-like count;
- QDQ-wrapped `Conv` count;
- QDQ `Conv` count whose weight path is dequantized from INT8/UINT8 initializer;
- estimated remaining FP32 Conv count and fraction;
- initializer dtype counts;
- QDQ/QOperator presence;
- `strict_int8_graph_pass` and `strict_int8_graph_reason`.

Example:

```bash
hpls-eval audit-onnx --model exports/model_int8.onnx --out compatibility/audits/model_int8_audit.json
```

Then connect it to the evaluation matrix:

```csv
model_id,runtime_id,precision,int8_audit_path
literace_segnet,tensorrt_gpu,INT8,audits/model_int8_audit.json
```

`evaluate` loads the JSON and carries these fields into `runtime_evidence_matrix.csv` and `condition_summary.csv`:

- `strict_int8_graph_pass`
- `strict_int8_graph_reason`
- `conv_total`
- `qlinear_conv_count`
- `qdq_wrapped_conv_count`
- `quantized_conv_like_count`
- `conv_quantized_fraction`
- `fp32_conv_fraction_estimated`
- `int8_graph_claim_allowed`

The strict pass defaults are:

- `conv_quantized_fraction >= 0.80`
- `fp32_conv_fraction_estimated <= 0.20`

These thresholds can be changed, but the paper should report them before results.

## Conversion success vs. execution success

Keep these separate:

1. ONNX export success
2. INT8 graph audit pass
3. Runtime build success
4. Provider execution profile
5. Accuracy/quality retention

A graph can pass conversion and fail actual INT8/provider execution. Conversely, a graph can contain QDQ structure but still fall back to CPU if the target execution provider does not support the relevant kernels.
