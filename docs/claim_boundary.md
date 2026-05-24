# Claim Boundary

HPLS-Eval is an evidence-first, non-clinical research prototype. It evaluates segmentation quality retention, boundary/small-region behavior, INT8 graph evidence, provider execution evidence, fallback, and latency stability. It does not validate medical diagnosis or deployment safety.

## Research positioning

Road-damage segmentation remains a motivating pilot domain. It provides prior evidence that small irregular regions, boundary degradation, threshold sensitivity, sparse masks, and backend-dependent precision behavior can matter in real segmentation evaluation. The follow-up research direction is narrowed toward medical-lesion-oriented evaluation because similar evaluation problems can arise in lesion segmentation.

This bridge is methodological. It is not a claim that a road-damage model transfers to medical data.

## Claim boundary table

| Allowed claim | Not allowed claim |
|---|---|
| Under the specified dataset, model set, precision settings, runtime paths, and measurement protocol, HPLS-Eval summarizes segmentation quality retention, boundary retention, small-region retention, runtime transfer gap, precision compatibility, INT8 graph-audit evidence, and provider-execution evidence. | The model is safe for safety-critical or medical deployment. |
| Road damage is used as a pilot domain and prior evidence. | The road-damage model is clinically valid. |
| Both road damage and medical lesion segmentation can involve small irregular regions and boundary-sensitive masks. | Road-damage performance implies medical lesion performance. |
| The same evaluation concepts can be reused after medical-domain validation. | INT8/TensorRT/CUDA/OpenVINO/NPU/TPU success implies diagnostic safety. |
| Runtime and low-precision instability should be measured before deployment-oriented claims. | Medical deployment is validated without medical data and domain-specific review. |
| A requested provider must be separated from actual provider execution. | A requested provider is equivalent to actual provider execution. |
| INT8 graph coverage must be separated from runtime execution. | An INT8 conversion log is equivalent to actual INT8 graph coverage or INT8 kernel execution. |
| TPU/XLA rows require backend-specific profiling evidence. | A TPU/XLA row is verified by ONNX Runtime provider-profile evidence alone. |

## Required wording

Use condition-specific wording such as:

> Under the measured TensorRT FP16 path, provider profiling showed a requested-provider unique-node fraction of X and fallback ratio of Y.

For domain bridging, use:

> Road-damage segmentation is used as a motivating pilot domain. The follow-up research direction is medical-lesion-oriented evaluation, but medical claims require medical-domain data and domain-specific review.

Avoid broad wording such as:

> The model runs on TensorRT/TPU.

> The road-damage model is ready for lesion segmentation.

> INT8/NPU success supports medical deployment.
