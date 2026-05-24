# Anti-overclaim Checklist

Before reporting any result, check the following.

## Domain bridge / clinical claim

- [ ] Did I state that road-damage segmentation is a pilot domain and prior evidence, not the final clinical target?
- [ ] Did I avoid implying that road-damage performance transfers to medical lesion performance?
- [ ] Did I keep `clinical_claim_allowed: false`?
- [ ] Did I avoid diagnostic-readiness or medical-deployment wording?
- [ ] Did I describe medical lesion segmentation as a future evaluation target requiring medical data and domain-specific review?
- [ ] Did I use `small_region_*` as the generic term and `small_lesion_*` only as a medical-extension alias?

## Runtime claim

- [ ] Did I record the requested provider?
- [ ] Did I attach a provider profile file for ONNX Runtime accelerated paths?
- [ ] Did I report `requested_provider_unique_node_fraction` and `requested_provider_event_fraction`?
- [ ] Did I report fallback counts using `fallback_unique_node_counts` and `fallback_event_counts`?
- [ ] Is `backend_claim_allowed` true?
- [ ] Is `hardware_claim_allowed` true?
- [ ] Did I avoid saying “ran on TensorRT/CUDA/NPU/TPU” when the profile shows partial fallback or weak evidence?

## INT8 claim

- [ ] Did ONNX export/conversion succeed?
- [ ] Did `hpls-eval audit-onnx` produce an audit JSON?
- [ ] Is `int8_audit_path` present in `compatibility.csv`?
- [ ] Does `condition_summary.csv` include `strict_int8_graph_pass`, `conv_quantized_fraction`, and `fp32_conv_fraction_estimated`?
- [ ] Did runtime build succeed?
- [ ] Did provider profile show actual execution on the intended backend?
- [ ] Did I separate INT8 graph structure from INT8 kernel execution?

## Boundary / small-region claim

- [ ] Did boundary means exclude empty-GT-boundary samples?
- [ ] Did I report `empty_gt_boundary_count`?
- [ ] Did I report `boundary_fp_empty_gt_count`?
- [ ] Did I avoid claiming boundary stability from blank images?
- [ ] Did I define the small-damage-region threshold before looking at results?

## Latency claim

- [ ] Did I report p50 and p99, not only a mean?
- [ ] Did I separate compile/build time from steady-state inference time?
- [ ] Did I record warmup policy and batch size somewhere?

## OpenVINO/NPU claim

- [ ] Did I record `requested_device`?
- [ ] Did I record `available_devices`?
- [ ] Did I record `actual_compiled_device`?
- [ ] Did I inspect `device_claim_status`?
- [ ] Did I avoid calling it NPU if AUTO selected CPU?

## TPU/XLA claim

- [ ] Did I attach `xla_profile_path` or `backend_profile_path`?
- [ ] Did I explicitly state that ORT provider profiles are not sufficient for TPU/XLA verification?
- [ ] Did I avoid ranking TPU rows against ORT-backed rows unless profiling evidence is comparable?

## Baseline claim

- [ ] Did the configured FP32 baseline produce `n_ok > 0` for every model?
- [ ] If `fail_on_missing_baseline` is true, did evaluation stop when baseline evidence was missing?
