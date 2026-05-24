# Reviewer Notes for v2.7

This revision is a presentation-polish update. It does not weaken the runtime or INT8 evidence gates from earlier versions.

## What changed

- Warning summaries now use stable all-caps codes. Prose warnings such as `Missing compatibility rows: 2` are summarized as `MISSING_COMPATIBILITY_ROWS`, and `Total missing prediction files: 4` is summarized as `TOTAL_MISSING_PREDICTIONS`.
- `reporting.group_related_warnings: true` groups related runtime evidence warnings by `model/runtime/precision`. This keeps TensorRT/CUDA/OpenVINO fallback reports readable without hiding the underlying warnings.
- Evidence paths are documented more explicitly: relative paths in `compatibility.csv` are resolved relative to the directory containing `compatibility.csv`, not the project root.
- Wide provider/fallback/device tables are split into `Provider placement table`, `Fallback table`, and `Device evidence table`.

## What did not change

- Requested provider is still separated from executed-provider evidence.
- Unique-node counts and node-event counts remain separated.
- Fallback ratios and backend/hardware claim gates remain in the raw CSV matrices.
- INT8 graph audit integration remains required for strong INT8 graph claims.
- Medical/clinical claims remain disabled.

## Path example

If the compatibility log is:

```text
compatibility/compatibility.csv
```

then this value:

```text
provider_profile_path=profiles/trt_profile.json
```

resolves to:

```text
compatibility/profiles/trt_profile.json
```

Do not write `provider_profile_path=compatibility/profiles/trt_profile.json` inside that CSV unless the file is intentionally nested under `compatibility/compatibility/`.
