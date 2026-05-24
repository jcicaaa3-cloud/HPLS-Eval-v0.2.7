# Research Plan Alignment

This prototype keeps the HPLS-Eval idea while preserving LiteRaceSegNet / road-damage segmentation as the motivating pilot domain. The follow-up research direction is narrowed toward medical-lesion-oriented evaluation, but the package remains a non-clinical evaluation harness.

## Narrative alignment

1. Road-damage segmentation is the pilot domain where small irregular regions, boundary loss, threshold sensitivity, sparse masks, and runtime/precision uncertainty were observed.
2. These observations motivate the evaluation axes: boundary retention, small-region retention, low-precision graph coverage, provider execution evidence, fallback, and tail latency.
3. Medical lesion segmentation is the intended future target because similar evaluation problems can appear around small lesions, ambiguous boundaries, sparse masks, low contrast, annotation uncertainty, and precision/runtime changes.
4. The bridge is methodological, not predictive. Road-damage performance does not imply medical lesion performance.

| Research concept | Prototype artifact |
|---|---|
| Road-damage pilot evidence | `domain_context: road_damage_pilot` |
| Medical-lesion-oriented follow-up | `domain_context: medical_lesion_target`, `claim_scope: medical_extension_planned` |
| Non-clinical boundary | `clinical_claim_allowed: false`, `clinical_claim_status` |
| FP32 / FP16 / INT8 precision axis | `precisions` in config |
| CPU / GPU / NPU / TPU-oriented runtime path | `runtime_paths` in config |
| Dice / IoU / boundary / small-region metrics | `src/hpls_eval/metrics.py` |
| BRS | `boundary_retention_brs` in `condition_summary.csv` |
| SRS | `small_region_retention_srs` and medical alias `small_lesion_retention_srs` |
| RTG | `runtime_transfer_gap_rtg` in `condition_summary.csv` |
| PCS | `precision_compatibility_pcs` in `condition_summary.csv` |
| Provider fallback evidence | `runtime_evidence_matrix.csv` and provider profile audit |
| INT8 graph evidence | `int8_audit_path` and integrated audit columns |
| Claim boundary | `docs/claim_boundary.md`, `docs/domain_bridge_road_to_medical.md` |

The prototype deliberately does not include invented predictions, invented latency, invented compatibility rows, invented hardware-profile evidence, or invented medical validation. The framework enforces that separation by requiring real prediction masks, real runtime logs, provider profiles, INT8 audit artifacts, and explicit domain/claim metadata before numeric claims are produced.
