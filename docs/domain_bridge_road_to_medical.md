# Domain Bridge: Road-Damage Pilot to Medical-Lesion-Oriented Evaluation

## Purpose

This document explains why the package keeps road-damage segmentation as a pilot domain while narrowing the follow-up research direction toward medical lesion segmentation. The goal is not to make a sudden domain switch. The goal is to preserve the road-damage evidence as a motivating case and use it to define a stricter evaluation problem for future medical-lesion-oriented studies.

## Why road damage first?

Road damage segmentation is a useful pilot domain because it exposes deployment-oriented segmentation failures in a visible and reproducible way. Cracks, potholes, and surface defects can be small, sparse, irregular, low-contrast, and boundary-sensitive. These properties make them suitable for testing whether a lightweight segmentation model remains stable under FP32, FP16, INT8, and runtime-dependent execution paths.

Road-damage experiments therefore remain as prior evidence and a reproducible pilot case. They are not discarded when the follow-up research scope moves toward medical lesion segmentation.

## What problem was observed in LiteRaceSegNet?

LiteRaceSegNet-style road-damage segmentation motivates the following evaluation concerns:

- small irregular region retention can degrade before global Dice/IoU clearly reveals the failure;
- boundary degradation can appear under threshold changes, low contrast, or precision changes;
- sparse masks can make aggregate metrics look stable even when small target regions are missed;
- backend-dependent precision behavior can change latency and output quality;
- requesting TensorRT, CUDA, OpenVINO, NPU, or TPU-like paths does not prove that the graph actually executed on that backend;
- INT8 conversion success does not prove Conv-like graph coverage or INT8 kernel execution.

These are evaluation problems, not claims of domain transfer.

## Why this problem is relevant to lesion segmentation

Medical lesion segmentation can also contain small objects, irregular shapes, ambiguous boundaries, low-contrast regions, sparse masks, and annotation uncertainty. Low-precision inference may affect boundary quality or small-region recall even when global overlap scores appear stable. Runtime conversion, unsupported operators, fallback, and tail latency can also affect deployment-oriented interpretation.

The bridge from road damage to medical lesions is therefore structural and methodological:

> small irregular region retention, boundary preservation, sparse-mask behavior, low-contrast confusion, annotation uncertainty, low-precision sensitivity, and runtime execution evidence should be measured explicitly.

This does **not** mean a road-damage model is suitable for medical images. It only motivates a stricter evaluation framework for a future medical-lesion-oriented target.

## Bilingual positioning statement

**English.** Road damage segmentation is used as a motivating pilot domain rather than as the final clinical target. The observed evaluation issues—small irregular region retention, boundary degradation, sparse-mask behavior, low-contrast confusion, and runtime-dependent precision loss—are structurally similar to those encountered in medical lesion segmentation. Therefore, this framework narrows the follow-up research scope toward medical lesion segmentation while retaining road-damage experiments as prior evidence and a reproducible pilot case.

**Korean.** 도로 손상 segmentation은 최종 의료 응용을 직접 주장하기 위한 대상이 아니라, 작은 불규칙 영역과 경계 손실 문제가 실제로 관찰된 선행 도메인으로 사용한다. 본 연구는 이 경험을 바탕으로 의료 병변 segmentation에서의 저정밀 추론 안정성, 경계 보존, 작은 병변 회수율, runtime compatibility를 평가하는 방향으로 연구 범위를 좁힌다.

## Claim boundary table

| Allowed | Not allowed |
|---|---|
| Road damage is used as a pilot domain. | The road-damage model is clinically valid. |
| Both road damage and medical lesion segmentation involve small irregular regions and boundary-sensitive masks. | Road-damage performance implies medical lesion performance. |
| The same evaluation concepts can be reused after medical-domain validation. | INT8/TensorRT/NPU success implies diagnostic safety. |
| Runtime and low-precision instability should be measured before deployment-oriented claims. | Medical deployment is validated without medical data and domain-specific review. |
| Road-damage experiments can serve as prior evidence for designing stricter metrics. | Road-damage masks, latency, or fallback logs prove medical robustness. |
| Medical-lesion-oriented evaluation can be planned as a future target. | Medical-lesion terminology means diagnostic readiness. |

## What claims are allowed?

Allowed wording should stay evidence-specific and condition-specific:

- Road damage is used as a pilot domain and a motivating case.
- Boundary degradation and small-region miss are evaluation issues observed in the pilot setting.
- Similar evaluation concerns can arise in medical lesion segmentation.
- HPLS-Eval/TRACE-style concepts can be reused after medical-domain data, model outputs, and domain-specific review are connected.
- Runtime and low-precision instability should be measured before deployment-oriented claims.
- Provider execution evidence and INT8 graph evidence must be separated from segmentation quality evidence.

## What claims are not allowed?

The following claims are forbidden in this prototype:

- The road-damage model is clinically valid.
- Road-damage performance implies medical lesion performance.
- A road-damage model can be directly deployed for medical lesion segmentation.
- INT8/TensorRT/CUDA/OpenVINO/NPU/TPU success implies diagnostic safety.
- Medical deployment is validated without medical data and domain-specific review.
- Medical-lesion wording proves clinical readiness.
- Hardware compatibility proves medical safety.

## How HPLS/TRACE will be narrowed to medical lesion segmentation

The narrowing should be staged rather than abrupt:

1. Keep road-damage segmentation as the pilot domain and motivating evidence.
2. Use pilot observations to justify the evaluation axes: boundary retention, small-region retention, low-precision sensitivity, runtime compatibility, fallback, and tail latency.
3. Replace the target dataset and task definition with medical lesion segmentation only in a future medical-domain run.
4. Keep `clinical_claim_allowed: false` unless a separate clinical validation protocol exists outside this package.
5. Report medical-lesion-oriented results as algorithmic/deployment-oriented evaluation only.
6. Do not merge road-damage and medical results into one universal ranking unless the study design explicitly supports cross-domain analysis.

## Terminology policy

`small_region_*` is the primary generic terminology used for road-damage and visual-anomaly experiments. `small_lesion_*` is retained as a medical-domain alias for future lesion segmentation experiments. These aliases must not be interpreted as evidence of clinical validity. CSV outputs keep aliases for continuity, while generated Markdown reports hide `small_lesion_*` by default unless `reporting.show_medical_alias_columns: true` is deliberately set for a medical-domain report.

Recommended mapping:

| Generic / pilot term | Medical-extension alias | Interpretation |
|---|---|---|
| `small_region_recall` | `small_lesion_recall` | Same measurement idea, different domain wording. |
| `small_region_retention_srs` | `small_lesion_retention_srs` | Alias only; not clinical evidence. |
| small irregular region | small lesion candidate region | Evaluation target after medical-domain validation. |
| safety-critical deployment risk | medical deployment-oriented risk | Research framing, not deployment approval. |

## Non-clinical disclaimer

This package does not provide clinical validation. Medical-lesion terminology, when present, refers to a future evaluation target and not to diagnostic readiness. Hardware or INT8 compatibility results must not be interpreted as evidence of medical safety or clinical deployment suitability.
