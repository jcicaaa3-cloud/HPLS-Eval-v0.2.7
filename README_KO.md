# HPLS-Eval

[English](README.md) · [日本語](README_JA.md)

작고 불규칙한 마스크를 다루는 세그멘테이션 모델용 평가 도구입니다. 같은 모델이라도 런타임(CPU, GPU, NPU, TPU)과 수치 정밀도(FP32, FP16, INT8)를 바꾸면 결과가 어떻게 달라지는지 잽니다.

모델이 아니고, 레포에 실험 결과도 들어 있지 않습니다. 정답 마스크, 런타임·정밀도별 예측 파일, 각 실행 기록을 넣으면 지표를 계산해서 표와 보고서를 만듭니다.

도로 손상 세그멘테이션 모델 [LiteRaceSegNet](https://github.com/jcicaaa3-cloud/LiteRaceSegNet-FINAL-)을 만들다가 시작했습니다. 그때 얇은 경계, 드문드문한 마스크, 작은 객체, 임계값 민감도, 백엔드 폴백 문제를 겪었고, 이것들을 따로따로 재려고 만들었습니다.

버전 0.2.7.

## 확인하는 항목

| 분류 | 내용 |
| --- | --- |
| 세그멘테이션 | Dice, IoU |
| 경계 | 경계 recall과 IoU. 정답 경계가 비어 있는 샘플은 평균에서 빼되 개수는 셉니다. |
| 작은 영역 | 작은 영역의 recall |
| 유지율 | FP32 기준(기본값은 CPU의 FP32) 대비 FP16·INT8 결과 |
| 지연 시간 | p50, p90, p95, p99, 최소, 최대 |
| 프로바이더 감사 | 요청한 실행 프로바이더가 아니라 실제로 노드를 실행한 프로바이더를 확인합니다(ONNX Runtime 프로파일 기준). |
| INT8 감사 | ONNX 그래프에서 양자화된 합성곱 노드를 셉니다. 파일 이름이 `int8`이라고 그대로 믿지 않습니다. |

요약 점수(BRS, SRS, RTG, PCS, HPLS 점수)의 정의는 [docs/protocol.md](docs/protocol.md)에 있습니다. 지연 시간은 추정하지 않습니다. 재지 않았으면 빈칸으로 남습니다.

## 설치

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .

pip install -e '.[onnx]'       # ONNX 그래프 감사
pip install -e '.[ort]'        # ONNX Runtime 프로파일
pip install -e '.[dev]'        # 테스트
```

## 사용법

```bash
hpls-eval scaffold --out ./my_project                      # 빈 템플릿 생성
hpls-eval validate --config configs/hpls_eval.yaml         # 입력 확인
hpls-eval evaluate --config configs/hpls_eval.yaml --out results/run_001
python -m pytest -q
```

감사:

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

## 입력

| 입력 | 역할 |
| --- | --- |
| 데이터셋 매니페스트 | 샘플 ID와 정답 마스크 연결 |
| 예측 파일 | `model_id/runtime_id/precision/sample_id` 구조로 둔 모델 출력 |
| 호환성 로그(`compatibility.csv`) | 실행 상태, 지연 시간, 요청한 프로바이더, 폴백, 감사 파일 경로 |

열 설명은 [docs/data_contract.md](docs/data_contract.md)에 있습니다. `compatibility.csv` 안의 상대 경로는 프로젝트 루트가 아니라 `compatibility.csv`가 있는 폴더를 기준으로 해석합니다.

## 출력

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

## 범위

임상용이 아닌 연구용 프로토타입입니다. 문서에 의료 병변 세그멘테이션이 나오는 건, 똑같이 작고 불규칙한 마스크를 다루기 때문에 나중에 평가해 볼 대상으로 적어 둔 것입니다. 환자 데이터도 임상 검증도 없고, 도로 손상 결과로는 의료 성능에 대해 아무것도 말할 수 없습니다. 기본 설정은 `clinical_claim_allowed: false`입니다.

## 문서

- [protocol.md](docs/protocol.md): 지표와 점수
- [data_contract.md](docs/data_contract.md): 입력 형식
- [runtime_logging_guide.md](docs/runtime_logging_guide.md): 런타임 결과 기록 방법
- [provider_execution_audit.md](docs/provider_execution_audit.md), [int8_graph_audit.md](docs/int8_graph_audit.md): 두 가지 감사
- [claim_boundary.md](docs/claim_boundary.md), [domain_bridge_road_to_medical.md](docs/domain_bridge_road_to_medical.md): 범위
- [security_notes.md](docs/security_notes.md): 체크포인트 안전하게 불러오기

## 라이선스

MIT
