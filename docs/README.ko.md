# Jetson-LLM

Google Colab(T4 GPU)에서 LLM을 학습(QLoRA)·양자화하고 NVIDIA **Jetson Orin Nano(8GB)** 에서 TensorRT-LLM으로 구동하는 온디바이스 LLM 배포 파이프라인.
모델: Llama-3 8B Instruct (리허설: TinyLlama-1.1B) · 전략: NF4(QLoRA 학습) → INT4 AWQ(추론) · 노션 "Jetson-LLM" 페이지의 실험 노트북 모음.

**작업 기간: 2025-12-15 ~ 2025-12-16** (파일 작성 시각 기준)

| 파일 (현재 영문명) | 원본 파일명 | 작성일 | 내용 |
|---|---|---|---|
| `01-dataset-synthesis.ipynb` | `Llama 데이터셋 만들기.ipynb` | 2025-12-16 | 자연어 명령→로봇 제어 코드 SFT 데이터셋 5만 건 합성 (robot_dataset.json) |
| `02-qlora-llama3-8b.ipynb` | `Quantization_0.ipynb` | 2025-12-16 | Llama-3 8B QLoRA(NF4+LoRA) 파인튜닝 — BF16/OOM 디버깅 기록 포함 |
| `03-tinyllama-awq-rehearsal.ipynb` | `양자화테스트.ipynb` | 2025-12-16 | TinyLlama로 TensorRT-LLM INT4 AWQ 변환→빌드→추론 전 과정 리허설 |
| `04-llama3-awq-orin-deploy.ipynb` | `양자화테스트_orin_초기버전.ipynb` | 2025-12-15 | Llama-3 8B INT4 AWQ 변환 + Jetson Orin(dustynv/tensorrt_llm 도커) 배포 절차 |

## 비고
- 제외한 원본: `양자화테스트_orin_초기버전.ipynb의 사본`(내용 중복), `Quantization 0.ipynb`(**HuggingFace 토큰·wandb API 키 평문 노출** — 클린본 `Quantization_0.ipynb`로 대체. 원본 덤프에는 아직 남아 있으므로 주의)
