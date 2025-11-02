## notebookf585393d8a_ver3.ipynb 튜닝 변경 요약

- **프롬프트 정비**  
  - 시스템 메시지를 4줄짜리 한국어 지침으로 재작성해 JSON 한 줄 답변 규칙을 강조했습니다.  
  - `create_user_block`는 보기들을 줄바꿈해 나열하고, 소문자 JSON 출력 요구 사항을 명시하도록 정리했습니다.
- **LoRA 설정 상향 조정**  
  - `LORA_R=32`, `LORA_ALPHA=64`, `LORA_DROPOUT=0.08`을 기본값으로 두고, 같은 이름의 환경변수로 조정 가능하도록 했습니다.  
  - A10G 24 GB VRAM 기준으로 안정적인 학습을 목표로 했으며, 필요 시 값만 바꿔 재사용할 수 있습니다.
- **훈련 루프 개편**  
  - `per_device_train_batch_size`를 GPU 메모리에 따라 자동 추정(기본 2)하고, 목표 누적 배치(`EFFECTIVE_BATCH_SIZE`)에 맞춰 `gradient_accumulation_steps`를 계산합니다.  
  - `TrainingArguments`에 `predict_with_generate`, `load_best_model_at_end`, `metric_for_best_model="accuracy"` 등을 추가하고, `EarlyStoppingCallback`(기본 patience=1)으로 8시간 안에서 효율적으로 멈추도록 구성했습니다.  
  - `compute_metrics`로 생성 결과를 파싱해 정확도를 계산하며, 로깅 주기는 기본 50 step입니다.
- **검증 샘플 확대**  
  - 검증셋은 층화 기반으로 최대 256개를 고정 추출해 `Trainer.evaluate`에 사용하고, 기존 32건 OOM 체크 루틴은 그대로 유지했습니다.
- **추론 신뢰성 향상**  
  - `decode_answer_text`가 JSON 스니펫 탐색, 키워드 패턴, 폴백 순으로 동작해 형식이 조금 어긋난 출력도 보정합니다.  
  - 추론 배치/토큰 수는 `PRED_BATCH_SIZE`, `MAX_NEW_TOKENS` 환경변수로 빠르게 조정할 수 있습니다.

### 실행 팁
1. (선택) `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`, `PER_DEVICE_BATCH`, `EFFECTIVE_BATCH_SIZE` 등 환경변수를 `conda activate vqa` 후 `export`로 지정해 필요 시 VRAM에 맞게 조정합니다.  
2. 노트북에서 학습 셀을 실행하면 약 8시간 예산 내에서 자동으로 최고 정확도 체크포인트를 저장하고, 결과는 `WORK_DIR/qwen25vl_lora`에 기록됩니다.  
3. 추론 전에는 `run_inference` 앞에서 `MAX_NEW_TOKENS`을 상황에 맞게 늘리거나 줄여 제출 파일 생성에 활용하세요.
