## Fast vs Extended Fine-Tuning Profiles

### notebookf585393d8a_ver3_1_1.ipynb (약 10분 타깃)
- 학습 데이터: `FAST_TRAIN_LIMIT`(기본 768)으로 클래스별 균등 샘플링하여 경량화, 검증도 `FAST_EVAL_SIZE`(기본 128)으로 축소.
- LoRA: 기본 `r=8 / alpha=16 / dropout=0.1` 경량 설정, 환경변수 `LORA_*`로 조정 가능.
- 시퀀스 길이: `MAX_SEQUENCE_LENGTH` 기본 768.
- TrainingArguments: `EPOCHS=1`, `LEARNING_RATE=3e-4`, `FAST_MAX_STEPS=120`, `evaluation_strategy='steps'`, `save_strategy='no'` 등으로 짧은 시간에 수렴 유도. 출력 디렉터리 `runs/qwen25vl_lora_fast`.
- 활용 팁: 더 빠르게 돌리고 싶으면 `FAST_MAX_STEPS`/`FAST_TRAIN_LIMIT`를 줄이고, 성능이 필요하면 값을 키우거나 `LEARNING_RATE`를 낮춰 안정성을 확보하세요.

### notebookf585393d8a_ver3_1_2.ipynb (약 4시간 타깃)
- 학습 데이터: 전체 train split 사용, 검증은 `VAL_EVAL_SIZE` 기본 256개 층화 샘플.
- LoRA: 기본 `r=32 / alpha=64 / dropout=0.05` 고용량 설정으로 표현력 강화.
- TrainingArguments: `EPOCHS=4`, `EFFECTIVE_BATCH_SIZE=24`, `predict_with_generate=True`, `load_best_model_at_end=True`, `metric_for_best_model='accuracy'`, `EarlyStoppingCallback(patience=2)` 포함. 출력 디렉터리 `runs/qwen25vl_lora_extended`.
- 평가 지표: 생성 결과를 JSON/문자 패턴으로 파싱해 정확도를 산출, 최적 체크포인트를 자동 보존.
- 활용 팁: 더 긴 학습이 가능하면 `NUM_EPOCHS`/`MAX_TRAIN_STEPS`를 늘리고, 더 작은 GPU에서는 `PER_DEVICE_BATCH`를 낮추면서 같은 Effective Batch를 유지하도록 `EFFECTIVE_BATCH_SIZE`를 조정하세요.
