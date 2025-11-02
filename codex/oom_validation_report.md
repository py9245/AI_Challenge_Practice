## notebookf585393d8a_ver3.ipynb OOM 및 전체 데이터 처리 검증

### 환경 가정
- AWS g5.xlarge (NVIDIA A10G 24 GB VRAM, 시스템 RAM 16 GB) 기준으로 분석했습니다.
- 노트북은 `BitsAndBytes` 4-bit 로딩 + LoRA 미세조정 구조를 사용하며, `gradient_checkpointing`과 `device_map="auto"`가 활성화되어 있습니다.

### 학습 경로 점검
- `train_test_split(train_df, test_size=0.1, stratify=answer)` → 학습 데이터 3,498건(총 3,887 × 0.9)을 `VqaFineTuneDataset`으로 전부 담습니다.  
  → `Trainer`가 기본 `RandomSampler`를 사용하므로 **세 epoch 동안 학습 데이터 전부가 반복**됩니다.
- 배치 설정  
  - GPU 메모리 ≥ 22 GB → `PER_DEVICE_BATCH=2`  
  - `TARGET_EFFECTIVE_BATCH=16` → `gradient_accumulation_steps=8`  
  - 1 epoch 최댓값 ≈ 3,498 / 16 ≈ 219 optimizer step → 3 epoch ≈ 657 step  
  - 각 스텝은 4-bit + 체크포인팅으로 24 GB VRAM 기준 안정 구간에 해당합니다.
- 추가 안전장치  
  - `max_grad_norm=0.3`, `weight_decay=0.01`, `warmup_ratio=0.1` 등 안정적 하이퍼파라미터  
  - `EarlyStoppingCallback(patience=1)` 활성화 → 검증 정확도가 향상되지 않으면 빠르게 중단  
  - `load_best_model_at_end=True` → 가장 높은 정확도 체크포인트 자동 저장

### 검증 & 평가
- 검증 세트는 층화 유지 상태로 **최대 256개 샘플**을 고정 선택 (`per_device_eval_batch_size=2`) → GPU 점유율 완만.  
- `predict_with_generate=True`로 평가 시에도 LoRA 가중치가 유지되며, `compute_metrics`가 생성된 텍스트를 파싱해 정확도를 계산합니다.

### 테스트 추론
- `run_inference`는 VRAM 용량에 따라 `PRED_BATCH_SIZE`를 자동 선택 (24 GB → 6).  
- `MAX_NEW_TOKENS=16`, `temperature=0.0`, `do_sample=False`로 짧은 답변만 생성 → batch 6 기준 VRAM 약 15 ~ 17 GB 내외 예상.  
- 전체 3,887건을 순회하며 결과를 리스트에 누적 후 제출 파일을 생성하므로 **테스트 전량 평가가 보장**됩니다.

### OOM 가능성 평가
- 4-bit 양자화 + LoRA + 배치 크기 2 + 체크포인팅 조합은 A10G 24 GB에서 일반적으로 여유가 있습니다.  
- 이미지 해상도가 매우 높은 샘플이 다수거나 `MAX_SEQUENCE_LENGTH=1024`에서 시각 토큰이 과도하게 늘어나면 VRAM이 23 GB 이상까지 치솟을 수 있습니다.  
  - 징후: `CUDA out of memory` 메시지.  
  - 대응: `os.environ`으로 `PER_DEVICE_BATCH=1`, `EFFECTIVE_BATCH_SIZE=12`, 또는 `LORA_R=24` 등으로 즉시 조정 가능.
- 시스템 RAM 16 GB 기준으로도 데이터로더는 큰 텐서를 보존하지 않고, PIL 이미지 로딩 후 즉시 텐서로 변환하므로 스왑 위험이 낮습니다.

### 재실행 체크리스트 (다른 환경에서 실행 시)
1. `conda activate vqa` → 필요 시 `pip install -r requirements.txt`로 의존성 재설치.  
2. 환경 변수로 VRAM에 맞춘 설정 확인  
   ```bash
   export PER_DEVICE_BATCH=2       # 24 GB 기준
   export EFFECTIVE_BATCH_SIZE=16  # 총 누적 배치
   export LORA_R=32 LORA_ALPHA=64 LORA_DROPOUT=0.08
   ```
3. 학습 셀 실행 후 `Training metrics`, `Eval metrics` 로그가 출력되고, `qwen25vl_lora/lora_adapter` 폴더에 LoRA 가중치가 생성되는지 확인.  
4. 추론 셀은 `Saved submission to .../submission.csv` 메시지와 함께 전체 테스트 평가가 완료되면 성공.

### 결론
- 현재 설정은 g5.xlarge(A10G 24 GB) 기준으로 **OOM 위험이 낮고, 학습/추론이 전체 데이터에 대해 수행되도록 구성**되어 있습니다.  
- 만약 더 작은 GPU 환경에서 실행한다면 `PER_DEVICE_BATCH`(또는 `PRED_BATCH_SIZE`)를 1로 줄이고, 필요 시 `LORA_*`와 `MAX_NEW_TOKENS`를 낮춰 VRAM을 확보하면 안정적으로 재현할 수 있습니다.
