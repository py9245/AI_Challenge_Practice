# G5DN_last.ipynb 사용 가이드

## 기본 설정
- GPU: AWS g5dn (Tesla T4 24GB x1)
- 모델: Qwen/Qwen2.5-VL-7B-Instruct, 4bit QLoRA + LoRA(r=16)
- 기본 시간 예산: 210분 (`TARGET_TRAIN_MINUTES=210`)
- 기본 배치: `per_device_train_batch_size=2`, `gradient_accumulation_steps=8`

## 주요 특징
- T4 VRAM에 맞춘 작은 per-device 배치와 자동 grad accumulation.
- 밝기/색감/좌우반전/소회전 증강 + 다중 밝기 앙상블(레벨 2개로 자동 제한)으로 일반화 유지.
- `StepTimerCallback`이 최근 step 시간을 보고하여 필요한 경우 `EST_STEP_TIME_SEC` 조정 가능.
- FAST 모드 (`FAST_MODE=1`)로 12샘플만 사용해 빠른 검증 후 전체 학습으로 전환.

## 권장 환경변수 예시
```python
os.environ.update({
    "PER_DEVICE_BATCH": "2",
    "TARGET_EFFECTIVE_BATCH": "16",
    "TARGET_TRAIN_MINUTES": "210",
    "EST_STEP_TIME_SEC": "3.5",
    "ENSEMBLE_BRIGHTNESS_LEVELS": "1.0,1.15,1.3",
})
```

## 실행 순서
1. 환경/데이터 준비 셀 실행 (HF 토큰은 노트북 내부 값 사용).
2. 학습 셀(#25) 실행 → 필요 시 `FAST_MODE` 또는 배치 관련 환경변수 조정.
3. 검증 셀(#29)에서 정확도 확인.
4. 제출 셀(#31)로 `submission.csv` 생성.

> 이 노트북도 HF 토큰이 하드코딩되어 있으니 외부 배포 전에는 반드시 제거하거나 무효화하세요.
