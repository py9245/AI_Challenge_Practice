# A100_last.ipynb 사용 가이드

## 기본 설정
- GPU: NVIDIA A100 40GB (단일)
- 모델: Qwen/Qwen2.5-VL-7B-Instruct, 4bit QLoRA + LoRA(r=16)
- 기본 시간 예산: 180분 (`TARGET_TRAIN_MINUTES=180`)
- 학습률: `7.5e-6`, `cosine` 스케줄러, `warmup_ratio=0.05`

## 주요 하이라이트
- 시간 예산 기반으로 `max_steps`를 자동 산정하여 3시간 이내 완료.
- 밝기/색감 증강(ColorJitter, flip, rotation) + 다중 밝기 앙상블(레벨 자동 클램프)로 일반화 강화.
- FAST 모드(`FAST_MODE=1`)로 10샘플만 빠르게 검증한 뒤, 전체 학습을 진행.
- `StepTimerCallback`으로 최근 step 시간 평균을 출력해 시간 추정을 갱신 가능.

## 권장 환경변수
```python
os.environ.update({
    "PER_DEVICE_BATCH": "8",
    "TARGET_EFFECTIVE_BATCH": "32",
    "TARGET_TRAIN_MINUTES": "180",
    "EST_STEP_TIME_SEC": "2.2",
    "ENSEMBLE_BRIGHTNESS_LEVELS": "1.0,1.2,1.35",
})
```

## 실행 순서
1. 환경/데이터 준비 셀 실행
2. 학습 셀(#25) 실행 (필요 시 `FAST_MODE` 조정)
3. 검증 셀(#29)로 정확도 확인
4. 제출 셀(#31)로 `submission.csv` 생성

> 노트북에는 Kaggle 환경을 기준으로 HF 토큰이 하드코딩돼 있으므로, 외부에 공유하기 전에 반드시 제거하거나 무효화하세요.
