# SSAFY AI 해커톤 준비과정 수기 작성 8일간 데일리 작성

## day_01(2025/10/22) :
- aws경험 전무 -> s3 버킷 생성, ec2 개발환경 구축 및 주피터 lab 구축완료, 간단한 딥러닝 테스트 완료
- kaggle경험 전무 -> kaggle 노트북 생성 및 데이터 셋 다운로드 주피터 노트북에 대한 이해
- AI 측면 -> cat dog 분류 완료 주피터 환경을 통해 작업 링크 : https://www.kaggle.com/code/yusin18/notebookcc4fca893b
- **과적합 발생, gpu노는 현상 발견 해결!!**

### 워크플로우 (10단계) - chatgpt-5
1. 환경/설정 → 시드·경로·하드웨어(T4×2, AMP).
2. 데이터 인덱싱 → 파일 리스트업, 파일명으로 라벨 생성, 중복/깨진파일 스킵.
3. EDA → 클래스 분포, 샘플 보기, 해상도 분포, 채널 통계(샘플링).
4. 전처리/증강 → RandomResizedCrop/AutoAugment(+ImageNet 정규화).
5. 커스텀 Dataset/Dataloader 구성.
6. 모델 빌더 → timm의 EfficientNet/ConvNeXt(가볍고 정확도↑).
7. 학습 유틸 → 손실/지표(Acc/F1), AMP, EarlyStopping, Cosine 스케줄러.
8. 5-Fold StratifiedCV 학습(각 fold 80/20), fold별 best ckpt 저장.
9. 테스트셋 평가(2,000장), 리포트/혼동행렬/샘플 예측.
10. 결과 취합/평균 점수 및 베스트 모델 저장.
