# 2025-ssafy-14 Kaggle 실행 가이드 (13개 섹션)

## 1. 작업 공간 인벤토리
- 루트 파일: 데이터셋 압축(`2025-ssafy-14.zip`), 추출된 폴더, 보조 PNG, 자리표시자 문서(`info.md`, `plan.md`).
- 핵심 리소스는 `2025-ssafy-14/` 내부에 존재하며 `train/`, `test/`, CSV 메타데이터, 참고용 `251023_Baseline.ipynb`가 포함됩니다.
- **사용 모듈**: `os`, `pathlib` 등 표준 라이브러리로 디렉터리 구조를 검사했습니다.

## 2. Kaggle 하드웨어 및 런타임 준비
- 대상 환경: 듀얼 NVIDIA T4 GPU, 메모리 30GB, `/kaggle/working` 최대 73GB.
- 단일 셀에서 python 패키지( `transformers`, `accelerate`, `einops`, `tiktoken`, `huggingface_hub`, `qwen-vl-utils`)와 한글 폰트를 위해 `apt-get install fonts-nanum`을 함께 수행합니다.
- `torch`로 CUDA 사용 가능 여부, 장치 개수, GPU 메모리를 확인해 모델 로딩 전 리소스를 점검합니다.
- **사용 모듈/도구**: `torch`, `pip`, `apt-get`, `fc-cache`.

## 3. 임포트 및 전역 설정
- 주요 라이브러리를 한 곳에서 불러오고, 랜덤 시드 및 시각화 규칙을 설정합니다.
- `matplotlib.font_manager`를 이용해 새로 설치된 나눔 글꼴을 등록하고, 사용 가능한 한글 글꼴(`NanumGothic` 등)을 자동으로 선택해 그래프 한글 깨짐을 방지합니다.
- Hugging Face 토큰은 Kaggle Secrets의 `HF_TOKEN` 환경변수에서 읽어옵니다.
- **사용 모듈**: `matplotlib`, `font_manager`, `pandas`, `numpy`, `torch`, `huggingface_hub`.

## 4. 압축 해제 전략
- 노트북이 `/kaggle/input`에서 `2025-ssafy-14.zip`을 찾아 `/kaggle/working/2025-ssafy-14/`로 자동 해제합니다.
- 이미 압축이 풀린 상태를 대비한 분기 로직으로 재사용성을 높였습니다.
- **사용 모듈**: 표준 라이브러리 `zipfile`.

## 5. 메타데이터 불러오기와 구조 확인
- `pandas`로 `train.csv`, `test.csv`, `sample_submission.csv`를 읽고, 절대 경로의 `image_path` 컬럼을 추가합니다.
- 훈련/테스트 각각 3,887개, 객관식 4지선다 구조임을 확인했습니다.
- **사용 모듈**: `pandas`(데이터프레임 조작, 미리보기 출력).

## 6. 데이터 무결성 점검
- 정답 분포(`a`:964, `b`:958, `c`:960, `d`:1005), 중복 ID, 결측치를 확인해 문제 없음을 검증했습니다.
- 무작위 256장 샘플로 이미지 폭 185~720px, 높이 225~720px(평균 약 588x658)을 파악했습니다.
- **사용 모듈**: `numpy`(통계), `Pillow`(이미지 크기 조사).

## 7. 시각적 샘플 검토
- `matplotlib` 격자로 임의의 학습 이미지를 띄워 조명, 구도 등을 수동 확인했습니다.
- 텍스트 출력으로 한국어 질문과 보기까지 함께 검토할 수 있습니다.
- **사용 모듈**: `matplotlib`, `Pillow`.

## 8. 검증 분할 설계
- `train_test_split`으로 90/10 계층형 검증 세트를 만들었습니다(약 389건).
- 인덱스를 리셋해 `.to_dict("records")` 변환 시 깔끔한 레코드를 유지합니다.
- **사용 모듈**: `scikit-learn`.

## 9. 프롬프트 템플릿 구성
- 시스템 메시지, 사용자 프롬프트, 유틸 함수를 정의해 JSON 형태의 응답(`{"answer": "a"}`)을 강제합니다.
- `qwen-vl-utils`는 추후 비전 전처리 확장 시 활용할 준비가 되어 있습니다.
- **사용 모듈**: `json`, `re`, `qwen-vl-utils`.

## 10. Qwen2.5-VL-7B 로딩
- `huggingface_hub` 로그인(`HF_TOKEN`) 후 `AutoProcessor`, `AutoModelForVision2Seq`를 `torch.float16`, `device_map="auto"` 설정으로 불러옵니다.
- FlashAttention을 우선 시도하고 실패하면 기본 어텐션으로 자동 전환합니다.
- **사용 모듈**: `huggingface_hub`, `transformers`, `accelerate`, `einops`, `tiktoken`, `torch`.

## 11. 추론 도우미 스택
- `tqdm` 진행률 표시와 함께 배치 추론을 실행하고, 정규식으로 JSON에서 정답 문자를 파싱합니다.
- 예외 시 기본값 `"a"`로 되돌리도록 안전장치를 두었습니다.
- **사용 모듈**: `torch`, `transformers`, `tqdm`, `Pillow`.

## 12. 검증 프로토콜
- 최대 128개의 계층 샘플에 대해 예측을 수행하고, 원문 응답과 정오표를 함께 저장합니다.
- 프롬프트 안정화 후 전체 검증 세트로 확장할 수 있습니다.
- **사용 모듈**: `pandas`, `numpy`, `tqdm`.

## 13. 제출 파일 작성과 인계
- 테스트 전체에 대해 추론한 뒤 `/kaggle/working/submission.csv`로 저장합니다.
- 추가 로그가 필요하면 `artifacts/` 디렉터리를 활용할 수 있도록 설계되어 있습니다.
- **사용 모듈**: `pandas`, `torch`, `transformers`.
