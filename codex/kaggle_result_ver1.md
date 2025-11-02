# Qwen2.5-VL-7B Kaggle Pipeline 핵심 포인트

1. **경량화된 Qwen2.5-VL-7B 적재와 LoRA 적용**  
   BitsAndBytes 4비트 NF4 양자화(`load_in_4bit=True`, `bnb_4bit_use_double_quant=True`, `bnb_4bit_compute_dtype=torch.float16`)로 기본 모델을 불러오고, `q_proj`·`k_proj`·`v_proj`·`o_proj` 등 핵심 선형 계층에 LoRA(r=16, alpha=32, dropout=0.05)를 연결해 T4 듀얼 GPU에서도 미세조정이 가능하도록 구성했습니다.

2. **한국어 멀티모달 프롬프트와 손실 마스킹 Collator**  
   한국어 시스템 메시지와 선택지 템플릿을 `build_messages`로 생성하고, `VqaFineTuneDataset`/`fine_tune_collate_fn`이 이미지 토큰과 JSON 정답을 포함한 대화를 `processor.apply_chat_template`로 인코딩합니다. 프롬프트 토큰 위치만 `labels=-100`으로 마스킹해 학습이 정답 JSON에만 집중되도록 했습니다.

3. **메모리 최적화된 검증·추론 파이프라인**  
   검증은 배치 1, `max_new_tokens` 축소, `use_cache=False` 등으로 VRAM을 제어하며 진행하고, `run_inference`는 온전한 JSON 응답을 강제(`decode_answer_text`)해 Kaggle 제출 파일을 즉시 생성할 수 있게 합니다. Left padding 유지와 이미지+텍스트 동시 인코딩으로 추론 일관성을 확보했습니다.

