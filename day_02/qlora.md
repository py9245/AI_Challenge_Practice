좋아, “LLM + 이미지 모델” 파인튜닝에서 바로 써먹는 핵심만 콤팩트하게 정리할게. 대회 환경(Colab T4 / AWS g4dn.xlarge) 기준의 **즉시 적용 값 + 코드 스니펫** 위주야.

---

# 0) 한 장 요약 (바로 적용)

* **LLM**: QLoRA(+Unsloth)로 4bit 로드 → LoRA(r=8~16, α=16~32, lora_dropout=0.1~0.3)만 학습. `q_proj,v_proj`(+여유되면 `k_proj,o_proj,gate,up,down`) 타깃. LR=1e-4, β2=0.95, wd=0.0, seq packing 켜기.
* **Vision**: `timm` 사전학습 백본(EfficientNet/ConvNeXt/DeiT-S). Augment(ResizeCrop+RandAug+Mixup/CutMix+LabelSmoothing) 기본. wd=0.05, cosine 스케줄, warmup 5%.
* **오버피팅**: 이미지=Aug 강도↑, Mixup/CutMix, Label smoothing, Stochastic Depth; LLM=LoRA rank↓, lora_dropout↑, 학습스텝↓, 템플릿/포맷 통일.
* **언더피팅**: 이미지=해상도↑, 에폭↑, rank/capacity↑, Mixup/CutMix 약화; LLM=rank↑, target_modules 추가, 일부 레이어 언프리즈(극약), 더 많은/깨끗한 데이터.
* **양자화**: LLM은 **nf4 + double quant**. 이미지 추론은 PTQ(INT8)로 지연 없이 이득, 학습엔 AMP(bf16) 우선.
* **정규화(regularization)**: 이미지=wd/label smoothing/mixup-cutmix/stochastic depth/EMA. LLM=LoRA dropout/gradient clip(1.0)/wd=0.
* **EDA**: 텍스트=토큰 길이·언어·중복·포맷 누수, 이미지=클래스 불균형·해상도/종횡비·객체 크기 분포·누락/손상 샘플.

---

## 1) 오버피팅/언더피팅 빠른 진단 → 즉시 처방

### 공통 지표

* **Train↓, Val↑**: 오버피팅
* **Train↑, Val↑ 둘다 높음**: 언더피팅/학습 불안정
* **LLM**: token-level loss/perplexity, task 별 BLEU/ROUGE/EM.
* **Vision**: Acc/F1, mAP, AUC-PR(불균형), 학습곡선·Confusion Matrix 체크.

### 처방(LLM)

* 오버피팅: `lora_dropout 0.2→0.3`, `rank 16→8`, 스텝/에폭↓, Prompt 템플릿 단일화, 학습셋 중복 제거.
* 언더피팅: `rank 8→16/32`, target_modules에 `k/o/up/down/gate` 추가, LR 살짝↑(1e-4→2e-4), max_steps↑, **packing** 켜기, 데이터 질/양 보강.

### 처방(Vision)

* 오버피팅: Mixup 0.2~0.4, CutMix 1.0, LabelSmoothing 0.1, StochasticDepth 0.1~0.2, Aug 강도↑, EarlyStopping.
* 언더피팅: 해상도 224→320/384, 에폭 30→100, 모델 S→B, Mixup/CutMix 약화, wd 완화(0.05→0.02), LR 스케줄 완만하게.

---

## 2) LLM 파인튜닝(QLoRA) 실전 템플릿

```python
# pip: unsloth, transformers, peft, bitsandbytes, accelerate, datasets
from unsloth import FastLanguageModel
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

base, tok = FastLanguageModel.from_pretrained(
    "unsloth/phi-3-mini-4k-instruct-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing=True,  # 메모리 절감
)

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(base, peft_cfg)

def format_sft(ex):
    # (인스트럭션/응답) 포맷을 '항상 같은 템플릿'으로
    sys = "You are a helpful assistant."
    prompt = f"<s>[SYSTEM]{sys}\n[USER]{ex['instruction']}\n[ASSISTANT]"
    ids = tok(prompt, add_special_tokens=False)
    labels = tok(ex["output"] + tok.eos_token, add_special_tokens=False)
    return {"input_ids": ids["input_ids"]+labels["input_ids"],
            "attention_mask": [1]* (len(ids["input_ids"])+len(labels["input_ids"]))}

collator = DataCollatorForLanguageModeling(tok, mlm=False)
args = TrainingArguments(
    per_device_train_batch_size=1, gradient_accumulation_steps=8,  # T4 안전
    learning_rate=1e-4, num_train_epochs=2, warmup_ratio=0.03,
    lr_scheduler_type="cosine", weight_decay=0.0, bf16=True,
    logging_steps=20, save_strategy="epoch", evaluation_strategy="epoch",
    gradient_checkpointing=True, max_grad_norm=1.0
)
# Trainer(dataset, compute_metrics=…) 로 학습
```

**양자화 포인트**

* 4bit **nf4** + `bnb_4bit_use_double_quant=True` 추천.
* 여기선 base가 이미 4bit. Full-precision로 되돌려 학습하지 않도록(메모리 폭발 방지).

---

## 3) Vision 분류/검출 기본기 (timm + 강력 정규화)

```python
# pip: timm==0.9+, torchmetrics, albumentations(optional)
import timm, torch, torch.nn as nn
from timm.data import create_transform
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler

model = timm.create_model("convnext_tiny.fb_in22k", pretrained=True, num_classes=NUM_CLS)
train_tf = create_transform(
    input_size=224, is_training=True, auto_augment="rand-m9-mstd0.5-inc1",
    mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), re_prob=0.25, re_mode='pixel'
)
val_tf   = create_transform(input_size=224, is_training=False)

criterion = timm.loss.CrossEntropyLossBinary(label_smoothing=0.1) if NUM_CLS==1 \
            else timm.loss.LabelSmoothingCrossEntropy(smoothing=0.1)

optimizer = create_optimizer_v2(model, opt='adamw', lr=5e-4, weight_decay=0.05)
sched, _ = create_scheduler(optimizer, t_initial=100*iters_per_epoch, warmup_t=5*iters_per_epoch,
                            sched='cosine')

# Mixup/CutMix
from timm.data import Mixup
mixup_fn = Mixup(prob=0.3, switch_prob=0.0, mode='batch',
                 mixup_alpha=0.2, cutmix_alpha=1.0, label_smoothing=0.1, num_classes=NUM_CLS)

# Stochastic Depth(이미 모델에 포함), AMP
scaler = torch.cuda.amp.GradScaler()
```

**검출**은 시간 대비 효율로 **YOLOv8/10s** 또는 **RT-DETR** 소형 권장(사전학습 가중치 사용). 불균형이면 `cls_pos_weight`/focal loss 고려.

---

## 4) 정규화(Regularization) 체크리스트

* **LLM**

  * LoRA: `r`(용량), `lora_dropout`(0.1~0.3), target_modules(확장/축소)로 규제 강도 조절
  * Gradient clipping=1.0, wd=0, β2=0.95, cosine 스케줄 + 짧은 warmup(1~3%)
  * Prompt 포맷 통일(누수 방지), 길이 패킹(효율↑, 과적합↓)
* **Vision**

  * **Label Smoothing 0.1**, **Mixup 0.2 / CutMix 1.0**, **RandAug**, **RandomErasing 0.25**
  * **Weight Decay 0.05**, **Stochastic Depth 0.1~0.2**, **EMA(0.9999)**(대회 막판에 좋음)
  * Cosine + warmup 5%, AMP(bf16/FP16)

---

## 5) 양자화(Quantization) 실전

### LLM (QLoRA)

* 메모리 절감: 4bit nf4 + double quant, gradient checkpointing.
* 성능 저하 시: `nf4→fp4` 비교, LoRA rank↑, 일부 레이어 언프리즈 소량(주의).
* 추론은 `load_in_4bit=True` 그대로. **INT8**은 메모리↑, 속도는 상황 따라.

### Vision

* **추론용 PTQ(INT8)**: 지연 거의 없이 5~15% 속도 개선(모델/하드웨어 의존).
* PyTorch 예시:

```python
import torch
from torch.ao.quantization import get_default_qconfig, prepare_fx, convert_fx
model.eval()
qconfig = get_default_qconfig("fbgemm")
prepared = prepare_fx(model, {"": qconfig})
with torch.no_grad():
    for x,_ in calib_loader: prepared(x.cuda())  # 짧게 캘리브레이션
quantized = convert_fx(prepared)
```

* 학습시엔 **AMP**가 보통 이득이 더 큼. QAT는 시간 넉넉할 때만.

---

## 6) EDA 체크리스트 & 스니펫

### 텍스트(Instruction Tuning)

* **길이**: 토큰 길이 분포(평균/95%/최대), 극단치 잘라내기(or truncation).
* **중복/누수**: 같은 입력-출력 중복, 테스트와의 중복 키워드.
* **언어/포맷**: 다국어 섞임 여부, 불량 포맷(미닫힘 토큰, role 태그 누락).
* **레이블 품질**: 반말/존댓말 혼재, 동일 질문 다중 정답 충돌.

```python
from transformers import AutoTokenizer
import json, numpy as np
tok = AutoTokenizer.from_pretrained("unsloth/phi-3-mini-4k-instruct-bnb-4bit")
lens=[]
with open("train.jsonl","r",encoding="utf-8") as f:
    for line in f:
        ex=json.loads(line)
        s=f"<s>[SYSTEM]You are helpful.\n[USER]{ex['instruction']}\n[ASSISTANT]{ex['output']}</s>"
        lens.append(len(tok(s, add_special_tokens=False)["input_ids"]))
print(np.percentile(lens,[50,90,95,99]), max(lens))
```

### 이미지

* **클래스 불균형**: 분포 히스토그램 → WeightedSampler/pos_weight.
* **해상도/종횡비**: Resize 전략 결정.
* **손상/잘못된 라벨**: 샘플링해서 사람 눈으로 점검(최소 50~100장).
* **객체 크기 분포(검출)**: 작은 객체 많으면 RandomCrop 지양, 멀티스케일 학습.

```python
import os, PIL.Image as Image, numpy as np, collections
root="train_images"
sizes, counts = [], collections.Counter()
for cls in os.listdir(root):
    for fn in os.listdir(os.path.join(root,cls)):
        p=os.path.join(root,cls,fn)
        try:
            w,h = Image.open(p).size
            sizes.append((w,h)); counts[cls]+=1
        except: pass
print("class dist:", counts)
wh = np.array(sizes); print("W,H p95:", np.percentile(wh,95,axis=0))
```

---

## 7) 하드웨어별 안전 하이퍼파라미터(시간 대비 효율)

* **Colab/Kaggle T4 16GB (LLM QLoRA)**

  * seq_len 1024, per_device_batch=1, grad_accum=8~16, r=16, steps 1~3e4 이내
  * bf16=True, gc=True, packing=True
* **g4dn.xlarge (T4, Vision 224)**

  * batch 64~128(AMP), wd=0.05, LR=5e-4, epochs=50~100, cosine+warmup 5%
  * Mixup 0.2, CutMix 1.0, smoothing 0.1, SD 0.1

---

## 8) 로그/얼리스탑/체크포인트

* **로그**: train/val loss, Top-1/F1(or BLEU/ROUGE/EM), 학습 시간/스텝당 토큰/이미지, GPU 메모리.
* **EarlyStopping**: patience 2~3(대회는 짧게). EMA 모델도 저장.
* **체크포인트**: 마지막 3개만 보관(용량 관리). LLM은 **LoRA 어댑터만 저장**.

---

## 9) 실패 패턴 → 디버깅 순서

1. 데이터 누락/포맷 불량 → 10샘플 눈검
2. 학습 곡선 비정상 → LR 10배 스윕 또는 warmup 증가
3. 메모리 OOM(LLM) → seq_len↓, accum↑, rank↓, target_modules 축소
4. 성능 정체 → 데이터 클린업/하드 라벨 재검토, 템플릿 일관화, rank↑

---

## 10) “가장 효율적인 선택” (대회용 추천)

* **LLM 지시튜닝**: Unsloth QLoRA + Phi-3/Llama-3-8B-instruct-4bit 중 하나, r=16, 1~2 epoch, 포맷 통일/패킹으로 스텝 효율↑.
* **이미지 분류**: `convnext_tiny` or `vit_small_patch16` + Mixup/CutMix + smoothing + cosine. 부족하면 해상도만 올리기.
* **검출**: YOLOv8s/RT-DETR-s 사전학습 가중치 + 데이터 정리(라벨 품질) > 어떤 하이퍼 트릭보다 큼.
* **시간이 진짜 없을 때**: LLM은 **어댑터 장착만**(이미 학습된 LoRA), Vision은 **EMA + TTA**로 막판 점수 1~2% 끌어올리기.

원하면 위 템플릿을 너의 데이터 구조에 맞춰 **10개 셀짜리 주피터 베이스라인**으로 바로 뽑아줄게. (LLM/이미지 각각)
