import json
from pathlib import Path

path = Path('notebookf585393d8a.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))


def set_cell(idx, code):
    nb['cells'][idx]['source'] = [(line + '\n') for line in code.rstrip().split('\n')]

cell6_code = '''import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import torch
from huggingface_hub import login
from transformers import AutoModelForVision2Seq, AutoProcessor

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams["axes.grid"] = False
pd.set_option("display.max_columns", None)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

INPUT_DIR = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")
ARTIFACT_DIR = WORK_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# ===== Hardware-aware Qwen2.5-VL-7B defaults for dual T4 16GB =====
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", 384))
SAVE_DIR = WORK_DIR / os.environ.get("SAVE_SUBDIR", "qwen25vl_lora_7b_full")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS_DEFAULT = float(os.environ.get("EPOCHS_DEFAULT", 3))
TRAIN_BATCH_T4 = int(os.environ.get("TRAIN_BATCH_T4", 1))   # per-device batch size
GRAD_ACCUM_T4 = int(os.environ.get("GRAD_ACCUM_T4", 12))    # accumulates to 24 samples with 2 GPUs
NUM_WORKERS_T4 = int(os.environ.get("NUM_WORKERS_T4", 4))
PRED_BATCH_T4 = int(os.environ.get("PRED_BATCH_T4", 2))
LEARNING_RATE_DEFAULT = float(os.environ.get("LEARNING_RATE_DEFAULT", 2.0e-5))
WARMUP_RATIO_DEFAULT = float(os.environ.get("WARMUP_RATIO_DEFAULT", 0.08))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("DEFAULT_MAX_NEW_TOKENS", 16))
VALIDATION_SAMPLE_SIZE = int(os.environ.get("VALIDATION_SAMPLE_SIZE", 256))
VALIDATION_EVAL_SIZE = int(os.environ.get("VALIDATION_EVAL_SIZE", 128))
TARGET_TRAIN_HOURS = float(os.environ.get("TARGET_TRAIN_HOURS", 5.0))
EST_STEP_TIME_SEC = float(os.environ.get("EST_STEP_TIME_SEC", 45.0))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", 10))
TRAIN_SAMPLE_LIMIT = int(os.environ.get("TRAIN_SAMPLE_LIMIT", 0))
TRAIN_SAMPLE_FRAC = float(os.environ.get("TRAIN_SAMPLE_FRAC", 1.0))
VAL_SAMPLE_LIMIT = int(os.environ.get("VAL_SAMPLE_LIMIT", 0))
VAL_SAMPLE_FRAC = float(os.environ.get("VAL_SAMPLE_FRAC", 1.0))

# ==============================
# Hugging Face Token Load
# ==============================
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("[OK] HF_TOKEN successfully loaded.")
else:
    print("[WARN] HF_TOKEN not found. Add it via Kaggle Secrets before downloading gated models.")

print(
    f"[Config] model={MODEL_ID} | image_size={IMAGE_SIZE} | per_device_batch={TRAIN_BATCH_T4} | "
    f"grad_accum={GRAD_ACCUM_T4} | lr={LEARNING_RATE_DEFAULT} | warmup_ratio={WARMUP_RATIO_DEFAULT} | "
    f"max_new_tokens={DEFAULT_MAX_NEW_TOKENS} | validation_sample={VALIDATION_SAMPLE_SIZE} | "
    f"train_limit={TRAIN_SAMPLE_LIMIT or 'all'} | train_frac={TRAIN_SAMPLE_FRAC}"
)
'''
set_cell(6, cell6_code)

cell16_code = '''train_split, val_split = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["answer"],
    random_state=SEED,
)

train_split = train_split.reset_index(drop=True)
val_split = val_split.reset_index(drop=True)

if 0 < TRAIN_SAMPLE_FRAC < 1.0:
    train_split = train_split.sample(frac=TRAIN_SAMPLE_FRAC, random_state=SEED)
if TRAIN_SAMPLE_LIMIT > 0 and len(train_split) > TRAIN_SAMPLE_LIMIT:
    train_split = train_split.sample(n=TRAIN_SAMPLE_LIMIT, random_state=SEED)
train_split = train_split.reset_index(drop=True)

if 0 < VAL_SAMPLE_FRAC < 1.0:
    val_split = val_split.sample(frac=VAL_SAMPLE_FRAC, random_state=SEED)
if VAL_SAMPLE_LIMIT > 0 and len(val_split) > VAL_SAMPLE_LIMIT:
    val_split = val_split.sample(n=VAL_SAMPLE_LIMIT, random_state=SEED)
val_split = val_split.reset_index(drop=True)

print(f"Training rows: {len(train_split)}")
print(f"Validation rows: {len(val_split)}")
print("Validation label balance:")
display(val_split["answer"].value_counts().sort_index())
'''
set_cell(16, cell16_code)

cell22_code = '''from torch.utils.data import Dataset
import json
import torch

MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQUENCE_LENGTH", 1024))

class VqaFineTuneDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "image_path": row["image_path"],
            "messages": build_messages(row),
            "answer": str(row["answer"]).lower(),
        }

def _build_conversation(sample):
    conversation = sample["messages"] + [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": json.dumps({"answer": sample["answer"]})}],
        }
    ]
    prompt_text = processor.apply_chat_template(
        sample["messages"], tokenize=False, add_generation_prompt=True
    )
    chat_text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    return prompt_text, chat_text, sample["image_path"]

if hasattr(processor, "tokenizer"):
    processor.tokenizer.padding_side = "left"

def fine_tune_collate_fn(batch):
    images, prompts, full_texts = [], [], []

    for sample in batch:
        prompt_text, chat_text, image_path = _build_conversation(sample)
        images.append(load_image(image_path))
        prompts.append(prompt_text)
        full_texts.append(chat_text)

    enc_prompt = processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )
    enc_full = processor(
        images=images,
        text=full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )

    labels = enc_full["input_ids"].clone()
    labels[enc_full["attention_mask"] == 0] = -100

    for i in range(labels.size(0)):
        full_mask = enc_full["attention_mask"][i].bool()
        prompt_len = int(enc_prompt["attention_mask"][i].sum().item())
        nonpad_idx = torch.nonzero(full_mask, as_tuple=False).squeeze(-1)
        prompt_idx = nonpad_idx[:prompt_len]
        labels[i, prompt_idx] = -100

    enc_full["labels"] = labels
    return enc_full

train_dataset = VqaFineTuneDataset(train_split)
if len(val_split) <= VALIDATION_EVAL_SIZE:
    val_eval_df = val_split.copy().reset_index(drop=True)
else:
    val_eval_df = val_split.sample(n=VALIDATION_EVAL_SIZE, random_state=SEED).reset_index(drop=True)

val_dataset = VqaFineTuneDataset(val_eval_df)
print(f"Train dataset size: {len(train_dataset)} | Eval dataset size: {len(val_dataset)}")
print(f"Validation eval subset capped at {VALIDATION_EVAL_SIZE} samples. Adjust VALIDATION_EVAL_SIZE as needed.")
'''
set_cell(22, cell22_code)

cell24_code = '''# ============================================================
# TrainingArguments tuned for dual T4 16GB
# ============================================================
import math
import os
import torch
from transformers import TrainingArguments, Trainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

torch.backends.cudnn.benchmark = True

if hasattr(processor, "tokenizer"):
    processor.tokenizer.padding_side = "left"

world_size = max(1, torch.cuda.device_count())
gpu_mem_gb = 0.0
cuda_name = "cpu"
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    gpu_mem_gb = props.total_memory / 1024 ** 3
    cuda_name = props.name

per_device_batch = int(os.environ.get("PER_DEVICE_TRAIN_BATCH", TRAIN_BATCH_T4))
grad_accum = int(os.environ.get("GRAD_ACCUM_STEPS", GRAD_ACCUM_T4))
epochs_requested = float(os.environ.get("NUM_EPOCHS", EPOCHS_DEFAULT))
learning_rate = float(os.environ.get("LEARNING_RATE", LEARNING_RATE_DEFAULT))
warmup_ratio = float(os.environ.get("WARMUP_RATIO", WARMUP_RATIO_DEFAULT))
num_workers = int(os.environ.get("NUM_DATALOADER_WORKERS", NUM_WORKERS_T4))
max_steps_override = int(os.environ.get("MAX_STEPS", 0))

target_hours = max(0.0, float(os.environ.get("TARGET_TRAIN_HOURS", TARGET_TRAIN_HOURS)))
est_step_time = max(0.1, float(os.environ.get("EST_STEP_TIME_SEC", EST_STEP_TIME_SEC)))

train_size = len(train_dataset)
effective_batch = max(1, per_device_batch) * max(1, grad_accum) * max(1, world_size)
steps_per_epoch = max(1, math.ceil(train_size / max(1, per_device_batch)))
optim_steps_per_epoch = max(1, math.ceil(train_size / effective_batch))

max_steps = max_steps_override if max_steps_override > 0 else -1
if max_steps == -1 and target_hours > 0:
    naive_total_steps = optim_steps_per_epoch * max(1, math.ceil(epochs_requested))
    target_steps = int(target_hours * 3600 / est_step_time)
    if target_steps > 0 and target_steps < naive_total_steps:
        max_steps = target_steps

epochs = int(max(1, math.ceil(epochs_requested)))
if max_steps > 0:
    epochs = max(1, min(epochs, math.ceil(max_steps / optim_steps_per_epoch)))

naive_total_time_hours = (optim_steps_per_epoch * epochs * est_step_time) / 3600
if max_steps > 0:
    capped_time_hours = (max_steps * est_step_time) / 3600
else:
    capped_time_hours = naive_total_time_hours

evaluation_strategy = "epoch" if len(val_dataset) > 0 else "no"

print(
    f"GPU: {world_size} x {cuda_name} (~{gpu_mem_gb:.1f} GB) | per_device_batch={per_device_batch} "
    f"grad_accum={grad_accum} effective_batch={effective_batch}"
)
print(
    f"epochs={epochs} (requested={epochs_requested}) | optim_steps/epoch~{optim_steps_per_epoch} | "
    f"max_steps={max_steps if max_steps > 0 else 'None'}"
)
print(
    f"Estimated wall-clock (naive) ≈ {naive_total_time_hours:.2f}h | with cap ≈ {capped_time_hours:.2f}h "
    f"(est_step_time={est_step_time}s)"
)
print(f"lr={learning_rate} | warmup_ratio={warmup_ratio} | dataloader_workers={num_workers}")

bf16_enabled = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

training_kwargs = dict(
    output_dir=str(SAVE_DIR),
    num_train_epochs=epochs,
    max_steps=max_steps,
    per_device_train_batch_size=per_device_batch,
    gradient_accumulation_steps=grad_accum,
    learning_rate=learning_rate,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type="linear",
    logging_steps=LOGGING_STEPS,
    save_strategy="epoch" if evaluation_strategy == "epoch" else "no",
    save_total_limit=1,
    load_best_model_at_end=False,
    bf16=bf16_enabled,
    fp16=not bf16_enabled,
    optim="adamw_bnb_8bit",
    dataloader_num_workers=num_workers,
    gradient_checkpointing=True,
    max_grad_norm=0.5,
    remove_unused_columns=False,
    report_to="none",
    evaluation_strategy=evaluation_strategy,
    per_device_eval_batch_size=per_device_batch,
)
if world_size > 1:
    training_kwargs["ddp_find_unused_parameters"] = False

training_args = TrainingArguments(**training_kwargs)

train_data_for_trainer = train_dataset
eval_data_for_trainer = val_dataset if evaluation_strategy != "no" else None

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data_for_trainer,
    eval_dataset=eval_data_for_trainer,
    data_collator=fine_tune_collate_fn,
)

train_result = trainer.train()
print("Training metrics:", getattr(train_result, "metrics", {}))

if eval_data_for_trainer is not None:
    try:
        eval_metrics = trainer.evaluate(eval_dataset=eval_data_for_trainer)
        print("Eval metrics:", eval_metrics)
    except Exception as exc:
        print(f"(info) Evaluation skipped: {exc}")

adapter_path = SAVE_DIR / "lora_adapter"
model.save_pretrained(adapter_path)
processor.save_pretrained(SAVE_DIR / "processor")
print(f"[INFO] LoRA fine-tuning complete. Adapter saved to {adapter_path}")
'''
set_cell(24, cell24_code)

path.write_text(json.dumps(nb, ensure_ascii=False), encoding='utf-8')