from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTConfig, SFTTrainer


# ============================================================
# Configuration
# ============================================================

MODEL_C = "HuggingFaceTB/SmolLM2-135M-Instruct"

TRAIN_FILE = "data/model_c_source/sft_train.jsonl"
VAL_FILE = "data/model_c_source/sft_val.jsonl"

OUTPUT_DIR = "models/support_adapter"

MAX_LENGTH = 512


# ============================================================
# Load dataset
# ============================================================

print("Loading Model C dataset...")

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VAL_FILE,
    },
)

print(f"Train examples: {len(dataset['train'])}")
print(f"Validation examples: {len(dataset['validation'])}")


# ============================================================
# Load tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer_c = AutoTokenizer.from_pretrained(MODEL_C)

if tokenizer_c.pad_token is None:
    tokenizer_c.pad_token = tokenizer_c.eos_token


# ============================================================
# Convert messages to training text
# ============================================================

def format_for_sft(example):
    return {
        "text": tokenizer_c.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


print("\nFormatting conversations...")

sft_dataset = dataset.map(
    format_for_sft,
    remove_columns=dataset["train"].column_names,
)

sft_train = sft_dataset["train"]
sft_val = sft_dataset["validation"]

print(f"Formatted train examples: {len(sft_train)}")
print(f"Formatted validation examples: {len(sft_val)}")


# ============================================================
# Load base model
# ============================================================

use_qlora = torch.cuda.is_available()

print("\nCUDA available:", use_qlora)

if use_qlora:

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base_c = AutoModelForCausalLM.from_pretrained(
        MODEL_C,
        quantization_config=quant_config,
        device_map="auto",
    )

    base_c = prepare_model_for_kbit_training(base_c)

else:

    base_c = AutoModelForCausalLM.from_pretrained(
        MODEL_C
    )


# ============================================================
# LoRA configuration
# ============================================================

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
    bias="none",
)


model_c = get_peft_model(
    base_c,
    lora_config,
)

print("\nTrainable parameters:")
model_c.print_trainable_parameters()


# ============================================================
# SFT configuration
# ============================================================

sft_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=2,

    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,

    learning_rate=2e-4,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",
    greater_is_better=False,

    dataset_text_field="text",

    max_length=MAX_LENGTH,

    packing=False,

    report_to="none",

    # CPU training
    use_cpu=True,
    bf16=False,
    fp16=False,
)


# ============================================================
# Trainer
# ============================================================

trainer = SFTTrainer(
    model=model_c,
    args=sft_args,
    train_dataset=sft_train,
    eval_dataset=sft_val,
    processing_class=tokenizer_c,
)


# ============================================================
# Train
# ============================================================

print("\nStarting Model C training...\n")

train_result = trainer.train()


# ============================================================
# Save adapter
# ============================================================

print("\nSaving Model C adapter...")

trainer.save_model(OUTPUT_DIR)
tokenizer_c.save_pretrained(OUTPUT_DIR)


# ============================================================
# Evaluation
# ============================================================

print("\nRunning validation evaluation...")

eval_result = trainer.evaluate()

print("\nTraining complete.")

print("\nFinal evaluation:")

for key, value in eval_result.items():
    print(f"{key}: {value}")

print(f"\nAdapter saved to: {OUTPUT_DIR}")