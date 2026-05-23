import os

# 🔒 FORCE SINGLE GPU, ALWAYS
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import json
import torch
import numpy as np
from sklearn.model_selection import KFold
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
N_SPLITS = 5
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 2e-4
MAX_LEN = 1024

OUTPUT_DIR = "/run/user/8638/unsc_llama_lora_outputs"
PERSIST_DIR = "/nas/home/abhajha/llm_network/leaders/unsc_llama_lora_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Indo-Pacific countries
SELECTED_COUNTRIES = {
    # Southeast Asia
    "Cambodia", "Thailand", "Viet Nam", "Myanmar", "Lao People's Democratic Republic",
    "Indonesia", "Malaysia", "Philippines", "Singapore", "Brunei Darussalam", "Timor-Leste",
    # East Asia
    "China", "Japan", "Republic of Korea", "Democratic People's Republic of Korea", "Mongolia",
    # South Asia
    "India", "Pakistan", "Bangladesh", "Sri Lanka", "Nepal", "Maldives", "Bhutan",
    # Oceania
    "Australia", "New Zealand", "Papua New Guinea", "Fiji",
    "Solomon Islands", "Vanuatu", "Samoa", "Tonga", "Tuvalu",
    "Kiribati", "Nauru", "Palau", "Marshall Islands",
    "Federated States of Micronesia",
    # Key Pacific powers
    "United States of America", "Russian Federation",
}

MAX_SPEECHES_PER_COUNTRY = 50  # use most recent N speeches per country

# =============================================================================
# LOAD DATA
# =============================================================================

with open("unsc_data.json", "r", encoding="utf-8") as f:
    unsc_data = json.load(f)

# Group speeches by country: {country: [(topic, speech), ...]}
COUNTRY_SPEECHES = {}
for record in unsc_data:
    topic = record.get("topic", "").strip()
    for sr in record.get("speech_records", []):
        country = sr["affiliation"]
        if country not in SELECTED_COUNTRIES:
            continue
        if sr.get("procedural", False):
            continue  # skip procedural speeches
        if country not in COUNTRY_SPEECHES:
            COUNTRY_SPEECHES[country] = []
        COUNTRY_SPEECHES[country].append({
            "topic": topic,
            "speech": sr["speech"].strip(),
        })

# Keep most recent N speeches per country, drop countries with fewer than N_SPLITS
for country in list(COUNTRY_SPEECHES.keys()):
    COUNTRY_SPEECHES[country] = COUNTRY_SPEECHES[country][-MAX_SPEECHES_PER_COUNTRY:]
    if len(COUNTRY_SPEECHES[country]) < N_SPLITS:
        print(f"  Dropping {country}: only {len(COUNTRY_SPEECHES[country])} speeches (need >= {N_SPLITS})")
        del COUNTRY_SPEECHES[country]

COUNTRY_NAMES = list(COUNTRY_SPEECHES.keys())
print(f"Countries: {len(COUNTRY_NAMES)}")
for c in COUNTRY_NAMES:
    print(f"  {c}: {len(COUNTRY_SPEECHES[c])} speeches")

# =============================================================================
# DATASET CONSTRUCTION
# =============================================================================

def build_instruction_examples(speeches_dict):
    """
    Converts country speeches into instruction-tuning examples.
    Each example: given a topic, respond as that country's representative.
    """
    examples = []
    for country, speeches in speeches_dict.items():
        for entry in speeches:
            examples.append({
                "instruction": f"Respond as the representative of {country} at the UN Security Council.",
                "input": entry["topic"] if entry["topic"] else "General debate.",
                "output": entry["speech"],
            })
    return examples


def tokenize_example(example):
    """
    Qwen 2.5 ChatML formatting.
    """
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a representative at the UN Security Council responding to geopolitical events.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{example['instruction']}\n\n{example['input']}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['output']}<|eot_id|>"
    )

    tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )

    return tokens


# =============================================================================
# TOKENIZER
# =============================================================================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# =============================================================================
# K-FOLD TRAINING
# =============================================================================

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

for fold_idx in range(1, N_SPLITS):  # resume from fold 1
    print("\n" + "=" * 70)
    print(f"TRAINING FOLD {fold_idx + 1} / {N_SPLITS}")
    print("=" * 70)

    train_speeches = {}
    test_speeches = {}

    fold_indices = {
        "train": {},
        "test": {},
    }

    for country in COUNTRY_NAMES:
        speeches = COUNTRY_SPEECHES[country]
        indices = np.arange(len(speeches))
        train_idx, test_idx = list(kf.split(indices))[fold_idx]

        train_speeches[country] = [speeches[i] for i in train_idx]
        test_speeches[country] = [speeches[i] for i in test_idx]

        fold_indices["train"][country] = train_idx.tolist()
        fold_indices["test"][country] = test_idx.tolist()

    # Build dataset
    train_examples = build_instruction_examples(train_speeches)
    print(f"  Training examples: {len(train_examples)}")
    train_dataset = Dataset.from_list(train_examples)
    train_dataset = train_dataset.map(
        tokenize_example,
        remove_columns=train_dataset.column_names,
    )

    # Model load (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": 0},
        quantization_config=bnb_config,
        trust_remote_code=False,
    )

    model.enable_input_require_grads()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=fold_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=NUM_EPOCHS,
        fp16=True,
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(fold_dir)
    tokenizer.save_pretrained(fold_dir)

    import shutil

    os.makedirs(PERSIST_DIR, exist_ok=True)

    shutil.copytree(
        fold_dir,
        os.path.join(PERSIST_DIR, f"fold_{fold_idx}"),
        dirs_exist_ok=True,
    )

    with open(
        os.path.join(PERSIST_DIR, f"fold_{fold_idx}", "split_indices.json"), "w"
    ) as f:
        json.dump(fold_indices, f, indent=2)

    metadata = {
        "fold": fold_idx,
        "train_counts": {k: len(v) for k, v in train_speeches.items()},
        "test_counts": {k: len(v) for k, v in test_speeches.items()},
    }

    with open(os.path.join(PERSIST_DIR, f"fold_{fold_idx}", "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Fold {fold_idx} finished and saved to {fold_dir} and {PERSIST_DIR}")

    # GPU cleanup
    import gc
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

print("\n" + "=" * 70)
print("ALL FOLDS COMPLETE")
print("=" * 70)
