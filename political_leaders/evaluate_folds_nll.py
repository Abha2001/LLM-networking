import os
import json
import math
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===============================
# CONFIG
# ===============================

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
PERSIST_DIR = "/nas/home/abhajha/llm_network/leaders/unsc_llama_lora_outputs"
MAX_LEN = 512
DEVICE_MAP = {"": 0}

# Same selection / cap as the finetuning script
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
MAX_SPEECHES_PER_COUNTRY = 50
N_SPLITS = 5

# Force single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ===============================
# LOAD DATA (must match training)
# ===============================

with open("unsc_data.json", "r", encoding="utf-8") as f:
    unsc_data = json.load(f)

COUNTRY_SPEECHES = {}
for record in unsc_data:
    topic = record.get("topic", "").strip()
    for sr in record.get("speech_records", []):
        country = sr["affiliation"]
        if country not in SELECTED_COUNTRIES:
            continue
        if sr.get("procedural", False):
            continue
        if country not in COUNTRY_SPEECHES:
            COUNTRY_SPEECHES[country] = []
        COUNTRY_SPEECHES[country].append({
            "topic": topic,
            "speech": sr["speech"].strip(),
        })

for country in list(COUNTRY_SPEECHES.keys()):
    COUNTRY_SPEECHES[country] = COUNTRY_SPEECHES[country][-MAX_SPEECHES_PER_COUNTRY:]
    if len(COUNTRY_SPEECHES[country]) < N_SPLITS:
        del COUNTRY_SPEECHES[country]


def format_prompt(country, entry):
    """Format a speech using the same LLaMA 3 template as training."""
    instruction = f"Respond as the representative of {country} at the UN Security Council."
    topic = entry["topic"] if entry["topic"] else "General debate."
    speech = entry["speech"]
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a representative at the UN Security Council responding to geopolitical events.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}\n\n{topic}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{speech}<|eot_id|>"
    )


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# ===============================
# EVALUATION FUNCTION
# ===============================

def eval_fold_nll(model, tokenizer, texts):
    model.eval()
    losses = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LEN,
                padding=False,
            ).to(model.device)

            out = model(**enc, labels=enc["input_ids"])
            losses.append(out.loss.item())
            del out, enc
            torch.cuda.empty_cache()

    return float(np.mean(losses)) if losses else math.nan

# ===============================
# RUN EVALUATION
# ===============================

eval_results = {}

fold_dirs = sorted(
    d for d in os.listdir(PERSIST_DIR)
    if d.startswith("fold_")
)

for fold in fold_dirs:
    print(f"\nEvaluating {fold}...")
    fold_path = os.path.join(PERSIST_DIR, fold)

    with open(os.path.join(fold_path, "split_indices.json")) as f:
        split = json.load(f)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        device_map=DEVICE_MAP,
    )
    model = PeftModel.from_pretrained(base, fold_path)
    model.eval()

    fold_losses = {}

    for country, idxs in split["test"].items():
        if country not in COUNTRY_SPEECHES:
            continue
        test_prompts = [format_prompt(country, COUNTRY_SPEECHES[country][i]) for i in idxs]
        fold_losses[country] = eval_fold_nll(model, tokenizer, test_prompts)

    eval_results[fold] = fold_losses

    # cleanup
    del model, base
    torch.cuda.empty_cache()
    import gc
    gc.collect()

# ===============================
# SAVE RESULTS
# ===============================

eval_path = os.path.join(PERSIST_DIR, "unsc_llama_evaluation_nll.json")
with open(eval_path, "w") as f:
    json.dump(eval_results, f, indent=2)

print(f"\nSaved evaluation to {eval_path}")

# ===============================
# SELECT BEST FOLD
# ===============================

fold_scores = {
    fold: np.mean([v for v in scores.values() if not math.isnan(v)])
    for fold, scores in eval_results.items()
}

best_fold = min(fold_scores, key=fold_scores.get)

best_info = {
    "best_fold": best_fold,
    "mean_nll": fold_scores[best_fold],
    "all_scores": fold_scores,
}

with open(os.path.join(PERSIST_DIR, "best_fold.json"), "w") as f:
    json.dump(best_info, f, indent=2)

print(f"\nBest fold: {best_fold}")
print(f"Mean held-out NLL: {fold_scores[best_fold]:.4f}")
