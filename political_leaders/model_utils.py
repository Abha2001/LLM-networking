# # merge_lora_fold2.py
# from transformers import AutoModelForCausalLM
# from peft import PeftModel

# BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# FOLD_PATH = "/nas/home/abhajha/llm_network/leaders/lora_outputs/fold_2"
# OUT_PATH = "./merged_fold_2"

# base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
# model = PeftModel.from_pretrained(base, FOLD_PATH)

# merged = model.merge_and_unload()
# merged.save_pretrained(OUT_PATH)

# print("✓ LoRA merged and saved to merged_fold_2/")

# merge_gemma_minimal.py
from transformers import AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
FOLD_PATH = "/nas/home/abhajha/llm_network/leaders/unsc_llama_lora_outputs/fold_4"
OUT_PATH = "./unsc_llama_merged_fold_4"

base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base, FOLD_PATH)

merged = model.merge_and_unload()
merged.save_pretrained(OUT_PATH)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(OUT_PATH)

print("✓ LoRA merged and saved")