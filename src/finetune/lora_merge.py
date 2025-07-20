import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Base model identifier
base_model_name = "Qwen/Qwen3-0.6B"
# Directory where your trained LoRA adapters are saved
adapter_path = "./Qwen3-0.6B-Finetune"
# Directory where the merged model will be saved
merged_model_path = "./Qwen3-0.6B-Finetune-Merged"

# Load the base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load the PeftModel with your adapters
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, adapter_path)

# Merge the adapters into the model
print("Merging model...")
model = model.merge_and_unload()

# Save the merged model and tokenizer
print(f"Saving merged model to {merged_model_path}...")
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
print("Done.")
