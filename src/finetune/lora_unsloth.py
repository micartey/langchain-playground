from unsloth import FastLanguageModel, apply_chat_template
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from unsloth.chat_templates import to_sharegpt
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq
import torch
import time

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

def formatting_prompts_func(examples):
    convos = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        if input_text:
            user_message = f"{instruction}\n\n{input_text}"
        else:
            user_message = instruction

        conversation = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": output}
        ]
        convos.append(conversation)
    return {"conversations": convos}


dataset = load_dataset("vicgalle/alpaca-gpt4", split = "train")
dataset = dataset.map(formatting_prompts_func, batched=True)

dataset = apply_chat_template(
    dataset,
    tokenizer,
)

# Train only on responses to help with stopping
# dataset = train_on_responses_only(
#     tokenizer,
#     dataset,
# )

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60, # The amount of training steps (default recommended: 60)
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()

# trainer.model.save_pretrained("new_model")

model.save_pretrained_gguf("model", tokenizer, maximum_memory_usage = 0.40)
