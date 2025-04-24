from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# Load model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

# Wrap model with LoRA
model = get_peft_model(model, peft_config)

# Load dataset from Hugging Face
dataset = load_dataset("ajibawa-2023/Maths-Grade-School")

# Prepare dataset
def tokenize_function(examples):
    return tokenizer(examples["output"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_tinyllama",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_steps=1000,
    logging_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
)

# Initialize Trainer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# Train model
trainer.train()

# Save model
model.save_pretrained("./finetuned_tinyllama")
tokenizer.save_pretrained("./finetuned_tinyllama")
