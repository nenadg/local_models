"""
Fine-tuning script for semantic reasoning task with TinyLlama.
This prepares a dataset and trains TinyLlama to better distinguish between
SAME, OPPOSITE, and DIFFERENT relationships between query pairs.
"""

import torch
import argparse
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

# Expanded training examples for semantic relationship classification
TRAINING_EXAMPLES = [
    # SAME relationship examples
    {
        "query1": "when was the euro introduced?",
        "query2": "when was the euro established?",
        "relationship": "SAME"
    },
    {
        "query1": "when was bitcoin created?",
        "query2": "when was bitcoin established?",
        "relationship": "SAME"
    },
    {
        "query1": "who wrote Hamlet?",
        "query2": "who is the author of Hamlet?",
        "relationship": "SAME"
    },
    {
        "query1": "what is the capital of France?",
        "query2": "what city is the capital of France?",
        "relationship": "SAME"
    },
    {
        "query1": "how do I bake bread?",
        "query2": "what's the process for making bread?",
        "relationship": "SAME"
    },
    {
        "query1": "when was the BAM monetary unit first introduced?",
        "query2": "when was the BAM monetary unit established?",
        "relationship": "SAME"
    },
    {
        "query1": "how tall is the Eiffel Tower?",
        "query2": "what is the height of the Eiffel Tower?",
        "relationship": "SAME"
    },
    
    # OPPOSITE relationship examples
    {
        "query1": "when was the bosnian convertible mark established?",
        "query2": "when was the bosnian convertible mark abolished?",
        "relationship": "OPPOSITE"
    },
    {
        "query1": "when did World War 2 begin?",
        "query2": "when did World War 2 end?",
        "relationship": "OPPOSITE"
    },
    {
        "query1": "what are the benefits of exercise?",
        "query2": "what are the drawbacks of exercise?",
        "relationship": "OPPOSITE"
    },
    {
        "query1": "how to increase computer performance?",
        "query2": "why is my computer running slowly?",
        "relationship": "OPPOSITE"
    },
    {
        "query1": "when did the BAM currency begin circulation?",
        "query2": "when was the BAM currency discontinued?",
        "relationship": "OPPOSITE"
    },
    {
        "query1": "how to start a business?",
        "query2": "how to close down a business?",
        "relationship": "OPPOSITE"
    },
    {
        "query1": "what causes inflation?",
        "query2": "what causes deflation?", 
        "relationship": "OPPOSITE"
    },

    # DIFFERENT relationship examples
    {
        "query1": "who was the main character in Titanic?",
        "query2": "who directed Titanic?",
        "relationship": "DIFFERENT"
    },
    {
        "query1": "how tall is the Eiffel Tower?",
        "query2": "what is the capital of Japan?",
        "relationship": "DIFFERENT"
    },
    {
        "query1": "what is the population of New York?",
        "query2": "what is the primary export of Brazil?",
        "relationship": "DIFFERENT"
    },
    {
        "query1": "who won the 2020 World Series?",
        "query2": "what is quantum computing?",
        "relationship": "DIFFERENT"
    },
    {
        "query1": "where is Mount Everest located?",
        "query2": "how does photosynthesis work?",
        "relationship": "DIFFERENT"
    },
    {
        "query1": "how do you make chocolate chip cookies?",
        "query2": "what is the history of the Roman Empire?",
        "relationship": "DIFFERENT"
    }
]

# Generate more examples by switching query1/query2 and creating permutations
def expand_training_data(examples):
    expanded = []
    
    for example in examples:
        # Add original example
        expanded.append(example)
        
        # Add reversed query example with same relationship
        reversed_example = {
            "query1": example["query2"],
            "query2": example["query1"],
            "relationship": example["relationship"]
        }
        expanded.append(reversed_example)
        
    return expanded

def format_prompt(example):
    """Format the example into a consistent instruction prompt"""
    system_prompt = "You are a logical reasoning assistant. Analyze the semantic relationship between pairs of questions."
    instruction = (
        "You must respond with EXACTLY ONE of these words: SAME, OPPOSITE, or DIFFERENT.\n\n"
        "- SAME: The queries are asking about the same type of information or event (e.g., both about creation)\n"
        "- OPPOSITE: The queries are asking about opposing information or events (e.g., creation vs. destruction)\n"
        "- DIFFERENT: The queries are asking about unrelated or tangential information\n\n"
        f"Query 1: {example['query1']}\nQuery 2: {example['query2']}\n\nRelationship:"
    )
    
    # Format for the model
    formatted = f"{system_prompt}\n\n{instruction}"
    
    return formatted

def format_output(example):
    """Format the expected output"""
    return example["relationship"]

def prepare_dataset(examples):
    """Prepare the dataset for training"""
    expanded_examples = expand_training_data(examples)
    
    # Create input and output pairs
    data = {
        "input": [format_prompt(ex) for ex in expanded_examples],
        "output": [format_output(ex) for ex in expanded_examples]
    }
    
    # Create dataset
    dataset = Dataset.from_dict(data)
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama for semantic reasoning")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                      help="Model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./finetuned_tinyllama_reasoning",
                      help="Directory to save fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-5,
                      help="Learning rate")
    
    args = parser.parse_args()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Set up LoRA configuration for efficient fine-tuning
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    
    # Prepare dataset
    dataset = prepare_dataset(TRAINING_EXAMPLES)
    
    # Create a validation split
    dataset = dataset.train_test_split(test_size=0.2)
    
    # Tokenize dataset
    def tokenize_function(examples):
        # Combine input and output with a separator
        texts = [inp + " " + out for inp, out in zip(examples["input"], examples["output"])]
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        return tokenized
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=args.lr,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        report_to="none"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator
    )
    
    # Train the model
    trainer.train()
    
    # Save the fine-tuned model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Model saved to {args.output_dir}")
    
    # Run a quick test on the fine-tuned model
    print("Testing fine-tuned model on a few examples:")
    test_examples = [
        {
            "query1": "when was the dollar introduced?",
            "query2": "when was the dollar abolished?",
            "relationship": "OPPOSITE"
        },
        {
            "query1": "what is the weather in London?",
            "query2": "what is the weather in Paris?",
            "relationship": "SAME"
        }
    ]
    
    model.eval()
    for example in test_examples:
        prompt = format_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1
            )
            
        output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Example: {example['query1']} vs {example['query2']}")
        print(f"Expected: {example['relationship']}")
        print(f"Predicted: {output_text.strip()}")
        print()

if __name__ == "__main__":
    main()