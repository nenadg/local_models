"""
Semantic Reasoning Test for TinyLlama

This script tests the ability of TinyLlama to analyze semantic relationships
between query pairs, particularly focusing on detecting semantic oppositions.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import random
import argparse

# Test pairs of queries
# Format: (query1, query2, expected_relationship)
# Relationships: "SAME", "OPPOSITE", "DIFFERENT"
TEST_PAIRS = [
    # Temporal opposites (creation vs termination)
    ("when was the bosnian convertible mark established?", 
     "when was the bosnian convertible mark abolished?", 
     "OPPOSITE"),
    
    # Temporal same (both about creation)
    ("when was the euro introduced?", 
     "when was the euro established?", 
     "SAME"),
    
    # Different subjects but same action
    ("when was bitcoin created?", 
     "when was ethereum created?", 
     "SAME"),
    
    # Same subject different aspects
    ("who was the main character in Titanic?", 
     "who directed Titanic?", 
     "DIFFERENT"),
    
    # Near-opposites
    ("when did World War 2 begin?", 
     "when did World War 2 end?", 
     "OPPOSITE"),
    
    # Complete opposites
    ("what are the benefits of exercise?", 
     "what are the drawbacks of exercise?", 
     "OPPOSITE"),
    
    # Completely different
    ("how tall is the Eiffel Tower?", 
     "what is the capital of Japan?", 
     "DIFFERENT"),
    
    # More subtle opposition
    ("how to increase computer performance?", 
     "why is my computer running slowly?", 
     "OPPOSITE"),
    
    # Rephrased same question
    ("what is the population of New York City?", 
     "how many people live in NYC?", 
     "SAME"),
    
    # Currency specific cases
    ("when was the BAM monetary unit first introduced?",
     "when was the BAM monetary unit established?",
     "SAME"),
     
    ("when did the BAM currency begin circulation?",
     "when was the BAM currency discontinued?",
     "OPPOSITE")
]

# Different system prompts to test which gets better results
SYSTEM_PROMPTS = [
    # Direct instruction prompt
    """You are a logical reasoning assistant. Analyze the semantic relationship between pairs of questions.
    You must respond with EXACTLY ONE of these words: SAME, OPPOSITE, or DIFFERENT.
    
    - SAME: The queries are asking about the same type of information or event (e.g., both about creation)
    - OPPOSITE: The queries are asking about opposing information or events (e.g., creation vs. destruction)
    - DIFFERENT: The queries are asking about unrelated or tangential information
    
    Only output the single word: SAME, OPPOSITE, or DIFFERENT.""",
    
    # More analytical prompt
    """You are a semantic analysis engine. Compare these two queries carefully.
    Determine if they are asking about:
    1. SAME type of information (e.g., both about when something started)
    2. OPPOSITE information (e.g., one about start, one about end)
    3. DIFFERENT topics entirely (unrelated queries)
    
    Respond with only one word: SAME, OPPOSITE, or DIFFERENT.""",
    
    # Chain of thought prompt
    """You are a logical reasoning assistant. 
    Analyze these two queries step by step:
    1. Identify the main subject of each query
    2. Identify the type of information being requested
    3. Determine if the information types are the same, opposite, or different
    
    Then output EXACTLY ONE of these classifications:
    - SAME: Both queries ask for the same type of information
    - OPPOSITE: The queries ask for semantically opposing information
    - DIFFERENT: The queries ask for unrelated information
    
    Your final answer must be only one word: SAME, OPPOSITE, or DIFFERENT."""
]

def create_prompt(system_content, query1, query2):
    """Create a prompt for the model with the given queries"""
    prompt = f"{system_content}\n\nAnalyze these two queries:\n\nQuery 1: {query1}\nQuery 2: {query2}\n\nRelationship:"
    return prompt

def run_reasoning_test(model, tokenizer, device="cpu", temperature=0.1):
    """Run the semantic reasoning test on all test pairs"""
    results = []
    success_count = 0
    
    # Try each system prompt to see which works best
    best_prompt = None
    best_prompt_score = 0
    
    for prompt_idx, system_prompt in enumerate(SYSTEM_PROMPTS):
        print(f"\nTesting system prompt #{prompt_idx+1}...")
        prompt_results = []
        prompt_success = 0
        
        # Shuffle test pairs for this prompt
        shuffled_pairs = TEST_PAIRS.copy()
        random.shuffle(shuffled_pairs)
        
        for i, (query1, query2, expected) in enumerate(shuffled_pairs):
            # Format and tokenize the prompt
            prompt = create_prompt(system_prompt, query1, query2)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate response
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=temperature,
                    top_p=0.95,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Extract response
            response_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            
            # Clean and normalize response
            cleaned_response = response.strip().upper()
            if "SAME" in cleaned_response:
                cleaned_response = "SAME"
            elif "OPPOSITE" in cleaned_response:
                cleaned_response = "OPPOSITE"
            elif "DIFFERENT" in cleaned_response:
                cleaned_response = "DIFFERENT"
            else:
                cleaned_response = "INVALID"
            
            # Check if correct
            is_correct = cleaned_response == expected
            if is_correct:
                prompt_success += 1
            
            # Store result
            prompt_results.append({
                "query1": query1,
                "query2": query2,
                "expected": expected,
                "response": response,
                "cleaned_response": cleaned_response,
                "is_correct": is_correct,
                "time": time.time() - start_time
            })
            
            # Print progress
            print(f"  Pair {i+1}/{len(shuffled_pairs)}: {'✓' if is_correct else '✗'} Expected: {expected}, Got: {cleaned_response}")
        
        # Calculate success rate for this prompt
        success_rate = prompt_success / len(shuffled_pairs) * 100
        print(f"Prompt #{prompt_idx+1} success rate: {success_rate:.1f}%")
        
        # Check if this is the best prompt
        if prompt_success > best_prompt_score:
            best_prompt = prompt_idx
            best_prompt_score = prompt_success
        
        # Add results to overall results
        results.append({
            "prompt_idx": prompt_idx,
            "success_rate": success_rate,
            "results": prompt_results
        })
    
    # Report best prompt
    if best_prompt is not None:
        print(f"\n===== BEST PERFORMING PROMPT: #{best_prompt+1} with {best_prompt_score}/{len(TEST_PAIRS)} correct =====")
        print(SYSTEM_PROMPTS[best_prompt])
    
    return results

def analyze_results(results):
    """Analyze performance on different types of relationships"""
    for prompt_result in results:
        prompt_idx = prompt_result["prompt_idx"]
        
        # Group by relationship type
        by_relationship = {"SAME": [], "OPPOSITE": [], "DIFFERENT": []}
        
        for result in prompt_result["results"]:
            by_relationship[result["expected"]].append(result)
        
        print(f"\nPrompt #{prompt_idx+1} performance by relationship type:")
        for rel_type, items in by_relationship.items():
            correct = sum(1 for item in items if item["is_correct"])
            if items:
                type_rate = correct / len(items) * 100
                print(f"  {rel_type}: {correct}/{len(items)} correct ({type_rate:.1f}%)")
            else:
                print(f"  {rel_type}: No test cases")

def main():
    parser = argparse.ArgumentParser(description="Test TinyLlama's semantic reasoning capabilities")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                      help="Model to use for testing")
    parser.add_argument("--device", type=str, default=None,
                      help="Device to use (cuda, cpu). Auto-detects if not specified.")
    parser.add_argument("--temperature", type=float, default=0.1,
                      help="Temperature for generation (lower = more deterministic)")
    
    args = parser.parse_args()
    
    # Determine device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Testing semantic reasoning with model: {args.model}")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    
    # Run the test
    results = run_reasoning_test(model, tokenizer, device, args.temperature)
    
    # Analyze results
    analyze_results(results)
    
    # Print a conclusion
    best_prompt_idx = max(range(len(results)), key=lambda i: results[i]["success_rate"])
    best_result = results[best_prompt_idx]
    
    print(f"\nCONCLUSION: TinyLlama semantic reasoning test achieved {best_result['success_rate']:.1f}% accuracy")
    print(f"Best prompt was #{best_prompt_idx+1}")
    
    # Check for ability to handle the specific BAM case
    bam_cases = [r for r in best_result["results"] 
                if "BAM" in r["query1"] and "BAM" in r["query2"]]
    
    if bam_cases:
        bam_correct = sum(1 for r in bam_cases if r["is_correct"])
        print(f"\nBAM currency specific cases: {bam_correct}/{len(bam_cases)} correct")

if __name__ == "__main__":
    main()