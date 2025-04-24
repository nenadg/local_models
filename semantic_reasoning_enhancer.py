import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import PeftModel, PeftConfig
from datetime import datetime

class SemanticReasoningEnhancer:
    """
    Enhances TinyLlama's semantic reasoning capabilities using fine-tuned model
    """
    def __init__(self, base_model_path, finetuned_model_path):
        """
        Initialize the semantic reasoning enhancer.

        Args:
            base_model_path: Path to the original base model
            finetuned_model_path: Path to the fine-tuned model
        """
        # Load base model and tokenizer
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

        # Load the fine-tuned model (using PEFT/LoRA)
        try:
            # Check if the fine-tuned model is a PEFT/LoRA model
            peft_config = PeftConfig.from_pretrained(finetuned_model_path)
            self.model = PeftModel.from_pretrained(self.base_model, finetuned_model_path)
        except:
            # Fallback to direct model loading if not using PEFT
            self.model = AutoModelForCausalLM.from_pretrained(finetuned_model_path)

        # Ensure we have the right tokenizer
        self.tokenizer = self.base_tokenizer

    def get_time(self):
        return datetime.now().strftime("[%d/%m/%y %H:%M:%S]")

    def format_reasoning_prompt(self, query1: str, query2: str) -> str:
        """
        Format the input for semantic relationship classification.

        Args:
            query1: First query
            query2: Second query

        Returns:
            Formatted prompt for model
        """
        prompt = (
            "You are a logical reasoning assistant. Analyze the semantic relationship "
            "between the following two queries. Respond with ONLY ONE WORD: "
            "SAME, OPPOSITE, or DIFFERENT.\n\n"
            f"Query 1: {query1}\n"
            f"Query 2: {query2}\n\n"
            "Relationship:"
        )
        return prompt

    def classify_relationship(self, query1: str, query2: str, max_tokens: int = 10) -> str:
        """
        Classify the semantic relationship between two queries.

        Args:
            query1: First query
            query2: Second query
            max_tokens: Maximum tokens to generate

        Returns:
            Relationship classification (SAME, OPPOSITE, or DIFFERENT)
        """
        # Prepare the prompt
        prompt = self.format_reasoning_prompt(query1, query2)

        # Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            do_sample=False,  # Use greedy decoding for classification
            temperature=0.1,
            top_p=0.95
        )

        # Decode the output
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )

        # Clean and standardize the response
        response = response.strip().upper()

        # Extract the first valid classification
        if "SAME" in response:
            return "SAME"
        elif "OPPOSITE" in response:
            return "OPPOSITE"
        elif "DIFFERENT" in response:
            return "DIFFERENT"
        
        # Fallback 
        return "UNKNOWN"

def integrate_semantic_reasoning(chat_instance, finetuned_model_path):
    """
    Integrate semantic reasoning capabilities into TinyLlamaChat.

    Args:
        chat_instance: The TinyLlamaChat instance
        finetuned_model_path: Path to the fine-tuned semantic reasoning model
    """
    try:
        # Create the semantic reasoning enhancer
        reasoning_enhancer = SemanticReasoningEnhancer(
            base_model_path=chat_instance.model_name,
            finetuned_model_path=finetuned_model_path
        )

        # Attach the enhancer to the chat instance
        chat_instance.semantic_reasoning = reasoning_enhancer

        # Optional: Add a method to the chat instance for semantic relationship testing
        def test_semantic_relationship(query1, query2):
            """
            Test the semantic relationship between two queries.
            
            Args:
                query1: First query
                query2: Second query
            
            Returns:
                Relationship classification
            """
            if not hasattr(chat_instance, 'semantic_reasoning'):
                print("Semantic reasoning not initialized.")
                return "UNKNOWN"
            
            return chat_instance.semantic_reasoning.classify_relationship(query1, query2)

        # Add the method to the chat instance
        chat_instance.test_semantic_relationship = test_semantic_relationship

        print(f"{reasoning_enhancer.get_time()} [Semantic Reasoning] Successfully integrated!")
        return True

    except Exception as e:
        print(f"{reasoning_enhancer.get_time()} [Semantic Reasoning] Integration failed: {e}")
        return False

# Example usage in main script:
# def main():
#     # In your main TinyLlamaChat initialization
#     chat = TinyLlamaChat(...)
    
#     # Path to your fine-tuned model
#     finetuned_model_path = "./finetuned_tinyllama_reasoning"
    
#     # Integrate semantic reasoning
#     integrate_semantic_reasoning(chat, finetuned_model_path)
    
#     # Now you can use semantic relationship testing
#     relationship = chat.test_semantic_relationship(
#         "when was the euro introduced?",
#         "when was the euro established?"
#     )
#     print(f"Relationship: {relationship}")  # Should print "SAME"

# # Recommended modifications to TinyLlamaChat class
# # Add this method to the class in tiny_llama_6_memory.py
# def add_semantic_reasoning_method(self):
#     """
#     Add semantic relationship testing capability to TinyLlamaChat.
#     This method should be called during initialization.
#     """
#     def test_semantic_relationship(query1, query2):
#         """
#         Test the semantic relationship between two queries.
        
#         Args:
#             query1: First query
#             query2: Second query
        
#         Returns:
#             Relationship classification
#         """
#         if not hasattr(self, 'semantic_reasoning'):
#             print("Semantic reasoning not initialized.")
#             return "UNKNOWN"
        
#         return self.semantic_reasoning.classify_relationship(query1, query2)
    
#     # Attach the method to the instance
#     self.test_semantic_relationship = test_semantic_relationship