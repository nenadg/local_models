#!/usr/bin/env python3
"""
Memory Importer for TinyLlama Chat
-----------------------------------
This script imports external text files into the memory system used by local_ai.py.
Each line in the file is processed and stored as a separate memory item.

Usage:
    python memory_importer.py --file path/to/file.txt [--memory_dir ./memory] [--batch_size 50]
"""

import os
import sys
import argparse
import torch
import numpy as np
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import necessary modules from the TinyLlama Chat project
try:
    from unified_memory import MemoryManager
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you're running this script from the TinyLlama Chat project directory")
    print("or that the project modules are in your Python path.")
    sys.exit(1)

class MemoryImporter:
    """
    Imports external content into the TinyLlama Chat memory system.
    """
    
    def __init__(self, 
                model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                memory_dir: str = "./memory",
                device: Optional[str] = None,
                batch_size: int = 50,
                enable_enhanced_embeddings: bool = True):
        """
        Initialize the memory importer.
        
        Args:
            model_name: Name of the model to use for embeddings
            memory_dir: Directory where memory data is stored
            device: Device to use for model inference ('cuda' or 'cpu')
            batch_size: Number of items to process in a batch
            enable_enhanced_embeddings: Whether to use enhanced embeddings
        """
        self.model_name = model_name
        self.memory_dir = memory_dir
        self.batch_size = batch_size
        self.enable_enhanced_embeddings = enable_enhanced_embeddings
        
        # Set up device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
            print(f"{self.get_time()} Using GPU for acceleration")
        else:
            self.device = "cpu"
            print(f"{self.get_time()} No GPU detected, using CPU (this will be slow)")
            
        # Set appropriate torch dtype
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Initialize memory system and model
        self._setup_memory_system()
        self._setup_model_and_tokenizer()
        self._setup_embedding_function()
        
        # Stats
        self.stats = {
            "items_added": 0,
            "items_skipped": 0,
            "batches_processed": 0,
            "time_elapsed": 0
        }
    
    def get_time(self) -> str:
        """Get formatted timestamp for logging."""
        return datetime.now().strftime("[%d/%m/%y %H:%M:%S] [MemoryImporter] ")
    
    def _setup_memory_system(self):
        """Set up the memory system with the same configuration as local_ai.py."""
        print(f"{self.get_time()} Initializing memory system in: {self.memory_dir}")
        
        self.memory_manager = MemoryManager(
            storage_path=self.memory_dir,
            embedding_dim=2048,  # Same as TinyLlama's hidden dimension
            enable_enhanced_embeddings=self.enable_enhanced_embeddings,
            max_enhancement_levels=4,
            auto_save=True,
            similarity_enhancement_factor=0.3
        )
    
    def _setup_model_and_tokenizer(self):
        """Load the model and tokenizer with appropriate settings."""
        print(f"{self.get_time()} Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.tokenizer.model_max_length = 2048  # Same as local_ai.py
        
        # Set up model loading options
        loading_options = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto" if self.device != "cpu" else None,
            "low_cpu_mem_usage": True,
        }
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **loading_options)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def _setup_embedding_function(self):
        """Set up embedding function for the memory system."""
        print(f"{self.get_time()} Setting up embedding function")
        
        # Define a function to generate embeddings for text
        def generate_embedding(text: str) -> np.ndarray:
            """Generate an embedding vector for a text string."""
            try:
                with torch.no_grad():
                    # Tokenize
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True
                    ).to(self.device)
                    
                    # Get model outputs
                    if hasattr(self.model, 'model'):
                        outputs = self.model.model(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask
                        )
                    else:
                        outputs = self.model(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            output_hidden_states=True
                        )
                    
                    # Get mean pooled representation
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
                    
                    return embedding
            except Exception as e:
                print(f"{self.get_time()} Error generating embedding: {e}")
                # Return a zero vector as fallback
                return np.zeros(2048)
        
        # Define a batch embedding function
        def generate_embeddings_batch(texts: List[str]) -> List[np.ndarray]:
            """Generate embeddings for multiple texts efficiently using batching."""
            # Skip empty inputs
            if not texts:
                return []
            
            try:
                # Tokenize all texts
                batch_tokenized = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    # Get model outputs
                    if hasattr(self.model, 'model'):
                        outputs = self.model.model(
                            input_ids=batch_tokenized['input_ids'],
                            attention_mask=batch_tokenized['attention_mask']
                        )
                    else:
                        outputs = self.model(
                            input_ids=batch_tokenized['input_ids'],
                            attention_mask=batch_tokenized['attention_mask'],
                            output_hidden_states=True
                        )
                    
                    # Get mean pooled representations
                    last_hidden_states = outputs.last_hidden_state
                    
                    # Mean pooling - take average of all token embeddings for each sequence
                    input_mask_expanded = batch_tokenized['attention_mask'].unsqueeze(-1).expand(last_hidden_states.size()).float()
                    sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    mean_pooled = sum_embeddings / sum_mask
                    
                    # Convert to numpy arrays
                    embeddings = mean_pooled.cpu().numpy()
                    
                    return list(embeddings)
                
            except Exception as e:
                print(f"{self.get_time()} Error in batch embedding: {e}")
                # Fall back to individual processing
                return [generate_embedding(text) for text in texts]
        
        # Set embedding functions in the memory manager
        self.memory_manager.set_embedding_function(
            function=generate_embedding,
            batch_function=generate_embeddings_batch
        )
    
    def _classify_content(self, text: str) -> Dict[str, str]:
        """
        Classify the content type to add appropriate metadata.
        
        Args:
            text: Text content to classify
            
        Returns:
            Metadata dictionary with classification
        """
        import re
        
        # Simple classifier for different content types
        metadata = {
            "source": "imported_content",
            "source_hint": "question",
            "timestamp": datetime.now().timestamp()
        }
        
        # Check for fitness-related keywords
        fitness_keywords = [
            "workout", "exercise", "muscle", "protein", "gym", "fitness", 
            "weight", "diet", "nutrition", "cardio", "fat loss", 
            "strength", "training", "motivation", "routine"
        ]
        
        if any(keyword in text.lower() for keyword in fitness_keywords):
            metadata["source_hint"] = "fitness_question"
            metadata["domain"] = "fitness"
        
        # Add more specific domain classification
        if "protein" in text.lower() or "diet" in text.lower() or "nutrition" in text.lower():
            metadata["domain"] = "nutrition"
        elif "motivation" in text.lower() or "quotes" in text.lower() or "mindset" in text.lower():
            metadata["domain"] = "motivation"
        elif "workout" in text.lower() or "exercise" in text.lower() or "routine" in text.lower():
            metadata["domain"] = "workout"
        
        return metadata
    
    def import_file(self, file_path: str) -> Dict[str, Any]:
        """
        Import a file into memory.
        
        Args:
            file_path: Path to the file to import
            
        Returns:
            Dictionary with import statistics
        """
        if not os.path.exists(file_path):
            print(f"{self.get_time()} Error: File not found: {file_path}")
            return self.stats
        
        print(f"{self.get_time()} Importing file: {file_path}")
        
        # Read file contents
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"{self.get_time()} Error reading file: {e}")
            return self.stats
        
        # Filter out empty lines
        lines = [line.strip() for line in lines if line.strip()]
        
        if not lines:
            print(f"{self.get_time()} No content found in file")
            return self.stats
        
        print(f"{self.get_time()} Found {len(lines)} items to import")
        
        # Process in batches
        start_time = time.time()
        
        for batch_idx in range(0, len(lines), self.batch_size):
            batch = lines[batch_idx:batch_idx + self.batch_size]
            batch_start_time = time.time()
            
            print(f"{self.get_time()} Processing batch {batch_idx//self.batch_size + 1}/{(len(lines) + self.batch_size - 1)//self.batch_size}")
            
            # Add each item to memory
            for item in batch:
                try:
                    # Classify content and create metadata
                    metadata = self._classify_content(item)
                    
                    # Add to memory
                    item_id = self.memory_manager.add(
                        content=item,
                        metadata=metadata
                    )
                    
                    if item_id:
                        self.stats["items_added"] += 1
                    else:
                        self.stats["items_skipped"] += 1
                
                except Exception as e:
                    print(f"{self.get_time()} Error adding item to memory: {e}")
                    self.stats["items_skipped"] += 1
            
            # Update batch stats
            self.stats["batches_processed"] += 1
            
            # Print batch progress
            batch_time = time.time() - batch_start_time
            items_per_second = len(batch) / max(0.001, batch_time)
            
            print(f"{self.get_time()} Batch complete: {len(batch)} items in {batch_time:.2f}s ({items_per_second:.1f} items/s)")
            
            # Force memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Update total stats
        self.stats["time_elapsed"] = time.time() - start_time
        print(f"{self.get_time()} Import complete!")
        print(f"{self.get_time()} Added: {self.stats['items_added']}, "
              f"Skipped: {self.stats['items_skipped']}, "
              f"Time: {self.stats['time_elapsed']:.2f}s")
        
        # Save memory at the end
        mem_stats = self.memory_manager.get_stats()
        print(f"{self.get_time()} Memory now contains {mem_stats['active_items']} items")
        
        return self.stats
    
    def cleanup(self):
        """Release resources."""
        print(f"{self.get_time()} Cleaning up resources...")
        
        # Clean up memory
        if hasattr(self, 'memory_manager'):
            self.memory_manager.cleanup()
        
        # Unload model
        if hasattr(self, 'model') and self.model is not None:
            self.model = None
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"{self.get_time()} Cleanup complete")


def main():
    """Main function to run the memory importer."""
    parser = argparse.ArgumentParser(description="Import a file into TinyLlama Chat memory")
    
    parser.add_argument("--file", type=str, required=True,
                      help="Path to file to import")
    parser.add_argument("--memory_dir", type=str, default="./memory",
                      help="Path to memory directory")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                      help="Model to use for embeddings")
    parser.add_argument("--device", type=str, default=None,
                      help="Device to use (cuda, cpu). If not specified, will autodetect.")
    parser.add_argument("--batch_size", type=int, default=50,
                      help="Batch size for processing")
    parser.add_argument("--no-enhanced-embeddings", action="store_true",
                      help="Disable enhanced embeddings")
    
    args = parser.parse_args()
    
    try:
        # Initialize importer
        importer = MemoryImporter(
            model_name=args.model,
            memory_dir=args.memory_dir,
            device=args.device,
            batch_size=args.batch_size,
            enable_enhanced_embeddings=not args.no_enhanced_embeddings
        )
        
        # Import file
        importer.import_file(args.file)
        
        # Clean up resources
        importer.cleanup()
        
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%d/%m/%y %H:%M:%S')}] [MemoryImporter] Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[{datetime.now().strftime('%d/%m/%y %H:%M:%S')}] [MemoryImporter] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()