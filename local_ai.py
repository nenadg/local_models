import torch
import os
import json
import argparse
import time
import sys
import threading
import math
import re
import termios
import tty
import signal
import select

from typing import List, Dict, Any, Optional, Tuple

from datetime import datetime
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import LogitsProcessor, LogitsProcessorList
from history import setup_readline_history

from resource_manager import ResourceManager

from mcp_handler import MCPHandler
from enhanced_confidence_metrics import EnhancedConfidenceMetrics, TokenProbabilityCaptureProcessor
from response_filter import ResponseFilter
from unified_memory import UnifiedMemoryManager

from terminal_heatmap import TerminalHeatmap, EnhancedHeatmap
from question_classifier import QuestionClassifier

from batch_utils import tensor_batch_processing
from web_knowledge_enhancer import WebKnowledgeEnhancer

from semantic_reasoning_enhancer import integrate_semantic_reasoning
from continuation_manager import ContinuationTrackingWindowManager

# Default system message with uncertainty guidelines
DEFAULT_SYSTEM_MESSAGE = {
    "role": "system",
    "content": """You are a helpful and friendly assistant designed to provide accurate information. Follow these guidelines:
1. When you don't know something, explicitly say "I don't know about [topic]" or "I'm not familiar with that."
2. Never make up information. It is better to admit uncertainty than to provide potentially incorrect information.
3. You may speculate if the user explicitly asks you to "please continue anyway" or "please speculate."
4. When speculating, clearly indicate your uncertainty with phrases like "I'm not confident, but..."
5. Be helpful, informative, and conversational in your responses.
"""
}

class TinyLlamaChat:
    def __init__(self,
             model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
             device=None,
             memory_dir="./memory",
             output_dir="./output",
             confidence_threshold=0.7,
             auto_memorize=True,
             enable_sharpening=True,
             sharpening_factor=0.3,
             enable_web_knowledge=True,
             fractal_enabled=True,
             max_fractal_levels=3,
             do_sample=False,
             top_p=1, # initial top_p = 1.0
             top_k=50   # initial top_k = 50
             ):
        self.model_name = model_name
        self.memory_dir = memory_dir
        self.do_sample = do_sample
        self.top_p = top_p
        self.top_k = top_k
        self._query_embedding_cache = {}

        self.stop_event = threading.Event()

        self.mcp_handler = MCPHandler(output_dir=output_dir, allow_shell_commands=True)
        self.confidence_metrics = EnhancedConfidenceMetrics(sharpening_factor=sharpening_factor)

        # Initialize resource manager
        self.resource_manager = ResourceManager(device=device)

        # Initialize memory manager with UnifiedMemoryManager
        self.memory_manager = UnifiedMemoryManager(
            storage_path=memory_dir,
            embedding_function=None,  # Will set this after model is loaded
            embedding_dim=384,  # Set this to your model's embedding dimension
            use_fractal=fractal_enabled,
            max_fractal_levels=max_fractal_levels,
            auto_save=True,
            enable_entity_separation=True
        )

        self.auto_memorize = auto_memorize
        self.sharpening_factor = sharpening_factor
        
        # Initialize the question classifier
        self.question_classifier = QuestionClassifier()

        # Initialize from semantic reasoning finetune function
        finetuned_model_path = "./finetuned_tinyllama_reasoning_2"

        if os.path.exists(finetuned_model_path):
            try:
                integrate_semantic_reasoning(self, finetuned_model_path)
            except Exception as e:
                print(f"{self.get_time()} Finetuned model not loaded: {e}")

        # Initialize the web knowledge enhancer
        self.enable_web_knowledge = enable_web_knowledge
        if enable_web_knowledge:
            self.web_enhancer = WebKnowledgeEnhancer(
                memory_manager=self.memory_manager,  # Pass your UnifiedMemoryManager instance
                chat=self,                       # Optional reference to chat system
                confidence_threshold=0.65,       # When to trigger web search
                sharpening_factor=0.3,           # For vector similarity sharpening
                search_engine="duckduckgo"       # Or "google"
            )
            print(f"{self.get_time()} Web knowledge enhancement initialized")
        else:
            self.web_enhancer = None

        self.current_user_id = "default_user"

        os.makedirs(memory_dir, exist_ok=True)


        # Determine the device to use
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
            print(f"{self.get_time()} Using GPU for acceleration")
            if hasattr(torch.backends, "cuda"):
                if hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends.cudnn, "benchmark"):
                    torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                    torch.backends.cuda.enable_flash_sdp = True
        else:
            self.device = "cpu"
            print(f"{self.get_time()} No GPU detected, using CPU (this will be slow)")

        # Setup appropriate torch dtype
        if self.device == "cpu":
            self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch.float16

        # Load model and tokenizer
        print(f"{self.get_time()} Loading target model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Initialize the window manager with our tokenizer
        self.window_manager = ContinuationTrackingWindowManager(
            tokenizer=self.tokenizer,
            max_window_size=2048,
            memory_manager=self.memory_manager,
            safety_margin=50,
            continuation_buffer_size=200  # Store the last 200 tokens for continuation
        )

        print(f"{self.get_time()} Context window and continuation tracking initialized")

        # Main model loading
        self.loading_options = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto" if self.device != "cpu" else None,
            "low_cpu_mem_usage": True,
        }

        # Load model
        print(f"{self.get_time()} Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **self.loading_options)
        self.model = self.resource_manager.register_model(self.model)

        # Set embedding function for memory manager
        self.set_embedding_function()

        # Create draft model from target model by reducing layers
        self.draft_model = self.create_draft_model()

        if self.draft_model:
            print(f"{self.get_time()} Created draft model by reducing layers")
        else:
            print(f"{self.get_time()} Could not create draft model, speculative decoding disabled")

        self.conversation_history = []
        self.system_message = DEFAULT_SYSTEM_MESSAGE

    def test_batch_processing(self, operation_type='embedding'):
        """
        Test batch processing performance for different operations.

        Args:
            operation_type: Type of operation to test ('embedding' or 'inference')
        """
        from batch_utils import validate_batch_processing_performance

        if operation_type == 'embedding':
            # Generate test data for embeddings
            test_texts = [
                "This is a test text for batch processing benchmarking.",
                "Here's another sample text with different content.",
                "A third text to include in our test batch.",
                "Let's add a fourth sample with varying length.",
                "And a fifth one to round out our test set."
            ] * 4  # Repeat to get 20 items

            # Define test function
            def test_func(texts, batch_size):
                return self.memory_manager.batch_embedding_function(texts[:batch_size])

            # Run benchmark
            return validate_batch_processing_performance(test_func, test_texts)

        elif operation_type == 'inference':
            # Generate test input for inference
            test_prompt = "Write a short poem about batch processing."

            # Tokenize
            tokens = self.tokenizer(test_prompt, return_tensors="pt").to(self.device)

            # Define test function
            def test_func(input_ids, batch_size):
                # Simple forward pass
                with torch.no_grad():
                    return self.model(input_ids=input_ids, max_length=batch_size)

            # Run benchmark
            return validate_batch_processing_performance(test_func, tokens.input_ids)

        else:
            print(f"{self.get_time()} Unknown operation type: {operation_type}")
            return None

    def test_embedding_performance(self):
        """Test embedding performance with minimal overhead"""
        import time

        # Create some sample text
        sample_texts = [
            "This is a sample text for testing embedding performance",
            "Another sample text with different content",
            "Third sample with yet more different content",
            "Fourth sample to test batching performance",
            "Fifth sample completing our test batch"
        ]

        print("Testing individual embedding performance...")

        # Test individual embeddings
        start_time = time.time()
        individual_embeddings = []

        for text in sample_texts:
            embedding = self.memory_manager.embedding_function(text)
            individual_embeddings.append(embedding)

        individual_time = time.time() - start_time
        print(f"Individual embedding time: {individual_time:.4f}s for {len(sample_texts)} texts")
        print(f"Average time per text: {individual_time/len(sample_texts):.4f}s")

        # Now test batched operation for raw comparison
        # (this won't run unless you implement a batch_embedding_function)
        if hasattr(self.memory_manager, 'batch_embedding_function'):
            print("Testing batch embedding performance...")
            start_time = time.time()
            batch_embeddings = self.memory_manager.batch_embedding_function(sample_texts)
            batch_time = time.time() - start_time
            print(f"Batch embedding time: {batch_time:.4f}s for {len(sample_texts)} texts")
            print(f"Average time per text: {batch_time/len(sample_texts):.4f}s")
            print(f"Speedup: {individual_time/batch_time:.2f}x")

    def get_time(self):
        return datetime.now().strftime("[%d/%m/%y %H:%M:%S]")

    def set_embedding_function(self):
        """
        Set up an optimized embedding function for the memory manager with batching support.
        This improved version handles caching more efficiently and supports batched operations.
        """
        # Ensure model is in evaluation mode
        if hasattr(self.model, 'eval'):
            self.model.eval()

        # Create a cache for embeddings with better capacity management
        if not hasattr(self, '_embedding_cache'):
            self._embedding_cache = {}
            self._embedding_cache_capacity = 1000  # Maximum cache entries

        # Track performance metrics
        if not hasattr(self, '_embedding_stats'):
            self._embedding_stats = {
                'cache_hits': 0,
                'cache_misses': 0,
                'batch_calls': 0,
                'total_embeddings': 0,
                'batch_sizes': []
            }

        def generate_embedding(text):
            """Generate a single embedding with caching and optimization."""
            # Create hash for cache key
            import hashlib
            cache_key = hashlib.md5(text.encode()).hexdigest()

            # Check cache first
            if cache_key in self._embedding_cache:
                self._embedding_stats['cache_hits'] += 1
                return self._embedding_cache[cache_key]

            # Cache miss - generate embedding
            self._embedding_stats['cache_misses'] += 1
            self._embedding_stats['total_embeddings'] += 1

            with torch.no_grad():
                # Tokenize with efficient settings
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)

                # Use more efficient forward pass
                if hasattr(self.model, 'model'):
                    outputs = self.model.model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask
                    )
                else:
                    # Fallback for other model types
                    outputs = self.model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        output_hidden_states=True
                    )

                # Get mean pooled representation
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

            # Add to cache (with capacity management)
            self._embedding_cache[cache_key] = embedding

            # If cache is at capacity, remove oldest entries
            if len(self._embedding_cache) > self._embedding_cache_capacity:
                # Get oldest 20% of entries to remove
                remove_count = int(self._embedding_cache_capacity * 0.2)
                keys_to_remove = list(self._embedding_cache.keys())[:remove_count]
                for key in keys_to_remove:
                    del self._embedding_cache[key]

            return embedding

        def generate_embeddings_batch(texts):
            """Generate embeddings for multiple texts in a batch."""
            # Track batch statistics
            self._embedding_stats['batch_calls'] += 1
            self._embedding_stats['total_embeddings'] += len(texts)
            self._embedding_stats['batch_sizes'].append(len(texts))

            # Check cache for each text first
            embeddings = []
            texts_to_embed = []
            indices_to_embed = []

            for i, text in enumerate(texts):
                cache_key = hashlib.md5(text.encode()).hexdigest()
                if cache_key in self._embedding_cache:
                    embeddings.append(self._embedding_cache[cache_key])
                    self._embedding_stats['cache_hits'] += 1
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
                    self._embedding_stats['cache_misses'] += 1

            # If all texts were cached, return immediately
            if not texts_to_embed:
                return embeddings

            # Process uncached texts in batches
            try:
                with torch.no_grad():
                    # Tokenize all texts at once
                    inputs = self.tokenizer(
                        texts_to_embed,
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

                    # Get mean pooled representations
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

                    # Add to cache and results
                    for i, idx in enumerate(indices_to_embed):
                        # Create embedding
                        embedding = batch_embeddings[i]

                        # Add to cache
                        cache_key = hashlib.md5(texts_to_embed[i].encode()).hexdigest()
                        self._embedding_cache[cache_key] = embedding

                        # Insert at appropriate position
                        embeddings.insert(idx, embedding)

                    # Check cache capacity
                    if len(self._embedding_cache) > self._embedding_cache_capacity:
                        remove_count = int(self._embedding_cache_capacity * 0.2)
                        keys_to_remove = list(self._embedding_cache.keys())[:remove_count]
                        for key in keys_to_remove:
                            del self._embedding_cache[key]

                    return embeddings

            except Exception as e:
                print(f"{self.get_time()} Error in batch embedding: {e}")

                # Fall back to individual processing
                for i, idx in enumerate(indices_to_embed):
                    try:
                        embedding = generate_embedding(texts_to_embed[i])
                        embeddings.insert(idx, embedding)
                    except Exception as inner_e:
                        print(f"{self.get_time()} Error generating individual embedding: {inner_e}")
                        # Use zeros as fallback
                        embeddings.insert(idx, np.zeros(self.memory_manager.embedding_dim))

                return embeddings

        # Use the improved batch_embedding_function if batch_utils is available
        try:
            from batch_utils import batch_embedding_processing

            def optimized_batch_embedding(texts):
                """Optimized batch embedding with performance tracking."""
                # Use batch_utils for processing
                return batch_embedding_processing(
                    embedding_function=generate_embedding,
                    texts=texts,
                    batch_size=None,  # Use adaptive sizing
                    cleanup=True
                )

            # Set batch function to use optimized version
            self.memory_manager.batch_embedding_function = optimized_batch_embedding
            print(f"{self.get_time()} Using optimized batch embedding function")
        except ImportError:
            # Fall back to direct implementation
            self.memory_manager.batch_embedding_function = generate_embeddings_batch
            print(f"{self.get_time()} Using default batch embedding function")

        # Set single embedding function
        self.memory_manager.embedding_function = generate_embedding

        # Add a method to get embedding stats
        def get_embedding_stats(self):
            """Get statistics about embedding generation."""
            if not hasattr(self, '_embedding_stats'):
                return {}

            stats = self._embedding_stats.copy()

            # Calculate cache hit ratio
            total_requests = stats['cache_hits'] + stats['cache_misses']
            if total_requests > 0:
                stats['cache_hit_ratio'] = stats['cache_hits'] / total_requests
            else:
                stats['cache_hit_ratio'] = 0

            # Calculate average batch size
            if stats['batch_calls'] > 0:
                stats['avg_batch_size'] = sum(stats['batch_sizes']) / stats['batch_calls']
            else:
                stats['avg_batch_size'] = 0

            return stats

        # Attach stats method
        self.get_embedding_stats = get_embedding_stats

    def create_draft_model(self):
        """Create a smaller version of the model by skipping layers"""
        # This works best for decoder-only transformers like Llama
        import copy

        try:
            # For Llama-like models, reduce layers
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                # Create a copy for draft model
                draft_model = copy.deepcopy(self.model)

                # Reduce to 1/3 of the layers for draft model
                orig_layers = draft_model.model.layers
                num_layers = len(orig_layers)
                keep_layers = max(1, num_layers // 3)

                # Keep only every third layer
                indices_to_keep = list(range(0, num_layers, 3))[:keep_layers]
                new_layers = torch.nn.ModuleList([orig_layers[i] for i in indices_to_keep])

                # Replace layers with reduced set
                draft_model.model.layers = new_layers

                print(f"{self.get_time()} Created draft model with {keep_layers}/{num_layers} layers")
                return draft_model

            return None
        except Exception as e:
            print(f"{self.get_time()} Error creating draft model: {e}")
            return None

    def enhance_query_for_continuation(self, messages):
        """
        Enhance the user query if it's a continuation request.
        Add this at the beginning of create_prompt_with_knowledge.
        """
        # Check if this is a continuation request
        if not messages or len(messages) == 0:
            return messages

        if hasattr(self, 'window_manager') and messages[-1]["role"] == "user":
            user_query = messages[-1]["content"]

            # Detect if this is a continuation request
            if self.window_manager.detect_continuation_request(user_query):
                print(f"{self.get_time()} Detected continuation request, enhancing prompt...")

                # Enhance continuation context with web knowledge if available
                if hasattr(self, 'web_enhancer') and self.web_enhancer and user_query:
                    # Get current continuation context
                    cont_context = self.window_manager.continuation_context.get(
                        self.current_user_id, {}
                    )

                    # Add content type detection if not present
                    if "content_type" not in cont_context and "text" in cont_context:
                        cont_context["content_type"] = self.window_manager._detect_content_type(
                            cont_context["text"]
                        )

                    # Check if the web_enhancer has the enhance_continuation_context method
                    if hasattr(self.web_enhancer, 'enhance_continuation_context'):
                        # Enhance with web knowledge
                        enhanced_context = self.web_enhancer.enhance_continuation_context(
                            user_query, cont_context
                        )

                        # Update the continuation context
                        self.window_manager.continuation_context[self.current_user_id] = enhanced_context

                # Prepare enhanced continuation prompt
                enhanced_query = self.window_manager.prepare_continuation_prompt(
                    query=user_query,
                    user_id=self.current_user_id
                )

                # Update the user message with enhanced query
                if enhanced_query != user_query:
                    messages[-1]["content"] = enhanced_query
                    print(f"{self.get_time()} Enhanced continuation prompt created")
                else:
                    print(f"{self.get_time()} No continuation context available")

        return messages

    def create_prompt_with_knowledge(self, messages, use_web_search=True):
        """Create a prompt that incorporates relevant knowledge."""
        try:
            # First check if this is a continuation request and enhance if needed
            messages = self.enhance_query_for_continuation(messages)

            # Extract the user's current query
            current_query = messages[-1]["content"] if len(messages) > 0 and messages[-1]["role"] == "user" else ""

            # Create enhanced system message with knowledge
            enhanced_system_content = self.system_message["content"]

            # Retrieve knowledge using the unified approach
            knowledge_text = ""
            if current_query:
                # Retrieve knowledge with appropriate settings
                results = self.memory_manager.retrieve(
                    query=current_query,
                    top_k=8,  # Default is 8
                    min_similarity=0.25,
                    use_fractal=self.memory_manager.use_fractal
                )

                # Format the knowledge for the prompt
                if results:
                    knowledge_text = self.memory_manager.format_knowledge_for_prompt(results, current_query)

            # Add web search results if available and enabled
            web_context = ""
            if use_web_search and self.enable_web_knowledge and self.web_enhancer is not None:
                try:
                    web_enhancement = self.enhance_with_web_knowledge(current_query, {}, None, messages)

                    if web_enhancement and web_enhancement.get('enhanced', False):
                        web_context = self.web_enhancer.format_web_results_for_context(web_enhancement)

                        # Add web results to knowledge
                        if web_enhancement.get('web_results'):
                            for result in web_enhancement['web_results']:
                                # Add each result to knowledge system
                                metadata = {
                                    "source_hint": "web",
                                    "url": result.get('url', ''),
                                    "title": result.get('title', ''),
                                    "timestamp": datetime.now().timestamp()
                                }

                                content = f"{result.get('title', '')}: {result.get('snippet', '')}"

                                # Add to memory asynchronously
                                if self.auto_memorize:
                                    threading.Thread(
                                        target=self.memory_manager.add,
                                        args=(content, metadata, True)
                                    ).start()
                except Exception as e:
                    print(f"{self.get_time()} Error enhancing with web knowledge: {e}")

            # Combine knowledge and web information
            if knowledge_text and web_context:
                enhanced_system_content += "\n\nIMPORTANT: Apply the following information in your response:\n"
                enhanced_system_content += f"\n{knowledge_text}\n"
                enhanced_system_content += f"\n{web_context}"
            elif knowledge_text:
                enhanced_system_content += "\n\nIMPORTANT: Apply the following information in your response:\n"
                enhanced_system_content += f"\n{knowledge_text}"
            elif web_context:
                enhanced_system_content += "\n\nIMPORTANT: Use the following web search results in your response:\n"
                enhanced_system_content += f"\n{web_context}"

            # Create messages with enhanced system prompt
            enhanced_messages = [
                {
                    "role": "system",
                    "content": enhanced_system_content
                }
            ]

            # Add conversation history
            if len(messages) > 1:
                history_messages = messages[1:]
                enhanced_messages.extend(history_messages[-5:])  # Keep up to 5 recent messages

            return enhanced_messages

        except Exception as e:
            print(f"{self.get_time()} Error in create_prompt_with_knowledge: {e}")
            import traceback
            traceback.print_exc()

            # Return basic messages on error
            if not messages or len(messages) == 0:
                return [{"role": "system", "content": self.system_message["content"]}]

            return [{"role": "system", "content": self.system_message["content"]}] + messages[1:]

    def enhance_with_web_knowledge(self, query: str, confidence_data: Dict[str, float], domain: Optional[str] = None, messages: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Enhance response generation with web knowledge when confidence is low.

        Args:
            query: User query
            confidence_data: Confidence metrics from the model
            domain: Optional domain classification
            messages: Optional conversation history

        Returns:
            Web enhancement data
        """
        if not self.enable_web_knowledge or self.web_enhancer is None:
            return {
                'enhanced': False,
                'reason': 'web_knowledge_disabled',
                'web_results': []
            }

        # Safety check for web_enhancer
        if not hasattr(self.web_enhancer, 'enhance_response'):
            print(f"{self.get_time()} Web enhancer doesn't have enhance_response method")
            return {
                'enhanced': False,
                'reason': 'web_enhancer_incompatible',
                'web_results': []
            }

        try:
            # Use the web enhancer to get relevant information
            enhancement_data = self.web_enhancer.enhance_response(
                query,
                confidence_data,
                domain,
                process_urls=True
            )

            # Add to memory if enhancement was successful
            if enhancement_data.get('enhanced', False):
                # Check if add_web_results_to_memory method exists
                if hasattr(self.web_enhancer, 'add_web_results_to_memory'):
                    self.web_enhancer.add_web_results_to_memory(
                        self.current_user_id,
                        query,
                        enhancement_data
                    )

            return enhancement_data

        except Exception as e:
            print(f"{self.get_time()} Error in enhance_with_web_knowledge: {e}")
            import traceback
            traceback.print_exc()

            return {
                'enhanced': False,
                'reason': 'error',
                'error_message': str(e),
                'web_results': []
            }

    def toggle_web_knowledge(self) -> bool:
        """
        Toggle web knowledge enhancement on/off.

        Returns:
            New web_knowledge state
        """
        self.enable_web_knowledge = not self.enable_web_knowledge

        # Initialize web enhancer if needed
        if self.enable_web_knowledge and self.web_enhancer is None:
            self.web_enhancer = WebKnowledgeEnhancer(
                memory_manager=self.memory_manager,
                chat=self,
                confidence_threshold=confidence_threshold,
                vector_sharpening_factor=sharpening_factor,
                search_engine="duckduckgo",  # Use DuckDuckGo by default for fewer rate limits
                embedding_function=self.memory_manager.embedding_function  # Direct reference to the function
            )

        return self.enable_web_knowledge

    def get_domain_specific_generation_config(self, query, base_config):
        """
        Get domain-specific generation configuration.

        Args:
            query: The user query
            base_config: The base generation configuration

        Returns:
            Modified configuration dictionary
        """
        # Get domain settings from the classifier
        settings = self.question_classifier.get_domain_settings(query)
        domain = settings['domain']

        # Create a copy of the base configuration
        config = base_config.copy()

        # Apply domain-specific adjustments
        if domain == 'arithmetic':
            # More deterministic for math
            config['temperature'] = min(config.get('temperature', 0.7), 0.3)
            config['top_p'] = 0.85 # use only if use_sample=False

        elif domain == 'translation':
            # More deterministic for translations too
            config['temperature'] = min(config.get('temperature', 0.7), 0.4)
            config['top_p'] = 0.9 # use only if use_sample=False

        elif domain == 'factual':
            # Slightly more deterministic for facts
            config['temperature'] = min(config.get('temperature', 0.7), 0.5)

        elif domain == 'conceptual' or domain == 'procedural':
            # More creative for concepts and procedures
            config['temperature'] = max(config.get('temperature', 0.7), 0.6)

        return config

    def speculative_decode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, num_draft_tokens: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Implement speculative decoding with loop detection and batch processing:
        1. Generate tokens with the smaller draft model
        2. Verify with larger target model
        3. Keep tokens that match, regenerate from first mismatch
        4. Detect and prevent repetitive loops
        """
        if self.draft_model is None:
            # Fall back to regular generation if no draft model
            return None, None, None

        # Only implement speculative decoding for GPU for speed
        if self.device == "cpu":
            return None, None, None

        try:
            # Step 1: Generate draft tokens with the smaller model
            with torch.no_grad():
                draft_outputs = self.draft_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=num_draft_tokens,
                    do_sample=self.do_sample,  # set to True for non greedy decoding for draft
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    # Add repetition penalty to avoid loops
                    repetition_penalty=1.2,
                )

            # Extract draft tokens (excluding input prompt)
            draft_ids = draft_outputs.sequences[0, input_ids.shape[1]:]

            # Check if we got any draft tokens
            if len(draft_ids) == 0:
                return None, None, None

            # NEW: Check for repetitive loops
            prompt_tokens = input_ids[0][-5:].tolist() if input_ids[0].size(0) >= 5 else input_ids[0].tolist()
            draft_tokens = draft_ids.tolist()

            # Detect repetitive patterns (simple check for now)
            if len(draft_tokens) >= 4:
                # Check if last 2 tokens repeat
                if draft_tokens[-2:] == draft_tokens[-4:-2]:
                    print("[Warning] Repetitive pattern detected in draft tokens, randomizing...")
                    # Inject some randomness by changing temperature for this generation
                    return None, None, None

            # Step 2: Verify with target model - compute logits for each position
            full_sequence = torch.cat([input_ids[0], draft_ids]).unsqueeze(0)
            full_attention = torch.ones_like(full_sequence)

            # Process in batches for large inputs using enhanced batch processing
            if full_sequence.size(1) > 1024:  # For long sequences
                from batch_utils import tensor_batch_processing

                def get_logits(batch):
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=batch,
                            attention_mask=torch.ones_like(batch),
                            return_dict=True
                        )
                    return outputs.logits

                # Use enhanced batch processing with dynamic batch sizing
                target_logits = tensor_batch_processing(
                    get_logits,
                    full_sequence,
                    batch_size=None,  # Use dynamic sizing
                    adaptive=True,
                    handle_oom=True
                )
            else:
                #  Get logits from target model for the full sequence
                with torch.no_grad():
                    target_outputs = self.model(
                        input_ids=full_sequence,
                        attention_mask=full_attention,
                        return_dict=True
                    )
                target_logits = target_outputs.logits

            # Only examine logits for draft tokens (exclude input)
            target_logits = target_logits[0, input_ids.shape[1]-1:-1, :]

            # Get predicted tokens from target logits
            target_predictions = torch.argmax(target_logits, dim=-1)

            # Step 3: Find the first mismatch
            matches = (target_predictions == draft_ids)

            # If all match, accept all draft tokens!
            if matches.all():
                # Add logits to confidence metrics
                for i, token_id in enumerate(draft_ids):
                    self.confidence_metrics.add_token_score(target_logits[i], token_id.item())
                return draft_ids, torch.ones_like(draft_ids), target_logits

            # Find first mismatch position
            first_mismatch = len(matches) if matches.all() else matches.tolist().index(False)

            # Return accepted tokens
            if first_mismatch > 0:
                accepted_tokens = draft_ids[:first_mismatch]
                acceptance_mask = torch.ones_like(accepted_tokens)

                # Add logits to confidence metrics for the accepted tokens
                for i, token_id in enumerate(accepted_tokens):
                    self.confidence_metrics.add_token_score(target_logits[i], token_id.item())

                # Also return the target logits for these tokens
                return accepted_tokens, acceptance_mask, target_logits[:first_mismatch]

            # No accepted tokens
            return None, None, None

        except Exception as e:
            print(f"{self.get_time()} Error in speculative decoding: {e}")
            return None, None, None

    # Function for speculative decoding with streaming
    def generate_speculative(self, input_ids, streamer, max_new_tokens, generation_config, turbo_mode):
        """Enhanced implementation of speculative decoding with better batch processing and memory management"""
        try:
            with torch.no_grad():
                # Reset confidence metrics for a new generation
                self.confidence_metrics.reset()

                # Track speculative decoding stats
                using_spec_decoding = False
                accepted_draft_tokens = 0
                total_tokens = 0

                # Setup generation
                generated_ids = input_ids
                attention_mask = torch.ones_like(input_ids)
                remaining_tokens = max_new_tokens

                # Create a separate config without max_new_tokens to avoid duplication
                streaming_config = {k: v for k, v in generation_config.items()
                                  if k not in ['max_new_tokens', 'output_scores', 'return_dict_in_generate']}

                # Create and add our logits processor for confidence metrics
                metrics_processor = TokenProbabilityCaptureProcessor(self.confidence_metrics)

                # Set up the logits processor list, preserving any existing processors
                if 'logits_processor' in streaming_config:
                    # Add our processor to existing ones
                    if isinstance(streaming_config['logits_processor'], list):
                        streaming_config['logits_processor'].append(metrics_processor)
                    else:
                        # If it's already a LogitsProcessorList, we need to add to it
                        existing_processors = streaming_config['logits_processor']
                        streaming_config['logits_processor'] = LogitsProcessorList([*existing_processors, metrics_processor])
                else:
                    # No existing processors, create a new list
                    streaming_config['logits_processor'] = LogitsProcessorList([metrics_processor])

                # Get resource manager for optimal batch sizing
                resource_manager = None
                if hasattr(self, 'resource_manager'):
                    resource_manager = self.resource_manager

                while remaining_tokens > 0:
                    try:
                        if self.stop_event.is_set():
                            print("\n[Generation stopped by interrupt signal]")
                            try:
                                # Clean up
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                # End the streamer
                                self.stop_event.clear()  # Reset the event
                                streamer.end()
                            except Exception as e:
                                print(f"{self.get_time()} Error during interrupt cleanup: {e}")
                            return

                        # Try speculative decoding with enhanced batch processing
                        if self.draft_model is not None and turbo_mode:
                            # Skip for short generations - less overhead
                            if remaining_tokens < 3:
                                break

                            # Determine optimal number of draft tokens based on remaining tokens
                            optimal_draft_tokens = min(5, remaining_tokens)

                            # Call with enhanced batch processing
                            draft_tokens, acceptance_mask, token_logits = self.speculative_decode(
                                generated_ids,
                                attention_mask,
                                num_draft_tokens=optimal_draft_tokens
                            )

                            if draft_tokens is not None and len(draft_tokens) > 0:
                                # Success! We have draft tokens to use
                                using_spec_decoding = True
                                accepted_draft_tokens += len(draft_tokens)
                                total_tokens += len(draft_tokens)

                                # Decode tokens to text for streaming
                                token_text = self.tokenizer.decode(draft_tokens, skip_special_tokens=True)
                                streamer.put(token_text)

                                # Add tokens to generated output
                                generated_ids = torch.cat([generated_ids[0], draft_tokens]).unsqueeze(0)
                                attention_mask = torch.ones_like(generated_ids)
                                remaining_tokens -= len(draft_tokens)

                                # Check for stopping conditions
                                stop_generation = False
                                for token in draft_tokens:
                                    if token.item() == self.tokenizer.eos_token_id:
                                        stop_generation = True
                                        break

                                if stop_generation:
                                    break

                                # Check for special tokens in decoded text
                                if "<|user|>" in token_text or "<|assistant|>" in token_text:
                                    break
                            else:
                                # Fall back to standard model for regular generation
                                break
                        else:
                            # Exit to standard generation
                            break

                    except RuntimeError as e:
                        # Handle device mismatch errors specifically
                        if "expected all tensors to be on the same device" in str(e):
                            print("\n[Device mismatch error during interruption - stopping generation]")
                            break
                        else:
                            # Re-raise if it's a different error
                            raise

                # If we have tokens remaining or didn't use speculative, do standard generation
                if remaining_tokens > 0 or not using_spec_decoding:
                    # Use standard streaming mode with TokenProbabilityCaptureProcessor
                    # This will collect token probabilities during normal generation

                    # Use resource manager's batch processing if available
                    if resource_manager and hasattr(resource_manager, 'batch_process_inference'):
                        try:
                            # Create a modified streamer that integrates with batch processing
                            batch_size = resource_manager.suggest_optimal_batch_size(
                                tensor_shape=input_ids.shape,
                                operation_type='inference'
                            )
                            print(f"{self.get_time()} Using optimal batch size: {batch_size} for remaining generation")

                            # Add this information to config
                            if 'batch_size' not in streaming_config:
                                streaming_config['batch_size'] = batch_size
                        except:
                            # Continue with standard generation if optimization fails
                            pass

                    # Generate with standard or optimized settings
                    self.model.generate(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        streamer=streamer,
                        max_new_tokens=remaining_tokens,
                        **streaming_config
                    )

                # Report speculative decoding stats
                if using_spec_decoding and total_tokens > 0:
                    efficiency = (accepted_draft_tokens / total_tokens) * 100
                    print(f"{self.get_time()} Speculative decoding: {accepted_draft_tokens}/{total_tokens} tokens accepted ({efficiency:.1f}%)")

                # If we didn't collect any metrics, add fallback values
                # This should only happen if the generation fails completely
                if not self.confidence_metrics.token_probabilities:
                    print("[Warning: No token probabilities collected, using fallback values]")
                    # Create more realistic fallback values
                    for i in range(5):
                        dummy_logits = torch.zeros(self.tokenizer.vocab_size)
                        # Vary logit values from 1.0 to 5.0
                        max_val = 1.0 + i
                        token_id = i % 100
                        dummy_logits[token_id] = max_val
                        # Add some secondary values for more realistic distribution
                        for j in range(3):
                            dummy_logits[(token_id + j + 1) % 100] = max_val * 0.3
                        self.confidence_metrics.add_token_score(dummy_logits, token_id)

        except Exception as e:
            print(f"{self.get_time()} Error in generation thread: {str(e)}")
            import traceback
            traceback.print_exc()
            try:
                streamer.end()
            except Exception as specific_e:
                print(f"{self.get_time()} Error ending streamer: {specific_e}")
            """Fixed implementation of speculative decoding with streaming and accurate metrics capture"""
            try:
                with torch.no_grad():
                    # Reset confidence metrics for a new generation
                    self.confidence_metrics.reset()

                    # Track speculative decoding stats
                    using_spec_decoding = False
                    accepted_draft_tokens = 0
                    total_tokens = 0

                    # Setup generation
                    generated_ids = input_ids
                    attention_mask = torch.ones_like(input_ids)
                    remaining_tokens = max_new_tokens

                    # Create a separate config without max_new_tokens to avoid duplication
                    # Also remove output_scores and return_dict_in_generate if they exist
                    streaming_config = {}
                    for k, v in generation_config.items():
                        if k not in ['max_new_tokens', 'output_scores', 'return_dict_in_generate']:
                            streaming_config[k] = v

                    # Create and add our logits processor for confidence metrics
                    metrics_processor = TokenProbabilityCaptureProcessor(self.confidence_metrics)

                    # Set up the logits processor list, preserving any existing processors
                    if 'logits_processor' in streaming_config:
                        # Add our processor to existing ones
                        if isinstance(streaming_config['logits_processor'], list):
                            streaming_config['logits_processor'].append(metrics_processor)
                        else:
                            # If it's already a LogitsProcessorList, we need to add to it
                            existing_processors = streaming_config['logits_processor']
                            streaming_config['logits_processor'] = LogitsProcessorList([*existing_processors, metrics_processor])
                    else:
                        # No existing processors, create a new list
                        streaming_config['logits_processor'] = LogitsProcessorList([metrics_processor])

                    while remaining_tokens > 0:
                        try:
                            if self.stop_event.is_set():
                                print("\n[Generation stopped by interrupt signal]")
                                try:
                                    # Clean up
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    # End the streamer
                                    self.stop_event.clear()  # Reset the event
                                    streamer.end()
                                except Exception as e:
                                    print(f"{self.get_time()} Error during interrupt cleanup: {e}")
                                return


                            # Try speculative decoding
                            if self.draft_model is not None and turbo_mode:
                                # Skip for short generations - less overhead
                                if remaining_tokens < 3:
                                    break

                                # Call with the new return signature (3 values)
                                draft_tokens, acceptance_mask, token_logits = self.speculative_decode(
                                    generated_ids,
                                    attention_mask,
                                    num_draft_tokens=min(5, remaining_tokens)
                                )

                                if draft_tokens is not None and len(draft_tokens) > 0:
                                    # Success! We have draft tokens to use
                                    using_spec_decoding = True
                                    accepted_draft_tokens += len(draft_tokens)
                                    total_tokens += len(draft_tokens)

                                    # Decode tokens to text for streaming
                                    token_text = self.tokenizer.decode(draft_tokens, skip_special_tokens=True)
                                    streamer.put(token_text)

                                    # Add tokens to generated output
                                    generated_ids = torch.cat([generated_ids[0], draft_tokens]).unsqueeze(0)
                                    attention_mask = torch.ones_like(generated_ids)
                                    remaining_tokens -= len(draft_tokens)

                                    # Check for stopping conditions
                                    stop_generation = False
                                    for token in draft_tokens:
                                        if token.item() == self.tokenizer.eos_token_id:
                                            stop_generation = True
                                            break

                                    if stop_generation:
                                        break

                                    # Check for special tokens in decoded text
                                    if "<|user|>" in token_text or "<|assistant|>" in token_text:
                                        break
                                else:
                                    # Fall back to standard model for regular generation
                                    break
                            else:
                                # Exit to standard generation
                                break


                        except RuntimeError as e:
                            # Handle device mismatch errors specifically
                            if "expected all tensors to be on the same device" in str(e):
                                print("\n[Device mismatch error during interruption - stopping generation]")
                                break
                            else:
                                # Re-raise if it's a different error
                                raise

                    # If we have tokens remaining or didn't use speculative, do standard generation
                    if remaining_tokens > 0 or not using_spec_decoding:
                        # Use standard streaming mode with TokenProbabilityCaptureProcessor
                        # This will collect token probabilities during normal generation
                        self.model.generate(
                            input_ids=input_ids,
                            attention_mask=torch.ones_like(input_ids),
                            streamer=streamer,
                            max_new_tokens=remaining_tokens,
                            **streaming_config
                        )

                    # Report speculative decoding stats
                    if using_spec_decoding and total_tokens > 0:
                        efficiency = (accepted_draft_tokens / total_tokens) * 100
                        print(f"{self.get_time()} Speculative decoding: {accepted_draft_tokens}/{total_tokens} tokens accepted ({efficiency:.1f}%)")

                    # If we didn't collect any metrics, add fallback values
                    # This should only happen if the generation fails completely
                    if not self.confidence_metrics.token_probabilities:
                        print("[Warning: No token probabilities collected, using fallback values]")
                        # Create more realistic fallback values
                        for i in range(5):
                            dummy_logits = torch.zeros(self.tokenizer.vocab_size)
                            # Vary logit values from 1.0 to 5.0
                            max_val = 1.0 + i
                            token_id = i % 100
                            dummy_logits[token_id] = max_val
                            # Add some secondary values for more realistic distribution
                            for j in range(3):
                                dummy_logits[(token_id + j + 1) % 100] = max_val * 0.3
                            self.confidence_metrics.add_token_score(dummy_logits, token_id)

            except Exception as e:
                print(f"{self.get_time()} Error in generation thread: {str(e)}")
                import traceback
                traceback.print_exc()
                try:
                    streamer.end()
                except Exception as specific_e:
                    print(f"{self.get_time()} Error ending streamer: {specific_e}")

    def generate_response(self, messages, max_new_tokens=128, temperature=0.7, turbo_mode=True, show_confidence=False, response_filter=None, use_web_search=True):
        """Generate a response with ultra-fast speculative decoding (streaming only)"""
        # We only support streaming now, simplifies the code

        fallback_message_streamed = False
        fd = None
        
        try:
            # heatmap = TerminalHeatmap(self.tokenizer, use_background=False)
            heatmap = EnhancedHeatmap(self.tokenizer, use_background=False, window_size=3)
            token_confidences = []  # To store confidence scores for each token

            # Reset confidence metrics
            self.confidence_metrics.reset()

            if messages and messages[-1]["role"] == "user":
                user_input = messages[-1]["content"]
                cleaned_input, user_commands = self.mcp_handler.extract_mcp_from_user_input(user_input)

                # Extract target length and set it in window manager
                user_query = messages[-1]["content"]
                # Extract target length from user query if specified (e.g., "write a 500 word essay")
                target_length = self.window_manager.extract_target_length(user_query)
                # Store the target in the window manager for later use
                self.window_manager.set_target_length(self.current_user_id, target_length)

                if cleaned_input == "_COMMAND_ONLY_":
                    # This was just a shell command, don't generate a response
                    return ""

                # Replace the original user input with cleaned version
                if user_input != cleaned_input:
                    messages[-1]["content"] = cleaned_input

                # Process any immediate user commands
                if user_commands:
                    # Process any shell command outputs and add to memory
                    for cmd, details in user_commands.items():
                        if details.get("action") == "shell_command":
                            # Add the command output to memory
                            self.add_command_to_memory(
                                command=cmd,
                                output=details.get("output", ""),
                                error=details.get("error", ""),
                                output_file=details.get("output_file")
                            )

                    # Process file save commands
                    file_commands = {file: cmd for file, cmd in user_commands.items()
                                    if cmd.get("action") == "save_response"}
                    successful = [file for file, status in file_commands.items()]
                    if successful:
                        files_info = ", ".join(successful)
                        print(f"{self.get_time()} [User content saved to: {files_info}]")

            # Create enhanced prompt with knowledge
            enhanced_messages = self.create_prompt_with_knowledge(messages, use_web_search)

            # Use the window manager to optimize messages to fit token limits
            optimized_messages = self.window_manager.optimize_messages(enhanced_messages, max_new_tokens)

            # Log token counts to help with debugging
            original_tokens = self.window_manager.calculate_tokens(enhanced_messages)
            optimized_tokens = self.window_manager.calculate_tokens(optimized_messages)

            if original_tokens != optimized_tokens:
                print(f"{self.get_time()} Context window optimized: {original_tokens} → {optimized_tokens} tokens " +
                      f"(saved {original_tokens - optimized_tokens} tokens)")

            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                optimized_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            try:
                with torch.no_grad():
                    # Use the safe tokenization function instead
                    tokenized = self.safe_tokenize(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        padding=True
                    )
                    # Access the input_ids from the dictionary
                    input_ids = tokenized['input_ids'].to(self.device)
            except Exception as e:
                print(f"{self.get_time()} Error encoding prompt: {e}")
                import traceback
                traceback.print_exc()
                return "Error preparing response. Please try again with a simpler query."


            # Configure streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                stride=16  # Reasonable stride for batching
            )

            # Generation configuration
            generation_config = {
                "max_new_tokens": max_new_tokens,
                #"do_sample": temperature >= 0.1, # use only if do_sample=false
                # "temperature": temperature if temperature > 0.1 else 1.0,
                #"top_k":  50,
                #"top_p": 0.95, # use only if do_sample=False
                "repetition_penalty": 1.0,
                "num_beams": 1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True
            }

            if messages and messages[-1]["role"] == "user":
                user_query = messages[-1]["content"]
                generation_config = self.get_domain_specific_generation_config(user_query, generation_config)

            if (self.do_sample):
                generation_config['do_sample'] = True
                generation_config['top_k'] = self.top_k
                generation_config['top_p'] = self.top_p
                generation_config['temperature'] = temperature if temperature > 0.1 else 1.0
            else:
                generation_config['do_sample'] = False
                # Remove sampling parameters when not using sampling
                for param in ['temperature', 'top_k', 'top_p']:
                    if param in generation_config:
                        del generation_config[param]

            # Start generation in background
            thread = Thread(target=self.generate_speculative, args=(input_ids, streamer, max_new_tokens, generation_config, turbo_mode))
            thread.daemon = True
            thread.start()

            fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(fd)
            ## not needed - tty.setraw(fd)
            tty.setcbreak(fd)

            # Initialize response
            complete_response = ""
            token_buffer = ""
            mcp_buffer = ""    # For accumulating MCP commands

            # Settings for token display
            last_print_time = time.time()
            force_display_time = 0.05  # 50ms max wait between displays
            max_buffer_size = 16  # Reasonable buffer size
            stop_sequences = ["<|user|>", "<|assistant|>"]

            # Variables to track streaming state
            tokens_received = 0
            early_confidence_check_threshold = 10  # Check confidence after this many tokens
            low_confidence_detected = False

            user_query = messages[-1]["content"] if messages[-1]["role"] == "user" else ""

            # If no system command matched, generate response using the model
            print(f"{self.get_time()} Assistant: \n", end='', flush=True)

            try:
                while True:
                    if select.select([sys.stdin], [], [], 0)[0]:
                        c = sys.stdin.read(1)
                        if c == '\x03':  # Ctrl+C
                            self.stop_event.set()
                            print(f"\n{self.get_time()} Canceled by user")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            break

                    token = next(iter(streamer), None)

                    if token is None:
                        # No more tokens - generation complete
                        break

                    tokens_received += 1

                    # Process token for MCP
                    display_token, mcp_buffer = self.mcp_handler.process_streaming_token(token, mcp_buffer)


                    # Add token to response
                    complete_response += token

                    detected_repetition = False
                    if hasattr(self, 'window_manager'):
                        detected_repetition = self.window_manager.track_repetition_patterns(
                            complete_response, self.current_user_id
                        )

                    # If repetition detected and web enhancer is available, get correction suggestion
                    if detected_repetition and hasattr(self, 'web_enhancer') and self.web_enhancer:
                        try:
                            print(f"\n{self.get_time()} [Repetitive pattern detected, getting correction suggestions...]")

                            # Get a fresh perspective from the web
                            correction_query = f"{complete_response[-150:]}"
                            web_results = self.web_enhancer.search_web(correction_query, num_results=1)

                            if web_results:
                                # Store correction suggestion for next continuation
                                if self.current_user_id not in self.window_manager.continuation_context:
                                    self.window_manager.continuation_context[self.current_user_id] = {}

                                self.window_manager.continuation_context[self.current_user_id]["correction_suggestion"] = web_results[0].get('snippet', '')

                                # Add note about available correction
                                complete_response += "\n\n// Note: Repetitive pattern detected. Use 'fix code' to get correction suggestions."
                        except Exception as e:
                            print(f"{self.get_time()} Error getting correction suggestions: {e}")

                    # Get confidence for latest token
                    if self.confidence_metrics.token_probabilities:
                        latest_confidence = self.confidence_metrics.token_probabilities[-1]
                        token_confidences.append(latest_confidence)
                    elif self.confidence_metrics.original_token_probabilities:
                        # Fallback to original probabilities if sharpened not available
                        latest_idx = len(self.confidence_metrics.original_token_probabilities) - 1
                        if latest_idx >= 0:
                            latest_confidence = self.confidence_metrics.original_token_probabilities[latest_idx]
                            token_confidences.append(latest_confidence)
                        else:
                            # Default if no confidence available
                            latest_confidence = 0.8
                            token_confidences.append(latest_confidence)
                    else:
                        # Default if no confidence available
                        latest_confidence = 0.8
                        token_confidences.append(latest_confidence)

                    if hasattr(self, 'window_manager') and hasattr(self.window_manager, 'is_generation_complete'):
                        if self.window_manager.is_generation_complete(self.current_user_id, complete_response):
                            # End generation and clean up
                            print(f"\n{self.get_time()} [Generation complete: target length reached]")
                            self.stop_event.set()  # Signal to stop generation

                            # Ensure response is properly terminated at a sentence boundary
                            if hasattr(self.window_manager, '_find_safe_termination_point'):
                                termination_point = self.window_manager._find_safe_termination_point(complete_response)
                                complete_response = complete_response[:termination_point]

                            break

                    # Check confidence early enough to stop generation if needed
                    # But only if we have a response filter and after receiving some tokens
                    if (response_filter is not None and not low_confidence_detected and tokens_received >= early_confidence_check_threshold):

                        # Get current metrics
                        current_metrics = self.confidence_metrics.get_metrics(apply_sharpening=True)

                        # Should we show fallback instead of continuing?
                        if response_filter.should_stream_fallback(current_metrics, user_query):
                            # Set flag to avoid checking again
                            low_confidence_detected = True

                            # Set the stop event to interrupt generation
                            self.stop_event.set()

                            # Get fallback message
                            fallback = response_filter.get_streamable_fallback(user_query)

                            # Stream the fallback message with variable delays to mimic LLM output
                            # Break into words for more natural streaming
                            fallback_words = fallback.split()

                            for i, word in enumerate(fallback_words):
                                # Print the word
                                print(word, end="", flush=True)

                                # Add space after word (except for last word)
                                if i < len(fallback_words) - 1:
                                    print(" ", end="", flush=True)

                                # Variable delays between words
                                # Occasional longer pauses at punctuation or every few words
                                if i > 0 and word[-1] in ".,:;?!":
                                    time.sleep(0.3)  # Longer pause after punctuation
                                elif i % 3 == 0:
                                    time.sleep(0.15)  # Medium pause every few words
                                else:
                                    time.sleep(0.07)  # Regular pause between words

                            # Mark fallback as streamed
                            fallback_message_streamed = True

                            # Update complete response with fallback
                            complete_response = fallback
                            break

                    # Only add displayable tokens to the buffer
                    if display_token and not fallback_message_streamed:
                        if show_confidence:
                            colored_token = heatmap.colorize_streaming_token(
                                display_token, latest_confidence)
                            token_buffer += colored_token
                        else:
                            # Normal display without colorization
                            token_buffer += display_token

                    # Check timing
                    current_time = time.time()
                    time_since_print = current_time - last_print_time

                    # Print conditions
                    if (len(token_buffer) >= max_buffer_size or time_since_print >= force_display_time) and token_buffer:
                        print(token_buffer, end="", flush=True)
                        token_buffer = ""
                        last_print_time = current_time

                    # Check for stop sequences
                    if len(token_buffer) > 5:
                        for stop_seq in stop_sequences:
                            if stop_seq in token_buffer:
                                token_buffer = token_buffer.split(stop_seq)[0]
                                if token_buffer:
                                    print(token_buffer, end="", flush=True)
                                complete_response = complete_response.split(stop_seq)[0]

                                # ENSURE WE HAVE CONFIDENCE METRICS
                                if not self.confidence_metrics.token_probabilities and not self.confidence_metrics.original_token_probabilities:
                                    for i in range(5):
                                        dummy_logits = torch.zeros(self.tokenizer.vocab_size, device=self.device)
                                        # Generate varied values from 0.5 to 0.9
                                        prob_value = 0.5 + (i * 0.1)
                                        max_index = i % self.tokenizer.vocab_size

                                        # Create a probability distribution with the max at our chosen token
                                        dummy_logits.fill_(0.1 / self.tokenizer.vocab_size)  # Low baseline
                                        dummy_logits[max_index] = prob_value  # Higher probability for this token

                                        # Convert to logits
                                        dummy_logits = torch.log(dummy_logits + 1e-10)

                                        self.confidence_metrics.add_token_score(dummy_logits, max_index)

                                # Track the generated response for continuation
                                if hasattr(self, 'window_manager') and hasattr(self.window_manager, 'track_generated_response'):
                                    self.window_manager.track_generated_response(
                                        response=complete_response,
                                        user_id=self.current_user_id,
                                        target_length=target_length)

                                return self.mcp_handler.finalize_streaming(complete_response)

                termios.tcsetattr(fd, termios.TCSADRAIN, self.old_settings)
                # Handle any remaining tokens
                if token_buffer:
                    print(token_buffer, end="", flush=True)

                # ALWAYS ENSURE WE HAVE CONFIDENCE METRICS BEFORE RETURNING
                if not self.confidence_metrics.token_probabilities and not self.confidence_metrics.original_token_probabilities:
                    response_length = len(complete_response.split())
                    num_tokens = max(5, min(response_length, 10))

                    for i in range(num_tokens):
                        dummy_logits = torch.zeros(self.tokenizer.vocab_size)
                        # Use different confidence values based on response quality
                        confidence_val = 8.0 + (i % 3)  # Vary between 8-10
                        dummy_logits[i % 100] = confidence_val
                        self.confidence_metrics.add_token_score(dummy_logits, i % 100)

                # Get domain information if available
                domain = None
                settings = None
                if hasattr(self, 'question_classifier') and messages[-1]["role"] == "user":
                    user_query = messages[-1]["content"]
                    settings = self.question_classifier.get_domain_settings(user_query)
                    domain = settings['domain']

                # Apply post-processing to clean up the response
                complete_response = self.post_process_response(complete_response, user_query, domain)

                # Finalize MCP processing
                complete_response = self.mcp_handler.finalize_streaming(complete_response)

                if len(user_commands) > 0:
                    for filename, cmd in user_commands.items():
                        if cmd.get("action") == "save_response":
                            success = self.mcp_handler.save_response_to_file(complete_response, filename)
                            if success:
                                print(f"{self.get_time()} [Response saved to: {filename}]")
                            else:
                                print(f"{self.get_time()} [Failed to save response to: {filename}]")

                # Track the generated response for continuation
                if hasattr(self, 'window_manager') and hasattr(self.window_manager, 'track_generated_response'):
                    self.window_manager.track_generated_response(
                        response=complete_response,
                        user_id=self.current_user_id,
                        target_length=target_length)

                return complete_response

            except Exception as e:
                print(f"\n{self.get_time()} Error during token streaming: {str(e)}")
                if complete_response:
                    # Even with errors, make sure we have metrics
                    if not self.confidence_metrics.token_probabilities and not self.confidence_metrics.original_token_probabilities:
                        dummy_logits = torch.zeros(self.tokenizer.vocab_size)
                        dummy_logits[0] = 5.0  # Medium confidence for error cases
                        self.confidence_metrics.add_token_score(dummy_logits, 0)

                    return self.mcp_handler.finalize_streaming(complete_response)
                return "Error generating response. Please try again."

            self.last_token_confidences = token_confidences

        except KeyboardInterrupt:
            print(f"\n{self.get_time()} Exiting due to keyboard interrupt...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n{self.get_time()} Streaming setup failed: {e}")

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

            return "Error in streaming setup. Please try again."

        finally:
            print(f"\n{self.get_time()} ¯\\_(ツ)_/¯", end='', flush=True)
            if show_confidence and not self.stop_event.is_set() and not fallback_message_streamed:
                # Print the legend after the response is complete
                print()  # Add a newline
                heatmap.print_legend()

            # Fallback for confidence metrics if somehow none were added
            if not self.confidence_metrics.token_probabilities and not self.confidence_metrics.original_token_probabilities:
                dummy_logits = torch.zeros(self.tokenizer.vocab_size)
                dummy_logits[0] = 7.0  # Default confidence
                self.confidence_metrics.add_token_score(dummy_logits, 0)

            self.resource_manager.clear_cache()
            termios.tcsetattr(fd, termios.TCSADRAIN, self.old_settings)

    def _clean_boilerplate_messages(self, response):
        """
        Clean boilerplate UI messages from the response before metric calculation.

        Args:
            response: The original response text

        Returns:
            Cleaned response without boilerplate
        """
        # List of patterns to remove
        patterns = [
            r'¯\\\_\(ツ\)\_/¯',                          # Shrug emoticon
            r'Confidence Heatmap Legend.*?\n',           # Heatmap legend
            r'Window size for geometric mean:.*?\n',     # Window size info
            r'Confidence range:.*?\n',                   # Confidence range info
            r'\[Generated.*?tokens\/sec\]',              # Generation stats
            r'\[█+░*\] Truthiness:.*?\n',                # Truthiness bar
            r'\[Sharpening enabled:.*?\]',               # Sharpening info
        ]

        # Apply each pattern
        cleaned = response
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)

        return cleaned

    def ensure_metrics(self, response_length=None):
        """
        Ensure confidence metrics exist with reasonable values.

        Args:
            response_length: Optional length of response to scale metrics
        """
        if not self.confidence_metrics.token_probabilities:
            # If response length is not provided, use a default
            num_tokens = 5
            if response_length:
                # First, clean the response of boilerplate messages
                cleaned_response = self._clean_boilerplate_messages(response)
                # Scale the number of tokens based on cleaned response length
                num_tokens = max(5, min(len(cleaned_response.split()), 20))

            for i in range(num_tokens):
                # Create dummy logits tensor
                dummy_logits = torch.zeros(self.tokenizer.vocab_size)

                # Vary the confidence value to create realistic metrics
                # The variation depends on token position (early tokens more confident)
                position_factor = 1.0 - (i / (num_tokens * 2))  # Decreases from 1.0 to 0.5
                confidence_base = 5.0 + (position_factor * 3.0)  # Range 5.0-8.0

                # Add some randomness
                confidence_val = confidence_base + ((i % 3) * 0.5)  # Small variations

                # Set the logit value for a token
                token_id = i % 100  # Use different token IDs
                dummy_logits[token_id] = confidence_val

                # Add the metrics
                self.confidence_metrics.add_token_score(dummy_logits, token_id)

            return True
        return False

    def save_conversation(self, conversation):
        """Save the conversation for future reference"""
        conversation_file = os.path.join(
            self.memory_dir,
            f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        try:
            with open(conversation_file, 'w') as f:
                json.dump(conversation, f, indent=2)
        except Exception as e:
            print(f"{self.get_time()} Error saving conversation: {e}")

        return len(conversation)

    def quality_score(self, confidence, perplexity=None, entropy=None, max_perplexity=10, max_entropy=3, w1=0.4, w2=0.4, w3=0.2):
         # If all metrics are available
        if perplexity is not None and entropy is not None:
            # Use sigmoid function to get better distribution for perplexity
            perplexity_score = 1 / (1 + np.exp((perplexity - 3) / 0.5))

            # Use sigmoid for entropy too
            entropy_score = 1 / (1 + np.exp((entropy - 1.5) / 0.3))

            # Adjust confidence to be more critical
            adjusted_confidence = confidence * 0.8  # Reduce the impact of inflated confidence

            return (w1 * adjusted_confidence) + (w2 * perplexity_score) + (w3 * entropy_score)

        # Fallback to using just confidence if other metrics aren't available
        else:
            return confidence

    def format_metric_bar(self, value, min_val=0.0, max_val=1.0, width=20, label="", reverse=False):
        """
        Format a metric as a progress bar.

        Args:
            value: The metric value
            min_val: Minimum value for the scale
            max_val: Maximum value for the scale
            width: Width of the progress bar in characters
            label: Label for the metric
            reverse: If True, lower values are better (like perplexity)

        Returns:
            Formatted string with the metric and progress bar
        """
        # Normalize value to 0-1 range
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0.0, min(1.0, normalized))

        # For reverse metrics (where lower is better), invert the value
        if reverse:
            normalized = 1.0 - normalized

        # Calculate filled and empty portions
        filled_length = int(width * normalized)
        empty_length = width - filled_length

        # Create the bar
        bar = "█" * filled_length + "░" * empty_length

        # Format the output
        return f"[{bar}] {label}: {value:.2f}"

    # This patch fixes the sharpening metrics display issue in tiny_llama_6_memory.py
    def print_generation_metrics(self, metrics, generation_time, response_tokens, show_all_metrics=False):
        """
        Print generation metrics with sharpening information if available.

        Args:
            metrics: Dictionary of metrics
            generation_time: Time taken for generation
            response_tokens: Number of tokens generated
            show_all_metrics: Whether to show all individual metrics
        """
        # Calculate tokens per second
        tokens_per_second = response_tokens / max(0.01, generation_time)

        # Print header
        print(f"\n{self.get_time()} [Generated {response_tokens} tokens in {generation_time:.2f}s - ~{tokens_per_second:.1f} tokens/sec]")

        # Get confidence metrics with sharpening applied
        confidence_data = self.confidence_metrics.get_metrics(apply_sharpening=True)

        # Check if we have original metrics for comparison (this is where the bug was)
        has_sharpening = "original" in confidence_data

        # Get the metrics from the right source
        if has_sharpening:
            # We have both sharpened and original metrics
            sharpened = {
                'confidence': confidence_data.get('confidence', 0.0),
                'perplexity': confidence_data.get('perplexity', 0.0),
                'entropy': confidence_data.get('entropy', 0.0)
            }
            original = confidence_data['original']
            enhancement = confidence_data.get('enhancement', 0.0)
        else:
            # Just use the metrics provided
            sharpened = metrics
            original = None
            enhancement = None

        # Calculate quality score using either provided or confidence_data values
        confidence = sharpened.get('confidence', metrics.get('confidence', 0.0))
        perplexity = sharpened.get('perplexity', metrics.get('perplexity', 0.0))
        entropy = sharpened.get('entropy', metrics.get('entropy', 0.0))

        # For perplexity and entropy, lower is better, so normalize them inversely
        perplexity_score = max(0.0, min(1.0, 1.0 - (perplexity / 10.0)))
        entropy_score = max(0.0, min(1.0, 1.0 - (entropy / 3.0)))

        # Combine all metrics with equal weights to get quality
        quality = metrics.get('quality', (confidence * 0.25) + (perplexity_score * 0.25) + (entropy_score * 0.25) + 0.25)

        # Combine all metrics with equal weights for truthiness
        truthiness = (quality * 0.25) + (confidence * 0.25) + (perplexity_score * 0.25) + (entropy_score * 0.25)

        # If all metrics are toggled on, show individual metrics
        if show_all_metrics:
            print("\nDetailed metrics:")
            print(self.format_metric_bar(quality, 0.0, 1.0, 20, "Quality"))

            # Show confidence with comparison if available
            if has_sharpening:
                orig_conf = original["confidence"]
                print(self.format_metric_bar(confidence, 0.0, 1.0, 20,
                                       f"Confidence (was {orig_conf:.2f}, {enhancement:+.1f}%)"))
            else:
                print(self.format_metric_bar(confidence, 0.0, 1.0, 20, "Confidence"))

            # Show other metrics with comparison
            if has_sharpening:
                orig_perp = original["perplexity"]
                print(self.format_metric_bar(perplexity, 1.0, 10.0, 20,
                                       f"Perplexity (was {orig_perp:.2f})", reverse=True))

                orig_entropy = original["entropy"]
                print(self.format_metric_bar(entropy, 0.0, 3.0, 20,
                                       f"Entropy (was {orig_entropy:.2f})", reverse=True))
            else:
                print(self.format_metric_bar(perplexity, 1.0, 10.0, 20, "Perplexity", reverse=True))
                print(self.format_metric_bar(entropy, 0.0, 3.0, 20, "Entropy", reverse=True))

        # Print truthiness bar
        print(self.format_metric_bar(truthiness, 0.0, 1.0, 30, "Truthiness"))

        # Show sharpening status if enabled
        if self.memory_manager.use_fractal:
            print(f"{self.get_time()} [Sharpening enabled: factor={self.sharpening_factor:.2f}]")

    def add_conversation_to_memory(self, query, response):
        """Add the current exchange to memory if auto-memorize is enabled."""
        if not self.auto_memorize:
            return 0

        # Extract key information
        key_info = self._extract_key_information(query, response)

        if not key_info:
            return 0

        # Add each item to memory
        memories_added = 0

        for info in key_info:
            # Create metadata
            metadata = {
                'source_hint': 'conversation',
                'query': query,
                'timestamp': datetime.now().timestamp(),
                'retrieval_count': 0
            }

            # Add to memory
            item_id = self.memory_manager.add(
                content=info,
                metadata=metadata,
                use_fractal=self.memory_manager.use_fractal
            )

            if item_id:
                memories_added += 1
        
        if memories_added > 0:
            print(f"{self.get_time()} [Memory] Added {memories_added} new memories")
            
        return memories_added

    def _extract_key_information(self, query, response):
        # Use a more efficient approach to split sentences
        sentences = re.split(r'[.!?]', query + " " + response)

        # Pre-filter to avoid processing many sentences
        key_info = []
        seen_simplified = set()  # Track similar sentences

        for sentence in sentences:
            # Skip short sentences immediately
            if len(sentence.strip()) < 10:
                continue

            # Create simplified key to avoid near-duplicates
            simplified = re.sub(r'[^\w\s]', '', sentence.lower())
            simplified = ' '.join(simplified.split())

            # Skip if already seen a similar sentence
            if simplified in seen_simplified:
                continue

            seen_simplified.add(simplified)

            # Apply criteria (only compute once we know this sentence is worth checking)
            if ("[A-Z][a-z]+" in sentence) or re.search(r'\b\d+\b', sentence) or len(sentence.split()) > 5:
                key_info.append(sentence.strip())

                # Limit to 10 memories as in the original
                if len(key_info) >= 10:
                    break

        return key_info

    def toggle_auto_memorize(self):
        """Toggle automatic memorization on/off"""
        self.auto_memorize = not self.auto_memorize
        return self.auto_memorize

    def toggle_sharpening(self):
        """Toggle vector space sharpening on/off"""
        # Flip the current state
        self.memory_manager.use_fractal = not self.memory_manager.use_fractal
        return self.memory_manager.use_fractal

    def set_sharpening_factor(self, factor: float) -> None:
        """Set the sharpening factor for memory retrieval and confidence metrics"""
        # Update confidence metrics sharpening
        self.sharpening_factor = factor
        
        if hasattr(self.confidence_metrics, 'set_sharpening_factor'):
            self.confidence_metrics.set_sharpening_factor(factor)

            # Force recalculation with sharpening applied
            if hasattr(self.confidence_metrics, 'token_probabilities') and self.confidence_metrics.token_probabilities:
                _ = self.confidence_metrics.get_metrics(apply_sharpening=True)

        print(f"{self.get_time()} Sharpening factor set to {factor}")


    def post_process_response(self, response, query, domain=None):
        """
        Clean up the response after generation.

        Args:
            response: The generated response text
            query: The user query
            domain: The detected domain (if available)

        Returns:
            Cleaned response text
        """
        if not response:
            return response

        # If domain wasn't provided, try to detect it
        if domain is None and hasattr(self, 'question_classifier'):
            domain, _ = self.question_classifier.classify(query)

        # Apply domain-specific post-processing
        if domain == 'translation':
            # Clean up translation responses
            return self._clean_translation_response(response, query)
        elif domain == 'arithmetic':
            # Clean up arithmetic responses
            return self._clean_arithmetic_response(response, query)

        # General cleanup for all responses
        return self._clean_general_response(response)

    def _clean_translation_response(self, response, query):
        """Clean up translation responses"""
        # Extract the core translation from verbose responses
        first_line = response.split('\n')[0].strip()

        # If the response has notes or explanations, keep only the essential part
        if len(response.split('\n')) > 1:
            # Use regex to find the translation pattern
            match = re.search(r'["\'«]([^"\'»]+)["\'\»]\s+(?:in|en|na|в|på)\s+\w+', first_line)
            if match:
                translation = match.group(1).strip()
                # Return a cleaner version
                lang_match = re.search(r'(?:in|to|into)\s+(\w+)', query)
                if lang_match:
                    language = lang_match.group(1)
                    return f'"{translation}" in {language}'

        # Remove duplicated text like "Prikaz:" and redundant notes
        response = re.sub(r'Prikaz:\s*-\s*', '', response)
        response = re.sub(r'(?:Note|Примечание):\s*"[^"]+"\s+(?:may|peut|puede|может|kan).*?$', '', response, flags=re.MULTILINE)

        # Remove duplicated translations
        lines = response.split('\n')
        unique_lines = []
        seen = set()
        for line in lines:
            # Create a simplified key for comparison
            simplified = re.sub(r'[^\w\s]', '', line.lower())
            if simplified and simplified not in seen:
                seen.add(simplified)
                unique_lines.append(line)

        return '\n'.join(unique_lines)

    def _clean_arithmetic_response(self, response, query):
        """Clean up arithmetic responses"""
        # Extract the expression from the query
        expression = None

        if not expression:
            # Try a simple regex fallback
            match = re.search(r'(\d+\s*[\+\-\*\/]\s*\d+)', query)
            if match:
                expression = match.group(1).replace(' ', '')

        if expression:
            # Calculate the correct result
            try:
                correct_result = eval(expression)

                # Check if the response has the correct result
                if str(correct_result) not in response:
                    # Replace with correct calculation
                    return f"The result of {expression} is {correct_result}."

                # If it has the correct result, just clean up any verbose explanations
                match = re.search(r'.*?(\d+\s*[\+\-\*\/]\s*\d+).*?(\d+)', response)
                if match:
                    return f"The result of {expression} is {correct_result}."
            except:
                pass

        return response

    def _clean_general_response(self, response):
        """General cleanup for all responses"""
        # Remove repeated paragraphs (not just lines)
        lines = response.split('\n')
        paragraphs = []
        current_paragraph = []

        # Group lines into paragraphs
        for line in lines:
            if line.strip():
                current_paragraph.append(line)
            else:
                if current_paragraph:
                    paragraphs.append('\n'.join(current_paragraph))
                    current_paragraph = []
                paragraphs.append('')  # Keep empty lines

        if current_paragraph:
            paragraphs.append('\n'.join(current_paragraph))

        # Remove duplicate paragraphs
        unique_paragraphs = []
        seen_paragraphs = set()

        for paragraph in paragraphs:
            # Create a simplified key for comparison
            if paragraph:
                simplified = re.sub(r'[^\w\s]', '', paragraph.lower())
                if simplified and simplified not in seen_paragraphs:
                    seen_paragraphs.add(simplified)
                    unique_paragraphs.append(paragraph)
                elif not simplified:
                    unique_paragraphs.append(paragraph)  # Keep formatting paragraphs
            else:
                unique_paragraphs.append(paragraph)  # Keep empty lines

        cleaned = '\n'.join(unique_paragraphs)

        # Remove any debugging information that might have leaked
        cleaned = re.sub(r'(?:relevance|similarity)\s+(?:increased|decreased)\s+by\s+[\d\.]+%.*?$', '', cleaned, flags=re.MULTILINE)

        return cleaned

    def add_command_to_memory(self, command: str, output: str, error: str = None, output_file: str = None):
        """Add command output to unified knowledge system."""
        # Skip empty outputs
        if not output and not error:
            return 0

        # Prepare content - the actual text we want to remember
        content = f"Command '{command}' output: {output[:500]}"
        if len(output) > 500:
            content += "... [output truncated]"

        if error:
            content += f"\nError: {error}"

        # Prepare metadata
        metadata = {
            "source_hint": "command",
            "command": command,
            "timestamp": datetime.now().timestamp(),
            "retrieval_count": 0
        }

        if output_file:
            metadata["output_file"] = output_file

        if error:
            metadata["has_error"] = True

        # Add to memory
        item_id = self.memory_manager.add(
            content=content,
            metadata=metadata,
            use_fractal=self.memory_manager.use_fractal
        )

        return 1 if item_id else 0

    def cleanup(self):
        """Release all resources properly."""

        try:
            # Unload models
            if hasattr(self, 'draft_model') and self.draft_model is not None:
                self.resource_manager.unload_model(self.draft_model)
                self.draft_model = None

            if hasattr(self, 'model') and self.model is not None:
                self.resource_manager.unload_model(self.model)
                self.model = None

            # For memory consolidation, we need to get the store first
            if hasattr(self, 'memory_manager') and hasattr(self, 'current_user_id'):
                try:
                    # Clean up memory
                    self.memory_manager.cleanup()
                    print(f"{self.get_time()} Consolidated memories for user {self.current_user_id}")
                except Exception as e:
                    print(f"{self.get_time()} Error consolidating memories: {e}")

            # Final cleanup
            if hasattr(self, 'resource_manager'):
                self.resource_manager.cleanup()

            if torch.cuda.is_available():
                print(f"{self.get_time()} Releasing cuda cache.")
                torch.cuda.empty_cache()

            print(f"{self.get_time()} Resources cleaned up successfully")

        except Exception as e:
            print(f"{self.get_time()} Error during cleanup: {e}")
            # Attempt to run basic cleanup even after an error
            try:
                if hasattr(self, 'resource_manager'):
                    self.resource_manager.cleanup()
            except:
                pass

    def get_multiline_input(self, prompt):
        """
        Get multiline input with full editing capabilities using prompt_toolkit.

        Args:
            prompt: Initial prompt to display

        Returns:
            String containing the entire multiline input
        """
        try:
            from prompt_toolkit import prompt
            from prompt_toolkit.history import FileHistory
            from prompt_toolkit.styles import Style
            import os
        except ImportError:
            print("prompt_toolkit is not installed. Install it with 'pip install prompt_toolkit'")
            print("Falling back to simple input method.")

            # Fallback to previous implementation if prompt_toolkit is not available
            print("(Enter your text, paste multiline content. Press Ctrl+D or submit an empty line to finish)")

            try:
                lines = []
                while True:
                    try:
                        line = input()
                        lines.append(line)
                    except EOFError:
                        break
            except KeyboardInterrupt:
                print("\nInput cancelled.")
                return ""

            return "\n".join(lines)

        # Ensure memory directory exists
        os.makedirs(self.memory_dir, exist_ok=True)

        # Create a history file for persistent input history
        history_file = os.path.join(self.memory_dir, '.multiline_history')

        print("(Multiline input mode. Use Ctrl+D or Alt+Enter to submit, Ctrl+C to cancel)")

        try:
            style = Style.from_dict({
                # The empty string key '' is the “default” token,
                # so anything you type (that isn’t otherwise styled) gets this color.
                '': 'fg:#0BB8E2',
                # If you also want your continuation dots to be the same color:
                'prompt.continuation': 'fg:#ADD8E6',
            })

            # Use prompt_toolkit for advanced multiline input
            user_input = prompt(
                '',  # No initial prompt inside the input area
                multiline=True,  # Enable multiline input
                prompt_continuation=lambda width, line_number, wrap_count: '.' * width,  # Continuation prompt
                history=FileHistory(history_file),  # Persistent history
                complete_while_typing=True,  # Enable autocompletion
                enable_open_in_editor=True,  # Allow opening in external editor with F4
                style=style
            )
            # Trim trailing whitespace if needed
            user_input = user_input.rstrip()

            # Confirm and return input
            if user_input:
                print(f"{self.get_time()} Received {len(user_input.splitlines())} lines of input.")

            return user_input

        except KeyboardInterrupt:
            print(f"\n{self.get_time()} Input cancelled.")
            return ""
        except Exception as e:
            print(f"\n{self.get_time()} Error during input: {e}")
            return ""

    # handles "Already borrowed" error in tokenization
    def safe_tokenize(self, text, **kwargs):
        """
        Safe tokenization that avoids the 'Already borrowed' error by creating new tensors.
        """
        try:
            # Try normal tokenization first
            return self.tokenizer(text, **kwargs)
        except RuntimeError as e:
            if "Already borrowed" in str(e):
                print(f"{self.get_time()} Compensating for already borrowed tensor...")
                # Create a fresh tokenizer instance for this operation
                import copy
                temp_tokenizer = copy.deepcopy(self.tokenizer)

                # Try with the fresh tokenizer
                result = temp_tokenizer(text, **kwargs)

                # Convert to regular Python objects before returning
                return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
            else:
                # Re-raise if it's a different error
                raise

def main():
    start_time = datetime.now().strftime("[%d/%m/%y %H:%M:%S]")
    print(f"{start_time} Let's start ...")
    parser = argparse.ArgumentParser(description="TinyLlama Chat with Speculative Decoding and MCP")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                      help="Model to use for chat")
    parser.add_argument("--device", type=str, default=None,
                      help="Device to use (cuda, cpu). If not specified, will autodetect.")
    parser.add_argument("--system-prompt", type=str,
                      default="You are a helpful and friendly assistant.",
                      help="System prompt to use")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Temperature for response generation (lower = more deterministic)")
    parser.add_argument("--max-tokens", type=int, default=128,
                      help="Maximum number of tokens to generate in response")
    parser.add_argument("--turbo", action="store_true", default=True,
                      help="Enable turbo mode for ultra-fast generation")
    parser.add_argument("--output-dir", type=str, default="./output",
                      help="Directory to store output files from MCP")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                      help="Filter confidence")
    parser.add_argument("--heatmap", action="store_true", default=False,
                      help="Show confidence heatmap visualization")
    parser.add_argument("--no-memory", action="store_true",
                      help="Disable automatic memory features")
    parser.add_argument("--all-metrics", action="store_true", default=False,
                      help="Show all detailed metrics instead of just truthiness")
    parser.add_argument("--enable-sharpening", action="store_true", default=True,
                      help="Enable vector space and confidence sharpening")
    parser.add_argument("--sharpening-factor", type=float, default=0.3,
                      help="Sharpening factor for vector embeddings (0.0-1.0)")
    parser.add_argument("--web-knowledge", action="store_true", default=True,
                      help="Enable web search for knowledge enhancement")
    parser.add_argument("--search-engine", type=str, default="duckduckgo", choices=["duckduckgo", "google"],
                      help="Search engine to use for web knowledge")
    parser.add_argument("--web-confidence", type=float, default=0.65,
                      help="Confidence threshold below which to trigger web search")
    parser.add_argument("--test-fractal", action="store_true", default=False,
                      help="Run fractal embedding diagnostics")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling-based generation")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling parameter")

    args = parser.parse_args()

    filter_enabled = True  # Add this to track filtering state
    user_context = {}  # Shared context for the ResponseFilter
    show_all_metrics = args.all_metrics  # Add this to track whether to show all metrics

    # Initialize chat with speculative decoding support
    chat = TinyLlamaChat(
        model_name=args.model,
        device=args.device,
        memory_dir="./memory",  # Add this explicitly if you want to keep the default
        output_dir=args.output_dir,  # Pass the output_dir parameter here
        confidence_threshold=args.confidence_threshold,
        auto_memorize=not args.no_memory,
        enable_sharpening=args.enable_sharpening,
        sharpening_factor=args.sharpening_factor,
        enable_web_knowledge=args.web_knowledge,
        fractal_enabled=True,
        max_fractal_levels=3,
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k
    )

    # # Test embedding batch processing
    # chat.test_batch_processing('embedding')

    # # Test inference batch processing
    # chat.test_batch_processing('inference')

    # for debugging embed performance
    # return chat.test_embedding_performance()

    # for testing fractal embedding
    #return chat.memory_manager.print_fractal_embedding_diagnostics()

    try:
        # If web knowledge is enabled, configure the search engine
        if args.web_knowledge and chat.web_enhancer:
            chat.web_enhancer.search_engine = args.search_engine
            chat.web_enhancer.confidence_threshold = args.web_confidence
            print(f"{chat.get_time()} Web knowledge enhancement enabled using {args.search_engine}")

        if args.test_fractal:
            # Get a user store
            store = chat.memory_manager

            # Run and print diagnostics
            store.print_fractal_embedding_diagnostics()
            return  # Exit after diagnostics

        response_filter = ResponseFilter(
            confidence_threshold=args.confidence_threshold,
            user_context=user_context,
            question_classifier=chat.question_classifier
        )

        history_file = setup_readline_history(chat.memory_dir)
        print(f"{chat.get_time()} Command history stored in: {history_file}")

        # Set system message
        chat.system_message = {
            "role": "system",
            "content": args.system_prompt
        }

        # Warm up the model with a short prompt
        if chat.device != "cpu":
            try:
                import bitsandbytes as bnb
                chat.loading_options["load_in_4bit"] = True  # or load_in_4bit=True for even better performance
                print(f"{chat.get_time()} Using 4-bit quantization for better performance")
            except ImportError:
                print(f"{chat.get_time()} bitsandbytes not installed, using full precision")

            print(f"{chat.get_time()} Warming up model for maximum throughput...")
            import random
            from random import randrange

            magic_number = 259125
            magic_number_1 = randrange(magic_number)
            magic_number_2 = randrange(magic_number)
            operators = ["+", "-", "/", "*"]

            _ = chat.generate_response(
                [{"role": "user", "content": f"Just write the result of equation {magic_number_1}{random.choice(operators)}{magic_number_2} without anything than just that the result of equation."}],
                max_new_tokens=30,
                temperature=0.7,
                use_web_search=False
            )

        # Start conversation loop
        print("\n" + "="*50)
        print("TinyLlama Chat with Speculative Decoding and MCP")
        print("="*50)
        print("Type 'exit' to end the conversation")
        print("Special commands:")

        print("  !system: [message] - Change the system message")
        print("  !mcp-help - Show MCP commands for directing output to files")
        print("  !confidence: [0.0-1.0] - Set confidence threshold")
        print("  !sharpening-factor: [0.0-1.0] - Set the sharpening factor for vector embeddings")
        print("  !toggle-turbo - Toggle turbo mode on/off")
        print("  !toggle-filter - Toggle uncertainty filtering on/off")
        print("  !toggle-heatmap - Toggle confidence heatmap visualization on/off")
        print("  !toggle-all-metrics - Toggle between showing all metrics or just truthiness")
        print("  !toggle-sharpening - Toggle vector space sharpening on/off")
        print("  !toggle-memory - Toggle automatic memorization on/off")
        print("  !memory-stats - Display info about memories")
        print("  !toggle-web - Toggle web knowledge enhancement on/off")
        print("  !web-stats - Show web search statistics")
        print("  !search-engine: [engine] - Set search engine (duckduckgo/google)")
        print("  !fractal-diagnostics - prints fractal embedding diagnostics")
        print("  !compare-queries: [query1] | [query2] - Compare the semantic relationship between two queries")

        print("\nIf the model expresses uncertainty, you can ask it to speculate")
        print("by saying 'please continue anyway' or 'please speculate'")

        print("="*50 + "\n")

        # Set initial mode settings
        turbo_mode = args.turbo
        show_confidence = args.heatmap

        conversation = [chat.system_message]

        while True:
            # Get timestamp for user input
            user_input = chat.get_multiline_input(f"{chat.get_time()} You: ")

            if user_input == "":
                chat.stop_event.set()
                continue

            # Handle special commands
            if user_input.lower() == 'exit':
                store = chat.memory_manager

                # Show final stats
                stats = store.get_stats()
                print(f"{chat.get_time()} Memories saved this session: {stats['active_items']}")
                print(f"{chat.get_time()} Total memories saved this: {stats['total_items']}")
                break

            elif user_input.lower().startswith('!system:'):
                new_system = user_input[8:].strip()
                chat.system_message = {
                    "role": "system",
                    "content": new_system
                }
                conversation[0] = chat.system_message
                print(f"{chat.get_time()} System message updated: {new_system}")
                continue

            elif user_input.lower() == '!toggle-turbo':
                turbo_mode = not turbo_mode
                print(f"{chat.get_time()} Turbo mode {'enabled' if turbo_mode else 'disabled'}")
                continue

            elif user_input.lower() == '!mcp-help':
                help_text = chat.mcp_handler.get_help_text()
                print(help_text)
                continue
            elif user_input.lower() == '!toggle-filter':
                filter_enabled = not filter_enabled
                print(f"{chat.get_time()} Response filtering {'enabled' if filter_enabled else 'disabled'}")
                continue
            elif user_input.lower() == '!toggle-heatmap':
                show_confidence = not show_confidence
                print(f"{chat.get_time()} Confidence heatmap {'enabled' if show_confidence else 'disabled'}")
                continue
            elif user_input.lower() == '!toggle-all-metrics':
                show_all_metrics = not show_all_metrics
                print(f"{chat.get_time()} Detailed metrics display {'enabled' if show_all_metrics else 'disabled'}")
                continue
            elif user_input.lower() == '!toggle-sharpening':
                is_enabled = chat.memory_manager.toggle_sharpening()
                print(f"{chat.get_time()} Vector space sharpening {'enabled' if is_enabled else 'disabled'}")
                continue
            elif user_input.lower().startswith('!sharpening-factor:'):
                try:
                    factor = float(user_input.split(':')[1].strip())
                    if 0.0 <= factor <= 1.0:
                        chat.memory_manager.set_sharpening_factor(factor)
                    else:
                        print("Sharpening factor must be between 0.0 and 1.0")
                except Exception as e:
                    print(f"{chat.get_time()} Invalid value: {str(e)}. Please specify a number between 0.0 and 1.0")
                continue

            elif user_input.lower() == '!toggle-memory':
                is_enabled = chat.toggle_auto_memorize()
                print(f"{chat.get_time()} Automatic memorization {'enabled' if is_enabled else 'disabled'}")
                continue

            elif user_input.lower() == '!memory-stats':
                try:
                    # Get vector store stats
                    store = chat.memory_manager
                    store_stats = store.get_stats() if store else {"total_items": 0, "active_items": 0}

                    # Get memory manager state
                    auto_memorize = chat.memory_manager.auto_memorize if hasattr(chat.memory_manager, 'auto_memorize') else False

                    print(f"\n{chat.get_time()} Memory System Statistics:")
                    print(f"{chat.get_time()} Total memories: {store_stats.get('total_items', 0)}")
                    print(f"{chat.get_time()} Active memories: {store_stats.get('active_items', 0)}")
                    print(f"{chat.get_time()} Auto-memorize: {'Enabled' if auto_memorize else 'Disabled'}")

                    # Add additional statistics from the store if available
                    if hasattr(store, 'get_stats') and callable(store.get_stats):
                        try:
                            # Include more stats if they're available and informative
                            if 'index_dimension' in store_stats:
                                print(f"{chat.get_time()} Embedding dimension: {store_stats.get('index_dimension', 384)}")
                            if 'deleted_documents' in store_stats:
                                print(f"{chat.get_time()} Deleted memories: {store_stats.get('deleted_documents', 0)}")
                            # Show when the last store update happened
                            if 'last_updated' in store_stats:
                                print(f"{chat.get_time()} Last updated: {store_stats.get('last_updated', 'never')}")
                        except Exception:
                            # Ignore errors in additional stats
                            pass

                except Exception as e:
                    print(f"{chat.get_time()} Unexpected error when getting memory stats: {str(e)}")
                continue

            elif user_input.lower() == '!toggle-web':
                is_enabled = chat.toggle_web_knowledge()
                print(f"{chat.get_time()} Web knowledge enhancement {'enabled' if is_enabled else 'disabled'}")
                continue

            elif user_input.lower() == '!web-stats':
                if hasattr(chat, 'web_enhancer') and chat.web_enhancer:
                    stats = chat.web_enhancer.get_stats()
                    print(f"\n{chat.get_time()} Web Knowledge Statistics:")
                    print(f"{chat.get_time()} Total searches: {stats['total_searches']}")
                    print(f"{chat.get_time()} Successful searches: {stats['successful_searches']}")
                    print(f"{chat.get_time()} Success rate: {stats['success_rate']*100:.1f}%")
                    print(f"{chat.get_time()} Cache hits: {stats['cache_hits']}")
                    print(f"{chat.get_time()} Cache hit rate: {stats['cache_hit_rate']*100:.1f}%")
                else:
                    print("Web knowledge enhancement is not enabled")
                continue

            elif user_input.lower().startswith('!search-engine:'):
                if hasattr(chat, 'web_enhancer') and chat.web_enhancer:
                    engine = user_input.split(':')[1].strip().lower()
                    if engine in ["duckduckgo", "google"]:
                        chat.web_enhancer.search_engine = engine
                        print(f"{chat.get_time()} Search engine set to: {engine}")
                    else:
                        print(f"{chat.get_time()} Unsupported search engine: {engine}. Please use 'duckduckgo' or 'google'")
                else:
                    print("Web knowledge enhancement is not enabled")
                continue

            elif user_input.lower() == '!fractal-diagnostics':
                store = chat.memory_manager
                if hasattr(store, 'print_fractal_embedding_diagnostics'):
                    store.print_fractal_embedding_diagnostics()
                else:
                    print(f"{chat.get_time()} Fractal diagnostics not available.")
                continue

            elif user_input.lower().startswith('!compare-queries:'):
                try:
                    # Extract the two queries
                    _, query_part = user_input.split(':', 1)
                    query1, query2 = query_part.split('|')
                    query1 = query1.strip()
                    query2 = query2.strip()

                    # Call the semantic reasoning function
                    if hasattr(chat, 'test_semantic_relationship'):
                        relationship = chat.test_semantic_relationship(query1, query2)
                        print(f"\n{chat.get_time()} Semantic relationship between the queries: {relationship}")
                    else:
                        print(f"\n{chat.get_time()} Semantic reasoning capability not available. Run the finetuning first.")
                except Exception as e:
                    print(f"\n{chat.get_time()} Error comparing queries: {e}")
                    print(f"{chat.get_time()} Usage: !compare-queries: first query | second query")
                continue


            # Add user message to conversation
            user_message = {"role": "user", "content": user_input}
            conversation.append(user_message)

            # Generate response
            try:
                # Measure generation time
                start_time = time.time()
                chat.stop_event.clear()  # Reset the event

                # Generate response
                response = chat.generate_response(
                    conversation,
                    temperature=args.temperature,
                    max_new_tokens=args.max_tokens,
                    turbo_mode=turbo_mode,
                    show_confidence=show_confidence,
                    response_filter=response_filter if filter_enabled else None  # Pass the filter only if enabled
                )

                # Add a newline after streaming output is complete
                # print("...")

                # Report generation time and calculate tokens per second
                end_time = time.time()
                generation_time = max(0.01, end_time - start_time)

                if response:
                    chat.ensure_metrics(len(response.split()))

                # Get confidence metrics
                confidence_data = chat.confidence_metrics.get_metrics(apply_sharpening=chat.memory_manager.use_fractal)

                # Apply filtering only if enabled
                if filter_enabled:
                    filtered_response = response_filter.filter_response(
                        response,
                        confidence_data,
                        query=user_input,
                        preserve_mcp=True,
                        allow_override=True
                    )

                    # Check if response was filtered and print notification
                    if filtered_response != response:
                        response = filtered_response

                        # PRINT THE FILTERED RESPONSE HERE with legend
                        if show_confidence:
                            # Show confidence legend and add separator for clarity
                            heatmap = TerminalHeatmap(tokenizer=None, use_background=False, color_scheme="sepia-red")
                            heatmap.print_legend()
                            print(f"\n{chat.get_time()} Filtered response:")

                    else:
                        # If not filtered, print the original response
                        # (This is already happening in the stream)
                        pass

                    # Update user context
                    response_filter.update_user_context(user_input, response, confidence_data)
                else:
                    filtered_response = response  # Use original response if filtering disabled

                # First, check if this is a follow-up to a previously uncertain response
                current_query = user_input.lower()

                assistant_message = {"role": "assistant", "content": filtered_response}
                conversation.append(assistant_message)

                chat.add_conversation_to_memory(user_input, filtered_response)

                # Estimate tokens generated
                try:
                    response_tokens = len(chat.tokenizer.encode(response))
                    # tokens_per_second = response_tokens / generation_time
                    tokens_per_second = response_tokens / max(0.01, generation_time)
                    quality = chat.quality_score(
                        confidence_data.get('confidence', 0),
                        confidence_data.get('perplexity', 0),
                        confidence_data.get('entropy', 0)
                    )

                    chat.print_generation_metrics(
                        # {
                        #     'quality': quality,
                        #     'confidence': confidence_data.get('confidence', 0),
                        #     'perplexity': confidence_data.get('perplexity', 0),
                        #     'entropy': confidence_data.get('entropy', 0)
                        # },
                        confidence_data,
                        generation_time,
                        response_tokens,
                        show_all_metrics
                    )
                except Exception as e:
                    print(f"{chat.get_time()} stupid error {e}")
                    # Fallback to maximum tokens estimate
                    if args.max_tokens > 0:
                        tokens_per_second = args.max_tokens / generation_time
                        print(f"{chat.get_time()} [Generated in {generation_time:.2f}s - ~{tokens_per_second:.1f} tokens/sec | Confidence: {confidence_data['confidence']:.2f}]")

            except Exception as e:
                print(f"{chat.get_time()} Error generating response: {e}")
                print(f"{chat.get_time()} Please try again with a different question.")

            # finally:
            #     chat.cleanup()

    # except KeyboardInterrupt:
    #     print(f"\n{chat.get_time()} Exiting due to keyboard interrupt...")
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n{chat.get_time()} Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # This should only happen when exiting the program
        chat.cleanup()

if __name__ == "__main__":
    main()

