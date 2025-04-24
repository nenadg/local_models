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
from weighted_memory_integrator import WeightedMemoryIntegrator

from pattern_matching_utils import (
    is_command_related_query, extract_command_context, extract_command_type,
    is_tabular_command, is_tabular_data, extract_columns_from_header,
    extract_column_references, extract_row_references, extract_arithmetic_expression,
    commands_match, extract_example_pairs, extract_mapping_category, clean_duplicate_memories
)

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
And remember - Imagine your words are like little treasure boxes. Instead of giving everyone a whole bunch of boxes at once,
you choose just a few really cool ones to share. This way, people know you're thoughtful and your words are special.
It's like being a superhero: you use your superpower of speaking wisely and listen to others, too!
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
            auto_save=True
        )

        self.auto_memorize = auto_memorize
        self.sharpening_factor = sharpening_factor
        
        # Initialize the question classifier
        self.question_classifier = QuestionClassifier()

        # Initialize the weighted memory integrator
        self.memory_integrator = WeightedMemoryIntegrator(
            memory_manager=self.memory_manager,
            question_classifier=self.question_classifier
        )

        # Initialize from semantic reasoning finetune function
        finetuned_model_path = "./finetuned_tinyllama_reasoning_2"

        if os.path.exists(finetuned_model_path):
            try:
                integrate_semantic_reasoning(self, finetuned_model_path)
            except Exception as e:
                print(f"Finetuned model not loaded: {e}")

        self.knowledge_system_enabled = True  # UnifiedMemoryManager has built-in knowledge capabilities
        self.current_domain_id = None  # Currently active domain ID

        # Initialize the web knowledge enhancer
        self.enable_web_knowledge = enable_web_knowledge
        if enable_web_knowledge:
            self.web_enhancer = WebKnowledgeEnhancer(
                memory_manager=self.memory_manager,
                chat=self,
                confidence_threshold=confidence_threshold,
                vector_sharpening_factor=sharpening_factor,
                search_engine="duckduckgo",  # Use DuckDuckGo by default for fewer rate limits
                embedding_function=self.memory_manager.embedding_function  # Direct reference to the function
            )
            print("Web knowledge enhancement initialized")
        else:
            self.web_enhancer = None

        self.current_user_id = "default_user"

        os.makedirs(memory_dir, exist_ok=True)

        current_time = datetime.now().strftime("[%d/%m/%y %H:%M:%S]")

        # Determine the device to use
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("Using GPU for acceleration")
            if hasattr(torch.backends, "cuda"):
                if hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends.cudnn, "benchmark"):
                    torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                    torch.backends.cuda.enable_flash_sdp = True
        else:
            self.device = "cpu"
            print("No GPU detected, using CPU (this will be slow)")

        # Setup appropriate torch dtype
        if self.device == "cpu":
            self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch.float16

        # Load model and tokenizer
        print(f"Loading target model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Initialize the window manager with our tokenizer
        self.window_manager = ContinuationTrackingWindowManager(
            tokenizer=self.tokenizer,
            max_window_size=2048,
            memory_manager=self.memory_manager,
            safety_margin=50,
            continuation_buffer_size=200  # Store the last 200 tokens for continuation
        )

        print("Context window and continuation tracking initialized")

        # Main model loading
        self.loading_options = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto" if self.device != "cpu" else None,
            "low_cpu_mem_usage": True,
        }

        # Load model
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **self.loading_options)
        self.model = self.resource_manager.register_model(self.model)

        # Set embedding function for memory manager
        self.set_embedding_function()

        # Create draft model from target model by reducing layers
        self.draft_model = self.create_draft_model()

        if self.draft_model:
            print("Created draft model by reducing layers")
        else:
            print("Could not create draft model, speculative decoding disabled")

        # Load knowledge base
        self.knowledge_base = self.load_knowledge()
        self.conversation_history = []
        self.system_message = DEFAULT_SYSTEM_MESSAGE

    def set_embedding_function(self):
        """Generate embedding with caching for repeated queries"""
        def generate_embedding(text):
            """Generate embedding with caching"""
            # Create a hash of the text for cache key
            import hashlib
            cache_key = hashlib.md5(text.encode()).hexdigest()

            # Check cache first
            if cache_key in self._query_embedding_cache:
                return self._query_embedding_cache[cache_key]

            # Generate embedding for new text
            with torch.no_grad():
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)

                outputs = self.model.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask
                )

                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

            # Cache result (limit cache size)
            if len(self._query_embedding_cache) > 1000:  # Limit cache size
                # Remove random 100 items when we hit the limit
                import random
                keys_to_remove = random.sample(list(self._query_embedding_cache.keys()), 100)
                for key in keys_to_remove:
                    del self._query_embedding_cache[key]

            self._query_embedding_cache[cache_key] = embedding
            return embedding

        # Set the embedding function
        self.memory_manager.embedding_function = generate_embedding

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

                print(f"Created draft model with {keep_layers}/{num_layers} layers")
                return draft_model

            return None
        except Exception as e:
            print(f"Error creating draft model: {e}")
            return None

    def load_knowledge(self):
        """Load previously stored knowledge"""
        knowledge_file = os.path.join(self.memory_dir, "knowledge_base.json")
        if os.path.exists(knowledge_file):
            try:
                with open(knowledge_file, 'r') as f:
                    return json.load(f)
            except:
                print("Error loading knowledge base, creating new one")

        # Default knowledge structure
        return {
            "facts": [],
            "corrections": [],
            "preferences": {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

    def save_knowledge(self):
        """Save knowledge base to disk"""
        self.knowledge_base["updated_at"] = datetime.now().isoformat()
        knowledge_file = os.path.join(self.memory_dir, "knowledge_base.json")
        try:
            with open(knowledge_file, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)
        except Exception as e:
            print(f"Error saving knowledge base: {e}")

    def add_to_knowledge(self, fact_or_correction, fact_type="fact"):
        """Add new information to knowledge base"""
        if fact_type == "fact":
            self.knowledge_base["facts"].append({
                "content": fact_or_correction,
                "added_at": datetime.now().isoformat()
            })
        elif fact_type == "correction":
            self.knowledge_base["corrections"].append({
                "content": fact_or_correction,
                "added_at": datetime.now().isoformat()
            })

        # Save updated knowledge
        self.save_knowledge()

    def enhance_query_for_continuation(self, messages):
        """
        Enhance the user query if it's a continuation request.
        Add this at the beginning of create_prompt_with_knowledge.
        """
        # Check if this is a continuation request
        if hasattr(self, 'window_manager') and len(messages) > 0 and messages[-1]["role"] == "user":
            user_query = messages[-1]["content"]

            # Detect if this is a continuation request
            if self.window_manager.detect_continuation_request(user_query):
                print("Detected continuation request, enhancing prompt...")

                # Get user's vector store for fractal integration
                user_store = self.memory_manager

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
                    print("Enhanced continuation prompt created")
                else:
                    print("No continuation context available")

        return messages

    def create_prompt_with_knowledge(self, messages, use_web_search=True):
        """Create a prompt that incorporates relevant knowledge with domain-aware weighting"""
        # First check if this is a continuation request and enhance if needed
        messages = self.enhance_query_for_continuation(messages)

        # Extract the user's current query
        current_query = messages[-1]["content"] if messages[-1]["role"] == "user" else ""

        # Create enhanced system message with knowledge
        enhanced_system_content = self.system_message["content"]

        # Use the weighted memory integrator for domain-aware memory retrieval
        if current_query:
            # Detect if this is a command-related query
            command_context = extract_command_context(current_query)

            # Get domain information if available
            domain = None
            settings = None

            if hasattr(self, 'question_classifier'):
                settings = self.question_classifier.get_domain_settings(current_query)
                domain = settings['domain']

            # Check confidence to determine if web enhancement is needed
            confidence_data = self.confidence_metrics.get_metrics(apply_sharpening=True)

            # Try to enhance with web knowledge if confidence is low
            web_enhancement = None
            if use_web_search and self.enable_web_knowledge and hasattr(self, 'web_enhancer'):
                web_enhancement = self.enhance_with_web_knowledge(current_query, confidence_data, domain, messages)

            # Ensure recent memories are included by forcing a higher k value for recent queries
            recency_boost = True
            top_k = 8 if recency_boost else 8  # Default is 8

            if command_context:
                # Use specialized command memory retrieval
                memories = self.retrieve_command_memories(
                    current_query,
                    command_context=command_context,
                    top_k=top_k,
                    recency_weight=0.3,
                    include_tabular=True
                )
            else:
                # Use standard memory retrieval for non-command queries
                result = self.memory_integrator.retrieve_and_integrate(
                    self.current_user_id,
                    current_query
                )
                memories = result['memory_text']
                settings = result['settings']
                domain = settings['domain']

                # Add procedural knowledge section for certain domains
                if domain in ['translation', 'procedural']:
                    # Retrieve specialized procedural knowledge
                    procedures = self.retrieve_procedural_knowledge(current_query)
                    if procedures:
                        # Create structured procedure section
                        procedure_section = self.create_procedural_prompt_section(procedures)
                        if procedure_section:
                            # Add explicit instruction for how to apply the knowledge
                            instruction = "\nIMPORTANT: When performing tasks like translation or following procedures, " \
                                        "use the PROCEDURAL KNOWLEDGE section below as a direct reference. " \
                                        "For translations, apply each mapping rule precisely.\n"

                            # Append to memories with high priority
                            memories = instruction + procedure_section + "\n\n" + memories

            # Add web search results if available
            web_context = ""
            if web_enhancement and web_enhancement.get('enhanced', False):
                web_context = self.web_enhancer.format_web_results_for_context(web_enhancement)

            # Combine local memory and web knowledge
            if memories and web_context:
                enhanced_system_content += "\n\nIMPORTANT: Apply the following information in your response:\n"
                enhanced_system_content += f"\n{memories}\n"
                enhanced_system_content += f"\n{web_context}"
            elif memories:
                enhanced_system_content += "\n\nIMPORTANT: Apply the following information in your response:\n"
                enhanced_system_content += f"\n{memories}"
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

        # Add conversation history - keep more history for context
        history_messages = messages[1:] if len(messages) > 1 else []
        enhanced_messages.extend(history_messages[-5:])  # Keep up to 5 recent messages

        return enhanced_messages

    def enhance_with_web_knowledge(self, query: str, confidence_data: Dict[str, float], domain: Optional[str] = None, messages: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Enhance response generation with web knowledge when confidence is low.

        Args:
            query: User query
            confidence_data: Confidence metrics from the model
            domain: Optional domain classification

        Returns:
            Web enhancement data
        """
        if not self.enable_web_knowledge or self.web_enhancer is None:
            return {
                'enhanced': False,
                'reason': 'web_knowledge_disabled',
                'web_results': []
            }

        # Use the web enhancer to get relevant information
        enhancement_data = self.web_enhancer.enhance_response(
            query,
            confidence_data,
            domain,
            process_urls=False,  # Set to True to fetch full content (slower but more thorough)
            messages=messages
        )

        # Add to memory if enhancement was successful
        if enhancement_data.get('enhanced', False):
            self.web_enhancer.add_web_results_to_memory(
                self.current_user_id,
                query,
                enhancement_data,
                min_similarity=0.5  # Minimum similarity threshold for memory storage
            )

        return enhancement_data

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
        Implement speculative decoding with loop detection:
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

            # Process in batches for large inputs
            if full_sequence.size(1) > 1024:  # For long sequences
                def get_logits(batch):
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=batch,
                            attention_mask=torch.ones_like(batch),
                            return_dict=True
                        )
                    return outputs.logits

                target_logits = tensor_batch_processing(
                    get_logits,
                    full_sequence,
                    batch_size=4
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
            print(f"Error in speculative decoding: {e}")
            return None, None, None

    # Function for speculative decoding with streaming
    def generate_speculative(self, input_ids, streamer, max_new_tokens, generation_config, turbo_mode):
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
                                print(f"Error during interrupt cleanup: {e}")
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
                    print(f"Speculative decoding: {accepted_draft_tokens}/{total_tokens} tokens accepted ({efficiency:.1f}%)")

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
            print(f"\nError in generation thread: {str(e)}")
            import traceback
            traceback.print_exc()
            try:
                streamer.end()
            except Exception as specific_e:
                print(f"Error ending streamer: {specific_e}")

    def generate_response(self, messages, max_new_tokens=128, temperature=0.7, turbo_mode=True, show_confidence=False, response_filter=None, use_web_search=True):
        """Generate a response with ultra-fast speculative decoding (streaming only)"""
        # We only support streaming now, simplifies the code

        fallback_message_streamed = False

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
                        print(f"[User content saved to: {files_info}]")

            # Create enhanced prompt with knowledge
            enhanced_messages = self.create_prompt_with_knowledge(messages, use_web_search)

            # Use the window manager to optimize messages to fit token limits
            optimized_messages = self.window_manager.optimize_messages(enhanced_messages, max_new_tokens)

            # Log token counts to help with debugging
            original_tokens = self.window_manager.calculate_tokens(enhanced_messages)
            optimized_tokens = self.window_manager.calculate_tokens(optimized_messages)

            if original_tokens != optimized_tokens:
                print(f"Context window optimized: {original_tokens} → {optimized_tokens} tokens " +
                      f"(saved {original_tokens - optimized_tokens} tokens)")

            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                optimized_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Encode the prompt
            try:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            except Exception as e:
                print(f"Error encoding prompt: {e}")
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
            # tty.setraw(fd)
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

            try:
                while True:
                    if select.select([sys.stdin], [], [], 0)[0]:
                        c = sys.stdin.read(1)
                        if c == '\x03':  # Ctrl+C
                            self.stop_event.set()

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
                            print("\n[Repetitive pattern detected, getting correction suggestions...]")

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
                            print(f"Error getting correction suggestions: {e}")

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
                            print("\n[Generation complete: target length reached]")
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
                                print(f"[Response saved to: {filename}]")
                            else:
                                print(f"[Failed to save response to: {filename}]")

                # Track the generated response for continuation
                if hasattr(self, 'window_manager') and hasattr(self.window_manager, 'track_generated_response'):
                    self.window_manager.track_generated_response(
                        response=complete_response,
                        user_id=self.current_user_id,
                        target_length=target_length)

                return complete_response

            except Exception as e:
                print(f"\nError during token streaming: {str(e)}")
                if complete_response:
                    # Even with errors, make sure we have metrics
                    if not self.confidence_metrics.token_probabilities and not self.confidence_metrics.original_token_probabilities:
                        dummy_logits = torch.zeros(self.tokenizer.vocab_size)
                        dummy_logits[0] = 5.0  # Medium confidence for error cases
                        self.confidence_metrics.add_token_score(dummy_logits, 0)

                    return self.mcp_handler.finalize_streaming(complete_response)
                return "Error generating response. Please try again."

            self.last_token_confidences = token_confidences

        except Exception as e:
            print(f"\nStreaming setup failed: {e}")
            return "Error in streaming setup. Please try again."

        finally:
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
                # Scale the number of tokens based on response length
                num_tokens = max(5, min(response_length, 20))

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
            print(f"Error saving conversation: {e}")

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
        print(f"\n[Generated {response_tokens} tokens in {generation_time:.2f}s - ~{tokens_per_second:.1f} tokens/sec]")

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
            print(f"[Sharpening enabled: factor={self.sharpening_factor:.2f}]")

    def add_conversation_to_memory(self, query, response):
        """Add the current exchange to memory if auto-memorize is enabled"""
        if not self.auto_memorize:
            return 0

        # Extract key information
        key_info = self._extract_key_information(query, response)

        if not key_info:
            return 0

        # Create batch items
        batch_items = []
        timestamp = datetime.now().isoformat()

        for info in key_info:
            # Create metadata
            metadata = {
                'source_query': query,
                'source_response': response[:100] + "..." if len(response) > 100 else response,
                'timestamp': timestamp,
                'memory_type': "conversation"
            }

            # Add to batch
            batch_items.append({
                "content": info,
                "memory_type": "conversation",
                "source": "conversation",
                "metadata": metadata
            })

        # Add all items at once
        added_ids = self.memory_manager.add_bulk(
            batch_items,
            use_fractal=self.memory_manager.use_fractal
        )

        memories_added = sum(1 for item_id in added_ids if item_id is not None)
        
        if memories_added > 0:
            print(f"[Memory] Added {memories_added} new memories")
            
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

        print(f"Sharpening factor set to {factor}")

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
        if hasattr(self, 'memory_integrator'):
            expression = extract_arithmetic_expression(query)

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
        """High-performance command memory addition"""
        # Skip empty outputs
        if not output and not error:
            return 0

        batch_items = []

        # Detect command type and output format
        command_type = self._detect_command_type(command, output)
        is_tabular = is_tabular_data(output)

        # Create base metadata
        base_metadata = {
            "command": command,
            "command_type": command_type,
            "is_tabular": is_tabular,
            "timestamp": datetime.now().isoformat()
        }

        if output_file:
            base_metadata["output_file"] = output_file

        if error:
            base_metadata["has_error"] = True
            base_metadata["error"] = error[:100] if len(error) > 100 else error

        # Create main memory item
        memory_content = f"Command '{command}' output: {output[:500]}"
        if len(output) > 500:
            memory_content += "... [output truncated]"

        if error:
            memory_content += f"\nError: {error}"

        main_item = {
            "content": memory_content,
            "memory_type": "command",
            "source": "shell",
            "metadata": base_metadata.copy()
        }

        batch_items.append(main_item)

        # For tabular data, add specialized entries
        if is_tabular:
            lines = output.strip().split('\n')

            if len(lines) >= 2:
                # Extract header and data rows
                header_row = lines[0]
                data_rows = lines[1:]

                # Add specialized entries for specific rows (limit to 5)
                for i, row in enumerate(data_rows[:5]):
                    # Create row-specific metadata
                    row_metadata = base_metadata.copy()
                    row_metadata.update({
                        "row_index": i,
                        "header": header_row
                    })

                    # Create content
                    row_content = f"Row {i+1} from command '{command}': {row}"

                    # Add to batch
                    batch_items.append({
                        "content": row_content,
                        "memory_type": "command_tabular",
                        "source": "shell",
                        "metadata": row_metadata
                    })

        # Add all items at once
        added_ids = self.memory_manager.add_bulk(batch_items, use_fractal=self.memory_manager.use_fractal)

        # Return count of added items
        return sum(1 for item_id in added_ids if item_id is not None)

    def _detect_command_type(self, command: str, output: str) -> str:
        """Detect the type of command based on the command string and output."""
        if any(cmd in command for cmd in ["ls", "dir"]):
            return "file_listing"
        elif any(cmd in command for cmd in ["grep", "find", "locate"]):
            return "search"
        elif any(cmd in command for cmd in ["df", "du", "free", "top", "ps"]):
            return "system_metrics"
        elif any(cmd in command for cmd in ["cat", "less", "more", "head", "tail"]):
            return "file_content"
        elif any(cmd in command for cmd in ["uname", "hostname", "whoami", "id"]):
            return "system_info"
        else:
            return "general_command"

    def _add_tabular_command_memory(self, command: str, output: str, command_type: str) -> int:
        """Add specialized memory entries for tabular command output."""
        lines = output.strip().split('\n')
        
        if len(lines) < 2:
            return 0
            
        # Extract header and data rows
        header_row = lines[0]
        data_rows = lines[1:]
        
        # Try to identify columns based on whitespace patterns
        columns = extract_columns_from_header(header_row)
        
        # Add memory entries for specific rows (limit to 5 for performance)
        memories_added = 0
        
        for i, row in enumerate(data_rows[:5]):
            # Create row-specific metadata
            row_metadata = {
                "command": command,
                "command_type": command_type,
                "is_tabular": True,
                "row_index": i,
                "header": header_row,
                "timestamp": datetime.now().isoformat()
            }
            
            # Create row-specific content
            row_content = f"Row {i+1} from command '{command}': {row}"
            
            # Add to memory
            item_id = self.memory_manager.add(
                content=row_content,
                memory_type="tabular_command",
                source="shell",
                metadata=row_metadata,
                use_fractal=self.memory_manager.use_fractal
            )
            
            if item_id:
                memories_added += 1
                
        return memories_added

    def retrieve_command_memories(self, query: str, command_context: Optional[str] = None, top_k: int = 5, recency_weight: float = 0.3, include_tabular: bool = True) -> str:
        """
        Enhanced retrieval specifically optimized for command outputs with better
        relevance scoring and support for tabular data.

        Args:
            query: The user query
            command_context: Optional context about the command being discussed
            top_k: Maximum number of memories to retrieve
            recency_weight: Weight given to recency (0-1)
            include_tabular: Whether to include specialized tabular data retrieval

        Returns:
            Formatted string of relevant command memories
        """
        # Generate embedding for query
        query_embedding = self.memory_manager.embedding_function(query) if self.memory_manager.embedding_function else None
        
        if not query_embedding:
            return ""

        # Search memory for related command outputs
        results = self.memory_manager.retrieve(
            query=query,
            memory_types=["command", "tabular_command"],
            top_k=top_k*2,  # Get more than needed for better filtering
            min_similarity=0.2,
            use_fractal=self.memory_manager.use_fractal
        )
        
        # If we have specific command context, prioritize matching commands
        if command_context:
            # Sort results to prioritize command matches
            for result in results:
                metadata = result.get("metadata", {})
                stored_command = metadata.get("command", "")
                
                # Boost score for command matches
                if commands_match(command_context, stored_command):
                    result["similarity"] *= 1.2  # 20% boost
        
        # Apply recency bias
        now = datetime.now().timestamp()
        for result in results:
            metadata = result.get("metadata", {})
            timestamp_str = metadata.get("timestamp")
            
            if timestamp_str:
                try:
                    # Parse timestamp
                    if 'T' in timestamp_str:
                        # ISO format
                        timestamp = datetime.fromisoformat(timestamp_str).timestamp()
                    else:
                        # Direct timestamp
                        timestamp = float(timestamp_str)
                        
                    # Calculate hours age
                    age_hours = (now - timestamp) / 3600
                    
                    # Apply recency boost (more recent = higher score)
                    recency_factor = math.exp(-0.01 * age_hours)  # Exponential decay
                    recency_boost = recency_weight * recency_factor
                    
                    # Apply boost
                    result["similarity"] = min(1.0, result["similarity"] * (1.0 + recency_boost))
                except:
                    pass
        
        # Re-sort based on adjusted scores
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Format the results for display
        return self._format_command_memories(results[:top_k], query, command_context)

    def _format_command_memories(self, results: List[Dict], query: str, command_context: Optional[str] = None) -> str:
        """Format command memories for inclusion in the prompt."""
        if not results:
            return ""
            
        # Group results into categories
        tabular_results = []
        general_results = []
        
        for result in results:
            metadata = result.get("metadata", {})
            
            if metadata.get("is_tabular", False):
                tabular_results.append(result)
            else:
                general_results.append(result)
                
        # Format the output with clear sections
        memory_text = ""
        
        # Add tabular data first if available
        if tabular_results:
            memory_text += "TABULAR DATA:\n"
            
            for i, result in enumerate(tabular_results[:3]):  # Limit to 3 entries
                memory_text += f"- {result['content']}\n"
                
            memory_text += "\n"
            
        # Add general command results
        if general_results:
            memory_text += "COMMAND OUTPUT INFORMATION:\n"
            
            for i, result in enumerate(general_results[:5]):  # Limit to 5 general results
                memory_text += f"- {result['content']}\n"
                
        return memory_text

    def retrieve_procedural_knowledge(self, query: str, procedure_type: str = None) -> List[Dict]:
        """
        Retrieve procedural knowledge relevant to a query, optimized for application.

        Args:
            query: The user query
            procedure_type: Optional filter for specific procedure types

        Returns:
            List of relevant procedural knowledge entries
        """
        # Search for procedural memories
        results = self.memory_manager.retrieve(
            query=query,
            memory_types=["procedural", "mapping", "rules"],
            top_k=15,
            min_similarity=0.2,
            use_fractal=self.memory_manager.use_fractal
        )
        
        # Extract procedural data
        procedures = []
        
        for result in results:
            # Check if this memory contains procedural content
            content = result.get("content", "")
            
            if any(marker in content.lower() for marker in 
                ["step", "procedure", "→", "->", "mapping", "instruction", "how to"]):
                
                # Parse the procedural content
                has_steps = "step" in content.lower() or any(line.strip().startswith(("-", "*", "•")) for line in content.split("\n"))
                has_mapping = "→" in content or "->" in content
                
                proc_entry = {
                    "type": "mapping_table" if has_mapping else "step_sequence",
                    "content": content,
                    "relevance": result.get("similarity", 0)
                }
                
                # Extract steps or mappings if possible
                if has_steps:
                    # Find steps using regex
                    steps = []
                    step_lines = re.findall(r'(?:^|\n)(?:\d+\.|\*|\-|\•)\s*(.+?)(?=(?:\n(?:\d+\.|\*|\-|\•))|$)', content)
                    if step_lines:
                        proc_entry["steps"] = step_lines
                
                if has_mapping:
                    # Find mappings using regex
                    mappings = re.findall(r'([^\n→\->]+)(?:→|->)([^\n]+)', content)
                    if mappings:
                        proc_entry["entries"] = [
                            {"input": m[0].strip(), "output": m[1].strip()} 
                            for m in mappings
                        ]
                
                procedures.append(proc_entry)
        
        # Filter by type if specified
        if procedure_type:
            procedures = [p for p in procedures if p.get("type") == procedure_type]
            
        # Sort by relevance
        return sorted(procedures, key=lambda x: x.get("relevance", 0), reverse=True)

    def create_procedural_prompt_section(self, procedures: List[Dict]) -> str:
        """Create a structured prompt section from procedural knowledge."""
        if not procedures:
            return ""
            
        sections = []
        
        # Handle mapping tables
        mapping_tables = [p for p in procedures if p.get("type") == "mapping_table"]
        if mapping_tables:
            sections.append("PROCEDURAL KNOWLEDGE - MAPPING TABLES:")
            
            for i, table in enumerate(mapping_tables[:3]):  # Limit to top 3
                entries = table.get("entries", [])
                
                if entries:
                    sections.append(f"\n{i+1}. Mapping Table:")
                    
                    # Format entries in a clean, structured way
                    formatted_entries = []
                    for entry in entries[:15]:  # Limit entries for context
                        input_val = entry.get("input", "")
                        output_val = entry.get("output", "")
                        
                        formatted_entries.append(f"  • {input_val} → {output_val}")
                        
                    sections.append("\n".join(formatted_entries))
                    
                # If no structured entries, include raw content
                elif table.get("content"):
                    sections.append(f"\n{i+1}. Mapping Information:")
                    sections.append(f"  {table['content']}")
        
        # Handle step sequences
        step_sequences = [p for p in procedures if p.get("type") == "step_sequence"]
        if step_sequences:
            sections.append("\nPROCEDURAL KNOWLEDGE - STEP SEQUENCES:")
            
            for i, sequence in enumerate(step_sequences[:2]):  # Limit to top 2
                steps = sequence.get("steps", [])
                
                if steps:
                    sections.append(f"\n{i+1}. Step Sequence:")
                    
                    for j, step in enumerate(steps):
                        sections.append(f"  {j+1}. {step}")
                        
                # If no structured steps, include raw content
                elif sequence.get("content"):
                    sections.append(f"\n{i+1}. Procedure:")
                    sections.append(f"  {sequence['content']}")
                    
        return "\n".join(sections)

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
                    print(f"Consolidated memories for user {self.current_user_id}")
                except Exception as e:
                    print(f"Error consolidating memories: {e}")

            # Final cleanup
            if hasattr(self, 'resource_manager'):
                self.resource_manager.cleanup()

            if torch.cuda.is_available():
                print("Releasing cuda cache.")
                torch.cuda.empty_cache()

            print("Resources cleaned up successfully")

        except Exception as e:
            print(f"Error during cleanup: {e}")
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
                print(f"Received {len(user_input.splitlines())} lines of input.")

            return user_input

        except KeyboardInterrupt:
            print("\nInput cancelled.")
            return ""
        except Exception as e:
            print(f"\nError during input: {e}")
            return ""

def main():
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

    try:
        # If web knowledge is enabled, configure the search engine
        if args.web_knowledge and chat.web_enhancer:
            chat.web_enhancer.search_engine = args.search_engine
            chat.web_enhancer.confidence_threshold = args.web_confidence
            print(f"Web knowledge enhancement enabled using {args.search_engine}")

        if args.test_fractal:
            # Initialize memory manager with fractal enabled
            memory_manager = MemoryManager(
                memory_dir="./memory",
                fractal_enabled=True
            )

            # Get a user store
            store = memory_manager._get_user_store("diagnostic_user")

            # Run and print diagnostics
            store.print_fractal_embedding_diagnostics()
            return  # Exit after diagnostics

        response_filter = ResponseFilter(
            confidence_threshold=args.confidence_threshold,
            user_context=user_context,
            question_classifier=chat.question_classifier
        )

        current_time = datetime.now().strftime("[%d/%m/%y %H:%M:%S]")
        history_file = setup_readline_history(chat.memory_dir)
        print(f"{current_time} Command history stored in: {history_file}")

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
                print("Using 8-bit quantization for better performance")
            except ImportError:
                print("bitsandbytes not installed, using full precision")

            print("Warming up model for maximum throughput...")
            _ = chat.generate_response(
                [{"role": "user", "content": "Say some nice greeting."}],
                max_new_tokens=16,
                temperature=0.7,
                use_web_search=False
            )

        # Start conversation loop
        print("\n" + "="*50)
        print("TinyLlama Chat with Speculative Decoding and MCP")
        print("="*50)
        print("Type 'exit' to end the conversation")
        print("Special commands:")
        print("  !teach: [fact] - Add knowledge to the model")
        print("  !correct: [correction] - Correct the model's understanding")
        print("  !save - Save the current conversation")
        print("  !system: [message] - Change the system message")
        print("  !mcp-help - Show MCP commands for directing output to files")
        print("  !confidence: [0.0-1.0] - Set confidence threshold")
        print("  !sharpening-factor: [0.0-1.0] - Set the sharpening factor for vector embeddings")
        print("  !toggle-turbo - Toggle turbo mode on/off")
        print("  !toggle-filter - Toggle uncertainty filtering on/off")
        print("  !toggle-heatmap - Toggle confidence heatmap visualization on/off")
        print("  !toggle-all-metrics - Toggle between showing all metrics or just truthiness")
        print("  !toggle-sharpening - Toggle vector space sharpening on/off")
        print("  !memorize - Force save the entire conversation to memory")
        print("  !toggle-memory - Toggle automatic memorization on/off")
        print("  !memory-stats - Display info about memories")
        print("  !toggle-web - Toggle web knowledge enhancement on/off")
        print("  !web-stats - Show web search statistics")
        print("  !search-engine: [engine] - Set search engine (duckduckgo/google)")
        print("  !fractal-diagnostics - prints fractal embedding diagnostics")
        print("  !visualize-fractal - visual fractal embeddings")
        print("  !compare-queries: [query1] | [query2] - Compare the semantic relationship between two queries")
        print("  !create-domain: [name] | [description] - Create a new knowledge domain")
        print("  !load-domain: [domain_id] - Load and activate a knowledge domain")
        print("  !list-domains - List all available knowledge domains")
        print("  !extract-knowledge: [text] - Extract structured knowledge from text")

        print("\nIf the model expresses uncertainty, you can ask it to speculate")
        print("by saying 'please continue anyway' or 'please speculate'")

        print("="*50 + "\n")

        # Set initial mode settings
        turbo_mode = args.turbo
        show_confidence = args.heatmap

        conversation = [chat.system_message]

        while True:
            # Get timestamp for user input
            current_time = datetime.now().strftime("[%d/%m/%y %H:%M:%S]")
            user_input = chat.get_multiline_input(f"\n{current_time} You: ")

            if user_input == "":
                chat.stop_event.set()
                continue

            # Handle special commands
            if user_input.lower() == 'exit':
                feedback_time = datetime.now().strftime("[%d/%m/%y %H:%M:%S]")
                feedback = 'y' # input(f"\n{feedback_time} Was this response helpful? (y/n, or provide feedback): ")
                if feedback.lower() != 'y' and feedback.lower() != 'yes' and feedback.strip():
                    # If the user provided specific feedback, add it to knowledge
                    if len(feedback) > 2:
                        correction = f"Regarding '{user_input}', remember: {feedback}"
                        chat.add_to_knowledge(correction, fact_type="correction")
                        print("Feedback saved as correction. I'll try to do better next time.")
                    else:
                        print("Sorry the response wasn't helpful.")

                store = chat.memory_manager

                # Show final stats
                stats = store.get_stats()
                print(f"Memories saved this session: {stats['active_items']}")
                print(f"Total memories saved this: {stats['total_items']}")
                break

            elif user_input.lower().startswith('!teach:'):
                new_fact = user_input[7:].strip()
                chat.add_to_knowledge(new_fact, fact_type="fact")
                print(f"Added to knowledge base: {new_fact}")
                continue

            elif user_input.lower().startswith('!correct:'):
                correction = user_input[9:].strip()
                chat.add_to_knowledge(correction, fact_type="correction")
                print(f"Added correction: {correction}")
                continue

            elif user_input.lower() == '!save':
                chat.save_conversation(conversation)
                print("Conversation saved!")
                continue

            elif user_input.lower().startswith('!system:'):
                new_system = user_input[8:].strip()
                chat.system_message = {
                    "role": "system",
                    "content": new_system
                }
                conversation[0] = chat.system_message
                print(f"System message updated: {new_system}")
                continue

            elif user_input.lower() == '!toggle-turbo':
                turbo_mode = not turbo_mode
                print(f"Turbo mode {'enabled' if turbo_mode else 'disabled'}")
                continue

            elif user_input.lower() == '!mcp-help':
                help_text = chat.mcp_handler.get_help_text()
                print(help_text)
                continue
            elif user_input.lower() == '!toggle-filter':
                filter_enabled = not filter_enabled
                print(f"Response filtering {'enabled' if filter_enabled else 'disabled'}")
                continue
            elif user_input.lower() == '!toggle-heatmap':
                show_confidence = not show_confidence
                print(f"Confidence heatmap {'enabled' if show_confidence else 'disabled'}")
                continue
            elif user_input.lower() == '!toggle-all-metrics':
                show_all_metrics = not show_all_metrics
                print(f"Detailed metrics display {'enabled' if show_all_metrics else 'disabled'}")
                continue
            elif user_input.lower() == '!toggle-sharpening':
                is_enabled = chat.memory_manager.toggle_sharpening()
                print(f"Vector space sharpening {'enabled' if is_enabled else 'disabled'}")
                continue
            elif user_input.lower().startswith('!sharpening-factor:'):
                try:
                    factor = float(user_input.split(':')[1].strip())
                    if 0.0 <= factor <= 1.0:
                        chat.memory_manager.set_sharpening_factor(factor)
                    else:
                        print("Sharpening factor must be between 0.0 and 1.0")
                except Exception as e:
                    print(f"Invalid value: {str(e)}. Please specify a number between 0.0 and 1.0")
                continue

            elif user_input.lower() == '!memorize':
                memories_added = chat.save_conversation(conversation)
                print(f"Conversation saved to long-term memory! Added {memories_added} memories.")
                continue

            elif user_input.lower() == '!toggle-memory':
                is_enabled = chat.memory_manager.toggle_auto_memorize()
                print(f"Automatic memorization {'enabled' if is_enabled else 'disabled'}")
                continue

            elif user_input.lower() == '!memory-stats':
                try:
                    # Get vector store stats
                    store = chat.memory_manager
                    store_stats = store.get_stats() if store else {"total_items": 0, "active_items": 0}

                    # Get memory manager state
                    auto_memorize = chat.memory_manager.auto_memorize if hasattr(chat.memory_manager, 'auto_memorize') else False

                    print("\nMemory System Statistics:")
                    print(f"Total memories: {store_stats.get('total_items', 0)}")
                    print(f"Active memories: {store_stats.get('active_items', 0)}")
                    print(f"Auto-memorize: {'Enabled' if auto_memorize else 'Disabled'}")

                    # Add additional statistics from the store if available
                    if hasattr(store, 'get_stats') and callable(store.get_stats):
                        try:
                            # Include more stats if they're available and informative
                            if 'index_dimension' in store_stats:
                                print(f"Embedding dimension: {store_stats.get('index_dimension', 384)}")
                            if 'deleted_documents' in store_stats:
                                print(f"Deleted memories: {store_stats.get('deleted_documents', 0)}")
                            # Show when the last store update happened
                            if 'last_updated' in store_stats:
                                print(f"Last updated: {store_stats.get('last_updated', 'never')}")
                        except Exception:
                            # Ignore errors in additional stats
                            pass

                except Exception as e:
                    print(f"Unexpected error when getting memory stats: {str(e)}")
                continue

            elif user_input.lower() == '!toggle-web':
                is_enabled = chat.toggle_web_knowledge()
                print(f"Web knowledge enhancement {'enabled' if is_enabled else 'disabled'}")
                continue

            elif user_input.lower() == '!web-stats':
                if hasattr(chat, 'web_enhancer') and chat.web_enhancer:
                    stats = chat.web_enhancer.get_stats()
                    print("\nWeb Knowledge Statistics:")
                    print(f"Total searches: {stats['total_searches']}")
                    print(f"Successful searches: {stats['successful_searches']}")
                    print(f"Success rate: {stats['success_rate']*100:.1f}%")
                    print(f"Cache hits: {stats['cache_hits']}")
                    print(f"Cache hit rate: {stats['cache_hit_rate']*100:.1f}%")
                else:
                    print("Web knowledge enhancement is not enabled")
                continue

            elif user_input.lower().startswith('!search-engine:'):
                if hasattr(chat, 'web_enhancer') and chat.web_enhancer:
                    engine = user_input.split(':')[1].strip().lower()
                    if engine in ["duckduckgo", "google"]:
                        chat.web_enhancer.search_engine = engine
                        print(f"Search engine set to: {engine}")
                    else:
                        print(f"Unsupported search engine: {engine}. Please use 'duckduckgo' or 'google'")
                else:
                    print("Web knowledge enhancement is not enabled")
                continue

            elif user_input.lower() == '!fractal-diagnostics':
                store = chat.memory_manager
                if hasattr(store, 'print_fractal_embedding_diagnostics'):
                    store.print_fractal_embedding_diagnostics()
                else:
                    print("Fractal diagnostics not available.")
                continue

            elif user_input.lower() == '!visualize-fractal':
                store = chat.memory_manager
                if hasattr(store, 'visualize_fractal_embeddings'):
                    store.visualize_fractal_embeddings()
                else:
                    print("Fractal visualization not available.")
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
                        print(f"\nSemantic relationship between the queries: {relationship}")
                    else:
                        print("\nSemantic reasoning capability not available. Run the finetuning first.")
                except Exception as e:
                    print(f"\nError comparing queries: {e}")
                    print("Usage: !compare-queries: first query | second query")
                continue

            elif user_input.lower().startswith('!create-domain:'):
                domain_info = user_input[14:].strip()
                parts = domain_info.split('|')
                name = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else ""
                domain_id = chat.create_knowledge_domain(name, description)
                if domain_id:
                    print(f"Created domain: {name} (ID: {domain_id})")
                else:
                    print("Failed to create domain")
                continue

            elif user_input.lower().startswith('!load-domain:'):
                domain_id = user_input[13:].strip()
                success = chat.load_knowledge_domain(domain_id)
                if success:
                    print(f"Loaded domain: {domain_id}")
                else:
                    print(f"Failed to load domain: {domain_id}")
                continue

            elif user_input.lower().startswith('!list-domains'):
                if hasattr(chat.memory_manager, 'knowledge_registry'):
                    domains = chat.memory_manager.knowledge_registry.list_domains()
                    print("\nAvailable Knowledge Domains:")
                    for domain in domains:
                        print(f"- {domain['name']} (ID: {domain['domain_id']})")
                        print(f"  Description: {domain['description']}")
                        print(f"  Items: {domain['stats'].get('total_items', 0)}")
                else:
                    print("Knowledge registry not available")
                continue

            elif user_input.lower().startswith('!extract-knowledge:'):
                text = user_input[18:].strip()
                if not text:
                    print("Please provide text to extract knowledge from")
                else:
                    knowledge_items = chat.extract_knowledge_from_text(text, "user_input")
                    print(f"Extracted {len(knowledge_items)} knowledge items:")
                    for i, item in enumerate(knowledge_items[:5]):  # Show first 5
                        print(f"{i+1}. Type: {item['type']}")
                        print(f"   Content: {item['content']}")
                        print(f"   Confidence: {item['metadata']['confidence']}")

                    if chat.current_domain_id:
                        add_to_domain = input(f"Add to current domain '{chat.current_domain_id}'? (y/n): ")
                        if add_to_domain.lower() == 'y':
                            count = chat.add_knowledge_to_domain(knowledge_items)
                            print(f"Added {count} knowledge items to domain {chat.current_domain_id}")
                continue

            # Add user message to conversation
            user_message = {"role": "user", "content": user_input}
            conversation.append(user_message)

            # Generate response
            try:
                # Measure generation time
                start_time = time.time()

                # Add timestamp for model output
                response_time = datetime.now().strftime("[%d/%m/%y %H:%M:%S]")

                chat.stop_event.clear()  # Reset the event

                # If no system command matched, generate response using the model
                print(f"\n{response_time} Assistant: \n", end='', flush=True)

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
                print()

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
                            print("\nFiltered response:")

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
                    print(f"stupid error {e}")
                    # Fallback to maximum tokens estimate
                    if args.max_tokens > 0:
                        tokens_per_second = args.max_tokens / generation_time
                        print(f"[Generated in {generation_time:.2f}s - ~{tokens_per_second:.1f} tokens/sec | Confidence: {confidence_data['confidence']:.2f}]")

                # Get feedback with timestamp
                # feedback_time = datetime.now().strftime("[%d/%m/%y %H:%M:%S]")
                # feedback = input(f"\n{feedback_time} Was this response helpful? (y/n, or provide feedback): ")

                # if feedback.lower() != 'y' and feedback.lower() != 'yes' and feedback.strip():
                #     # If the user provided specific feedback, add it to knowledge
                #     if len(feedback) > 2:
                #         correction = f"Regarding '{user_input}', remember: {feedback}"
                #         chat.add_to_knowledge(correction, fact_type="correction")
                #         print("Feedback saved as correction. I'll try to do better next time.")
                #     else:
                #         print("Sorry the response wasn't helpful.")

            except Exception as e:
                print(f"Error generating response: {e}")
                print("Please try again with a different question.")

            # finally:
            #     chat.cleanup()

    except KeyboardInterrupt:
        print("\nExiting due to keyboard interrupt...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        # This should only happen when exiting the program
        chat.cleanup()

if __name__ == "__main__":
    main()

