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
import argparse
import select
import tty

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
from memory_manager import MemoryManager, VectorStore

from terminal_heatmap import TerminalHeatmap, EnhancedHeatmap
from question_classifier import QuestionClassifier
from weighted_memory_integrator import WeightedMemoryIntegrator

from pattern_matching_utils import (
    is_command_related_query, extract_command_context, extract_command_type,
    is_tabular_command, is_tabular_data, extract_columns_from_header,
    extract_column_references, extract_row_references, extract_arithmetic_expression,
    commands_match, extract_example_pairs, extract_mapping_category
)

from batch_utils import tensor_batch_processing
from web_knowledge_enhancer import WebKnowledgeEnhancer

from semantic_reasoning_enhancer import integrate_semantic_reasoning
from rolling_window_manager import RollingWindowManager

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
It’s like being a superhero: you use your superpower of speaking wisely and listen to others, too!
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
             max_fractal_levels=3
             ):
        self.model_name = model_name
        self.memory_dir = memory_dir
        self.stop_event = threading.Event()

        self.mcp_handler = MCPHandler(output_dir=output_dir, allow_shell_commands=True)
        self.confidence_metrics = EnhancedConfidenceMetrics(sharpening_factor=sharpening_factor)

        # Initialize resource manager
        self.resource_manager = ResourceManager(device=device)

        # Initialize memory manager
        self.memory_manager = MemoryManager(
            memory_dir=memory_dir,
            device=device,
            auto_memorize=auto_memorize,
            sharpening_enabled=enable_sharpening,
            sharpening_factor=sharpening_factor,
            fractal_enabled=fractal_enabled,  # Enable fractal embeddings
            max_fractal_levels=max_fractal_levels   # Number of fractal levels
        )

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
            except:
                print(f"Finetuned model not loaded, it doesn't exist in {finetuned_model_path}")

        self.knowledge_system_enabled = self.memory_manager.initialize_knowledge_system()
        self.current_domain_id = None  # Currently active domain ID

        # Initialize the web knowledge enhancer
        self.enable_web_knowledge = enable_web_knowledge
        if enable_web_knowledge:
            self.web_enhancer = WebKnowledgeEnhancer(
                memory_manager=self.memory_manager,
                chat=self,
                confidence_threshold=confidence_threshold,
                vector_sharpening_factor=sharpening_factor,
                search_engine="duckduckgo"  # Use DuckDuckGo by default for fewer rate limits
            )
            print("Web knowledge enhancement initialized")
        else:
            self.web_enhancer = None

        self.current_user_id = "default_user"

        if self.knowledge_system_enabled:
            self._initialize_knowledge_integration()

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
        self.window_manager = RollingWindowManager(
            tokenizer=self.tokenizer,
            max_window_size=2048,  # Default context window size
            memory_manager=self.memory_manager,
            safety_margin=50  # Provide some buffer
        )

        print("Context window management initialized")


        # Set pad_token if not already set and different from eos_token
        # FIXES: The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's attention_mask to obtain reliable results
        # if self.tokenizer.pad_token is None or self.tokenizer.pad_token == self.tokenizer.eos_token:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        #     # Use a special config to silence the warning
        #     self.generation_attention_mask = torch.ones((1, 1))

        # Main model loading
        loading_options = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto" if self.device != "cpu" else None,
            "low_cpu_mem_usage": True,
        }

        # Load model
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **loading_options)
        self.model = self.resource_manager.register_model(self.model)

        # Create draft model from target model by reducing layers
        self.draft_model = self.create_draft_model()

        if self.draft_model:
            print("Created draft model by reducing layers")
        else:
            print("Could not create draft model, speculative decoding disabled")

        # Load memory/knowledge base
        self.knowledge_base = self.load_knowledge()
        self.conversation_history = []
        self.system_message = DEFAULT_SYSTEM_MESSAGE

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

    def create_prompt_with_knowledge(self, messages, use_web_search=True):
        """Create a prompt that incorporates relevant knowledge with domain-aware weighting"""
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
            confidence_data = self.confidence_metrics.get_metrics(apply_sharpening=self.memory_manager.sharpening_enabled)

            # Try to enhance with web knowledge if confidence is low
            web_enhancement = None
            if use_web_search and hasattr(self, 'enable_web_knowledge') and self.enable_web_knowledge and hasattr(self, 'web_enhancer'):
                web_enhancement = self.enhance_with_web_knowledge(current_query, confidence_data, domain, messages)

            # Ensure recent memories are included by forcing a higher k value for recent queries
            recency_boost = True
            top_k = 8 if recency_boost else 8  # Increase from default 8

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
                confidence_threshold=self.confidence_metrics.get_metrics()['confidence'],
                vector_sharpening_factor=self.memory_manager.sharpening_factor,
                search_engine="duckduckgo"
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
                    do_sample=False,  # set to True for non greedy decoding for draft
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
                        if self.stop_event.is_set(): # or self.interrupt_handler.check():
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
                stride=8  # Reasonable stride for batching
            )

            # Generation configuration
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature >= 0.1, # use only if do_sample=false
                "temperature": temperature if temperature > 0.1 else 1.0,
                "top_k": 50,
                "top_p": 0.95, # use only if do_sample=False
                "repetition_penalty": 1.0,
                "num_beams": 1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True
            }

            if messages and messages[-1]["role"] == "user":
                user_query = messages[-1]["content"]
                generation_config = self.get_domain_specific_generation_config(user_query, generation_config)

            # Start generation in background
            thread = Thread(target=self.generate_speculative, args=(input_ids, streamer, max_new_tokens, generation_config, turbo_mode))
            thread.daemon = True
            thread.start()

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
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
                    token = next(iter(streamer), None)

                    if token is None:
                        # No more tokens - generation complete
                        break

                    if select.select([sys.stdin], [], [], 0)[0]:
                        c = sys.stdin.read(1)
                        if c == '\x03':  # Ctrl+C
                            self.stop_event.set()

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            break

                    tokens_received += 1

                    # Process token for MCP
                    display_token, mcp_buffer = self.mcp_handler.process_streaming_token(token, mcp_buffer)


                    # Add token to response
                    complete_response += token

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

                    # Check confidence early enough to stop generation if needed
                    # But only if we have a response filter and after receiving some tokens
                    if (response_filter is not None and not low_confidence_detected and tokens_received >= early_confidence_check_threshold):

                        # Get current metrics
                        current_metrics = self.confidence_metrics.get_metrics(apply_sharpening=self.memory_manager.sharpening_enabled)

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

                                return self.mcp_handler.finalize_streaming(complete_response)

                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
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
            # termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


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
        confidence_data = self.confidence_metrics.get_metrics(apply_sharpening=self.memory_manager.sharpening_enabled)

        # Check if we have original metrics for comparison (this is where the bug was)
        has_sharpening = "original" in confidence_data and self.memory_manager.sharpening_enabled

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
        if self.memory_manager.sharpening_enabled:
            print(f"[Sharpening enabled: factor={self.memory_manager.sharpening_factor:.2f}]")

    def add_conversation_to_memory(self, query, response):
        """Add the current exchange to memory if auto-memorize is enabled"""
        memories_added = self.memory_manager.add_memory(
            self.current_user_id,
            query,
            response,
            pre_sharpen=self.memory_manager.sharpening_enabled  # Use pre-sharpening if enabled
        )
        if memories_added > 0:
            print(f"[Memory] Added {memories_added} new memories")

    def toggle_sharpening(self):
        """Toggle vector space sharpening on/off"""
        return self.memory_manager.toggle_sharpening()

    def set_sharpening_factor(self, factor: float) -> None:
        """Set the sharpening factor for memory retrieval and confidence metrics"""
        # Update confidence metrics sharpening
        if hasattr(self.confidence_metrics, 'set_sharpening_factor'):
            self.confidence_metrics.set_sharpening_factor(factor)

            # Force recalculation with sharpening applied
            if hasattr(self.confidence_metrics, 'token_probabilities') and self.confidence_metrics.token_probabilities:
                _ = self.confidence_metrics.get_metrics(apply_sharpening=True)

        # Update memory manager sharpening
        self.memory_manager.set_sharpening_factor(factor)

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
        """
        Add shell command output to memory with enhanced indexing for tabular data and structured outputs.

        Args:
            command: The shell command that was executed
            output: Command output text
            error: Command error text (if any)
            output_file: Path to file where full output was saved (for large outputs)
        """
        # Skip empty outputs
        if not output and not error:
            return 0

        # Detect command type and output format for specialized processing
        command_type, is_tabular = self._detect_command_type(command, output)

        # For tabular data, use specialized indexing
        if is_tabular:
            return self._index_tabular_data(command, output, command_type)

        # For other command types, use improved chunking and metadata
        return self._index_general_command_output(command, output, error, output_file, command_type)

    def _detect_command_type(self, command: str, output: str) -> tuple:
        """
        Detect the type of command and whether its output is tabular.

        Args:
            command: The shell command
            output: Command output text

        Returns:
            Tuple of (command_type, is_tabular)
        """
        # Identify command type more precisely
        if any(cmd in command for cmd in ["ls", "dir"]):
            command_type = "file_listing"
        elif any(cmd in command for cmd in ["grep", "find", "locate"]):
            command_type = "search"
        elif any(cmd in command for cmd in ["df", "du", "free", "top", "ps"]):
            command_type = "system_metrics"
        elif any(cmd in command for cmd in ["cat", "less", "more", "head", "tail"]):
            command_type = "file_content"
        elif any(cmd in command for cmd in ["uname", "hostname", "whoami", "id"]):
            command_type = "system_info"
        else:
            command_type = "general_command"

        # Detect if output is likely tabular
        is_tabular = is_tabular_data(output)

        return command_type, is_tabular

    def _index_tabular_data(self, command: str, output: str, command_type: str) -> int:
        """
        Create specialized memory entries for tabular command output.

        Args:
            command: The shell command
            output: Command output text
            command_type: Type of command

        Returns:
            Number of memories added
        """
        total_memories = 0
        lines = output.strip().split('\n')

        if len(lines) < 2:
            return 0

        # Extract header and data rows
        header_row = lines[0]
        data_rows = lines[1:]

        # Try to identify columns based on whitespace patterns
        columns = extract_columns_from_header(header_row)

        # Create a structured memory for the entire table
        table_memory_text = f"Table from command '{command}':\n{output}"
        table_metadata = {
            "command": command,
            "command_type": command_type,
            "is_tabular": True,
            "column_count": len(columns),
            "row_count": len(data_rows),
            "timestamp": datetime.now().isoformat()
        }

        # Add overall table memory
        memories_added = self.memory_manager.add_memory(
            self.current_user_id,
            f"Tabular data from command: {command}",
            table_memory_text,
            memory_type="tabular_data",
            attributes=table_metadata,
            pre_sharpen=self.memory_manager.sharpening_enabled
        )
        total_memories += memories_added

        # For each row, create a more specific memory with structured data
        for i, row in enumerate(data_rows):
            # Try to extract structured data from the row
            row_data = self._extract_row_data(row, columns, header_row)

            # Skip if we couldn't parse the row
            if not row_data:
                continue

            # Create row-specific memory
            row_memory = f"Row {i+1} from '{command}' output:\n"
            for col, val in row_data.items():
                row_memory += f"{col}: {val}\n"

            row_metadata = {
                "command": command,
                "command_type": command_type,
                "is_tabular": True,
                "row_index": i,
                "timestamp": datetime.now().isoformat(),
                "values": row_data
            }

            # Add row memory with detailed metadata
            memories_added = self.memory_manager.add_memory(
                self.current_user_id,
                f"Row details from command: {command}",
                row_memory,
                memory_type="tabular_row",
                attributes=row_metadata
            )
            total_memories += memories_added

        # For system metrics commands, add specialized memories for key metrics
        if command_type == "system_metrics" and "df" in command:
            # Create specific memories for filesystem usage metrics
            for i, row in enumerate(data_rows):
                row_data = self._extract_row_data(row, columns, header_row)
                if not row_data:
                    continue

                # Focus on filesystem usage
                if "Filesystem" in columns and "Use%" in columns:
                    filesystem = row_data.get("Filesystem", "unknown")
                    usage = row_data.get("Use%", "").strip("%")
                    size = row_data.get("Size", "")

                    metric_memory = f"Filesystem '{filesystem}' is using {usage}% of its {size} capacity."

                    self.memory_manager.add_memory(
                        self.current_user_id,
                        f"Storage metric from command: {command}",
                        metric_memory,
                        memory_type="system_metric",
                        attributes={
                            "metric_type": "filesystem_usage",
                            "filesystem": filesystem,
                            "usage_percent": usage,
                            "total_size": size,
                            "timestamp": datetime.now().isoformat()
                        },
                        pre_sharpen=self.memory_manager.sharpening_enabled
                    )
                    total_memories += 1

        if total_memories > 0:
            print(f"[Memory] Added {total_memories} tabular data memories from command: '{command}'")

        return total_memories

    def _extract_row_data(self, row: str, columns: list, header_row: str) -> dict:
        """
        Extract structured data from a row based on column positions in the header.

        Args:
            row: Row string
            columns: List of column names
            header_row: Original header row for position reference

        Returns:
            Dictionary mapping column names to values
        """
        result = {}

        # Try to split by whitespace alignment first
        # This works for well-formatted tables with aligned columns
        row_parts = row.split()

        # If we have exactly the same number of parts as columns, direct mapping
        if len(row_parts) == len(columns):
            for i, col in enumerate(columns):
                result[col] = row_parts[i]
            return result

        # For more complex alignment, try position-based extraction
        try:
            # Find column boundaries based on header
            col_positions = []
            current_pos = 0
            for col in columns:
                col_pos = header_row.find(col, current_pos)
                if col_pos == -1:
                    break
                col_positions.append(col_pos)
                current_pos = col_pos + len(col)

            # Add end position
            col_positions.append(len(header_row) + 1)

            # Extract column values using positions
            for i in range(len(col_positions) - 1):
                start = col_positions[i]
                end = col_positions[i+1]

                # Make sure row is long enough
                if start < len(row):
                    value = row[start:min(end, len(row))].strip()
                    result[columns[i]] = value
        except Exception:
            # Fall back to simple splitting if position-based fails
            if not result and len(row_parts) > 0:
                # Just assign values to columns until we run out of either
                for i in range(min(len(columns), len(row_parts))):
                    result[columns[i]] = row_parts[i]

                # If we have more row parts than columns, append remaining to the last column
                if len(row_parts) > len(columns) and len(columns) > 0:
                    last_col = columns[-1]
                    result[last_col] = " ".join([result.get(last_col, "")] + row_parts[len(columns):])

        return result

    def _index_general_command_output(self, command: str, output: str, error: str, output_file: str, command_type: str) -> int:
        """
        Index general (non-tabular) command output with improved chunking.

        Args:
            command: The shell command
            output: Command output text
            error: Command error text
            output_file: Path to output file
            command_type: Type of command

        Returns:
            Number of memories added
        """
        total_memories = 0

        # Create a descriptive memory entry for the complete output
        memory_text = f"Shell command '{command}' was executed and returned: "

        # Handle large outputs with better chunking
        if output:
            if len(output) > 500:
                preview = output[:500].strip() + "..."
                memory_text += f"\nOutput preview: {preview}"
                if output_file:
                    memory_text += f"\nFull output saved to: {output_file}"

                # Create chunks for better retrieval
                chunks = self._chunk_command_output(output, command_type)

                # Add each chunk as a separate memory
                for i, chunk in enumerate(chunks):
                    chunk_memory = f"Part {i+1} of output from command '{command}':\n{chunk}"

                    memories_added = self.memory_manager.add_memory(
                        self.current_user_id,
                        f"Command output chunk {i+1} from: {command}",
                        chunk_memory,
                        memory_type=f"{command_type}_chunk",
                        attributes={
                            "command": command,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "timestamp": datetime.now().isoformat()
                        },
                        pre_sharpen=self.memory_manager.sharpening_enabled
                    )
                    total_memories += memories_added
            else:
                memory_text += f"\nOutput: {output}"

        # Add error if present
        if error:
            memory_text += f"\nError: {error}"

            # Add error as separate memory for better retrieval
            error_memory = f"Error from command '{command}':\n{error}"

            memories_added = self.memory_manager.add_memory(
                self.current_user_id,
                f"Command error from: {command}",
                error_memory,
                memory_type="command_error",
                attributes={
                    "command": command,
                    "command_type": command_type,
                    "timestamp": datetime.now().isoformat()
                },
                pre_sharpen=self.memory_manager.sharpening_enabled
            )
            total_memories += memories_added

        # Add the complete memory
        memories_added = self.memory_manager.add_memory(
            self.current_user_id,
            f"Information from {command_type}: {command}",
            memory_text,
            memory_type=command_type,
            attributes={
                "command": command,
                "has_error": error is not None,
                "output_file": output_file,
                "output_length": len(output) if output else 0,
                "timestamp": datetime.now().isoformat()
            },
            pre_sharpen=self.memory_manager.sharpening_enabled
        )
        total_memories += memories_added

        if total_memories > 0:
            print(f"[Memory] Added {total_memories} memories from command: '{command}'")

        return total_memories

    def _chunk_command_output(self, output: str, command_type: str) -> list:
        """
        Chunk large command outputs intelligently based on content structure.

        Args:
            output: Command output text
            command_type: Type of command

        Returns:
            List of output chunks
        """
        # For large outputs, we want semantically meaningful chunks
        lines = output.split('\n')

        # Different chunking strategies based on command type
        if command_type == "file_listing":
            # Group by directories or file types
            return self._chunk_by_sections(lines, chunk_size=20)
        elif command_type == "search":
            # Group by match patterns
            return self._chunk_by_sections(lines, chunk_size=15, empty_line_delimiter=True)
        elif command_type == "file_content":
            # Group by paragraphs or sections
            return self._chunk_by_paragraphs(lines)
        else:
            # Default chunking by fixed size with overlap
            return self._chunk_by_size(lines, chunk_size=25, overlap=5)

    def _chunk_by_sections(self, lines: list, chunk_size: int = 20, empty_line_delimiter: bool = False) -> list:
        """
        Chunk lines by logical sections.

        Args:
            lines: List of text lines
            chunk_size: Maximum lines per chunk
            empty_line_delimiter: Whether empty lines indicate section boundaries

        Returns:
            List of chunked text sections
        """
        chunks = []
        current_chunk = []

        for line in lines:
            # Check for section delimiter
            if empty_line_delimiter and not line.strip():
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                continue

            current_chunk.append(line)

            # Check size limit
            if len(current_chunk) >= chunk_size:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []

        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def _chunk_by_paragraphs(self, lines: list, max_chunk_size: int = 30) -> list:
        """
        Chunk lines by paragraphs (empty line separated).

        Args:
            lines: List of text lines
            max_chunk_size: Maximum lines per chunk

        Returns:
            List of chunked paragraphs
        """
        chunks = []
        current_chunk = []

        for line in lines:
            # If empty line and we have content, it's a paragraph boundary
            if not line.strip() and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                continue

            current_chunk.append(line)

            # Check size limit
            if len(current_chunk) >= max_chunk_size:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []

        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def _chunk_by_size(self, lines: list, chunk_size: int = 25, overlap: int = 5) -> list:
        """
        Chunk lines by fixed size with overlap.

        Args:
            lines: List of text lines
            chunk_size: Number of lines per chunk
            overlap: Number of overlapping lines between chunks

        Returns:
            List of text chunks
        """
        chunks = []

        if not lines:
            return chunks

        for i in range(0, len(lines), chunk_size - overlap):
            chunk = lines[i:i + chunk_size]
            if chunk:
                chunks.append('\n'.join(chunk))

        return chunks

    def parse_procedural_content(self, content: str, command: str = "") -> List[Dict]:
        """
        Parse content specifically for procedural knowledge (steps, mappings, rules).

        Args:
            content: Text content containing procedural knowledge
            command: Original command that generated the content (for context)

        Returns:
            List of structured procedural knowledge entries
        """
        procedures = []

        # Look for mapping tables (like English to Katakana)
        mapping_pattern = r'(\w+)\s*(?:→|->)\s*([^\s(]+)(?:\s*\(([^)]+)\))?'
        mappings = re.findall(mapping_pattern, content)

        if mappings:
            # Group related mappings together
            mapping_dict = {}
            for source, target, note in mappings:
                source = source.strip().lower()
                category = extract_mapping_category(source, content)

                if category not in mapping_dict:
                    mapping_dict[category] = []

                mapping_dict[category].append({
                    "input": source,
                    "output": target.strip(),
                    "note": note.strip() if note else ""
                })

            # Create structured procedural entries
            for category, items in mapping_dict.items():
                procedures.append({
                    "type": "mapping_table",
                    "category": category,
                    "entries": items,
                    "source_command": command,
                    "example_pairs": extract_example_pairs(content)
                })

        # Look for numbered/bulleted step-by-step instructions
        steps_pattern = r'(?:^|\n)(?:\d+\.|\*|\-)\s+(.+?)(?=(?:\n(?:\d+\.|\*|\-)\s+)|$)'
        steps = re.findall(steps_pattern, content)

        if steps:
            procedures.append({
                "type": "step_sequence",
                "steps": steps,
                "source_command": command
            })

        return procedures

    def retrieve_procedural_knowledge(self, query: str, procedure_type: str = None) -> List[Dict]:
        """
        Retrieve procedural knowledge relevant to a query, optimized for application.

        Args:
            query: The user query
            procedure_type: Optional filter for specific procedure types

        Returns:
            List of relevant procedural knowledge entries
        """
        # First use vector search to find potentially relevant memories
        query_embedding = self.memory_manager.generate_embedding(query)
        store = self.memory_manager._get_user_store(self.current_user_id)

        if not store or not hasattr(store, 'index') or store.index is None:
            return [];

        results = store.search(
            query_embedding, top_k=15, min_similarity=0.2
        )

        # Extract procedure-related content
        procedures = []
        for result in results:
            # Check if this memory contains procedural content markers
            has_markers = any(marker in result['text'].lower() for marker in
                            ["step", "→", "->", "mapping", "instruction", "how to"])

            if has_markers:
                # Parse the procedural content
                parsed = self.parse_procedural_content(result['text'],
                                                    result.get('metadata', {}).get('source_query', ''))

                # Add to our collection with the similarity score
                for proc in parsed:
                    proc['relevance'] = result['similarity']
                    procedures.append(proc)

        # Filter by type if specified
        if procedure_type:
            procedures = [p for p in procedures if p.get('type') == procedure_type]

        # Sort by relevance and return
        return sorted(procedures, key=lambda x: x.get('relevance', 0), reverse=True)

    def create_procedural_prompt_section(self, procedures: List[Dict]) -> str:
        """
        Create a structured prompt section from procedural knowledge.

        Args:
            procedures: List of procedural knowledge entries

        Returns:
            Formatted prompt section for procedural knowledge
        """
        if not procedures:
            return ""

        sections = []

        # Handle mapping tables
        mapping_tables = [p for p in procedures if p.get('type') == 'mapping_table']
        if mapping_tables:
            sections.append("PROCEDURAL KNOWLEDGE - MAPPING TABLES:")

            for i, table in enumerate(mapping_tables[:3]):  # Limit to top 3 for context window
                entries = table.get('entries', [])
                category = table.get('category', 'General').title()

                sections.append(f"\n{i+1}. {category} Mappings:")

                # Format entries in a clean, structured way
                formatted_entries = []
                for entry in entries[:15]:  # Limit entries for context
                    input_val = entry.get('input', '')
                    output_val = entry.get('output', '')
                    note = f" ({entry.get('note')})" if entry.get('note') else ""

                    formatted_entries.append(f"  • {input_val} → {output_val}{note}")

                sections.append("\n".join(formatted_entries))

                # Add examples if available
                examples = table.get('example_pairs', [])
                if examples:
                    sections.append("\n  Examples:")
                    for ex_input, ex_output in examples[:3]:
                        sections.append(f"  • \"{ex_input}\" → \"{ex_output}\"")

        # Handle step sequences
        step_sequences = [p for p in procedures if p.get('type') == 'step_sequence']
        if step_sequences:
            sections.append("\nPROCEDURAL KNOWLEDGE - STEP SEQUENCES:")

            for i, sequence in enumerate(step_sequences[:2]):
                steps = sequence.get('steps', [])
                sections.append(f"\n{i+1}. Step Sequence:")

                for j, step in enumerate(steps):
                    sections.append(f"  {j+1}. {step}")

        return "\n".join(sections)

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
        # Get user's vector store
        store = self.memory_manager._get_user_store(self.current_user_id)

        # Generate embedding for query
        query_embedding = self.memory_manager.generate_embedding(query)

        # First pass: Get candidate memories with standard search
        # Get more results than needed for re-ranking
        # candidate_results = store.search(
        #     query_embedding,
        #     top_k=top_k*3,
        #     min_similarity=0.2
        # )

        candidate_results = store.enhanced_fractal_search(
            query_embedding,
            top_k=10,
            multi_level_search=True,
            level_weights=[1.0, 0.7, 0.5, 0.3]
        )

        # Process based on command context if provided
        if command_context:
            # Extract command type from context
            command_type = extract_command_type(command_context)

            # Get tabular data if requested
            if include_tabular and is_tabular_command(command_context):
                tabular_results = self._retrieve_tabular_data(query, command_context, top_k)
                # Combine with standard results
                candidate_results.extend(tabular_results)

        # Re-rank results with enhanced scoring
        scored_results = self._score_command_memories(
            candidate_results,
            query,
            command_context,
            recency_weight
        )

        # Take top results after scoring
        top_results = scored_results[:top_k]

        # Format the results
        return self._format_command_memories(top_results, query, command_context)

    def _retrieve_tabular_data(self, query: str, command_context: str, top_k: int) -> List[Dict]:
        """
        Specialized retrieval for tabular data with column/row awareness.

        Args:
            query: User query
            command_context: Command context
            top_k: Maximum results to return

        Returns:
            List of relevant tabular data memories
        """
        store = self.memory_manager._get_user_store(self.current_user_id)
        results = []

        # Try to extract column/row references from the query
        column_references = extract_column_references(query)
        row_references = extract_row_references(query)

        # If we have specific column/row references, prioritize those memories
        if column_references or row_references:
            # Get all memories with tabular metadata
            all_memories = store.documents
            all_metadata = store.metadata

            # Find memories that match the column/row references
            for i, (memory, metadata) in enumerate(zip(all_memories, all_metadata)):
                # Skip non-tabular memories
                if not metadata.get('is_tabular', False):
                    continue

                # Check command match if context provided
                if command_context and 'command' in metadata:
                    if not commands_match(command_context, metadata['command']):
                        continue

                # Check column references
                columns_match = False
                if column_references and 'values' in metadata:
                    values = metadata.get('values', {})
                    for col in column_references:
                        if col in values:
                            columns_match = True
                            break

                # Check row references
                row_match = False
                if row_references and 'row_index' in metadata:
                    row_idx = metadata.get('row_index', -1)
                    if any(ref == row_idx for ref in row_references):
                        row_match = True

                # Add if either columns or rows match
                if columns_match or row_match:
                    # Calculate a base similarity (we'll refine later)
                    results.append({
                        'text': memory,
                        'similarity': 0.7,  # Start with high base similarity for matches
                        'metadata': metadata,
                        'index': i
                    })

        # If no specialized results or not enough, supplement with semantic search
        if len(results) < top_k:
            # Generate specialized tabular query embedding
            tabular_query = f"tabular data from command {command_context}: {query}"
            tabular_embedding = self.memory_manager.generate_embedding(tabular_query)

            # Search for tabular data memories
            # semantic_results = store.search(
            #     tabular_embedding,
            #     top_k=top_k*2,
            #     min_similarity=0.2
            # )
            semantic_results = store.enhanced_fractal_search(
                tabular_embedding,
                top_k=10,
                multi_level_search=True,
                level_weights=[1.0, 0.7, 0.5, 0.3]
            )

            # Filter for tabular data and add to results
            for result in semantic_results:
                metadata = result.get('metadata', {})
                if metadata.get('is_tabular', False) or metadata.get('memory_type', '') in ['tabular_data', 'tabular_row', 'system_metrics']:
                    # Check if already in results
                    if not any(r.get('index', -1) == result.get('index', -2) for r in results):
                        results.append(result)

        return results

    def _score_command_memories(self, results: List[Dict], query: str, command_context: Optional[str] = None, recency_weight: float = 0.3) -> List[Dict]:
        """
        Enhanced scoring for command memories with recency bias.

        Args:
            results: Search results to score
            query: User query
            command_context: Optional command context
            recency_weight: Weight for recency (0-1)

        Returns:
            List of scored results
        """
        now = datetime.now()

        for result in results:
            # Start with base similarity
            base_similarity = result.get('similarity', 0)
            result['original_similarity'] = base_similarity

            # Initialize components
            semantic_score = base_similarity
            recency_score = 0.5  # Default mid-value
            command_match_score = 0.5  # Default mid-value
            type_match_score = 0.5  # Default mid-value
            metadata_match_score = 0.0  # Default zero

            # Extract metadata
            metadata = result.get('metadata', {})

            # Calculate recency score
            if 'timestamp' in metadata:
                try:
                    timestamp = datetime.fromisoformat(metadata['timestamp'])
                    # Calculate time difference in hours
                    time_diff = (now - timestamp).total_seconds() / 3600

                    # Exponential decay formula for recency
                    # 1.0 for very recent, decaying over time
                    recency_score = math.exp(-0.01 * time_diff)
                except (ValueError, TypeError):
                    # Default mid-value if timestamp parsing fails
                    recency_score = 0.5

            # Calculate command match score
            if command_context and 'command' in metadata:
                stored_command = metadata['command']
                if commands_match(command_context, stored_command):
                    command_match_score = 1.0
                else:
                    command_match_score = 0.2

            # Calculate type match score
            if command_context:
                context_type = self._extract_command_type_from_context(command_context)
                memory_type = metadata.get('command_type', metadata.get('memory_type', ''))

                if context_type == memory_type:
                    type_match_score = 1.0
                elif context_type in memory_type or memory_type in context_type:
                    type_match_score = 0.8
                else:
                    type_match_score = 0.3

            # Check for metadata matches with query terms
            query_terms = set(query.lower().split())
            for key, value in metadata.items():
                # Skip non-relevant metadata fields
                if key in ['timestamp', 'memory_type', 'command_type']:
                    continue

                # Convert value to string if needed
                if not isinstance(value, str):
                    value = str(value)

                # Check for query term matches in metadata
                value_lower = value.lower()
                for term in query_terms:
                    if term in value_lower:
                        metadata_match_score += 0.1  # Increment score for each match

            # Cap metadata match score
            metadata_match_score = min(metadata_match_score, 1.0)

            # Special boost for tabular data when relevant
            is_tabular = metadata.get('is_tabular', False)
            if is_tabular and self._contains_tabular_query_terms(query):
                tabular_boost = 0.2
            else:
                tabular_boost = 0.0

            # Calculate final score: weighted combination of components
            final_score = (
                (1 - recency_weight) * semantic_score +  # Semantic similarity
                recency_weight * recency_score +         # Recency bias
                0.2 * command_match_score +              # Command match bonus
                0.15 * type_match_score +                # Type match bonus
                0.1 * metadata_match_score +             # Metadata match bonus
                tabular_boost                            # Tabular data boost
            )

            # Ensure score is in valid range
            result['final_score'] = max(0.0, min(1.0, final_score))

            # Store component scores for debugging/analysis
            result['score_components'] = {
                'semantic': semantic_score,
                'recency': recency_score,
                'command_match': command_match_score,
                'type_match': type_match_score,
                'metadata_match': metadata_match_score,
                'tabular_boost': tabular_boost
            }

        # Sort by final score
        sorted_results = sorted(results, key=lambda x: x.get('final_score', 0), reverse=True)

        return sorted_results

    def _contains_tabular_query_terms(self, query: str) -> bool:
        """
        Check if query contains terms related to tabular data.

        Args:
            query: User query

        Returns:
            Boolean indicating if query relates to tabular data
        """
        tabular_terms = [
            'table', 'row', 'column', 'cell', 'header', 'value',
            'field', 'record', 'entry', 'data', 'metric', 'percentage',
            'filesystem', 'memory', 'storage', 'cpu', 'disk'
        ]

        query_lower = query.lower()
        return any(term in query_lower for term in tabular_terms)

    def _format_command_memories(self, results: List[Dict], query: str, command_context: Optional[str] = None) -> str:
        """
        Format command memories for inclusion in the prompt.

        Args:
            results: Scored search results
            query: User query
            command_context: Optional command context

        Returns:
            Formatted string of memories
        """
        if not results:
            return ""

        # Detect if query is about tabular data
        is_tabular_query = self._contains_tabular_query_terms(query)

        # Group results into categories
        tabular_results = []
        metric_results = []
        general_results = []

        for result in results:
            metadata = result.get('metadata', {})

            # Categorize by result type
            if metadata.get('is_tabular', False) or metadata.get('memory_type', '') in ['tabular_data', 'tabular_row']:
                tabular_results.append(result)
            elif metadata.get('memory_type', '') == 'system_metric':
                metric_results.append(result)
            else:
                general_results.append(result)

        # Format the output with clear sections
        memory_text = ""

        # Format tabular data first if relevant
        if tabular_results and is_tabular_query:
            memory_text += "TABULAR DATA:\n"

            # Check if we have both table and row memories
            table_memories = [r for r in tabular_results if metadata_refers_to_full_table(r.get('metadata', {}))]
            row_memories = [r for r in tabular_results if not metadata_refers_to_full_table(r.get('metadata', {}))]

            # If we have both, organize by table then rows
            if table_memories and row_memories:
                # First add table overview
                for i, result in enumerate(table_memories[:1]):  # Just add the first table for context
                    memory_text += f"- {result['text']}\n\n"

                # Then add specific rows with clear references
                memory_text += "SPECIFIC ROWS:\n"
                for i, result in enumerate(row_memories[:5]):  # Limit to 5 rows
                    # Extract row index if available
                    metadata = result.get('metadata', {})
                    row_idx = metadata.get('row_index', i)

                    # Format with explicit row reference
                    memory_text += f"- Row {row_idx + 1}: {result['text']}\n"
            else:
                # Just add all tabular results
                for i, result in enumerate(tabular_results[:5]):  # Limit to 5 entries
                    memory_text += f"- {result['text']}\n"

            memory_text += "\n"

        # Add system metrics if any
        if metric_results:
            memory_text += "SYSTEM METRICS:\n"
            for i, result in enumerate(metric_results[:3]):  # Limit to 3 metrics
                memory_text += f"- {result['text']}\n"
            memory_text += "\n"

        # Add general command results
        if general_results or (not is_tabular_query and tabular_results):
            memory_text += "COMMAND OUTPUT INFORMATION:\n"

            # Add remaining results
            remaining = general_results
            if not is_tabular_query:
                remaining += tabular_results

            for i, result in enumerate(remaining[:5]):  # Limit to 5 general results
                memory_text += f"- {result['text']}\n"

        return memory_text

    def metadata_refers_to_full_table(metadata: Dict) -> bool:
        """
        Check if metadata refers to a full table or just a row.

        Args:
            metadata: Result metadata

        Returns:
            Boolean indicating if metadata refers to full table
        """
        # Check memory type
        if metadata.get('memory_type', '') == 'tabular_data':
            return True

        # Check if it has column and row counts (table overview)
        if 'column_count' in metadata and 'row_count' in metadata:
            return True

        # Check if it has row index (specific row)
        if 'row_index' in metadata:
            return False

        # Check memory text for table indicators
        memory_text = metadata.get('memory_text', '')
        if isinstance(memory_text, str) and any(indicator in memory_text.lower() for indicator in ['table from', 'tabular data', 'command output']):
            return True

        return False

    def cleanup(self):
        """Release all resources properly."""

        try:
            # self.uninstall_global_kb_handler()
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
                    # Get the user's vector store
                    store = self.memory_manager._get_user_store(self.current_user_id)
                    # Now call consolidate_memories on the store object
                    store.consolidate_memories()
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

    def load_knowledge_domain(self, domain_id: str) -> bool:
        """Load a knowledge domain and set it as active."""
        if not self.knowledge_system_enabled:
            print("Knowledge management system not available")
            return False

        domain = self.memory_manager.get_domain(domain_id)
        if domain:
            self.current_domain_id = domain_id
            return True
        return False

    def create_knowledge_domain(self, name: str, description: str = "") -> Optional[str]:
        """Create a new knowledge domain."""
        if not self.knowledge_system_enabled:
            print("Knowledge management system not available")
            return None

        domain_id = self.memory_manager.create_domain(name, description)
        if domain_id:
            # Set as current domain
            self.current_domain_id = domain_id
        return domain_id

    def extract_knowledge_from_text(self, text: str, source: str = None) -> List[Dict[str, Any]]:
        """Extract knowledge from text."""
        if not self.knowledge_system_enabled:
            print("Knowledge management system not available")
            return []

        # Check if we have a knowledge extractor
        if not hasattr(self.web_enhancer, 'knowledge_extractor') or not self.web_enhancer.knowledge_extractor:
            try:
                from knowledge_extractor import KnowledgeExtractor
                self.web_enhancer.knowledge_extractor = KnowledgeExtractor(
                    embedding_function=self.memory_manager.generate_embedding,
                    enable_fractal_validation=self.memory_manager.fractal_enabled
                )
            except ImportError:
                print("KnowledgeExtractor not available")
                return []

        # Extract knowledge
        domain = self.current_domain_id or "general"
        return self.web_enhancer.knowledge_extractor.extract_knowledge(text, source, domain)

    def add_knowledge_to_domain(self, knowledge_items: List[Dict[str, Any]], domain_id: str = None) -> int:
        """Add knowledge items to a domain."""
        if not self.knowledge_system_enabled:
            print("Knowledge management system not available")
            return 0

        # Use current domain if not specified
        domain_id = domain_id or self.current_domain_id
        if not domain_id:
            print("No domain specified and no current domain")
            return 0

        # Get domain
        domain = self.memory_manager.get_domain(domain_id)
        if not domain:
            print(f"Domain {domain_id} not found")
            return 0

        # Add knowledge
        return domain.add_knowledge(knowledge_items)

    def _initialize_knowledge_integration(self):
        """Connect all knowledge components together."""
        if not self.knowledge_system_enabled:
            return

        # Connect web enhancer to knowledge system
        if self.web_enhancer:
            # Give web enhancer access to current domain
            self.web_enhancer.get_current_domain = lambda: (
                self.memory_manager.get_domain(self.current_domain_id)
                if self.current_domain_id else None
            )

            # Give web enhancer a reference to the knowledge validator
            if hasattr(self.memory_manager, 'knowledge_validator'):
                self.web_enhancer.knowledge_validator = self.memory_manager.knowledge_validator

        # Set up knowledge extraction from web results
        self._ensure_knowledge_extractor()

        print("Knowledge integration initialized")

    def _ensure_knowledge_extractor(self):
        """Ensure a knowledge extractor is available."""
        if not hasattr(self, 'knowledge_extractor'):
            try:
                from knowledge_extractor import KnowledgeExtractor
                self.knowledge_extractor = KnowledgeExtractor(
                    embedding_function=self.memory_manager.generate_embedding,
                    enable_fractal_validation=self.memory_manager.fractal_enabled
                )
            except ImportError:
                print("KnowledgeExtractor not available")
                self.knowledge_extractor = None

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
        max_fractal_levels=3
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

                store = chat.memory_manager._get_user_store(chat.current_user_id)

                # Show final stats
                stats = store.get_stats()
                print(f"Total memories saved this session: {stats['active_documents']}")
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
                    store = chat.memory_manager._get_user_store(chat.current_user_id)
                    store_stats = store.get_stats() if store else {"total_documents": 0, "active_documents": 0}

                    # Get memory manager state
                    auto_memorize = chat.memory_manager.auto_memorize if hasattr(chat.memory_manager, 'auto_memorize') else False

                    print("\nMemory System Statistics:")
                    print(f"Total memories: {store_stats.get('total_documents', 0)}")
                    print(f"Active memories: {store_stats.get('active_documents', 0)}")
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
                store = chat.memory_manager._get_user_store(chat.current_user_id)
                if hasattr(store, 'print_fractal_embedding_diagnostics'):
                    store.print_fractal_embedding_diagnostics()
                else:
                    print("Fractal diagnostics not available.")
                continue

            elif user_input.lower() == '!visualize-fractal':
                store = chat.memory_manager._get_user_store(chat.current_user_id)
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
                confidence_data = chat.confidence_metrics.get_metrics(apply_sharpening=chat.memory_manager.sharpening_enabled)

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

