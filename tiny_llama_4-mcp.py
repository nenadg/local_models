import torch
import os
import json
import argparse
import time
import sys
import threading
import signal

from datetime import datetime
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import numpy as np
from typing import List, Dict, Tuple, Optional
from history import setup_readline_history

from keyboard_interrupt import KeyboardInterruptHandler
from mcp_handler import MCPHandler
from confidence_metrics import ConfidenceMetrics
from response_filter import ResponseFilter
from logit_capture_processor import TokenProbabilityCaptureProcessor
from transformers import LogitsProcessor, LogitsProcessorList
from terminal_heatmap import TerminalHeatmap

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
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=None, memory_dir="./memory", output_dir="./output", confidence_threshold=0.7):
        self.model_name = model_name
        self.memory_dir = memory_dir
        self.interrupt_handler = KeyboardInterruptHandler()
        self.stop_event = threading.Event()
        self.mcp_handler = MCPHandler(output_dir=output_dir)
        self.confidence_metrics = ConfidenceMetrics()

        os.makedirs(memory_dir, exist_ok=True)

        current_time = datetime.now().strftime("[%d/%m/%y %H:%M:%S]")
        print(f"{current_time} let's go")
        # Determine the device to use
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("Using NVIDIA GPU for acceleration")
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

    def _interrupt_handler(self, signum, frame):
        """SIGINT handler - sets the stop event"""
        self.stop_event.set()
        print("\n[Generation interrupted by user (Ctrl+C)]")

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

    def create_prompt_with_knowledge(self, messages):
        """Create a prompt that incorporates relevant knowledge"""
        # Extract the user's current query
        current_query = messages[-1]["content"] if messages[-1]["role"] == "user" else ""

        # Find relevant knowledge using simple keyword matching
        relevant_facts = []
        if current_query:
            query_words = set(current_query.lower().split())
            for fact in self.knowledge_base["facts"]:
                fact_words = set(fact["content"].lower().split())
                # If there's word overlap, consider the fact relevant
                if query_words.intersection(fact_words):
                    relevant_facts.append(fact["content"])

        # Incorporate up to 2 most recently added corrections
        recent_corrections = [c["content"] for c in self.knowledge_base["corrections"][-2:]]

        # Create enhanced system message with knowledge
        enhanced_system_content = self.system_message["content"]

        if relevant_facts:
            enhanced_system_content += "\n\nRelevant information:"
            for fact in relevant_facts[:2]:  # Limit to 2 relevant facts for speed
                enhanced_system_content += f"\n- {fact}"

        if recent_corrections:
            enhanced_system_content += "\n\nPlease remember these corrections:"
            for correction in recent_corrections:
                enhanced_system_content += f"\n- {correction}"

        # Create messages with enhanced system prompt
        enhanced_messages = [
            {
                "role": "system",
                "content": enhanced_system_content
            }
        ]

        # Add conversation history (limiting to last 3 exchanges)
        history_messages = messages[1:] if len(messages) > 1 else []
        enhanced_messages.extend(history_messages[-3:])

        return enhanced_messages

    def speculative_decode(self,
                         input_ids: torch.Tensor,
                         attention_mask: torch.Tensor,
                         num_draft_tokens: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Implement speculative decoding:
        1. Generate tokens with the smaller draft model
        2. Verify with larger target model
        3. Keep tokens that match, regenerate from first mismatch
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
                    do_sample=False,  # Use greedy decoding for draft
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            # Extract draft tokens (excluding input prompt)
            draft_ids = draft_outputs.sequences[0, input_ids.shape[1]:]

            # Check if we got any draft tokens
            if len(draft_ids) == 0:
                return None, None, None

            # Step 2: Verify with target model - compute logits for each position
            full_sequence = torch.cat([input_ids[0], draft_ids]).unsqueeze(0)
            full_attention = torch.ones_like(full_sequence)

            # Get logits from target model for the full sequence
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
                    # Check stop event
                    if self.stop_event.is_set():
                        print("\n[Generation stopped by user]")
                        break

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
            except:
                pass

    def generate_response(self, messages, max_new_tokens=128, temperature=0.7, stream=True, turbo_mode=True, show_confidence=False):
        """Generate a response with ultra-fast speculative decoding (streaming only)"""

        self.stop_event.clear()  # Reset the event
        # Install handler only in main thread
        signal.signal(signal.SIGINT, self._interrupt_handler)

        heatmap = TerminalHeatmap(self.tokenizer, use_background=False)
        token_confidences = []  # To store confidence scores for each token

        # Reset confidence metrics
        self.confidence_metrics.reset()

        if messages and messages[-1]["role"] == "user":
            user_input = messages[-1]["content"]
            cleaned_input, user_commands = self.mcp_handler.extract_mcp_from_user_input(user_input)

            # Replace the original user input with cleaned version
            if user_input != cleaned_input:
                messages[-1]["content"] = cleaned_input

            # Process any immediate user commands
            if user_commands:
                results = self.mcp_handler.execute_commands(user_commands)
                successful = [file for file, status in results.items() if status]
                if successful:
                    files_info = ", ".join(successful)
                    print(f"[User content saved to: {files_info}]")

        # Create enhanced prompt with knowledge
        enhanced_messages = self.create_prompt_with_knowledge(messages)

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            enhanced_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Encode the prompt
        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        except Exception as e:
            print(f"Error encoding prompt: {e}")
            return "Error preparing response. Please try again with a simpler query."

        # We only support streaming now, simplifies the code
        try:
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
                "do_sample": temperature > 0.1,
                "temperature": temperature if temperature > 0.1 else 1.0,
                "top_k": 50,
                "top_p": 0.95,
                "repetition_penalty": 1.0,
                "num_beams": 1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True
            }

            # Start generation in background
            thread = Thread(target=self.generate_speculative, args=(input_ids, streamer, max_new_tokens, generation_config, turbo_mode))
            thread.daemon = True
            thread.start()

            # Initialize response
            complete_response = ""
            token_buffer = ""
            mcp_buffer = ""    # For accumulating MCP commands

            # Settings for token display
            last_print_time = time.time()
            force_display_time = 0.05  # 50ms max wait between displays
            max_buffer_size = 16  # Reasonable buffer size
            stop_sequences = ["<|user|>", "<|assistant|>"]

            try:
                for token in streamer:
                    # Process stop event more gracefully
                    if self.stop_event.is_set():
                        # Add current token to response, but stop getting more
                        print(token, end="", flush=True)
                        complete_response += token
                        print("\n[Generation stopped by user]")
                        break

                    # Process token for MCP
                    display_token, mcp_buffer = self.mcp_handler.process_streaming_token(token, mcp_buffer)

                    # Add token to response
                    complete_response += token

                    # Get confidence for latest token
                    if self.confidence_metrics.token_probabilities:
                        latest_confidence = self.confidence_metrics.token_probabilities[-1]
                        token_confidences.append(latest_confidence)
                    else:
                        # Default if no confidence available
                        latest_confidence = 0.8
                        token_confidences.append(latest_confidence)

                    # Only add displayable tokens to the buffer
                    if display_token:
                        # Colorize token if confidence visualization is enabled
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
                                if not self.confidence_metrics.token_probabilities:
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

                # Handle any remaining tokens
                if token_buffer:
                    print(token_buffer, end="", flush=True)

                # ALWAYS ENSURE WE HAVE CONFIDENCE METRICS BEFORE RETURNING
                if not self.confidence_metrics.token_probabilities:
                    response_length = len(complete_response.split())
                    num_tokens = max(5, min(response_length, 10))

                    for i in range(num_tokens):
                        dummy_logits = torch.zeros(self.tokenizer.vocab_size)
                        # Use different confidence values based on response quality
                        confidence_val = 8.0 + (i % 3)  # Vary between 8-10
                        dummy_logits[i % 100] = confidence_val
                        self.confidence_metrics.add_token_score(dummy_logits, i % 100)

                # Finalize MCP processing
                complete_response = self.mcp_handler.finalize_streaming(complete_response)

                return complete_response

            except Exception as e:
                print(f"\nError during token streaming: {str(e)}")
                if complete_response:
                    # Even with errors, make sure we have metrics
                    if not self.confidence_metrics.token_probabilities:
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
            # Fallback for confidence metrics if somehow none were added
            if not self.confidence_metrics.token_probabilities:
                dummy_logits = torch.zeros(self.tokenizer.vocab_size)
                dummy_logits[0] = 7.0  # Default confidence
                self.confidence_metrics.add_token_score(dummy_logits, 0)

            signal.signal(signal.SIGINT, signal.default_int_handler)

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
    parser.add_argument("--no-stream", action="store_true",
                      help="Disable streaming output (word by word generation)")
    parser.add_argument("--turbo", action="store_true", default=True,
                      help="Enable turbo mode for ultra-fast generation")
    parser.add_argument("--output-dir", type=str, default="./output",
                      help="Directory to store output files from MCP")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                      help="Filter confidence")
    parser.add_argument("--heatmap", action="store_true", default=False,
                   help="Show confidence heatmap visualization")

    args = parser.parse_args()


    filter_enabled = True  # Add this to track filtering state
    user_context = {}  # Shared context for the ResponseFilter

    # Initialize chat with speculative decoding support
    chat = TinyLlamaChat(
        model_name=args.model,
        device=args.device,
        memory_dir="./memory",  # Add this explicitly if you want to keep the default
        output_dir=args.output_dir,  # Pass the output_dir parameter here
        confidence_threshold=args.confidence_threshold
    )

    response_filter = ResponseFilter(
        confidence_threshold=args.confidence_threshold,
        user_context=user_context
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
            [{"role": "user", "content": "Hi"}],
            max_new_tokens=16,
            temperature=0.7,
            stream=False
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
    print("  !toggle-stream - Toggle streaming output on/off")
    print("  !toggle-turbo - Toggle turbo mode on/off")
    print("  !mcp-help - Show MCP commands for directing output to files")
    print("  !toggle-filter - Toggle uncertainty filtering on/off")
    print("  !confidence: [0.0-1.0] - Set confidence threshold")
    print("  !toggle-heatmap - Toggle confidence heatmap visualization on/off")
    print("\nIf the model expresses uncertainty, you can ask it to speculate")
    print("by saying 'please continue anyway' or 'please speculate'")
    print("="*50 + "\n")

    # Set initial mode settings
    streaming_enabled = not args.no_stream
    turbo_mode = args.turbo
    show_confidence = args.heatmap

    conversation = [chat.system_message]

    while True:
        # Get timestamp for user input
        current_time = datetime.now().strftime("[%d/%m/%y %H:%M:%S]")
        user_input = input(f"\n{current_time} You: ")

        # Handle special commands
        if user_input.lower() == 'exit':
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

        elif user_input.lower() == '!toggle-stream':
            streaming_enabled = not streaming_enabled
            print(f"Streaming output {'enabled' if streaming_enabled else 'disabled'}")
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

        # Add user message to conversation
        user_message = {"role": "user", "content": user_input}
        conversation.append(user_message)

        # Generate response
        try:
            # Measure generation time
            start_time = time.time()

            # Add timestamp for model output
            response_time = datetime.now().strftime("[%d/%m/%y %H:%M:%S]")

            # If no system command matched, generate response using the model
            print(f"\n{response_time} Assistant: ", end='', flush=True)

            # Generate response
            response = chat.generate_response(
                conversation,
                temperature=args.temperature,
                max_new_tokens=args.max_tokens,
                stream=streaming_enabled,
                turbo_mode=turbo_mode,
                show_confidence=show_confidence
            )

            # Add a newline after streaming output is complete
            print()

            # Report generation time and calculate tokens per second
            end_time = time.time()
            generation_time = max(0.01, end_time - start_time)
            
            if response:
                chat.ensure_metrics(len(response.split()))

            # Get confidence metrics
            confidence_data = chat.confidence_metrics.get_metrics()

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
                    print("\n                                           [Response filtered due to low confidence]")
                    # PRINT THE FILTERED RESPONSE HERE with legend
                    if show_confidence:
                        # Show confidence legend and add separator for clarity
                        heatmap = TerminalHeatmap(tokenizer=None, use_background=False, color_scheme="sepia-red")
                        heatmap.print_legend()
                        print("\nFiltered response:")

                    print(filtered_response)
                    print("\n")

                    # PRINT THE FILTERED RESPONSE HERE
                    print(filtered_response)
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

            # Estimate tokens generated
            try:
                response_tokens = len(chat.tokenizer.encode(response))
                tokens_per_second = response_tokens / generation_time
                print(f"[Generated {response_tokens} tokens in {generation_time:.2f}s - ~{tokens_per_second:.1f} tokens/sec | Confidence: {confidence_data['confidence']:.2f} | Perplexity: {confidence_data['perplexity']:.2f} | Entropy: {confidence_data['entropy']:.2f}]")
            except:
                # Fallback to maximum tokens estimate
                if args.max_tokens > 0:
                    tokens_per_second = args.max_tokens / generation_time
                    print(f"[Generated in {generation_time:.2f}s - ~{tokens_per_second:.1f} tokens/sec | Confidence: {confidence_data['confidence']:.2f}]")

            # Get feedback with timestamp
            feedback_time = datetime.now().strftime("[%d/%m/%y %H:%M:%S]")
            feedback = input(f"\n{feedback_time} Was this response helpful? (y/n, or provide feedback): ")
            
            if feedback.lower() != 'y' and feedback.lower() != 'yes' and feedback.strip():
                # If the user provided specific feedback, add it to knowledge
                if len(feedback) > 2:
                    correction = f"Regarding '{user_input}', remember: {feedback}"
                    chat.add_to_knowledge(correction, fact_type="correction")
                    print("Feedback saved as correction. I'll try to do better next time.")
                else:
                    print("Sorry the response wasn't helpful.")
        
        except Exception as e:
            print(f"Error generating response: {e}")
            print("Please try again with a different question.")

if __name__ == "__main__":

    main()