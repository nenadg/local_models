"""
TinyLlama Chat with integrated memory, streaming, and MCP support.
Provides a more coherent integration of memory, confidence, and response generation.
"""

import torch
import os
import json
import argparse
import time
import sys
import threading
import re
import termios
import tty
import signal
import select
import hashlib
import traceback
import gc

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import numpy as np

from history import setup_readline_history
from resource_manager import ResourceManager
from mcp_handler import MCPHandler
from enhanced_confidence_metrics import EnhancedConfidenceMetrics, TokenProbabilityCaptureProcessor
from response_filter import ResponseFilter
from terminal_heatmap import EnhancedHeatmap
from question_classifier import QuestionClassifier
from topic_shift_detector import TopicShiftDetector

# Import our refactored memory manager
from unified_memory import MemoryManager
from mcp_prompt_completer import MCPCompleter
from token_buffer import TokenBuffer

from memory_utils import (
    extract_topics,
    classify_content,
    save_to_memory,
    format_memories_by_category
)

from web_search_integration import WebSearchIntegration, integrate_web_search

# Default system message template
DEFAULT_SYSTEM_MESSAGE = """You are a helpful and friendly assistant designed to provide accurate information and actions.

Follow these guidelines:
1. When you don't know something, explicitly say "I don't know about [topic]" or "I'm not familiar with that."
2. Never make up information. It is better to admit uncertainty than to provide incorrect information.
3. You may speculate if human explicitly asks you to "please continue anyway" or "please speculate."
4. When speculating, clearly indicate your uncertainty with phrases like "I'm not confident, but..."
5. Be helpful, informative, and breif in your responses.

You can also access to local operating system.

When you need current system information:
- Use !{command} syntax to execute commands
- For example, use !{date} to get the current date
- Commands will be executed and results shown automatically
- Always execute commands rather than just mentioning them

Available commands include:
- !{date} - Get current date and time
- !{pwd} - Show current directory
- !{ls} - List files
- !{whoami} - Show current user
- !{touch [filename]} - create a file
- !{df -h} - show the size of local partitions
"""

class MemoryEnhancedChat:
    """
    Improved chat interface with proper memory integration and streaming.
    """

    def __init__(self,
                model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                device: Optional[str] = None,
                memory_dir: str = "./memory",
                output_dir: str = "./output",
                confidence_threshold: float = 0.45,
                enable_memory: bool = True,
                enable_enhanced_embeddings: bool = True,
                do_sample: bool = True,
                top_p: float = 1.0,
                top_k: int = 50,
                system_message: Optional[str] = None,
                no_filter: bool = False):
        """
        Initialize the chat interface.

        Args:
            model_name: Name of the model to use
            device: Device to use (cuda/cpu)
            memory_dir: Directory for storing memory data
            output_dir: Directory for output files
            confidence_threshold: Threshold for confidence filtering
            enable_memory: Whether to enable memory
            enable_enhanced_embeddings: Whether to use enhanced embeddings
            do_sample: Whether to use sampling for generation
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            system_message: Custom system message
        """
        self.model_name = model_name
        self.memory_dir = memory_dir
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.enable_memory = enable_memory
        self.enable_enhanced_embeddings = enable_enhanced_embeddings
        self.do_sample = do_sample
        self.top_p = top_p
        self.top_k = top_k
        self.memory_debug = False  # Set to True for memory diagnostics
        self.no_filter = no_filter
        self.enable_web_search = True

        # Set system message
        self.system_message = system_message or DEFAULT_SYSTEM_MESSAGE

        # Set up components
        self._setup_device(device)
        self._setup_resource_manager()
        self._setup_model_and_tokenizer()
        self._setup_memory_system()
        self._setup_utility_components()
        self._setup_web_search()

        # Initialize stats and state
        self.memory_stats = {"items_added": 0, "retrievals": 0}
        self.stop_event = threading.Event()

    def _setup_device(self, device: Optional[str]):
        """Set up the device for processing."""
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
            print(f"{self.get_time()} Using GPU for acceleration")

            # Enable optimizations for CUDA
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

        # Set appropriate torch dtype
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

    def _setup_resource_manager(self):
        """Enhanced resource manager setup with device awareness."""
        self.resource_manager = ResourceManager(device=self.device)

        # Add device validation
        if self.device == "cuda" and torch.cuda.is_available():
            # Set memory fraction to avoid OOM
            torch.cuda.set_per_process_memory_fraction(0.9)

            # Enable memory efficient attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except:
                pass

    def _setup_utility_components(self):
        """Set up utility components for the chat system."""

        # Output file handler with memory integration
        self.mcp_handler = MCPHandler(
            output_dir=self.output_dir,
            allow_shell_commands=True,
            memory_manager=self.memory_manager  # Pass memory manager here
        )

        self.mcp_completer = MCPCompleter(self.output_dir)

        # Confidence metrics tracker
        self.confidence_metrics = EnhancedConfidenceMetrics()

        # Question classifier for domain-specific handling
        self.question_classifier = QuestionClassifier()

        # Initialize topic shift detector
        self.topic_detector = TopicShiftDetector(
            similarity_threshold=0.55,          # Adjust this threshold based on testing
            memory_manager=self.memory_manager  # Reuse memory manager for embeddings
        )

        # Initialize response filter
        if self.no_filter:
            # Use very lenient thresholds with no filter
            print(f"{self.get_time()} Loading response filter with minimal settings.")

            self.response_filter = ResponseFilter(
                confidence_threshold=0.05,   # Much lower
                entropy_threshold=8.0,       # Much higher
                perplexity_threshold=60.0,   # Much higher
                user_context={"model_name": self.model_name},
                question_classifier=self.question_classifier,
                window_size=3,
                use_relative_filtering=False, # Disable pattern detection
                pattern_detection_weight=0.1, # Much lower
                token_count_threshold=200     # Higher threshold


                # confidence_threshold=0.1,  # Much lower than default 0.45
                # entropy_threshold=5.0,     # Much higher than default 3.5
                # perplexity_threshold=40.0, # Much higher than default 25.0
                # user_context={"model_name": self.model_name},
                # question_classifier=self.question_classifier,
                # window_size=3,
                # use_relative_filtering=True,
                # pattern_detection_weight=0.3,  # Lower from 0.6
                # token_count_threshold=100      # Higher from 60
            )
        else:
            print(f"{self.get_time()} Loading response filter with default settings.")

            self.response_filter = ResponseFilter(
                confidence_threshold=self.confidence_threshold,
                entropy_threshold=3.5,         # Up from 2.5
                perplexity_threshold=25.0,     # Up from 15.0
                user_context={"model_name": self.model_name},
                question_classifier=self.question_classifier,
                window_size=3,
                use_relative_filtering=True,
                pattern_detection_weight=0.6,
                token_count_threshold=60       # Up from 30
            )

        self.response_filter.question_classifier = self.question_classifier

    def _setup_model_and_tokenizer(self):
        """Load the model and tokenizer with appropriate settings."""
        print(f"{self.get_time()} Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        # self.tokenizer.model_max_length = self.tokenizer.model_max_length 8192 # 2048  # Set appropriate context window, max - 2048 for tinyllama

        # Set up model loading options
        loading_options = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto" if self.device != "cpu" else None,
            "low_cpu_mem_usage": True,
        }

        # Load main model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **loading_options)
        self.model = self.resource_manager.register_model(self.model)

        # Force everything to the same device
        self.model = self.model.to(self.device)

        # Ensure all model parameters are on the same device
        for param in self.model.parameters():
            param.data = param.data.to(self.device)

        # Check if this is a Qwen model
        is_qwen = "qwen" in self.model_name.lower()

        custom_template = self.tokenizer.chat_template

        if custom_template:
            if is_qwen:
                print(f"{self.get_time()} Using Qwen to template memories.")
                # Qwen-specific template modification
                # Insert memory content before the user's message
                custom_template = custom_template.replace(
                    "{% for message in messages %}",
                    """{% for message in messages %}
                    {% if message['role'] == 'memory' %}
                    <|im_start|>assistant\n<think>\n:
                    {{ message['content'] }}
                    \n\n</think>\n\n'
                    <|im_end>
                    {% continue %}
                    {% endif %}"""
                )
            else:
                # Add handling for memory role
                custom_template = custom_template.replace(
                    "{% for message in messages %}",
                    "{% for message in messages %}{% if message['role'] == 'memory' %}{{ message['content'] }}{% endif %}"
                )
            # custom_template += "{% endif %}"

            self.tokenizer.chat_template = custom_template

        # Optimize for inference
        self.resource_manager.optimize_for_inference()

    def _setup_memory_system(self):
        """Set up the memory system with embedding capabilities."""
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            storage_path=self.memory_dir,
            embedding_dim=2048,  # Adjust based on model's hidden size (initially 384)
            enable_enhanced_embeddings=self.enable_enhanced_embeddings,
            max_enhancement_levels=4,
            auto_save=True
        )

        # Create embedding function
        self._setup_embedding_function()

        # Update embedding dimension based on model
        self._update_embedding_dimension()

    def _setup_embedding_function(self):
        # Set embedding functions in memory manager
        self.memory_manager.set_embedding_function(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

    def _setup_web_search(self):
        """Set up web search integration."""
        self.web_search = WebSearchIntegration(
            memory_manager=self.memory_manager,
            question_classifier=self.question_classifier
        )
        print(f"{self.get_time()} Web search integration initialized")

    def _update_embedding_dimension(self):
        """Update embedding dimension based on loaded model."""
        # Get model's hidden size
        if hasattr(self.model.config, 'hidden_size'):
            embedding_dim = self.model.config.hidden_size
        elif hasattr(self.model.config, 'd_model'):
            embedding_dim = self.model.config.d_model
        elif hasattr(self.model.config, 'n_embd'):
            embedding_dim = self.model.config.n_embd

        # print(f"{self.get_time()} Model config attributes: {dir(self.model.config)}")
        print(f"{self.get_time()} Detected model embedding dimension: {embedding_dim}")

        if embedding_dim is None:
            # Force detection by running a test inference
            test_input = self.tokenizer("test", return_tensors="pt").to(self.device)
            with torch.no_grad():
                test_output = self.model(**test_input, output_hidden_states=True)
                if hasattr(test_output, 'hidden_states'):
                    embedding_dim = test_output.hidden_states[-1].shape[-1]
                    print(f"Detected dimension from test inference: {embedding_dim}")

        # Update memory manager if dimension differs
        if self.memory_manager.embedding_dim != embedding_dim:
            print(f"{self.get_time()} Updating memory embedding dimension: {self.memory_manager.embedding_dim} -> {embedding_dim}")
            self.memory_manager.update_embedding_dimension(embedding_dim)


    def _check_and_perform_web_search(self, query: str) -> Dict[str, any]:
        """Check if web search is needed and perform it."""
        if not hasattr(self, 'web_search') or not self.enable_memory:
            return {'searched': False}

        # Check if search is needed
        should_search, reason = self.web_search.should_search(query)

        if should_search and self.enable_web_search:
            print(f"{self.get_time()} Performing web search: {reason}")
            return self.web_search.process_query(query)

        return {'searched': False}


    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages for Gemma 3 models.
        """
        # Check if model is Gemma based on name
        is_gemma = "gemma" in self.model_name.lower()

        if is_gemma:
            prompt = ""
            system_instruction = None
            memory_content = None

            # First pass to extract system and memory
            for message in messages:
                if message["role"] == "system":
                    system_instruction = message["content"]
                elif message["role"] == "memory":
                    memory_content = message["content"]

            # Start with system instruction if available
            if system_instruction:
                prompt += f"{system_instruction}\n\n"

            # Add memory content right before the user query
            if memory_content:
                prompt += f"<knowledge>\n{memory_content}\n</knowledge>\n\n"

            # Now add conversation turns
            for message in messages:
                if message["role"] == "user":
                    prompt += f"<start_of_turn>user\n{message['content']}<end_of_turn>\n"
                elif message["role"] == "assistant":
                    prompt += f"<start_of_turn>model\n{message['content']}<end_of_turn>\n"
                # Skip system and memory messages as they've already been handled

            # Add final prompt for generation
            prompt += "<start_of_turn>model\n"

            return prompt
        else:
            # Original format for other models
            prompt = ""

            # Extract system message (usually the first message)
            system_message = next((m["content"] for m in messages if m["role"] == "system"), None)

            # Add system message first if available
            if system_message:
                prompt += f"{system_message}\n\n"

            # Process remaining messages
            for message in messages:
                role = message["role"]
                content = message["content"]

                # Skip system message (already added)
                if role == "system":
                    continue

                # Handle memory blocks
                elif role == "memory":
                    prompt += f"{content}\n\n"

                # Handle user messages
                elif role == "user":
                    prompt += f"User: {content}\n\n"

                # Handle assistant messages
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"

            # Add final assistant prompt
            prompt += "Assistant: "

            return prompt

    def _save_command_output_to_memory(self, command: str, output: str) -> None:
        """
        Enhanced method to save command output to memory with knowledge classification.
        For use in MemoryEnhancedChat.

        Args:
            command: The command that was executed
            output: Output text from the command
        """
        if not self.enable_memory or not output:
            return

        try:
            # Classify the output content
            classification = classify_content(output, self.question_classifier)

            # For date-related commands, ensure proper categorization
            if re.search(r'date|time', command, re.IGNORECASE):
                classification['main_category'] = 'declarative'
                classification['subcategory'] = 'temporal'
                classification['confidence'] = 0.95

                # Format for clear retrieval
                formatted_content = f"Today's date is {output.strip()}."

                # Create multiple variations for better retrieval
                save_to_memory(
                    memory_manager=self.memory_manager,
                    content=formatted_content,
                    classification=classification
                )

                # Additional formats
                variations = [
                    f"The current date is {output.strip()}.",
                    f"Today is {output.strip()}.",
                    f"Right now it is {output.strip()}."
                ]

                for variation in variations:
                    save_to_memory(
                        memory_manager=self.memory_manager,
                        content=variation,
                        classification=classification
                    )

                # Log success
                print(f"{self.get_time()} Date/time command output saved with {len(variations) + 1} variations")

            # For directory listing commands
            elif re.search(r'ls|dir', command, re.IGNORECASE):
                classification['main_category'] = 'declarative'
                classification['subcategory'] = 'file_system'
                classification['confidence'] = 0.9

                # Format for retrieval
                formatted_content = f"The output of command '{command}' shows these files: {output.strip()}"

                # Save to memory
                save_to_memory(
                    memory_manager=self.memory_manager,
                    content=formatted_content,
                    classification=classification
                )

                print(f"{self.get_time()} Directory listing saved to memory")

            # For other commands, use standard processing
            else:
                # Format for retrieval
                formatted_content = f"The output of command '{command}' is: {output.strip()}"

                # Save to memory
                save_to_memory(
                    memory_manager=self.memory_manager,
                    content=formatted_content,
                    classification=classification
                )

                print(f"{self.get_time()} Command output saved to memory")

        except Exception as e:
            print(f"{self.get_time()} Error saving command output to memory: {e}")

    def get_time(self) -> str:
        """Get formatted timestamp for logging."""
        return datetime.now().strftime("[%d/%m/%y %H:%M:%S] [Main] ")

    def toggle_memory(self) -> bool:
        """Toggle memory system on/off."""
        self.enable_memory = not self.enable_memory
        return self.enable_memory

    def chat(self, messages: List[Dict[str, str]], max_new_tokens: int = 128, temperature: float = 0.7, enable_memory: Optional[bool] = None, show_confidence: bool = False) -> str:
        """
        Generate a response with memory integration and streaming.

        Args:
            messages: List of conversation messages
            max_new_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            enable_memory: Override default memory setting
            show_confidence: Whether to show confidence heatmap

        Returns:
            Generated response
        """
        # Reset for new generation
        self.stop_event.clear()
        self.confidence_metrics.reset()

        # Get the most recent user query
        user_query = messages[-1]["content"] if messages[-1]["role"] == "user" else ""

        # Process MCP commands in user query if present
        if user_query:
            cleaned_input, user_commands = self.mcp_handler.extract_mcp_from_user_input(user_query)

            # Check if we have command outputs to process
            for cmd_name, cmd_info in user_commands.items():
                if cmd_info.get("action") == "shell_command" and cmd_info.get("output"):
                    # Save command output to memory
                    self._save_command_output_to_memory(cmd_name, cmd_info.get("output", ""))

                    # Enhance the query with context
                    enhanced_query = (
                        f"{cleaned_input} (The command {cmd_name} returned: {cmd_info.get('output', '')})"
                    )
                    messages[-1]["content"] = enhanced_query

            # Replace the original user input with cleaned version
            if user_query != cleaned_input:
                messages[-1]["content"] = cleaned_input

            # Handle command-only inputs
            if cleaned_input == "_COMMAND_ONLY_":
                return "Command processed."

        # Create memory-enhanced messages if memory is enabled
        memory_enabled = self.enable_memory if enable_memory is None else enable_memory

        if memory_enabled and user_query:
            web_results = self._check_and_perform_web_search(user_query)
            if web_results.get('searched') and web_results.get('saved_to_memory', 0) > 0:
                print(f"{self.get_time()} Added {web_results['saved_to_memory']} web results to memory")
                # Add slight delay to ensure memory is indexed
                time.sleep(0.1)

            memory_enhanced_messages, retrieved_memories = self._integrate_memory(messages, user_query)
            messages = memory_enhanced_messages

            # Pass memory results to response filter if provided
            if self.response_filter and hasattr(self.response_filter, 'user_context'):
                self.response_filter.user_context['memory_details'] = retrieved_memories

                # Log memory retrieval for debugging
                if retrieved_memories:
                    max_sim = max([m.get('similarity', 0) for m in retrieved_memories], default=0)
                    print(f"{self.get_time()} Retrieved {len(retrieved_memories)} memories, max similarity: {max_sim:.3f}")

        # Setup streaming
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tokens_received = 0
        response = ""

        # Apply chat template
        is_gemma = "gemma" in self.model_name.lower()
        is_qwen = "qwen" in self.model_name.lower()

        print(f"{self.get_time()} Model used: {self.model_name.lower()}")

        if is_gemma:
            prompt = self._format_chat_prompt(messages)
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=is_qwen
            )

        if self.memory_debug:
            print(f"\n============ DEBUG PROMPT ============\n{prompt}\n====================================\n")

        # Tokenize prompt
        try:
            tokenized = self._safe_tokenize(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding=True
            )
            input_ids = tokenized['input_ids'].to(self.device)
        except Exception as e:
            print(f"{self.get_time()} Error encoding prompt: {e}")
            return "Error preparing response."

        try:
            tty.setcbreak(fd)

            # Initialize streaming components
            heatmap = EnhancedHeatmap(self.tokenizer, use_background=False, window_size=3)
            token_confidences = []

            # INTEGRATION: Initialize TokenBuffer with tokenizer - SIMPLIFIED VERSION
            token_buffer = TokenBuffer(self.tokenizer)

            # Separate display buffer for ANSI-colored output
            display_buffer = ""

            # Initialize MCP buffer
            mcp_buffer = ""

            # Set up streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                stride=16
            )

            # Get domain-specific generation config
            generation_config = self._get_domain_config(
                user_query,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

            # Start generation in a separate thread
            thread = Thread(
                target=self._generate_with_streaming,
                args=(input_ids, streamer, generation_config)
            )
            thread.daemon = True
            thread.start()

            # Settings for token display
            last_print_time = time.time()
            force_display_time = 0.05  # 50ms max wait
            max_buffer_size = 16

            # OPTIMIZATION: Precompile stop sequence detection regexes
            stop_sequences = ["<|user|>", "<|assistant|>"]
            stop_regexes = [re.compile(re.escape(seq)) for seq in stop_sequences]

            # Variables to track streaming state
            low_confidence_detected = False

            # OPTIMIZATION: Adaptive pattern detection intervals
            pattern_check_interval = 20
            next_pattern_check = 40

            # Print assistant marker
            print(f"{self.get_time()} Assistant: \n", end='', flush=True)

            # Streaming loop
            while True:
                # Check for user interrupt
                if select.select([sys.stdin], [], [], 0)[0]:
                    c = sys.stdin.read(1)
                    if c == '\x03':  # Ctrl+C
                        self.stop_event.set()
                        print(f"\n{self.get_time()} Canceled by user")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        break

                # Get next token
                token = next(iter(streamer), None)

                # End of generation
                if token is None:
                    break

                tokens_received += 1

                # INTEGRATION: Add token to TokenBuffer
                token_buffer.add_token(token)

                # Process token for MCP
                display_token, mcp_buffer = self.mcp_handler.process_streaming_token(token, mcp_buffer)

                # Get confidence for latest token
                if self.confidence_metrics.token_probabilities:
                    latest_confidence = self.confidence_metrics.token_probabilities[-1]
                    token_confidences.append(latest_confidence)
                else:
                    # Default if no confidence available
                    latest_confidence = 0.8
                    token_confidences.append(latest_confidence)


                # Only add displayable tokens to the display buffer
                if display_token:
                    if show_confidence:
                        colored_token = heatmap.colorize_streaming_token(
                            display_token, latest_confidence)
                        display_buffer += colored_token
                    else:
                        # Normal display without colorization
                        display_buffer += display_token

                # Check timing
                current_time = time.time()
                time_since_print = current_time - last_print_time

                # Print buffer when it's full or enough time has passed
                if (len(display_buffer) >= max_buffer_size or
                    time_since_print >= force_display_time) and display_buffer:
                    print(display_buffer, end="", flush=True)
                    display_buffer = ""
                    last_print_time = current_time

                # Check for stop sequences in the buffer
                for regex in stop_regexes:
                    if regex.search(display_buffer):
                        display_buffer = regex.split(display_buffer)[0]
                        if display_buffer:
                            print(display_buffer, end="", flush=True)
                            display_buffer = ""

                        # Also need to truncate the token buffer
                        full_text = token_buffer.get_text()
                        for seq in stop_sequences:
                            if seq in full_text:
                                # Need to rebuild token_buffer with truncated content
                                truncated_text = full_text.split(seq)[0]
                                token_buffer = TokenBuffer(self.tokenizer)
                                for char in truncated_text:
                                    token_buffer.add_token(char)
                                break

                # OPTIMIZATION: Periodic memory cleanup during generation
                if tokens_received % 100 == 0:
                    self.resource_manager.clear_cache()

            self.tokens_received = tokens_received
            # Handle any remaining display buffer
            if display_buffer:
                print(display_buffer, end="", flush=True)

            # Ensure confidence metrics exist
            self._ensure_confidence_metrics(len(token_buffer))

            # INTEGRATION: Get full text from TokenBuffer
            complete_response = token_buffer.get_text()

            # Finalize the response
            response = self.mcp_handler.finalize_streaming(complete_response)


            # Process MCP commands in the AI's response
            if response and hasattr(self.mcp_handler, 'execute_response_commands'):
                original_response = response
                response = self.mcp_handler.execute_response_commands(response, save_to_memory=True)

                # Log if commands were executed
                if response != original_response:
                    command_count = len(re.findall(r'!{[^}]+}', original_response))
                    print(f"\n{self.get_time()} Executed {command_count} command(s) from AI response")

            # Post-process the response
            response = self._post_process_response(response)

            if response:
                # Extract and execute any MCP commands in the response
                cleaned_response, response_commands = self.mcp_handler.extract_mcp_from_user_input(response)

                # If commands were executed, append their output to the response
                command_outputs = []
                for cmd_name, cmd_info in response_commands.items():
                    if cmd_info.get("output"):
                        command_outputs.append(f"\nCommand output for {cmd_name}:\n{cmd_info.get('output')}")

                if command_outputs:
                    response = cleaned_response + "\n" + "\n".join(command_outputs)

            # Save to memory if enabled
            if memory_enabled and user_query and response:
                print("\n")
                self._save_to_memory(user_query, response)

            return response

        except KeyboardInterrupt:
            print(f"\n{self.get_time()} Generation interrupted by user")
            return "Generation interrupted."
        except Exception as e:
            print(f"\n{self.get_time()} Error during generation: {e}")
            traceback.print_exc()
            return f"Error during generation: {str(e)}"

        finally:
            # Clean up terminal settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

            prompt_tokens = len(tokenized['input_ids'][0])
            response_tokens_count = len(token_buffer) if 'token_buffer' in locals() else 0
            total_tokens = prompt_tokens + response_tokens_count

            # Get max context window from tokenizer
            max_tokens = self.tokenizer.model_max_length

            # Display legend with context window usage
            heatmap.print_legend(current_tokens=total_tokens, max_tokens=max_tokens)

            print()  # Add a newline
            confidence_metrics = self.confidence_metrics.get_metrics()
            normalized_metrics = self.response_filter.normalize_confidence_metrics(confidence_metrics)
            should_filter, reason, details = self.response_filter.should_filter(normalized_metrics, response, user_query, tokens_received)
            # print("DETAILS", json.dumps(details))
            # Get confidence metrics
            metrics = details

            # Generate and display confidence indicator
            indicator = self.response_filter.get_confidence_indicator(metrics)
            print(indicator)

            # OPTIMIZATION: More aggressive cleanup
            self.resource_manager.clear_cache()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _integrate_memory(self, messages: List[Dict[str, str]], query: str) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """
        Enhanced method to integrate relevant memories into conversation messages.
        For use in MemoryEnhancedChat.
        """
        try:
            # Classify query
            query_classification = classify_content(query, self.question_classifier)
            main_category = query_classification.get("main_category", "unknown")
            subcategory = query_classification.get("subcategory")

            # Retrieve memories
            memories = self.memory_manager.retrieve(
                query=query,
                top_k=10,
                min_similarity=0.3,
                metadata_filter={"main_category": main_category, "subcategory": subcategory},
            )

            # Check if we're using Gemma or Qwen
            is_gemma = "gemma" in self.model_name.lower()
            is_qwen = "qwen" in self.model_name.lower()

            # Format memory content appropriately for the model type
            if is_gemma:
                memory_text = ""
                for memory in memories:
                    content = memory.get("content", "").strip()
                    similarity = memory.get("similarity", 0)
                    if similarity > 0.5:  # Only include highly relevant memories
                        memory_text += f"- {content}\n"
            elif is_qwen:
                # Wrap memory in Qwen-specific format
                # memory_text = "<|im_start|>memories\n"
                memory_text = ""
                for memory in memories:
                    content = memory.get("content", "").strip()
                    similarity = memory.get("similarity", 0)
                    if similarity > 0.5:  # Only include highly relevant memories
                        memory_text += f"- {content}\n"

                # memory_text += f"<|im_end|>"
            else:
                # Standard formatting for other models
                memory_text = format_memories_by_category(memories, main_category, subcategory)
                memory_text = "IMPORTANT: Pay careful attention to the HIGHLY RELEVANT INFORMATION section below, which directly answers the query.\n\n" + memory_text

            # Create memory-enhanced messages
            memory_enhanced_messages = messages.copy()

            # Find the right place to insert memory
            last_user_idx = next((i for i in reversed(range(len(memory_enhanced_messages)))
                                 if messages[i]["role"] == "user"), None)

            if last_user_idx is not None:
                memory_enhanced_messages.insert(last_user_idx, {
                    "role": "system",
                    "content": memory_text
                })

            # Update retrieval stats
            self.memory_stats["retrievals"] += 1

            if self.memory_debug:
                print('memories', json.dumps(memory_enhanced_messages), "\n\n\n\n\n ---------", memories)

            return memory_enhanced_messages, memories

        except Exception as e:
            print(f"{self.get_time()} Error integrating memory: {e}")
            import traceback
            traceback.print_exc()
            return messages, []

    def _save_to_memory(self, query: str, response: str) -> bool:
        """
        Enhanced method to save conversation exchange to memory with knowledge classification.
        For use in MemoryEnhancedChat.

        Args:
            query: User query
            response: Generated response

        Returns:
            Success status
        """
        try:
            # Get current metrics to check for hallucination
            current_metrics = self.confidence_metrics.get_metrics()
            normalized_metrics = self.response_filter.normalize_confidence_metrics(current_metrics)

            # Check if response was flagged as hallucination
            should_filter, reason, details = self.response_filter.should_filter(
                normalized_metrics, response, query, len(response.split())
            )

            # Check for hallucination flag
            is_hallucination = details.get('is_likely_hallucination', False)
            # Classify response
            response_classification = classify_content(response, self.question_classifier)

            # Add hallucination metadata if detected
            if is_hallucination:
                response_classification['hallucination_detected'] = True
                response_classification['hallucination_score'] = details.get('uncertainty_score', 0.5)
                response_classification['reliability_penalty'] = 0.7  # Apply 70% penalty

                # print(f"{self.get_time()} Warning: Storing potentially hallucinated response with penalty")

            # Save response
            response_result = save_to_memory(
                memory_manager=self.memory_manager,
                content=response.strip(),
                classification=response_classification,
                related_content=query.strip()
            )

            memories_added = (1 if response_result.get("saved", False) else 0)

            self.memory_stats["items_added"] += memories_added
            print(f"{self.get_time()} Added {memories_added} new memories")

            return True

        except Exception as e:
            print(f"{self.get_time()} Error saving to memory: {e}")
            return False

    def _generate_with_streaming(self, input_ids, streamer, generation_config):
        """Generate with proper device management."""
        try:
            # Ensure input_ids are on correct device
            if hasattr(input_ids, 'device') and input_ids.device.type != self.device:
                print(f"{self.get_time()} Moving input_ids from {input_ids.device} to {self.device}")
                input_ids = input_ids.to(self.device)

            # Create attention mask on same device
            attention_mask = torch.ones_like(input_ids, device=self.device)

            # Clear CUDA cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Reset confidence metrics
            if hasattr(self.confidence_metrics, 'reset'):
                self.confidence_metrics.reset()

            # Setup streaming config with device-aware logits processor
            streaming_config = {k: v for k, v in generation_config.items()
                               if k not in ['max_new_tokens', 'output_scores', 'return_dict_in_generate']}

            # Add logits processor for confidence tracking
            from transformers import LogitsProcessorList
            if 'logits_processor' not in streaming_config:
                streaming_config['logits_processor'] = LogitsProcessorList()

            metrics_processor = TokenProbabilityCaptureProcessor(self.confidence_metrics)
            streaming_config['logits_processor'].append(metrics_processor)

            # Ensure use_cache is set
            if 'use_cache' not in streaming_config:
                streaming_config['use_cache'] = True

            # Generate with proper device handling
            try:
                self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    streamer=streamer,
                    max_new_tokens=generation_config.get('max_new_tokens', 128),
                    **streaming_config
                )
            except RuntimeError as e:
                if "device" in str(e).lower():
                    print(f"{self.get_time()} Device error during generation, attempting recovery...")

                    # Force all model parameters to correct device
                    self.model = self.model.to(self.device)

                    # Retry generation
                    self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        streamer=streamer,
                        max_new_tokens=generation_config.get('max_new_tokens', 128),
                        **streaming_config
                    )
                else:
                    raise

            # Clear cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"{self.get_time()} Error in generation thread: {str(e)}")
            traceback.print_exc()
            try:
                streamer.end()
            except Exception:
                pass

            # Clean up on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _get_domain_config(self, query, max_new_tokens=128, temperature=0.7):
        """Enhanced domain configuration with better error handling."""

        # Base configuration
        config = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.0,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True
        }

        # Special handling for Gemma 3
        if "gemma" in self.model_name.lower():
            # Use more conservative temperature
            temperature = min(temperature, 0.5)

            # Add stricter constraints for Gemma 3
            config.update({
                'do_sample': True,
                'temperature': temperature,
                'top_k': 40,  # Reduced from default
                'top_p': 0.85,  # More conservative
                'repetition_penalty': 1.2,  # Increased to avoid loops
                # Add token constraints if you know the ID for non-English tokens
                # 'bad_words_ids': [[id1], [id2], ...],
            })

            return config

        # Add sampling parameters
        if self.do_sample:
            config.update({
                'do_sample': True,
                'top_k': self.top_k,
                'top_p': self.top_p,
                'temperature': temperature if temperature > 0.1 else 1.0
            })

        # Get domain settings with error handling
        try:
            settings = self.question_classifier.get_domain_settings(query)
            domain = settings['domain']

            # Apply domain-specific adjustments
            domain_configs = {
                'arithmetic': {'temperature': 0.3, 'top_p': 0.85},
                'translation': {'temperature': 0.4, 'top_p': 0.9},
                'factual': {'temperature': 0.5, 'top_p': 0.95},
                'conceptual': {'temperature': 0.7, 'top_p': 0.9},
                'procedural': {'temperature': 0.6, 'top_p': 0.9}
            }

            if domain in domain_configs and self.do_sample:
                for key, value in domain_configs[domain].items():
                    config[key] = min(config.get(key, value), value)

        except Exception as e:
            print(f"{self.get_time()} Error getting domain settings: {e}")
            traceback.print_exc()

        return config

    def _ensure_confidence_metrics(self, token_count=10):
        """
        Ensure confidence metrics exist with reasonable values.

        Args:
            token_count: Number of tokens to simulate
        """
        if (not hasattr(self.confidence_metrics, 'token_probabilities') or
            not self.confidence_metrics.token_probabilities):

            # Create dummy metrics
            for i in range(token_count):
                # Create dummy logits tensor
                dummy_logits = torch.zeros(self.tokenizer.vocab_size)

                # Vary confidence values
                position_factor = 1.0 - (i / (token_count * 2))
                confidence_base = 5.0 + (position_factor * 3.0)
                confidence_val = confidence_base + ((i % 3) * 0.5)

                # Set the logit value for a token
                token_id = i % 100
                dummy_logits[token_id] = confidence_val

                # Add the metrics
                self.confidence_metrics.add_token_score(dummy_logits, token_id)

    def _post_process_response(self, response: str) -> str:
        """
        Clean up the response after generation.

        Args:
            response: The generated response

        Returns:
            Cleaned response
        """
        if not response:
            return response

        # Remove duplicate paragraphs
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
            if paragraph:
                # Create a simplified key for comparison
                simplified = re.sub(r'[^\w\s]', '', paragraph.lower())
                if simplified and simplified not in seen_paragraphs:
                    seen_paragraphs.add(simplified)
                    unique_paragraphs.append(paragraph)
                elif not simplified:
                    unique_paragraphs.append(paragraph)
            else:
                unique_paragraphs.append(paragraph)

        cleaned = '\n'.join(unique_paragraphs)

        # Remove any debugging information
        cleaned = re.sub(r'(?:relevance|similarity)\s+(?:increased|decreased)\s+by\s+[\d\.]+%.*?', '', cleaned, flags=re.MULTILINE)

        return cleaned

    def _safe_tokenize(self, text, **kwargs):
        """Safe tokenization with proper device placement."""
        try:
            # Tokenize normally first
            result = self.tokenizer(text, **kwargs)

            # Ensure all tensors are on the correct device
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, torch.Tensor):
                        result[key] = value.to(self.device)

            return result

        except RuntimeError as e:
            if "Already borrowed" in str(e):
                print(f"{self.get_time()} Handling 'Already borrowed' error...")

                # Create fresh tokenizer and tokenize
                import copy
                temp_tokenizer = copy.deepcopy(self.tokenizer)
                result = temp_tokenizer(text, **kwargs)

                # Ensure tensors are on correct device
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, torch.Tensor):
                            result[key] = value.to(self.device)

                return result
            else:
                raise


    def validate_model_device_consistency(self):
        """Validate that all model components are on the same device."""
        devices = set()

        for name, param in self.model.named_parameters():
            devices.add(param.device)
            if len(devices) > 1:
                print(f"{self.get_time()} Warning: Model has parameters on multiple devices: {devices}")
                print(f"Parameter {name} is on {param.device}")
                break

        if len(devices) == 1:
            actual_device = list(devices)[0]
            expected_device = torch.device(self.device)
            if actual_device != expected_device:
                print(f"{self.get_time()} Warning: Model device {actual_device} != expected {expected_device}")
                return False
            else:
                print(f"{self.get_time()} Model device consistency validated: {actual_device}")
                return True

        return len(devices) == 1

    def get_multiline_input(self, prompt: str = "") -> str:
        """
        Get multiline input from the user.

        Args:
            prompt: Prompt to display

        Returns:
            User input as string
        """
        try:
            from prompt_toolkit import prompt
            from prompt_toolkit.history import FileHistory
            from prompt_toolkit.styles import Style
        except ImportError:
            print("prompt_toolkit not installed. Install with 'pip install prompt_toolkit'")
            print("Falling back to simple input method.")

            # Simple fallback
            print("(Enter your text. Press Ctrl+D or empty line to finish)")

            try:
                lines = []
                while True:
                    try:
                        line = input()
                        if not line:
                            break
                        lines.append(line)
                    except EOFError:
                        break
            except KeyboardInterrupt:
                print("\nInput cancelled.")
                return ""

            return "\n".join(lines)

        # Create history file
        os.makedirs(self.memory_dir, exist_ok=True)
        history_file = os.path.join(self.memory_dir, '.multiline_history')

        print("(Multiline input mode. Use Ctrl+D or Alt+Enter to submit, Ctrl+C to cancel)")

        try:
            style = Style.from_dict({
                '': 'fg:#0BB8E2',
                'prompt.continuation': 'fg:#ADD8E6',
            })

            # Get input with prompt_toolkit
            user_input = prompt(
                '',
                multiline=True,
                prompt_continuation=lambda width, line_number, wrap_count: '.' * width,
                history=FileHistory(history_file),
                complete_while_typing=True,
                enable_open_in_editor=True,
                style=style,
                completer=self.mcp_completer
            )

            # Trim trailing whitespace
            user_input = user_input.rstrip()

            # Confirm input
            if user_input:
                print(f"{self.get_time()} Received {len(user_input.splitlines())} lines of input.")

            return user_input

        except KeyboardInterrupt:
            print(f"\n{self.get_time()} Input cancelled.")
            return ""
        except Exception as e:
            print(f"\n{self.get_time()} Error during input: {e}")
            return ""

    def cleanup(self):
        """Release all resources properly."""
        try:
            if hasattr(self, 'model') and self.model is not None:
                self.resource_manager.unload_model(self.model)
                self.model = None

            # Final cleanup
            if hasattr(self, 'resource_manager'):
                self.resource_manager.cleanup()

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"{self.get_time()} Resources cleaned up successfully")

        except Exception as e:
            print(f"{self.get_time()} Error during cleanup: {e}")
            traceback.print_exc()
            # Attempt basic cleanup
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

def main():
    """Main application function for the chat interface."""
    parser = argparse.ArgumentParser(description="Memory-Enhanced Chat")

    # Basic parameters
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                       help="Model to use for chat")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda, cpu). If not specified, will autodetect.")
    parser.add_argument("--system-prompt", type=str,
                       default=DEFAULT_SYSTEM_MESSAGE,
                       help="System prompt to use")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for response generation")
    parser.add_argument("--max-tokens", type=int, default=128,
                       help="Maximum number of tokens to generate")

    # Feature flags
    parser.add_argument("--no-memory", action="store_true",
                       help="Disable memory features")
    parser.add_argument("--heatmap", action="store_true", default=True,
                       help="Show confidence heatmap")
    parser.add_argument("--do-sample", action="store_true", default=True,
                        help="Enable sampling-based generation")

    # Advanced settings
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="Directory for output files")
    parser.add_argument("--confidence-threshold", type=float, default=0.45,
                       help="Confidence threshold for filtering")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--no-filter", action="store_true", default=False,
                        help="Disable response filtering")

    # Parse args
    args = parser.parse_args()

    # Initialize user_context for ResponseFilter
    user_context = {}

    # Initialize chat
    chat = MemoryEnhancedChat(
        model_name=args.model,
        device=args.device,
        memory_dir="./memory",
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        enable_memory=not args.no_memory,
        enable_enhanced_embeddings=True,
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        system_message=args.system_prompt,
        no_filter=args.no_filter
    )

    # Set up history
    history_file = setup_readline_history(chat.memory_dir)
    print(f"{chat.get_time()} Command history stored in: {history_file}")

    # Warm up the model with a short prompt
    if chat.device == "cuda":
        print(f"{chat.get_time()} Warming up model...")
        import random

        _ = chat.chat(
            [{"role": "user", "content": f"Calculate the result of {random.randint(1, 100)} + {random.randint(1, 100)}.  Print just result without any explanation or preamble, like in this example: 242+4=246." }],
            max_new_tokens=20,
            temperature=0.7
        )

    # Display welcome header
    print("\n" + "="*50)
    print("Memory-Enhanced Chat")
    print("="*50)
    print("Type 'exit' or 'q' to end the conversation")
    print("Special commands:")
    print("  !system: [message] - Change the system message")
    print("  !toggle-memory - Toggle memory system on/off")
    print("  !toggle-memory-debug - Toggle memory debugging")
    print("  !toggle-heatmap - Toggle confidence heatmap visualization")
    print("  !toggle-filter - Toggle confidence filtering")
    print("  !toggle-websearch - Toggle web search on/off")
    print("  !search: [query] - Force web search for a query")
    print("  !mcp-help - Show MCP commands for file output")
    print("  !memory-stats - Display info about memories")

    print("="*50 + "\n")

    # Initialize chat state
    show_confidence = args.heatmap
    filter_enabled = True
    conversation = [{"role": "system", "content": args.system_prompt}]

    try:
        # Main conversation loop
        while True:
            # Get user input
            user_input = chat.get_multiline_input(f"{chat.get_time()} You: ")

            if not user_input:
                continue

            # Check for topic shift
            if len(conversation) > 2:  # Only check after we have at least one exchange
                is_shift, similarity = chat.topic_detector.is_topic_shift(user_input)

                if is_shift:
                    # Store context before reset
                    last_context = {
                        'last_user_query': conversation[-2].get('content') if len(conversation) > 2 else '',
                        'last_assistant_response': conversation[-1].get('content') if len(conversation) > 1 else ''
                    }

                    # Reset conversation but keep system message
                    system_message = conversation[0]
                    conversation = [system_message]

                    # Add shift notification to user context
                    if chat.response_filter and hasattr(chat.response_filter, 'user_context'):
                        chat.response_filter.user_context['topic_shift'] = {
                            'similarity': similarity,
                            'previous_context': last_context
                        }

                    print(f"{chat.get_time()} Topic shift detected (similarity: {similarity:.3f})")

            # Handle special commands
            if user_input.lower() == 'exit' or user_input.lower() == 'q':
                # Show memory stats
                stats = chat.memory_manager.get_stats()
                print(f"{chat.get_time()} Total memories: {stats['active_items']}")
                print(f"{chat.get_time()} Memories added this session: {chat.memory_stats['items_added']}")
                break

            elif user_input.lower().startswith('!system:'):
                new_system = user_input[8:].strip()
                conversation[0] = {"role": "system", "content": new_system}
                print(f"{chat.get_time()} System message updated")
                continue

            elif user_input.lower() == '!mcp-help':
                help_text = chat.mcp_handler.get_help_text()
                print(help_text)
                continue

            elif user_input.lower() == '!toggle-memory':
                is_enabled = chat.toggle_memory()
                print(f"{chat.get_time()} Memory system {'enabled' if is_enabled else 'disabled'}")
                continue

            if user_input.lower() == '!toggle-memory-debug':
                chat.memory_debug = not chat.memory_debug
                print(f"{chat.get_time()} Memory debugging {'enabled' if chat.memory_debug else 'disabled'}")
                continue

            elif user_input.lower() == '!toggle-heatmap':
                show_confidence = not show_confidence
                print(f"{chat.get_time()} Confidence heatmap {'enabled' if show_confidence else 'disabled'}")
                continue

            elif user_input.lower() == '!toggle-filter':
                filter_enabled = not filter_enabled
                print(f"{chat.get_time()} Response filtering {'enabled' if filter_enabled else 'disabled'}")
                continue

            elif user_input.lower() == '!memory-stats':
                stats = chat.memory_manager.get_stats()
                print(f"\n{chat.get_time()} Memory Statistics:")
                print(f"  Total items: {stats['total_items']}")
                print(f"  Active items: {stats['active_items']}")
                print(f"  Items added this session: {chat.memory_stats['items_added']}")
                print(f"  Retrievals this session: {chat.memory_stats['retrievals']}")
                print(f"  Enhanced embeddings: {'Enabled' if stats['enhanced_enabled'] else 'Disabled'}")
                continue

            elif user_input.lower() == '!rebuild-memory':
                print(f"{chat.get_time()} Rebuilding memory indices...")
                # Force rebuild of memory indices
                chat.memory_manager._rebuild_index()
                chat.memory_manager._rebuild_enhanced_indices()
                print(f"{chat.get_time()} Memory indices rebuilt")
                continue

            elif user_input.lower() == '!toggle-websearch':
                if hasattr(chat, 'web_search'):
                    chat.enable_web_search = not getattr(chat, 'enable_web_search', True)
                    print(f"{chat.get_time()} Web search {'enabled' if chat.enable_web_search else 'disabled'}")
                else:
                    print(f"{chat.get_time()} Web search not available")
                continue

            elif user_input.lower().startswith('!search:'):
                query = user_input[8:].strip()
                if query and hasattr(chat, 'web_search'):
                    results = chat.web_search.process_query(query)
                    print(f"{chat.get_time()} Search results: {results['results_found']} found, {results['saved_to_memory']} saved")
                    for i, content in enumerate(results['content'][:3]):
                        print(f"  {i+1}. {content['title']} - {content['domain']}")
                continue

            # Add user message to conversation
            user_message = {"role": "user", "content": user_input}
            conversation.append(user_message)

            # Generate response
            start_time = time.time()

            response = chat.chat(
                messages=conversation,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                show_confidence=show_confidence
            )

            # Add assistant response to conversation
            assistant_message = {"role": "assistant", "content": response}
            conversation.append(assistant_message)

            # Calculate tokens and timing
            generation_time = time.time() - start_time
            token_count = chat.tokens_received
            tokens_per_second = token_count / max(0.01, generation_time)

            # Display generation stats
            print(f"\n{chat.get_time()} Generated ~{token_count} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/sec)")

    except KeyboardInterrupt:
        print(f"\n{chat.get_time()} Exiting due to keyboard interrupt...")

    except Exception as e:
        print(f"\n{chat.get_time()} Unexpected error: {e}")
        traceback.print_exc()

    finally:
        chat.cleanup()
        sys.exit(0)

if __name__ == "__main__":
    main()