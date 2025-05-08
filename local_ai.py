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
from speculative_decoder import SpeculativeDecoder, SpeculativeDecodingStats
from topic_shift_detector import TopicShiftDetector

# Import our refactored memory manager
from unified_memory import MemoryManager
from mcp_prompt_completer import MCPCompleter
from token_buffer import TokenBuffer

# Default system message template
DEFAULT_SYSTEM_MESSAGE = """You are a helpful and friendly assistant designed to provide accurate information. Follow these guidelines:
1. When you don't know something, explicitly say "I don't know about [topic]" or "I'm not familiar with that."
2. Never make up information. It is better to admit uncertainty than to provide potentially incorrect information.
3. You may speculate if the user explicitly asks you to "please continue anyway" or "please speculate."
4. When speculating, clearly indicate your uncertainty with phrases like "I'm not confident, but..."
5. Be helpful, informative, and conversational in your responses.
"""

class MemoryEnhancedChat:
    """
    Improved chat interface with proper memory integration, streaming, and speculative decoding.
    """

    def __init__(self,
                model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                device: Optional[str] = None,
                memory_dir: str = "./memory",
                output_dir: str = "./output",
                confidence_threshold: float = 0.45,
                enable_memory: bool = True,
                similarity_enhancement_factor: float = 0.3,
                enable_enhanced_embeddings: bool = True,
                do_sample: bool = True,
                top_p: float = 1.0,
                top_k: int = 50,
                enable_draft_model: bool = False,
                system_message: Optional[str] = None):
        """
        Initialize the chat interface.

        Args:
            model_name: Name of the model to use
            device: Device to use (cuda/cpu)
            memory_dir: Directory for storing memory data
            output_dir: Directory for output files
            confidence_threshold: Threshold for confidence filtering
            enable_memory: Whether to enable memory
            similarity_enhancement_factor: Factor for enhancing similarity (0.0-1.0)
            enable_enhanced_embeddings: Whether to use enhanced embeddings
            do_sample: Whether to use sampling for generation
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            enable_draft_model: Whether to enable draft model for speculative decoding
            system_message: Custom system message
        """
        self.model_name = model_name
        self.memory_dir = memory_dir
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.enable_memory = enable_memory
        self.similarity_enhancement_factor = similarity_enhancement_factor
        self.enable_enhanced_embeddings = enable_enhanced_embeddings
        self.do_sample = do_sample
        self.top_p = top_p
        self.top_k = top_k
        self.enable_draft_model = enable_draft_model
        self.memory_debug = False  # Set to True for memory diagnostics

        # Set system message
        self.system_message = system_message or DEFAULT_SYSTEM_MESSAGE

        # Set up components
        self._setup_device(device)
        self._setup_resource_manager()
        self._setup_model_and_tokenizer()
        self._setup_memory_system()
        self._setup_utility_components()
        self._setup_speculative_decoding()

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
        # Resource manager for efficient memory usage
        self.resource_manager = ResourceManager(device=self.device)

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
        self.confidence_metrics = EnhancedConfidenceMetrics(
            sharpening_factor=self.similarity_enhancement_factor
        )

        # Question classifier for domain-specific handling
        self.question_classifier = QuestionClassifier()

        # Initialize topic shift detector
        self.topic_detector = TopicShiftDetector(
            similarity_threshold=0.55,          # Adjust this threshold based on testing
            memory_manager=self.memory_manager  # Reuse memory manager for embeddings
        )

        # Initialize response filter
        self.response_filter = ResponseFilter(
            confidence_threshold=self.confidence_threshold,
            entropy_threshold=3.5,         # Up from 2.5
            perplexity_threshold=25.0,     # Up from 15.0
            user_context={},
            question_classifier=self.question_classifier,
            sharpening_factor=self.similarity_enhancement_factor,
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
        self.tokenizer.model_max_length = 2048  # Set appropriate context window, max - 2048 for tinyllama

        # Set up model loading options
        loading_options = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto" if self.device != "cpu" else None,
            "low_cpu_mem_usage": True,
        }

        # Load main model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **loading_options)
        self.model = self.resource_manager.register_model(self.model)

        custom_template = self.tokenizer.chat_template
        if custom_template:
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
            auto_save=True,
            similarity_enhancement_factor=self.similarity_enhancement_factor
        )

        # Create embedding function
        self._setup_embedding_function()

    def _setup_embedding_function(self):
        """Set up embedding function for the memory system."""
        from batch_utils import tensor_batch_processing

        # Ensure model is in evaluation mode
        self.model.eval()

        # Detect model's hidden dimension
        # For TinyLlama, this is 2048
        try:
            if hasattr(self.model, 'config'):
                if hasattr(self.model.config, 'hidden_size'):
                    embedding_dim = self.model.config.hidden_size
                elif hasattr(self.model.config, 'hidden_dim'):
                    embedding_dim = self.model.config.hidden_dim
                elif hasattr(self.model.config, 'd_model'):
                    embedding_dim = self.model.config.d_model
                else:
                    # Default for TinyLlama if we can't detect it
                    embedding_dim = 2048
            else:
                embedding_dim = 2048

            print(f"{self.get_time()} Detected embedding dimension: {embedding_dim}")

            # Update memory manager's embedding dimension
            if hasattr(self.memory_manager, 'embedding_dim'):
                if self.memory_manager.embedding_dim != embedding_dim:
                    print(f"{self.get_time()} Updating memory manager embedding dimension: "
                          f"{self.memory_manager.embedding_dim} → {embedding_dim}")

                    # Use the comprehensive update method
                    self.memory_manager.update_embedding_dimension(embedding_dim)

        except Exception as e:
            print(f"{self.get_time()} Error detecting embedding dimension: {e}")
            embedding_dim = 2048  # Fall back to TinyLlama's dimension

        # Define the core embedding operation that processes a batch of texts
        def batch_embedding_operation(batch_tokenized):
            """Process a batch of tokenized texts to generate embeddings."""
            with torch.no_grad():
                # Get model outputs
                if hasattr(self.model, 'model'):
                    outputs = self.model.model(
                        input_ids=batch_tokenized['input_ids'].to(self.device),
                        attention_mask=batch_tokenized['attention_mask'].to(self.device)
                    )
                else:
                    outputs = self.model(
                        input_ids=batch_tokenized['input_ids'].to(self.device),
                        attention_mask=batch_tokenized['attention_mask'].to(self.device),
                        output_hidden_states=True
                    )

                # Get mean pooled representation
                last_hidden_states = outputs.last_hidden_state
                # Mean pooling - take average of all token embeddings for each sequence
                input_mask_expanded = batch_tokenized['attention_mask'].to(self.device).unsqueeze(-1).expand(last_hidden_states.size()).float()
                sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask

                return mean_pooled

        # Create single embedding function
        def generate_embedding(text: str) -> np.ndarray:
            """Generate an embedding vector for a text string."""

            # Generate embedding
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
                return np.zeros(self.memory_manager.embedding_dim)

        # Create a batch embedding function using tensor_batch_processing
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
                )

                # Get resource manager batch settings
                batch_settings = self.resource_manager.get_batch_settings('embedding')

                # Use tensor_batch_processing for efficient processing
                pooled_outputs = tensor_batch_processing(
                    tensor_op=batch_embedding_operation,
                    input_tensor=batch_tokenized['input_ids'],  # We'll use input_ids for batching
                    batch_dim=0,  # Batch along the first dimension
                    batch_size=batch_settings.get('batch_size', 8),
                    cleanup=batch_settings.get('cleanup', True),
                    adaptive=batch_settings.get('adaptive', True),
                    handle_oom=batch_settings.get('handle_oom', True)
                )

                # Convert to numpy arrays
                embeddings = pooled_outputs.cpu().numpy()

                return list(embeddings)

            except Exception as e:
                print(f"{self.get_time()} Error in batch embedding: {e}")
                # Fall back to individual processing
                return [generate_embedding(text) for text in texts]

        # Set embedding functions
        self.memory_manager.set_embedding_function(
            function=generate_embedding,
            batch_function=generate_embeddings_batch
        )

    def _setup_speculative_decoding(self):
        """Set up speculative decoding with draft model if enabled."""
        # Initialize stats tracker
        self.spec_stats = SpeculativeDecodingStats()

        # Initialize speculative decoder
        self.spec_decoder = SpeculativeDecoder(
            main_model=self.model,
            draft_model=None,  # Will be set later if enabled
            tokenizer=self.tokenizer,
            confidence_metrics=self.confidence_metrics,
            device=self.device,
            stats=self.spec_stats,
            repetition_penalty=1.2,
            max_draft_tokens=5,
            batch_processor=None
        )

        # Create draft model if enabled
        if self.enable_draft_model:
            print(f"{self.get_time()} Creating draft model for speculative decoding...")
            self.draft_model = self.spec_decoder.create_draft_model(self.model)

            if self.draft_model:
                print(f"{self.get_time()} Successfully created draft model")
                self.spec_decoder.draft_model = self.draft_model
            else:
                print(f"{self.get_time()} Could not create draft model, speculative decoding disabled")
                self.enable_draft_model = False
        else:
            self.draft_model = None

    def _save_command_output_to_memory(self, command: str, output: str) -> None:
        """
        Save command output to memory with optimized formatting for retrieval.

        Args:
            command: The command that was executed
            output: Output text from the command
        """
        if not self.enable_memory or not output:
            return

        try:
            # Create formatted content for better retrievability
            formatted_content = ""

            # Special handling for date-related commands
            if re.search(r'date|time', command, re.IGNORECASE):
                output_clean = output.strip()

                # Format as clear statements
                formatted_content = f"Today's date is {output_clean}."

                # Add additional formats for better retrieval
                additional_formats = [
                    f"The current date is {output_clean}.",
                    f"Today is {output_clean}.",
                    f"Right now it is {output_clean}."
                ]
            else:
                # For other commands, format clearly
                formatted_content = f"The output of command '{command}' is: {output.strip()}"

            # Create metadata
            metadata = {
                'source': 'command_output',
                'command': command,
                'type': 'fact',
                'timestamp': datetime.now().timestamp()
            }

            # Add primary format to memory
            memory_id = self.memory_manager.add(
                content=formatted_content,
                metadata=metadata
            )

            # For date commands, add alternative formats as well
            if 'additional_formats' in locals() and additional_formats:
                for format in additional_formats:
                    self.memory_manager.add(
                        content=format,
                        metadata=metadata  # Same metadata for all formats
                    )
                print(f"{self.get_time()} Command output saved with {len(additional_formats) + 1} format variations")
            else:
                print(f"{self.get_time()} Command output saved to memory (ID: {memory_id})")

        except Exception as e:
            print(f"{self.get_time()} Error saving command output to memory: {e}")

    # def _save_command_output_to_memory(self, command: str, output: str):
    #     """Save command output to memory for future reference."""
    #     if not self.enable_memory or not output:
    #         return

    #     try:
    #         # Detect content type
    #         content_type = "tabular" if '\t' in output or re.search(r'\S\s{2,}\S', output) else "text"

    #         # Create metadata
    #         metadata = {
    #             "source": "command_output",
    #             "command": command,
    #             "timestamp": datetime.now().timestamp(),
    #             "content_type": content_type
    #         }

    #         # Add to memory
    #         memory_id = self.memory_manager.add(
    #             content=output,
    #             metadata=metadata
    #         )

    #         if memory_id:
    #             print(f"{self.get_time()} Command output saved to memory (ID: {memory_id})")
    #     except Exception as e:
    #         print(f"{self.get_time()} Error saving command output to memory: {e}")

    def get_time(self) -> str:
        """Get formatted timestamp for logging."""
        return datetime.now().strftime("[%d/%m/%y %H:%M:%S] [Main] ")

    def toggle_memory(self) -> bool:
        """Toggle memory system on/off."""
        self.enable_memory = not self.enable_memory
        return self.enable_memory

    def toggle_draft_model(self) -> bool:
        """Toggle draft model on/off for speculative decoding."""
        if self.enable_draft_model and self.draft_model is not None:
            # Disable draft model
            self.enable_draft_model = False
            print(f"{self.get_time()} Draft model disabled")
            return False
        elif not self.enable_draft_model and self.draft_model is None:
            # Try to create draft model
            print(f"{self.get_time()} Creating draft model...")
            self.draft_model = self.spec_decoder.create_draft_model(self.model)

            if self.draft_model:
                self.enable_draft_model = True
                self.spec_decoder.draft_model = self.draft_model
                print(f"{self.get_time()} Draft model enabled")
                return True
            else:
                print(f"{self.get_time()} Failed to create draft model")
                return False
        elif not self.enable_draft_model and self.draft_model is not None:
            # Re-enable existing draft model
            self.enable_draft_model = True
            self.spec_decoder.draft_model = self.draft_model
            print(f"{self.get_time()} Draft model re-enabled")
            return True
        else:
            # No change needed
            return True

    def set_similarity_enhancement(self, factor: float) -> None:
        """Set the similarity enhancement factor for memory and confidence."""
        # Validate range
        factor = max(0.0, min(1.0, factor))

        # Update settings
        self.similarity_enhancement_factor = factor

        # Update confidence metrics
        if hasattr(self.confidence_metrics, 'set_sharpening_factor'):
            self.confidence_metrics.set_sharpening_factor(factor)

        # Update memory manager
        self.memory_manager.similarity_enhancement_factor = factor

        print(f"{self.get_time()} Similarity enhancement factor set to {factor}")

    def chat(self,
            messages: List[Dict[str, str]],
            max_new_tokens: int = 128,
            temperature: float = 0.7,
            enable_memory: Optional[bool] = None,
            show_confidence: bool = False) -> str:
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

        fallback_shown = False

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

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
                args=(input_ids, streamer, generation_config, self.enable_draft_model)
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
            ignore_fallback_answer = False

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

                # Check confidence to possibly stop generation - ONLY IF NO FALLBACK YET
                if (self.response_filter is not None and
                    not low_confidence_detected and
                    not fallback_shown and
                    tokens_received >= 10):  # Check after reasonable context

                    # Get current metrics
                    current_metrics = self.confidence_metrics.get_metrics(apply_sharpening=True)

                    # Should we show fallback instead of continuing?
                    if self.response_filter.should_stream_fallback(
                        current_metrics,
                        user_query,
                        token_buffer.get_text(),
                        tokens_received
                    ):
                        # Set flags to avoid checking again
                        low_confidence_detected = True
                        ignore_fallback_answer = True
                        fallback_shown = True

                        # Stop generation
                        self.stop_event.set()

                        # FIXED: Clear the display buffer before showing fallback
                        display_buffer = ""

                        # FIXED: Get fallback message and show it
                        fallback = self.response_filter.get_streamable_fallback(user_query)
                        print(fallback, end="", flush=True)

                        # FIXED: Clear token_buffer and replace with fallback for final response
                        token_buffer.clear()
                        for c in fallback:
                            token_buffer.add_token(c)

                        break

                # Check for pattern detection at adaptive intervals - ONLY IF NO FALLBACK YET
                if tokens_received >= next_pattern_check and not fallback_shown:
                    pattern_results = self.response_filter.detect_repetitive_patterns(token_buffer.get_text())
                    if pattern_results['has_repetitive_patterns']:
                        # Set flags
                        ignore_fallback_answer = True
                        fallback_shown = True

                        # Stop generation
                        self.stop_event.set()

                        # FIXED: Clear the display buffer before showing fallback
                        display_buffer = ""

                        # Get fallback message
                        fallback = self.response_filter.get_streamable_fallback(
                            user_query, reason="repetitive_pattern", details=pattern_results)
                        print(fallback, end="", flush=True)

                        # FIXED: Clear token_buffer and replace with fallback for final response
                        token_buffer.clear()
                        for c in fallback:
                            token_buffer.add_token(c)

                        break

                    # OPTIMIZATION: Adaptive backoff for pattern checking
                    pattern_check_interval = min(pattern_check_interval * 2, 100)
                    next_pattern_check = tokens_received + pattern_check_interval

                    # OPTIMIZATION: Explicit garbage collection after pattern checking
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Only add displayable tokens to the display buffer if no fallback shown
                if display_token and not fallback_shown:
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

            # Handle any remaining display buffer
            if display_buffer and not fallback_shown:
                print(display_buffer, end="", flush=True)

            # Ensure confidence metrics exist
            self._ensure_confidence_metrics(len(token_buffer))

            # INTEGRATION: Get full text from TokenBuffer
            complete_response = token_buffer.get_text()

            # Finalize the response
            response = self.mcp_handler.finalize_streaming(complete_response)

            # Post-process the response
            response = self._post_process_response(response)

            # Apply response filtering if not already handled during streaming
            if self.response_filter is not None and not fallback_shown:
                current_metrics = self.confidence_metrics.get_metrics(apply_sharpening=True)
                # Update the user context with information about this exchange
                if hasattr(self.response_filter, 'update_user_context'):
                    self.response_filter.update_user_context(user_query, response, current_metrics)
                # Apply filtering
                response = self.response_filter.filter_response(
                    response=response,
                    metrics=current_metrics,
                    query=user_query,
                    preserve_mcp=True,
                    allow_override=True
                )

            # Save to memory if enabled
            if memory_enabled and user_query and response and not ignore_fallback_answer:
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

            # Show heatmap legend if enabled
            if show_confidence and not self.stop_event.is_set() and not fallback_shown:
                # Calculate token usage
                prompt_tokens = len(tokenized['input_ids'][0])
                response_tokens_count = len(token_buffer) if 'token_buffer' in locals() else 0
                total_tokens = prompt_tokens + response_tokens_count

                # Get max context window from tokenizer
                max_tokens = self.tokenizer.model_max_length

                # Display legend with context window usage
                heatmap.print_legend(current_tokens=total_tokens, max_tokens=max_tokens)

                print()  # Add a newline
                confidence_metrics = self.confidence_metrics.get_metrics(apply_sharpening=True)
                normalized_metrics = self.response_filter.normalize_confidence_metrics(confidence_metrics)
                should_filter, reason, details = self.response_filter.should_filter(normalized_metrics, response, user_query, tokens_received)

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

    def _format_tabular_memory(self, content: str) -> str:
        """Format tabular data for better readability and comprehension."""
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return content

        # Extract header and data rows
        header = lines[0]
        data_rows = lines[1:]

        # Create a structured representation
        formatted = "TABLE DATA:\n"
        formatted += f"Headers: {header}\n"
        formatted += "Data rows:\n"

        # Add each row with a number for reference
        for i, row in enumerate(data_rows, 1):
            formatted += f"Row {i}: {row}\n"

        # Add guidance for interpretation
        formatted += "\nWhen answering questions about this table, refer to specific columns by name and rows by number.\n"

        return formatted

    def _format_combined_memories(self, memories: List[Dict[str, Any]], query: str) -> str:
        """Format combined memories with better visibility and explicit instructions."""

        output = "MEMORY CONTEXT (USE THIS INFORMATION TO ANSWER THE QUERY):\n"

        # Make command outputs more visible
        if any(memory.get("metadata", {}).get("source") == "command_output" for memory in memories):
            output += "\nCOMMAND OUTPUT (THIS IS CURRENT REAL-TIME DATA - USE THIS DIRECTLY):\n"
            for memory in [m for m in memories if m.get("metadata", {}).get("source") == "command_output"]:
                command = memory.get("metadata", {}).get("command", "unknown command")
                content = memory.get("content", "").strip()
                output += f"Output from command '{command}':\n{content}\n\n"

        # More explicit instructions for the model
        output += "\nINSTRUCTIONS FOR USING MEMORY:\n"
        output += "1. The information above is FACTUAL and CURRENT.\n"
        output += "2. You MUST use this information to answer the query directly.\n"
        output += "3. If date or time information is present, use it explicitly rather than saying you don't know.\n"
        output += "4. Command outputs represent real-time data - treat them as current and accurate.\n\n"

        return output

    # def _format_combined_memories(self, memories: List[Dict[str, Any]], query: str) -> str:
    #     """Format combined memories with proper context separation."""

    #     # Separate different types of memories
    #     command_memories = []
    #     conversation_memories = []
    #     direct_answers = []

    #     for memory in memories:
    #         metadata = memory.get("metadata", {})
    #         source_hint = metadata.get("source_hint", "")

    #         if metadata.get("source") == "command_output":
    #             command_memories.append(memory)
    #         elif source_hint == "direct_answer":
    #             direct_answers.append(memory)
    #         elif source_hint in ["conversation", "qa_pair"]:
    #             conversation_memories.append(memory)

    #     # Build formatted output
    #     output = "MEMORY CONTEXT:\n"

    #     # Add command outputs first
    #     if command_memories:
    #         output += "\nCOMMAND OUTPUT INFORMATION:\n"
    #         for memory in command_memories:
    #             command = memory.get("metadata", {}).get("command", "unknown command")
    #             content = memory.get("content", "").strip()
    #             output += f"Output from command '{command}':\n"
    #             indented_content = "\n".join(f"  {line}" for line in content.split("\n"))
    #             output += f"{indented_content}\n\n"

    #     # Add direct answers
    #     if direct_answers:
    #         output += "\nRELEVANT INFORMATION:\n"
    #         for memory in direct_answers:
    #             content = memory.get("content", "").strip()
    #             similarity = memory.get("similarity", 0)
    #             if not content.startswith("Q:") and not content.startswith("- Q:"):
    #                 output += f"- [{similarity:.2f}] {content}\n"

    #     # Add conversation context only if specifically requested
    #     if conversation_memories and len(query.split()) > 10:  # Only for longer queries
    #         output += "\nPREVIOUS CONVERSATION CONTEXT:\n"
    #         for memory in conversation_memories:
    #             content = memory.get("content", "").strip()
    #             # Skip if it's in the awkward Q&A format
    #             if not content.startswith("Q:") and not content.startswith("- Q:"):
    #                 output += f"- {content}\n"

    #     # Add guidance
    #     output += "\nUse the above information to help answer the query if relevant. Do not repeat the information verbatim.\n"

    #     return output

    def _integrate_memory(self, messages: List[Dict[str, str]], query: str) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """
        Integrate relevant memories into conversation messages.

        Args:
            messages: Current conversation messages
            query: Current user query

        Returns:
            Tuple of (updated messages, retrieved memories)
        """
        try:
            # First check for command outputs related to the query
            command_memories = self.memory_manager.retrieve(
                query=query,
                top_k=2,
                min_similarity=0.1,
                metadata_filter={"source": "command_output"}
            )

            memories = []

            # Get domain information
            domain_settings = self.question_classifier.get_domain_settings(query)
            domain = domain_settings.get('domain', 'unknown')

            # Adjust memory retrieval based on domain
            if domain == 'arithmetic':
                # For math, prioritize exact examples
                memories = self.memory_manager.retrieve(
                    query=query,
                    top_k=3,
                    min_similarity=0.6,
                    metadata_filter={"source_hint": ["arithmetic", "qa_pair"]}
                )
            elif domain == 'translation':
                # For translation, look for similar language pairs
                memories = self.memory_manager.retrieve(
                    query=query,
                    top_k=8,
                    min_similarity=0.4,
                    metadata_filter={"source_hint": ["translation", "language"]}
                )
            else:
                # Standard retrieval for other domains
                memories = self.memory_manager.retrieve(
                    query=query,
                    top_k=5,
                    min_similarity=0.25
                )

            # Combine the results, prioritizing command outputs
            combined_memories = []
            command_ids = []

            # Process command outputs first
            for memory in command_memories:
                # Format based on content_type if available
                memory_type = memory.get("metadata", {}).get("content_type", "text")
                if memory_type == "tabular":
                    # Special formatting for tables
                    memory["formatted_content"] = self._format_tabular_memory(memory["content"])
                else:
                    memory["formatted_content"] = memory["content"]
                combined_memories.append(memory)
                command_ids.append(memory["id"])

            # Then add general memories that aren't duplicates
            for memory in memories:
                if memory["id"] not in command_ids:
                    combined_memories.append(memory)

            # Format memories for context if we have any
            if combined_memories:
                memory_text = self._format_combined_memories(combined_memories, query)

                # Create memory-enhanced messages - but with a crucial change
                memory_enhanced_messages = messages.copy()

                # Instead of modifying system message, add memory as a "memory" role message
                # right before the user query
                last_user_idx = next((i for i in reversed(range(len(messages)))
                                     if messages[i]["role"] == "user"), None)

                if last_user_idx is not None:
                    memory_enhanced_messages.insert(last_user_idx, {
                        "role": "memory",  # Use a custom role
                        "content": memory_text
                    })

                if self.memory_debug:
                    print(f"\n{self.get_time()} Memory diagnostic: Retrieved {len(memories)} items")
                    print("-----------------------------------")
                    for i, memory in enumerate(memories):
                        content = memory.get("content", "").strip()
                        # Truncate long content for display
                        if len(content) > 100:
                            content = content[:97] + "..."

                        similarity = memory.get("similarity", 0)
                        source = memory.get("metadata", {}).get("source", "unknown")
                        timestamp = memory.get("metadata", {}).get("timestamp", 0)

                        # Format timestamp as readable date
                        if timestamp:
                            time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
                        else:
                            time_str = "unknown"

                        print(f"Memory #{i+1} (sim: {similarity:.3f}, source: {source}, time: {time_str})")
                        print(f"Content: {content}")
                        print("-----------------------------------")

                return memory_enhanced_messages, combined_memories
            # if combined_memories:
            #     # Use custom formatter for combined memories
            #     memory_text = self._format_combined_memories(combined_memories, query)
            #     memory_text += f"\n\nNote: Query classified as {domain} domain."

            #     # Create memory-enhanced messages
            #     memory_enhanced_messages = messages.copy()

            #     # Extract system message
            #     original_system = messages[0]["content"]

            #     # Remove existing memory context if present
            #     if "MEMORY CONTEXT:" in original_system:
            #         system_parts = original_system.split("MEMORY CONTEXT:")
            #         system_content = system_parts[0].strip()
            #     else:
            #         system_content = original_system

            #     # Add memory context
            #     system_with_memory = f"{system_content}\n\n{memory_text}"

            #     # Update system message
            #     memory_enhanced_messages[0]["content"] = system_with_memory

            #     # Update stats
            #     self.memory_stats["retrievals"] += 1

            #     return memory_enhanced_messages, combined_memories

            return messages, []

        except Exception as e:
            print(f"{self.get_time()} Error integrating memory: {e}")
            traceback.print_exc()
            return messages, []

    def _save_to_memory(self, query: str, response: str) -> bool:
        """
        Save conversation exchange to memory, separating query and response.

        Args:
            query: User query
            response: Generated response

        Returns:
            Success status
        """
        try:
            # 1. Store the query separately
            query_metadata = {
                'source': 'user_query',
                'type': 'question',
                'timestamp': datetime.now().timestamp()
            }
            query_id = self.memory_manager.add(
                content=query.strip(),
                metadata=query_metadata
            )

            # 2. Store the response separately
            response_metadata = {
                'source': 'assistant_response',
                'type': 'answer',
                'related_query': query.strip(),
                'timestamp': datetime.now().timestamp()
            }
            response_id = self.memory_manager.add(
                content=response.strip(),
                metadata=response_metadata
            )

            # 3. Extract and store key statements from the response
            key_statements = self._extract_key_statements(response)
            for statement in key_statements:
                statement_metadata = {
                    'source': 'extracted_fact',
                    'type': 'fact',
                    'derived_from': 'response',
                    'timestamp': datetime.now().timestamp()
                }
                self.memory_manager.add(
                    content=statement,
                    metadata=statement_metadata
                )

            # Update stats
            memories_added = 2 + len(key_statements)
            self.memory_stats["items_added"] += memories_added
            print(f"{self.get_time()} Added {memories_added} new memories")
            return True
        except Exception as e:
            print(f"{self.get_time()} Error saving to memory: {e}")
            return False

    def _extract_key_statements(self, text: str) -> List[str]:
        """
        Extract key factual statements from text.
        Focus on simple declarative sentences and clear facts.
        """
        statements = []

        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            # Skip short sentences
            if len(sentence) < 10:
                continue

            # Look for factual declarative sentences
            if re.match(r'^[A-Z].*?(?:is|are|was|were|has|have)\b', sentence):
                # Check if it contains an actual assertion
                if " is " in sentence or " are " in sentence or " was " in sentence:
                    statements.append(sentence)

            # Special handling for date/time information
            if re.search(r'\b(?:today|current date|the date)\b', sentence.lower()):
                if re.search(r'\b(?:is|was)\b', sentence.lower()):
                    statements.append(sentence)

        # Limit to 3 most important statements
        return statements[:3]

    def _extract_memory_worthy_content(self, query: str, response: str) -> List[Tuple[str, str]]:
        """
        Extract content worthy of memorization.

        Args:
            query: User query
            response: Generated response

        Returns:
            List of (content, source_hint) tuples
        """
        memories = []

        # For short exchanges, memorize complete Q&A
        if len(query) <= 100 and len(response) <= 200:
            # Store as natural conversation context
            conversation_text = f"{query.strip()} {response.strip()}"
            memories.append((conversation_text, "conversation"))

            # Also store response separately for direct retrieval
            if len(response.strip()) > 20:
                memories.append((response.strip(), "direct_answer"))

            return memories

        # Extract factual statements
        factual_pattern = r'(?:is|are|was|were|has|have|had|will be)\s+([A-Z][a-z]+\s+(?:is|are|was|were|has|have|had)\s+[^.!?]+[.!?])'
        matches = re.findall(factual_pattern, response)
        for match in matches:
            if len(match) > 10 and len(match) < 200:
                memories.append((match, "factual"))

        # Extract definitions
        definition_pattern = r'([A-Z][a-z]+\s+(?:refers to|is defined as|means|is|are)\s+[^.!?]+[.!?])'
        matches = re.findall(definition_pattern, response)
        for match in matches:
            if len(match) > 10 and len(match) < 200:
                memories.append((match, "definition"))

        # Extract lists
        lines = response.split('\n')
        list_items = []
        in_list = False
        list_topic = ""

        for line in lines:
            # Check for list headers
            if re.match(r'^[^-*\d]*:$', line) or re.match(r'^[^-*\d]*:$', line.strip()):
                list_topic = line.strip()
                in_list = True
                list_items = []
            # Check for list items
            elif in_list and (line.strip().startswith('-') or line.strip().startswith('*') or
                             re.match(r'^\d+\.', line.strip())):
                list_items.append(line.strip())
            # List ended
            elif in_list and line.strip() and not (line.strip().startswith('-') or
                                                  line.strip().startswith('*') or
                                                  re.match(r'^\d+\.', line.strip())):
                if list_topic and list_items and len(list_items) >= 2:
                    content = f"{list_topic}\n" + "\n".join(list_items)
                    if 20 < len(content) < 300:
                        memories.append((content, "list"))
                in_list = False

        # If no structured content found, extract key sentences
        if not memories:
            sentences = re.split(r'[.!?]', response)
            key_sentences = []

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue

                # Look for sentences with informational value
                if (re.search(r'\b(?:key|important|significant|essential|critical)\b', sentence) or
                    re.search(r'\b(?:always|never|must|should|typically|generally)\b', sentence)):
                    key_sentences.append(sentence + ".")

            # Take top 3 most informative sentences
            for sentence in key_sentences[:3]:
                if 10 < len(sentence) < 200:
                    memories.append((sentence, "key_info"))

        # De-duplicate memories
        unique_memories = []
        seen_content = set()

        for content, source_hint in memories:
            # Normalize to avoid near-duplicates
            normalized = re.sub(r'\s+', ' ', content.lower())
            if normalized not in seen_content:
                seen_content.add(normalized)
                unique_memories.append((content, source_hint))

        # Limit total memories from a single exchange
        return unique_memories[:5]  # Cap at 5 memories per exchange

    def _generate_with_streaming(self, input_ids, streamer, generation_config, use_draft_model):
        """
        Generate a response with streaming output - optimized version.

        Args:
            input_ids: Input token IDs
            streamer: Text streamer for output
            generation_config: Generation configuration
            use_draft_model: Whether to use draft model
        """
        try:
            # OPTIMIZATION: Clear CUDA cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Reset confidence metrics
            if hasattr(self.confidence_metrics, 'reset'):
                self.confidence_metrics.reset()

            # Setup logits processor for confidence tracking
            from transformers import LogitsProcessorList

            # OPTIMIZATION: Create a copy of generation_config
            # to avoid modifying the original
            streaming_config = {k: v for k, v in generation_config.items()
                               if k not in ['max_new_tokens', 'output_scores', 'return_dict_in_generate']}

            # Add TokenProbabilityCaptureProcessor for confidence tracking
            if 'logits_processor' not in streaming_config:
                streaming_config['logits_processor'] = LogitsProcessorList()

            # Create the processor for capturing token probabilities
            metrics_processor = TokenProbabilityCaptureProcessor(self.confidence_metrics)
            streaming_config['logits_processor'].append(metrics_processor)

            # OPTIMIZATION: Check if input sequence is very long
            input_length = input_ids.shape[1]
            if input_length > 1024:
                # For very long inputs, reduce max_new_tokens to avoid OOM
                max_tokens = generation_config.get('max_new_tokens', 128)
                adjusted_max_tokens = min(max_tokens, max(30, 2048 - input_length))

                if adjusted_max_tokens < max_tokens:
                    print(f"{self.get_time()} Reducing max tokens from {max_tokens} to {adjusted_max_tokens} due to long input")

                # Create a modified config with adjusted max_tokens
                if use_draft_model and self.draft_model is not None:
                    self.spec_decoder.generate_with_speculative_decoding(
                        input_ids=input_ids,
                        streamer=streamer,
                        max_new_tokens=adjusted_max_tokens,
                        generation_config=streaming_config,
                        stop_event=self.stop_event
                    )
                else:
                    self.model.generate(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        streamer=streamer,
                        max_new_tokens=adjusted_max_tokens,
                        **streaming_config
                    )
            else:
                # Use standard generation with original max_tokens
                # Use speculative decoding if enabled
                if use_draft_model and self.draft_model is not None:
                    # Generate with speculative decoding
                    self.spec_decoder.generate_with_speculative_decoding(
                        input_ids=input_ids,
                        streamer=streamer,
                        max_new_tokens=generation_config.get('max_new_tokens', 128),
                        generation_config=streaming_config,
                        stop_event=self.stop_event
                    )
                else:
                    # OPTIMIZATION: Set use_cache=True for better performance
                    if 'use_cache' not in streaming_config:
                        streaming_config['use_cache'] = True

                    # Use standard generation
                    self.model.generate(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        streamer=streamer,
                        max_new_tokens=generation_config.get('max_new_tokens', 128),
                        **streaming_config
                    )

            # OPTIMIZATION: Clear cache after generation is complete
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"{self.get_time()} Error in generation thread: {str(e)}")
            traceback.print_exc()
            try:
                streamer.end()
            except Exception:
                pass

            # OPTIMIZATION: Ensure memory is cleaned up on error
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
        """
        Safe tokenization that handles 'Already borrowed' errors.

        Args:
            text: Text to tokenize
            **kwargs: Arguments for tokenizer

        Returns:
            Tokenized result
        """
        try:
            # Try normal tokenization
            return self.tokenizer(text, **kwargs)
        except RuntimeError as e:
            if "Already borrowed" in str(e):
                print(f"{self.get_time()} Handling 'Already borrowed' error...")

                # Create a fresh tokenizer instance
                import copy
                temp_tokenizer = copy.deepcopy(self.tokenizer)

                # Tokenize with fresh tokenizer
                result = temp_tokenizer(text, **kwargs)

                # Convert tensors to fresh copies
                return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
            else:
                # Re-raise other errors
                raise

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
            # First, ensure memory manager creates a cache
            # Clean up memory
            if hasattr(self, 'memory_manager'):
                self.memory_manager.cleanup()

            # Unload models
            if hasattr(self, 'draft_model') and self.draft_model is not None:
                self.resource_manager.unload_model(self.draft_model)
                self.draft_model = None

            # Unload models
            if hasattr(self, 'draft_model') and self.draft_model is not None:
                self.resource_manager.unload_model(self.draft_model)
                self.draft_model = None

            if hasattr(self, 'model') and self.model is not None:
                self.resource_manager.unload_model(self.model)
                self.model = None

            # Clear speculative decoder references
            if hasattr(self, 'spec_decoder') and self.spec_decoder:
                self.spec_decoder.draft_model = None
                self.spec_decoder.main_model = None

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
                if hasattr(self, 'resource_manager'):
                    self.resource_manager.cleanup()
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
    parser.add_argument("--draft-model", action="store_true", default=False,
                       help="Enable draft model for speculative decoding")

    # Advanced settings
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="Directory for output files")
    parser.add_argument("--confidence-threshold", type=float, default=0.45,
                       help="Confidence threshold for filtering")
    parser.add_argument("--enhancement-factor", type=float, default=0.3,
                       help="Similarity enhancement factor (0.0-1.0)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling parameter")

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
        similarity_enhancement_factor=args.enhancement_factor,
        enable_enhanced_embeddings=True,
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        enable_draft_model=args.draft_model,
        system_message=args.system_prompt
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
    print("  !toggle-draft - Toggle draft model for speculative decoding")
    print("  !toggle-filter - Toggle confidence filtering")
    print("  !mcp-help - Show MCP commands for file output")
    print("  !memory-stats - Display info about memories")
    print("  !enhance-factor: [0.0-1.0] - Set similarity enhancement factor")

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

            elif user_input.lower() == '!toggle-draft':
                is_enabled = chat.toggle_draft_model()
                print(f"{chat.get_time()} Draft model {'enabled' if is_enabled else 'disabled'}")
                continue

            elif user_input.lower().startswith('!enhance-factor:'):
                try:
                    factor = float(user_input.split(':')[1].strip())
                    if 0.0 <= factor <= 1.0:
                        chat.set_similarity_enhancement(factor)
                        # Update response filter too
                        self.response_filter.sharpening_factor = factor
                    else:
                        print(f"{chat.get_time()} Factor must be between 0.0 and 1.0")
                except Exception as e:
                    print(f"{chat.get_time()} Invalid value: {e}")
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
            token_count = len(response.split())
            tokens_per_second = token_count / max(0.01, generation_time)

            # Display generation stats
            print(f"\n{chat.get_time()} Generated ~{token_count} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/sec)")

    except KeyboardInterrupt:
        print(f"\n{chat.get_time()} Exiting due to keyboard interrupt...")

    except Exception as e:
        print(f"\n{chat.get_time()} Unexpected error: {e}")
        traceback.print_exc()

    finally:
        # Final cleanup
        """Setup signal handlers to ensure clean shutdown."""
        # def handle_exit(signum, frame):
        #     print("\nReceived exit signal, cleaning up...")
        #     chat.cleanup()
        #     sys.exit(0)

        # # Register signal handlers
        # signal.signal(signal.SIGINT, handle_exit)  # Ctrl+C
        # signal.signal(signal.SIGTERM, handle_exit)  # Termination signal

        # if hasattr(signal, 'SIGBREAK'):  # Windows Ctrl+Break
        #     signal.signal(signal.SIGBREAK, handle_exit)
        # else:
        chat.cleanup()
        sys.exit(0)

if __name__ == "__main__":
    main()