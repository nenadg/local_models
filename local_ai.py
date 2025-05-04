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

# Import our refactored memory manager
from unified_memory import MemoryManager

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
                confidence_threshold: float = 0.7,
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

        # Confidence metrics tracker
        self.confidence_metrics = EnhancedConfidenceMetrics(
            sharpening_factor=self.similarity_enhancement_factor
        )

        # Question classifier for domain-specific handling
        self.question_classifier = QuestionClassifier()

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

        # Optimize for inference
        self.resource_manager.optimize_for_inference()

    def _setup_memory_system(self):
        """Set up the memory system with embedding capabilities."""
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            storage_path=self.memory_dir,
            embedding_dim=2048,  # Adjust based on model's hidden size (initially 384)
            enable_enhanced_embeddings=self.enable_enhanced_embeddings,
            max_enhancement_levels=3,
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

    def _save_command_output_to_memory(self, command: str, output: str):
        """Save command output to memory for future reference."""
        if not self.enable_memory or not output:
            return

        try:
            # Detect content type
            content_type = "tabular" if '\t' in output or re.search(r'\S\s{2,}\S', output) else "text"

            # Create metadata
            metadata = {
                "source": "command_output",
                "command": command,
                "timestamp": datetime.now().timestamp(),
                "content_type": content_type
            }

            # Add to memory
            memory_id = self.memory_manager.add(
                content=output,
                metadata=metadata
            )

            if memory_id:
                print(f"{self.get_time()} Command output saved to memory (ID: {memory_id})")
        except Exception as e:
            print(f"{self.get_time()} Error saving command output to memory: {e}")

    def get_time(self) -> str:
        """Get formatted timestamp for logging."""
        return datetime.now().strftime("[%d/%m/%y %H:%M:%S] Main")

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
            show_confidence: bool = False,
            response_filter: Optional[ResponseFilter] = None) -> str:
        """
        Generate a response with memory integration and streaming.

        Args:
            messages: List of conversation messages
            max_new_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            enable_memory: Override default memory setting
            show_confidence: Whether to show confidence heatmap
            response_filter: Optional response filter

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
            messages = self._integrate_memory(messages, user_query)

        # Setup streaming
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)

            # Initialize streaming components
            heatmap = EnhancedHeatmap(self.tokenizer, use_background=False, window_size=3)
            token_confidences = []

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

            # Initialize response tracking
            complete_response = ""
            token_buffer = ""
            mcp_buffer = ""

            # Settings for token display
            last_print_time = time.time()
            force_display_time = 0.05  # 50ms max wait
            max_buffer_size = 16
            stop_sequences = ["<|user|>", "<|assistant|>"]

            # Variables to track streaming state
            tokens_received = 0
            low_confidence_detected = False

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

                # Check confidence to possibly stop generation
                if (response_filter is not None and
                    not low_confidence_detected and
                    tokens_received >= 10):  # Check after reasonable context

                    # Get current metrics
                    current_metrics = self.confidence_metrics.get_metrics(apply_sharpening=True)

                    # Should we show fallback instead of continuing?
                    if response_filter.should_stream_fallback(current_metrics, user_query):
                        # Set flag to avoid checking again
                        low_confidence_detected = True

                        # Stop generation
                        self.stop_event.set()

                        # Get fallback message
                        fallback = response_filter.get_streamable_fallback(user_query)
                        print(fallback, end="", flush=True)

                        # Update complete response
                        complete_response = fallback
                        break

                # Only add displayable tokens to the buffer
                if display_token and not low_confidence_detected:
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

                # Print buffer when it's full or enough time has passed
                if (len(token_buffer) >= max_buffer_size or
                    time_since_print >= force_display_time) and token_buffer:
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

                            # Make sure we have confidence metrics
                            self._ensure_confidence_metrics(10)

                            return self.mcp_handler.finalize_streaming(complete_response)

            # Handle any remaining tokens
            if token_buffer:
                print(token_buffer, end="", flush=True)

            # Ensure confidence metrics exist
            self._ensure_confidence_metrics(len(complete_response.split()))

            # Finalize the response
            response = self.mcp_handler.finalize_streaming(complete_response)

            # Post-process the response
            response = self._post_process_response(response)

            # Apply response filtering if provided and not already handled during streaming
            if response_filter is not None and not low_confidence_detected:
                current_metrics = self.confidence_metrics.get_metrics(apply_sharpening=True)
                # Update the user context with information about this exchange
                if hasattr(response_filter, 'update_user_context'):
                    response_filter.update_user_context(user_query, response, current_metrics)
                # Apply filtering
                response = response_filter.filter_response(
                    response=response,
                    metrics=current_metrics,
                    query=user_query,
                    preserve_mcp=True,
                    allow_override=True
                )

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

            # Show heatmap legend if enabled
            if show_confidence and not self.stop_event.is_set() and not low_confidence_detected:
                print()  # Add a newline
                heatmap.print_legend()

            # Clean up resources
            self.resource_manager.clear_cache()

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
        """Format combined memories with priority to command outputs."""
        # Separate command and general memories
        command_memories = []
        general_memories = []

        for memory in memories:
            if memory.get("metadata", {}).get("source") == "command_output":
                command_memories.append(memory)
            else:
                general_memories.append(memory)

        # Build formatted output
        output = "MEMORY CONTEXT:\n"

        # Add command outputs with special formatting
        if command_memories:
            output += "\nCOMMAND OUTPUT INFORMATION:\n"
            for i, memory in enumerate(command_memories):
                # Get command that produced this output
                command = memory.get("metadata", {}).get("command", "unknown command")
                content = memory.get("formatted_content", memory.get("content", ""))

                output += f"Output from command '{command}':\n"
                # Add indentation to command output for clarity
                indented_content = "\n".join(f"  {line}" for line in content.split("\n"))
                output += f"{indented_content}\n\n"

        # Add general memories
        if general_memories:
            output += "\nRELEVANT KNOWLEDGE:\n"
            for memory in general_memories:
                content = memory.get("content", "").strip()
                similarity = memory.get("similarity", 0)

                # Add the memory with its similarity score
                output += f"- [{similarity:.2f}] {content}\n"

        # Add instruction for using the memory
        output += "\nUse the information above to respond to the query. For table data, refer to specific columns and rows.\n"

        return output

    def _integrate_memory(self, messages: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
        """
        Integrate relevant memories into conversation messages.

        Args:
            messages: Current conversation messages
            query: Current user query

        Returns:
            Updated messages with memory context
        """
        try:
            # First check for command outputs related to the query
            command_memories = self.memory_manager.retrieve(
                query=query,
                top_k=2,
                min_similarity=0.1,
                metadata_filter={"source": "command_output"}
            )

            # Retrieve relevant memories
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
                combined_memories.append(memory)
                command_ids.append(memory["id"])

            # Add other relevant memories that aren't duplicates
            for memory in memories:
                if memory["id"] not in command_ids:
                    combined_memories.append(memory)

            # Add command memories first (they're more relevant for commands)
            for memory in command_memories:
                # Format based on content_type if available
                memory_type = memory.get("metadata", {}).get("content_type", "text")
                if memory_type == "tabular":
                    # Special formatting for tables
                    memory["formatted_content"] = self._format_tabular_memory(memory["content"])
                else:
                    memory["formatted_content"] = memory["content"]
                combined_memories.append(memory)

            # Then add general memories that aren't duplicates
            command_ids = [m["id"] for m in command_memories]

            for memory in memories:
                if memory["id"] not in command_ids:
                    combined_memories.append(memory)

            # Format memories for context if we have any
            if combined_memories:
                # Use custom formatter for combined memories
                memory_text = self._format_combined_memories(combined_memories, query)

                # Create memory-enhanced messages
                memory_enhanced_messages = messages.copy()

                # Extract system message
                original_system = messages[0]["content"]

                # Remove existing memory context if present
                if "MEMORY CONTEXT:" in original_system:
                    system_parts = original_system.split("MEMORY CONTEXT:")
                    system_content = system_parts[0].strip()
                else:
                    system_content = original_system

                # Add memory context
                system_with_memory = f"{system_content}\n\n{memory_text}"

                # Update system message
                memory_enhanced_messages[0]["content"] = system_with_memory

                # Update stats
                self.memory_stats["retrievals"] += 1

                return memory_enhanced_messages

            return messages
        except Exception as e:
            print(f"{self.get_time()} Error integrating memory: {e}")
            traceback.print_exc()
            return messages

    def _save_to_memory(self, query: str, response: str) -> bool:
        """
        Save conversation exchange to memory.

        Args:
            query: User query
            response: Generated response

        Returns:
            Success status
        """
        try:
            # Extract key information
            memories = self._extract_memory_worthy_content(query, response)

            if not memories:
                return False

            # Add each memory item
            memories_added = 0

            for memory_text, source_hint in memories:
                # Create metadata
                metadata = {
                    'source': 'conversation',
                    'source_hint': source_hint,
                    'query': query,
                    'timestamp': datetime.now().timestamp()
                }

                # Add to memory
                item_id = self.memory_manager.add(
                    content=memory_text,
                    metadata=metadata
                )

                if item_id:
                    memories_added += 1

            # Update stats
            if memories_added > 0:
                self.memory_stats["items_added"] += memories_added
                print(f"{self.get_time()} Added {memories_added} new memories")
                return True

            return False
        except Exception as e:
            print(f"{self.get_time()} Error saving to memory: {e}")
            return False

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
            memories.append((f"Q: {query}\nA: {response}", "qa_pair"))
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
        Generate a response with streaming output.

        Args:
            input_ids: Input token IDs
            streamer: Text streamer for output
            generation_config: Generation configuration
            use_draft_model: Whether to use draft model
        """
        try:
            # Reset confidence metrics
            if hasattr(self.confidence_metrics, 'reset'):
                self.confidence_metrics.reset()

            # Setup logits processor for confidence tracking
            from transformers import LogitsProcessorList

            streaming_config = {k: v for k, v in generation_config.items()
                               if k not in ['max_new_tokens', 'output_scores', 'return_dict_in_generate']}

            # Add TokenProbabilityCaptureProcessor for confidence tracking
            if 'logits_processor' not in streaming_config:
                streaming_config['logits_processor'] = LogitsProcessorList()

            # Create the processor for capturing token probabilities
            metrics_processor = TokenProbabilityCaptureProcessor(self.confidence_metrics)
            streaming_config['logits_processor'].append(metrics_processor)

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
                # Use standard generation
                self.model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    streamer=streamer,
                    max_new_tokens=generation_config.get('max_new_tokens', 128),
                    **streaming_config
                )

        except Exception as e:
            print(f"{self.get_time()} Error in generation thread: {str(e)}")
            traceback.print_exc()
            try:
                streamer.end()
            except Exception:
                pass

    def _get_domain_config(self, query, max_new_tokens=128, temperature=0.7):
        """
        Get domain-specific generation configuration.

        Args:
            query: User query
            max_new_tokens: Maximum tokens to generate
            temperature: Temperature for sampling

        Returns:
            Generation configuration dictionary
        """
        # Base configuration
        config = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.0,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True
        }

        # Set sampling parameters if needed
        if self.do_sample:
            config['do_sample'] = True
            config['top_k'] = self.top_k
            config['top_p'] = self.top_p
            config['temperature'] = temperature if temperature > 0.1 else 1.0
        else:
            config['do_sample'] = False
            del config['top_k']
            del config['top_k']
            del config['temperature']

        # Get domain settings if classifier is available
        if self.do_sample and hasattr(self, 'question_classifier') and self.question_classifier:
            settings = self.question_classifier.get_domain_settings(query)
            domain = settings['domain']

            # Apply domain-specific adjustments
            if domain == 'arithmetic':
                # More deterministic for math
                config['temperature'] = min(config.get('temperature', 0.7), 0.3)
                config['top_p'] = 0.85
            elif domain == 'translation':
                # More deterministic for translations
                config['temperature'] = min(config.get('temperature', 0.7), 0.4)
                config['top_p'] = 0.9
            elif domain == 'factual':
                # Slightly more deterministic for facts
                config['temperature'] = min(config.get('temperature', 0.7), 0.5)
            elif domain == 'conceptual' or domain == 'procedural':
                # More creative for concepts and procedures
                config['temperature'] = max(config.get('temperature', 0.7), 0.6)

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
            print(prompt)
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
                style=style
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
                print(f"{self.get_time()} Creating memory quickstart cache for next boot...")
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
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
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

    # Initialize response filter
    response_filter = ResponseFilter(
        confidence_threshold=args.confidence_threshold,
        user_context=user_context,
        question_classifier=chat.question_classifier,
        sharpening_factor=args.enhancement_factor
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
    print("Type 'exit' to end the conversation")
    print("Special commands:")
    print("  !system: [message] - Change the system message")
    print("  !toggle-memory - Toggle memory system on/off")
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

            # Handle special commands
            if user_input.lower() == 'exit':
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
                        response_filter.sharpening_factor = factor
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
                show_confidence=show_confidence,
                response_filter=response_filter if filter_enabled else None
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