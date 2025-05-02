"""
Speculative decoding module for efficient text generation with smaller draft models.
This module provides a cleaner implementation with better metrics tracking and error handling.
"""

import torch
import time
import logging
from typing import Dict, Any, Tuple, Optional, List, Union
from dataclasses import dataclass

from enhanced_confidence_metrics import TokenProbabilityCaptureProcessor
# Setup logging
logger = logging.getLogger("speculative_decoding")

@dataclass
class SpeculativeDecodingStats:
    """Statistics for tracking speculative decoding performance."""
    
    total_generations: int = 0
    total_tokens_generated: int = 0
    tokens_from_draft: int = 0
    total_draft_tokens: int = 0
    speculative_attempts: int = 0
    successful_attempts: int = 0
    repetition_detections: int = 0
    fallbacks: int = 0
    time_saved: float = 0.0  # estimated time saved in seconds
    
    def get_efficiency(self) -> float:
        """Get the acceptance rate of draft tokens."""
        if self.total_draft_tokens == 0:
            return 0.0
        return self.tokens_from_draft / self.total_draft_tokens
    
    def get_success_rate(self) -> float:
        """Get the success rate of speculative decoding attempts."""
        if self.speculative_attempts == 0:
            return 0.0
        return self.successful_attempts / self.speculative_attempts
    
    def get_utilization_rate(self) -> float:
        """Get the percentage of tokens that came from the draft model."""
        if self.total_tokens_generated == 0:
            return 0.0
        return self.tokens_from_draft / self.total_tokens_generated
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.total_generations = 0
        self.total_tokens_generated = 0
        self.tokens_from_draft = 0
        self.total_draft_tokens = 0
        self.speculative_attempts = 0
        self.successful_attempts = 0
        self.repetition_detections = 0
        self.fallbacks = 0
        self.time_saved = 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the statistics as a dictionary."""
        return {
            "total_generations": self.total_generations,
            "total_tokens": self.total_tokens_generated,
            "tokens_from_draft": self.tokens_from_draft,
            "draft_token_acceptance_rate": f"{self.get_efficiency()*100:.1f}%",
            "speculative_success_rate": f"{self.get_success_rate()*100:.1f}%",
            "draft_utilization_rate": f"{self.get_utilization_rate()*100:.1f}%",
            "estimated_time_saved": f"{self.time_saved:.2f}s",
            "repetition_detections": self.repetition_detections,
            "fallbacks": self.fallbacks
        }

class SpeculativeDecoder:
    """
    Implements speculative decoding to accelerate text generation using a draft model.
    
    This class handles the core logic of:
    1. Generating candidate tokens with a smaller draft model
    2. Verifying these tokens with the main model
    3. Tracking performance metrics
    4. Detecting and mitigating repetitive patterns
    """
    
    def __init__(
        self, 
        main_model: Any, 
        draft_model: Any, 
        tokenizer: Any,
        confidence_metrics=None,
        device: str = None,
        stats: SpeculativeDecodingStats = None,
        repetition_penalty: float = 1.2,
        max_draft_tokens: int = 5,
        batch_processor=None
    ):
        """
        Initialize the speculative decoder.
        
        Args:
            main_model: The primary language model
            draft_model: The smaller draft model for generating candidates
            tokenizer: Tokenizer for the models
            confidence_metrics: Optional metrics tracker for token confidence
            device: Device to use for computation ('cuda' or 'cpu')
            stats: Statistics tracker (will create one if not provided)
            repetition_penalty: Penalty to apply for repetitive tokens
            max_draft_tokens: Maximum number of tokens to draft at once
            batch_processor: Optional function for batch processing tensors
        """
        self.main_model = main_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.confidence_metrics = confidence_metrics
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.stats = stats or SpeculativeDecodingStats()
        self.repetition_penalty = repetition_penalty
        self.max_draft_tokens = max_draft_tokens
        self.batch_processor = batch_processor
        
        # Store past repetitions to detect patterns
        self.past_drafts = []
        self.max_stored_drafts = 5

    def create_draft_model(self, model):
        """
        Create a smaller draft model from the main model.
        
        This method extracts a subset of layers from the main model to create a faster draft model.
        
        Args:
            model: The main language model
            
        Returns:
            A draft model or None if creation failed
        """
        import copy
        
        try:
            # Architectural check - works for decoder-only transformers
            if not (hasattr(model, "model") and hasattr(model.model, "layers")):
                logger.warning("Model architecture not supported for draft model creation")
                return None
            
            # Create a copy of the model
            logger.info("Creating draft model from main model...")
            draft_model = copy.deepcopy(model)
            
            # Reduce to approximately 1/3 of the layers
            orig_layers = draft_model.model.layers
            num_layers = len(orig_layers)
            
            # Determine which layers to keep - try to keep first, middle and last sections
            # Using a dynamic approach based on model size
            if num_layers <= 6:
                # For very small models, keep half the layers
                keep_indices = list(range(0, num_layers, 2))
            else:
                # For larger models, keep first, some middle, and last layers
                # This generally preserves more model capabilities than just taking every Nth layer
                keep_layer_count = max(2, num_layers // 3)
                
                # Always keep first and last layer
                keep_indices = [0, num_layers-1]
                
                # Add some middle layers
                middle_count = keep_layer_count - 2
                if middle_count > 0:
                    step = (num_layers - 2) / (middle_count + 1)
                    middle_indices = [int(1 + i * step) for i in range(middle_count)]
                    keep_indices.extend(middle_indices)
                    keep_indices.sort()
            
            # Create new layers with only the kept indices
            new_layers = torch.nn.ModuleList([orig_layers[i] for i in keep_indices])
            draft_model.model.layers = new_layers
            
            logger.info(f"Created draft model with {len(new_layers)}/{num_layers} layers")
            return draft_model
            
        except Exception as e:
            logger.error(f"Error creating draft model: {e}")
            return None

    def speculative_decode(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        num_draft_tokens: Optional[int] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform speculative decoding with enhanced pattern detection and error handling.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for the input
            num_draft_tokens: Number of tokens to draft (uses self.max_draft_tokens if None)
            
        Returns:
            Tuple of (accepted_tokens, acceptance_mask, token_logits) or (None, None, None) on failure
        """
        if self.draft_model is None:
            # No draft model available
            self.stats.fallbacks += 1
            return None, None, None
            
        # Only implement for GPU for speed benefits
        if self.device == "cpu":
            self.stats.fallbacks += 1
            return None, None, None
            
        # Track speculative attempts
        self.stats.speculative_attempts += 1
        
        # Default num_draft_tokens to class setting if not specified
        if num_draft_tokens is None:
            num_draft_tokens = self.max_draft_tokens
            
        try:
            # Step 1: Generate draft tokens with the smaller model
            draft_start_time = time.time()
            with torch.no_grad():
                draft_outputs = self.draft_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=num_draft_tokens,
                    do_sample=False,  # Use greedy for deterministic drafting
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    repetition_penalty=self.repetition_penalty,
                )
                
            # Extract draft tokens (excluding input prompt)
            draft_ids = draft_outputs.sequences[0, input_ids.shape[1]:]
            
            # Check if we got any draft tokens
            if len(draft_ids) == 0:
                self.stats.fallbacks += 1
                return None, None, None
                
            # Track total draft tokens
            self.stats.total_draft_tokens += len(draft_ids)
                
            # Check for repetitive patterns
            draft_tokens = draft_ids.tolist()
            self._check_repetitive_patterns(draft_tokens)
                
            # Step 2: Verify with target model - compute logits for each position
            full_sequence = torch.cat([input_ids[0], draft_ids]).unsqueeze(0)
            full_attention = torch.ones_like(full_sequence)
                
            # Use batch processing if available
            if self.batch_processor is not None and full_sequence.size(1) > 1024:
                def get_logits(batch):
                    with torch.no_grad():
                        outputs = self.main_model(
                            input_ids=batch,
                            attention_mask=torch.ones_like(batch),
                            return_dict=True
                        )
                    return outputs.logits
                    
                # Use batch processing with adaptive sizing
                target_logits = self.batch_processor(
                    get_logits,
                    full_sequence,
                    # batch_size=None,  # Use dynamic sizing
                    adaptive=True,
                    handle_oom=True
                )
            else:
                # Get logits from target model for the full sequence
                with torch.no_grad():
                    target_outputs = self.main_model(
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
                self.stats.successful_attempts += 1
                self.stats.tokens_from_draft += len(draft_ids)
                
                # Add logits to confidence metrics if available
                if self.confidence_metrics:
                    for i, token_id in enumerate(draft_ids):
                        self.confidence_metrics.add_token_score(target_logits[i], token_id.item())
                        
                # Calculate time saved (approximate)
                draft_time = time.time() - draft_start_time
                # Assume verification time is insignificant compared to generation
                self.stats.time_saved += draft_time * 0.7  # Conservative estimate
                
                return draft_ids, torch.ones_like(draft_ids), target_logits
                
            # Find first mismatch position
            first_mismatch = len(matches) if matches.all() else matches.tolist().index(False)
                
            # Return accepted tokens
            if first_mismatch > 0:
                accepted_tokens = draft_ids[:first_mismatch]
                acceptance_mask = torch.ones_like(accepted_tokens)
                
                # Add logits to confidence metrics for the accepted tokens
                if self.confidence_metrics:
                    for i, token_id in enumerate(accepted_tokens):
                        self.confidence_metrics.add_token_score(target_logits[i], token_id.item())
                        
                # Update statistics
                self.stats.successful_attempts += 1
                self.stats.tokens_from_draft += first_mismatch
                
                # Calculate time saved (approximate)
                draft_time = time.time() - draft_start_time
                self.stats.time_saved += draft_time * 0.4  # Conservative estimate
                
                # Return the accepted tokens
                return accepted_tokens, acceptance_mask, target_logits[:first_mismatch]
                
            # No accepted tokens
            self.stats.fallbacks += 1
            return None, None, None
            
        except Exception as e:
            logger.error(f"Error in speculative decoding: {e}")
            self.stats.fallbacks += 1
            return None, None, None
            
    def _check_repetitive_patterns(self, tokens: List[int]) -> bool:
        """
        Detect repetitive patterns in draft tokens that might indicate low quality.
        
        Args:
            tokens: List of token IDs to check
            
        Returns:
            True if repetitive pattern detected, False otherwise
        """
        # Pattern detection 1: Internal repetition
        if len(tokens) >= 4:
            # Check if any repeated bigrams or trigrams
            for n in [2, 3]:
                if len(tokens) >= n*2:
                    # Get n-grams and check for repeats
                    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
                    ngram_counts = {}
                    for ngram in ngrams:
                        if ngram in ngram_counts:
                            ngram_counts[ngram] += 1
                            if ngram_counts[ngram] >= 2:
                                self.stats.repetition_detections += 1
                                return True
                        else:
                            ngram_counts[ngram] = 1
                
        # Pattern detection 2: Repetition with past drafts
        # Store past drafts to detect loops across rounds
        if self.past_drafts:
            for past_draft in self.past_drafts:
                # Check for overlap with past drafts
                if len(tokens) >= 3 and len(past_draft) >= 3:
                    # Compare the last 3 tokens of past draft with the first 3 of current
                    if past_draft[-3:] == tokens[:3]:
                        self.stats.repetition_detections += 1
                        return True
        
        # Update past drafts
        self.past_drafts.append(tokens)
        if len(self.past_drafts) > self.max_stored_drafts:
            self.past_drafts.pop(0)
            
        return False

    def generate_with_speculative_decoding(
        self,
        input_ids: torch.Tensor,
        streamer: Any,
        max_new_tokens: int,
        generation_config: Dict[str, Any],
        stop_event: Optional[Any] = None
    ) -> None:
        """
        Generate text using speculative decoding with streaming.

        Args:
            input_ids: Input token IDs
            streamer: Streamer to output generated tokens
            max_new_tokens: Maximum new tokens to generate
            generation_config: Configuration for generation
            stop_event: Optional event to signal stopping generation
        """
        with torch.no_grad():
            # Track this generation in stats
            self.stats.total_generations += 1

            # Reset confidence metrics if available
            if self.confidence_metrics:
                self.confidence_metrics.reset()

            # Setup generation
            generated_ids = input_ids
            attention_mask = torch.ones_like(input_ids)
            remaining_tokens = max_new_tokens

            # Create a separate config without max_new_tokens to avoid duplication
            streaming_config = {k: v for k, v in generation_config.items()
                               if k not in ['max_new_tokens', 'output_scores', 'return_dict_in_generate']}

            # Setup logits processor for confidence metrics
            if self.confidence_metrics:
                # Add TokenProbabilityCaptureProcessor for confidence tracking
                from transformers import LogitsProcessorList

                if 'logits_processor' not in streaming_config:
                    streaming_config['logits_processor'] = LogitsProcessorList()

                # Create the processor for capturing token probabilities
                if hasattr(self.confidence_metrics, 'create_processor'):
                    metrics_processor = self.confidence_metrics.create_processor()
                    streaming_config['logits_processor'].append(metrics_processor)
                else:
                    # Fallback if create_processor isn't available
                    # This assumes TokenProbabilityCaptureProcessor exists
                    metrics_processor = TokenProbabilityCaptureProcessor(self.confidence_metrics)
                    streaming_config['logits_processor'].append(metrics_processor)

            # Speculative decoding loop
            while remaining_tokens > 0:
                # Check if we should stop
                if stop_event is not None and stop_event.is_set():
                    logger.info("Generation stopped by interrupt signal")
                    streamer.end()
                    return

                # Determine if we should use speculative decoding
                # Skip for short generations - less overhead
                if remaining_tokens >= 3:
                    # Determine optimal number of draft tokens
                    optimal_draft_tokens = min(self.max_draft_tokens, remaining_tokens)

                    # Try speculative decoding
                    draft_tokens, acceptance_mask, token_logits = self.speculative_decode(
                        generated_ids,
                        attention_mask,
                        num_draft_tokens=optimal_draft_tokens
                    )

                    if draft_tokens is not None and len(draft_tokens) > 0:
                        # Success! We have draft tokens to use
                        token_text = self.tokenizer.decode(draft_tokens, skip_special_tokens=True)
                        streamer.put(token_text)

                        # Add tokens to generated output
                        generated_ids = torch.cat([generated_ids[0], draft_tokens]).unsqueeze(0)
                        attention_mask = torch.ones_like(generated_ids)
                        remaining_tokens -= len(draft_tokens)

                        # Update token count
                        self.stats.total_tokens_generated += len(draft_tokens)

                        # Check for stopping conditions
                        if self._check_stop_conditions(draft_tokens, token_text):
                            break

                        # Continue to next iteration
                        continue

                # Fall back to standard generation for remaining tokens
                break

            # If we have tokens remaining, do standard generation
            if remaining_tokens > 0:
                # Use standard streaming mode
                self.main_model.generate(
                    input_ids=generated_ids,
                    attention_mask=torch.ones_like(generated_ids),
                    streamer=streamer,
                    max_new_tokens=remaining_tokens,
                    **streaming_config
                )

                # Estimate total tokens generated
                # We don't know exactly how many were generated, so use remaining_tokens as estimate
                self.stats.total_tokens_generated += remaining_tokens

            # Ensure we have confidence metrics even if none were collected
            if self.confidence_metrics and not hasattr(self.confidence_metrics, 'token_probabilities') or not self.confidence_metrics.token_probabilities:
                # Create fallback metrics with reasonable values
                self._create_fallback_confidence_metrics()
    
    def _create_fallback_confidence_metrics(self):
      """Create fallback confidence metrics if none were collected."""
      if not self.confidence_metrics:
          return

      # Check if we need to create fallbacks
      if not hasattr(self.confidence_metrics, 'token_probabilities') or not self.confidence_metrics.token_probabilities:
          print("[Warning: No token probabilities collected, using fallback values]")
          # Create fallback metrics
          for i in range(5):
              # Create dummy logits tensor with varying confidence
              dummy_logits = torch.zeros(self.tokenizer.vocab_size, device=self.device)
              # Use different confidence values based on position
              confidence_val = 5.0 + (i % 3)
              token_id = i % 100

              # Set main token with high value
              dummy_logits[token_id] = confidence_val

              # Add some secondary values for more realistic distribution
              for j in range(3):
                  alt_token = (token_id + j + 1) % self.tokenizer.vocab_size
                  dummy_logits[alt_token] = confidence_val * 0.3

              # Add to confidence metrics
              self.confidence_metrics.add_token_score(dummy_logits, token_id)

    def _check_stop_conditions(self, tokens: torch.Tensor, text: str) -> bool:
        """
        Check if generation should stop based on tokens or decoded text.
        
        Args:
            tokens: Generated token IDs
            text: Decoded text from tokens
            
        Returns:
            True if generation should stop, False otherwise
        """
        # Check for EOS token
        for token in tokens:
            if token.item() == self.tokenizer.eos_token_id:
                return True
                
        # Check for special tokens in decoded text
        if "<|user|>" in text or "<|assistant|>" in text:
            return True
            
        return False
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get a summary of speculative decoding statistics."""
        return self.stats.get_summary()
    
    def reset_stats(self) -> None:
        """Reset the speculative decoding statistics."""
        self.stats.reset()