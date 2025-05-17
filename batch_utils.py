"""
Enhanced batch processing utilities for TinyLlama Chat system.
Provides functions for efficient processing of large tensor operations with dynamic batch sizing.
"""

import torch
import gc
import numpy as np
import time
import logging
from typing import Callable, Any, List, Optional, Tuple, Dict, Union

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('batch_utils')

def estimate_optimal_batch_size(tensor_shape: Tuple[int, ...],
                               dtype: torch.dtype = torch.float32,
                               target_memory_usage: float = 0.7,
                               min_batch_size: int = 1,
                               max_batch_size: int = 32) -> int:
    """
    Estimate the optimal batch size based on tensor shape and available GPU memory.

    Args:
        tensor_shape: Shape of a single item in the batch
        dtype: Tensor data type
        target_memory_usage: Target fraction of available memory to use (0.0-1.0)
        min_batch_size: Minimum batch size to return
        max_batch_size: Maximum batch size to return

    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        return max_batch_size  # On CPU, default to max batch size

    try:
        # Get available GPU memory
        device = torch.cuda.current_device()
        available_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = available_memory - allocated_memory

        # Calculate memory per item based on tensor shape and dtype
        element_size = torch.tensor([], dtype=dtype).element_size()
        elements_per_item = np.prod(tensor_shape)
        memory_per_item = element_size * elements_per_item

        # Calculate optimal batch size (with 20% buffer for workspace)
        target_memory = free_memory * target_memory_usage * 0.8
        optimal_batch_size = max(min_batch_size, min(max_batch_size,
                                               int(target_memory / memory_per_item)))

        logger.debug(f"Estimated optimal batch size: {optimal_batch_size} for shape {tensor_shape}")
        return optimal_batch_size

    except Exception as e:
        logger.warning(f"Error estimating batch size: {e}, falling back to default")
        return 8  # Default fallback

def batch_embed_texts(
    texts: List[str],
    tokenizer,
    model,
    device: str = "cuda",
    batch_size: int = 32,
    adaptive: bool = True,
    handle_oom: bool = True,
    max_length: int = 512
) -> List[np.ndarray]:
    """
    Universal batch embedding function for all text embedding operations.
    Updated to work with various model architectures.

    Args:
        texts: List of texts to embed
        tokenizer: Tokenizer to use
        model: Model to generate embeddings
        device: Device to use (cuda/cpu)
        batch_size: Size of batches (calculated dynamically if adaptive is True)
        adaptive: Whether to adapt batch size based on memory
        handle_oom: Whether to handle OOM errors by reducing batch size
        max_length: Maximum token length for truncation

    Returns:
        List of embedding arrays
    """
    if not texts:
        return []

    # Define the embedding operation that works with different architectures
    def batch_embedding_operation(batch_tokenized):
        with torch.no_grad():
            # Get model outputs
            outputs = model(**batch_tokenized)

            # Different models expose embeddings differently
            if hasattr(outputs, 'last_hidden_state'):
                # For models like BERT, RoBERTa, T5, etc.
                hidden_states = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # For models that return hidden states in a tuple
                hidden_states = outputs.hidden_states[-1]  # Use the last layer
            else:
                # For causal LM models like GPT, BLOOM, etc.
                # Extract the embeddings from the model itself
                if hasattr(model, 'transformer'):
                    # Run the transformer directly
                    transformer_outputs = model.transformer(**batch_tokenized)
                    if hasattr(transformer_outputs, 'last_hidden_state'):
                        hidden_states = transformer_outputs.last_hidden_state
                    else:
                        # Fall back to logits
                        logits = outputs.logits
                        # Convert to simple embedding by mean pooling
                        hidden_states = torch.mean(logits, dim=1, keepdim=True).transpose(1, 2)
                elif hasattr(model, 'model') and hasattr(model.model, 'encoder'):
                    # Try to access encoder outputs for encoder-decoder models
                    encoder_outputs = model.model.encoder(**batch_tokenized)
                    hidden_states = encoder_outputs.last_hidden_state
                else:
                    # Last resort - use logits
                    logits = outputs.logits
                    hidden_states = torch.mean(logits, dim=1, keepdim=True).transpose(1, 2)

            # Mean pooling - take attention mask into account for weighted averaging
            attention_mask = batch_tokenized.get('attention_mask', torch.ones_like(batch_tokenized['input_ids'])).to(device)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            return mean_pooled

    try:
        # Tokenize all texts
        batch_tokenized = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(device)

        # Estimate optimal batch size if adaptive
        if adaptive:
            tensor_shape = tuple(batch_tokenized['input_ids'].shape[1:])
            batch_size = estimate_optimal_batch_size(
                tensor_shape=tensor_shape,
                dtype=torch.float16 if device == "cuda" else torch.float32,
                max_batch_size=batch_size
            )

        # Process in batches efficiently
        pooled_outputs = tensor_batch_processing(
            tensor_op=batch_embedding_operation,
            input_tensor=batch_tokenized,
            batch_size=batch_size,
            cleanup=True,
            adaptive=adaptive,
            handle_oom=handle_oom
        )

        # Convert to numpy arrays
        if isinstance(pooled_outputs, torch.Tensor):
            embeddings = pooled_outputs.cpu().numpy()
            return list(embeddings)
        else:
            # Handle case where tensor_batch_processing returns a list
            return [output.cpu().numpy() for output in pooled_outputs]

    except Exception as e:
        print(f"Error in batch embedding: {e}")
        import traceback
        traceback.print_exc()

        # Fall back to individual processing
        return [embed_single_text(text, tokenizer, model, device, max_length) for text in texts]

def embed_single_text(text: str, tokenizer, model, device: str = "cuda", max_length: int = 512) -> np.ndarray:
    """Single text embedding function that works with various model architectures."""
    try:
        with torch.no_grad():
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(device)

            # Get model outputs
            outputs = model(**inputs)

            # Different models expose embeddings differently
            if hasattr(outputs, 'last_hidden_state'):
                # For models like BERT, RoBERTa, T5, etc.
                hidden_states = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # For models that return hidden states in a tuple
                hidden_states = outputs.hidden_states[-1]  # Use the last layer
            else:
                # For causal LM models like GPT, BLOOM, etc.
                # Extract the embeddings from the model itself
                if hasattr(model, 'transformer'):
                    # Run the transformer directly to get embeddings
                    transformer_outputs = model.transformer(**inputs)
                    if hasattr(transformer_outputs, 'last_hidden_state'):
                        hidden_states = transformer_outputs.last_hidden_state
                    else:
                        # Fall back to logits
                        logits = outputs.logits
                        # Convert to a simple embedding by mean pooling
                        hidden_states = torch.mean(logits, dim=1, keepdim=True).transpose(1, 2)
                elif hasattr(model, 'model') and hasattr(model.model, 'encoder'):
                    # Try to access encoder outputs for encoder-decoder models
                    encoder_outputs = model.model.encoder(**inputs)
                    hidden_states = encoder_outputs.last_hidden_state
                else:
                    # Last resort fallback - use logits
                    logits = outputs.logits
                    hidden_states = torch.mean(logits, dim=1, keepdim=True).transpose(1, 2)

            # Get mean pooled representation
            # Mean pooling - take attention mask into account for weighted averaging
            attention_mask = inputs.get('attention_mask', torch.ones_like(inputs['input_ids']))
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask

            return embedding.cpu().numpy()[0]

    except Exception as e:
        print(f"Error generating embedding: {e}")
        import traceback
        traceback.print_exc()

        # Get embedding dimension from model config
        if hasattr(model, 'config'):
            if hasattr(model.config, 'hidden_size'):
                dim = model.config.hidden_size
            elif hasattr(model.config, 'd_model'):
                dim = model.config.d_model
            else:
                dim = 2048  # Default fallback
        else:
            dim = 2048

        # Return a zero vector as fallback
        return np.zeros(dim)
        
def tensor_batch_processing(tensor_op: Callable[[torch.Tensor], torch.Tensor],
                          input_tensor: torch.Tensor,
                          batch_dim: int = 0,
                          batch_size: Optional[int] = None,
                          cleanup: bool = True,
                          adaptive: bool = True,
                          handle_oom: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Enhanced: Process a large tensor in batches with adaptive sizing and better error handling.

    Args:
        tensor_op: Function that processes a tensor batch
        input_tensor: The input tensor to process
        batch_dim: Dimension to batch along
        batch_size: Size of batches (will be calculated dynamically if None and adaptive=True)
        cleanup: Whether to run cleanup between batches
        adaptive: Whether to adapt batch size dynamically
        handle_oom: Whether to handle OOM errors by reducing batch size

    Returns:
        Processed tensor with results combined, or list of tensors if they cannot be combined
    """
    # Get total size along batch dimension
    total_size = input_tensor.shape[batch_dim]

    # Determine batch size if not provided and adaptive is True
    if batch_size is None and adaptive:
        # Get shape of a single item
        item_shape = list(input_tensor.shape)
        item_shape[batch_dim] = 1
        batch_size = estimate_optimal_batch_size(tuple(item_shape), input_tensor.dtype)
        logger.info(f"Using dynamically calculated batch size: {batch_size}")
    elif batch_size is None:
        # Default batch size if not adapting
        batch_size = 8

    # Track performance
    start_time = time.time()
    processed_items = 0

    # Create list for output batches
    output_batches = []

    # Process in batches
    current_batch_size = batch_size
    start_idx = 0

    while start_idx < total_size:
        batch_start_time = time.time()

        # Get end index for current batch
        end_idx = min(start_idx + current_batch_size, total_size)
        actual_batch_size = end_idx - start_idx

        # Create indexing tuple to select along the batch dimension
        idx = [slice(None)] * input_tensor.ndim
        idx[batch_dim] = slice(start_idx, end_idx)

        # Get batch
        batch = input_tensor[tuple(idx)]

        try:
            # Process batch
            processed_batch = tensor_op(batch)

            # Save batch result
            output_batches.append(processed_batch)

            # Update tracking
            processed_items += actual_batch_size
            batch_duration = time.time() - batch_start_time

            if batch_duration > 0:
                logger.debug(f"Processed batch {start_idx}:{end_idx} ({actual_batch_size} items) "
                           f"in {batch_duration:.2f}s ({actual_batch_size/batch_duration:.1f} items/s)")

            # Move to next batch
            start_idx = end_idx

            # Clean up memory if requested
            if cleanup:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except RuntimeError as e:
            # Handle out of memory errors if enabled
            if handle_oom and "CUDA out of memory" in str(e) and current_batch_size > 1:
                # Reduce batch size by half and retry
                current_batch_size = max(1, current_batch_size // 2)

                # Clean up the failed attempt
                del batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.warning(f"OOM error, reducing batch size to {current_batch_size} and retrying")

                # Don't advance start_idx so we retry with smaller batch
            else:
                # Re-raise if not OOM or if we can't reduce batch size further
                raise

    # Log overall performance
    total_duration = time.time() - start_time
    if total_duration > 0:
        logger.info(f"Processed {processed_items} items in {total_duration:.2f}s "
                   f"({processed_items/total_duration:.1f} items/s)")

    # Combine results
    try:
        # Try to concatenate along batch dimension
        result = torch.cat(output_batches, dim=batch_dim)
        return result
    except Exception as e:
        logger.warning(f"Could not concatenate results: {e}")
        # If concatenation fails, just return the list of batches
        return output_batches

def batch_embedding_processing(embedding_function: Callable[[str], np.ndarray],
                             texts: List[str],
                             batch_size: Optional[int] = None,
                             cleanup: bool = True,
                             parallel: bool = False,
                             max_workers: int = 4) -> List[np.ndarray]:
    """
    Enhanced: Process a list of texts to generate embeddings in batches with better performance.

    Args:
        embedding_function: Function that processes a text to an embedding
        texts: List of texts to embed
        batch_size: Size of batches (dynamically calculated if None)
        cleanup: Whether to run cleanup between batches
        parallel: Whether to process batches in parallel
        max_workers: Maximum number of worker threads if parallel is True

    Returns:
        List of embedding arrays
    """
    import math

    # Calculate batch size if not provided
    if batch_size is None:
        # Estimate based on text lengths
        avg_length = sum(len(text) for text in texts) / max(1, len(texts))
        # Heuristic: longer texts need smaller batches
        if avg_length > 1000:
            batch_size = 4
        elif avg_length > 500:
            batch_size = 8
        elif avg_length > 200:
            batch_size = 16
        else:
            batch_size = 32

        logger.info(f"Using batch size {batch_size} for texts with avg length {avg_length:.1f}")

    # Create list for output embeddings
    output_embeddings = []
    total_texts = len(texts)

    # For parallel processing
    if parallel and total_texts > 1:
        import concurrent.futures

        # Adjust max_workers based on batch count
        batch_count = math.ceil(total_texts / batch_size)
        actual_workers = min(max_workers, batch_count)

        logger.info(f"Processing {batch_count} batches with {actual_workers} workers")

        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Function to process a single batch
            def process_batch(batch_texts):
                batch_embeddings = []
                for text in batch_texts:
                    try:
                        embedding = embedding_function(text)
                        batch_embeddings.append(embedding)
                    except Exception as e:
                        logger.error(f"Error generating embedding: {e}")
                        # Use zeros as fallback
                        if len(output_embeddings) > 0:
                            # Use same shape as previous embeddings
                            embedding = np.zeros_like(output_embeddings[0])
                        else:
                            # Assume 384-dimensional embeddings if we don't know
                            embedding = np.zeros(384)
                        batch_embeddings.append(embedding)
                return batch_embeddings

            # Submit batch processing tasks
            futures = []
            for start_idx in range(0, total_texts, batch_size):
                end_idx = min(start_idx + batch_size, total_texts)
                batch_texts = texts[start_idx:end_idx]
                futures.append(executor.submit(process_batch, batch_texts))

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                output_embeddings.extend(future.result())

                # Run cleanup if needed
                if cleanup:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    else:
        # Process in batches sequentially
        start_time = time.time()
        processed_items = 0

        for start_idx in range(0, total_texts, batch_size):
            batch_start_time = time.time()

            # Get end index for current batch
            end_idx = min(start_idx + batch_size, total_texts)
            actual_batch_size = end_idx - start_idx

            # Get batch
            batch = texts[start_idx:end_idx]

            # Process each text in the batch
            batch_embeddings = []
            for text in batch:
                try:
                    embedding = embedding_function(text)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Error generating embedding: {e}")
                    # Use zeros as fallback
                    if len(output_embeddings) > 0:
                        # Use same shape as previous embeddings
                        embedding = np.zeros_like(output_embeddings[0])
                    else:
                        # Assume 384-dimensional embeddings if we don't know
                        embedding = np.zeros(384)
                    batch_embeddings.append(embedding)

            # Add to output
            output_embeddings.extend(batch_embeddings)

            # Update tracking
            processed_items += actual_batch_size
            batch_duration = time.time() - batch_start_time

            if batch_duration > 0:
                logger.debug(f"Processed batch {start_idx}:{end_idx} ({actual_batch_size} items) "
                           f"in {batch_duration:.2f}s ({actual_batch_size/batch_duration:.1f} items/s)")

            # Clean up memory if requested
            if cleanup:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Log overall performance
        total_duration = time.time() - start_time
        if total_duration > 0:
            logger.info(f"Processed {processed_items} texts in {total_duration:.2f}s "
                       f"({processed_items/total_duration:.1f} texts/s)")

    return output_embeddings

def validate_batch_processing_performance(test_function: Callable,
                                        test_data: Any,
                                        batch_sizes: List[int] = [1, 2, 4, 8, 16, 32]):
    """
    Validate batch processing performance by testing different batch sizes.

    Args:
        test_function: Function to test with various batch sizes
        test_data: Data to process in batches
        batch_sizes: List of batch sizes to test

    Returns:
        Dictionary with performance metrics for each batch size
    """
    results = {}

    logger.info(f"Testing {len(batch_sizes)} batch sizes: {batch_sizes}")

    for batch_size in batch_sizes:
        try:
            logger.info(f"Testing batch size: {batch_size}")

            # Ensure clean start
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                start_mem = torch.cuda.memory_allocated()

            # Test performance
            start_time = time.time()
            _ = test_function(test_data, batch_size)
            duration = time.time() - start_time

            # Get memory usage
            if torch.cuda.is_available():
                end_mem = torch.cuda.memory_allocated()
                peak_mem = torch.cuda.max_memory_allocated()
                memory_used = peak_mem - start_mem
            else:
                memory_used = 0

            # Store results
            results[batch_size] = {
                'duration': duration,
                'memory_used': memory_used,
                'items_per_second': len(test_data) / duration if duration > 0 else 0
            }

            logger.info(f"Batch size {batch_size}: {duration:.2f}s, "
                       f"{memory_used/(1024*1024):.1f}MB, "
                       f"{results[batch_size]['items_per_second']:.1f} items/s")

        except Exception as e:
            logger.error(f"Error testing batch size {batch_size}: {e}")
            results[batch_size] = {'error': str(e)}

    # Find optimal batch size
    valid_sizes = [size for size in batch_sizes if size in results and 'error' not in results[size]]
    if valid_sizes:
        optimal_size = max(valid_sizes, key=lambda s: results[s]['items_per_second'])
        logger.info(f"Optimal batch size: {optimal_size} "
                   f"({results[optimal_size]['items_per_second']:.1f} items/s)")
    else:
        logger.warning("Could not determine optimal batch size")

    return results