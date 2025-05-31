"""
Consolidated batch processing utilities for TinyLlama Chat system.
Combines functionality from batch_utils.py, resource_manager.py batch methods,
and memory system batch processing into a unified system.
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

class BatchProcessor:
    """
    Unified batch processing system with configuration management,
    performance optimization, and resource-aware processing.
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize the batch processor.

        Args:
            device: Device to use ('cuda', 'cpu', or 'auto' for detection)
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_cuda = self.device.startswith("cuda")

        # Batch configuration for different operation types
        self.batch_settings = {
            'embedding': {
                'batch_size': 8,
                'cleanup': True,
                'adaptive': True,
                'parallel': False,
                'max_workers': 4,
                'target_memory_usage': 0.6
            },
            'inference': {
                'batch_size': 4,
                'cleanup': True,
                'adaptive': True,
                'handle_oom': True,
                'target_memory_usage': 0.5
            },
            'memory_processing': {
                'batch_size': 32,
                'cleanup': True,
                'adaptive': True,
                'max_length': 512,
                'target_memory_usage': 0.7
            }
        }

        # Performance tracking
        self.performance_history = {}

    def get_batch_settings(self, operation_type: str) -> dict:
        """
        Get batch settings for a specific operation type.

        Args:
            operation_type: Type of operation ('embedding', 'inference', 'memory_processing')

        Returns:
            Dictionary of batch settings
        """
        return self.batch_settings.get(operation_type, self.batch_settings['embedding']).copy()

    def update_batch_settings(self, operation_type: str, settings: dict):
        """
        Update batch settings for an operation type.

        Args:
            operation_type: Type of operation
            settings: Settings to update
        """
        if operation_type in self.batch_settings:
            self.batch_settings[operation_type].update(settings)
        else:
            self.batch_settings[operation_type] = settings

    def estimate_optimal_batch_size(self,
                                   tensor_shape: Tuple[int, ...],
                                   dtype: torch.dtype = torch.float32,
                                   operation_type: str = 'embedding',
                                   min_batch_size: int = 1,
                                   max_batch_size: int = 32) -> int:
        """
        Estimate optimal batch size based on tensor shape and available memory.

        Args:
            tensor_shape: Shape of a single item in the batch
            dtype: Tensor data type
            operation_type: Type of operation for target memory usage
            min_batch_size: Minimum batch size to return
            max_batch_size: Maximum batch size to return

        Returns:
            Optimal batch size
        """
        if not self.is_cuda:
            return max_batch_size  # On CPU, default to max batch size

        try:
            # Get target memory usage from settings
            settings = self.get_batch_settings(operation_type)
            target_memory_usage = settings.get('target_memory_usage', 0.7)

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

            logger.debug(f"Estimated optimal batch size: {optimal_batch_size} for {operation_type}")
            return optimal_batch_size

        except Exception as e:
            logger.warning(f"Error estimating batch size: {e}, falling back to default")
            return self.batch_settings.get(operation_type, {}).get('batch_size', 8)

    def batch_embed_texts(self,
                         texts: List[str],
                         tokenizer,
                         model,
                         device: str = None,
                         batch_size: int = None,
                         adaptive: bool = None,
                         handle_oom: bool = True,
                         max_length: int = 512,
                         target_dim: Optional[int] = None) -> List[np.ndarray]:
        """
        Generate embeddings for texts in batches with adaptive sizing.

        Args:
            texts: List of texts to embed
            tokenizer: Tokenizer to use
            model: Model to generate embeddings
            device: Device override (uses self.device if None)
            batch_size: Batch size override
            adaptive: Adaptive sizing override
            handle_oom: Whether to handle OOM errors
            max_length: Maximum token length
            target_dim: Target embedding dimension

        Returns:
            List of embedding arrays
        """
        device = device or self.device
        settings = self.get_batch_settings('embedding')

        # Apply overrides
        batch_size = batch_size or settings['batch_size']
        adaptive = adaptive if adaptive is not None else settings['adaptive']

        if not texts:
            return []

        # For very few texts, process individually
        if len(texts) <= 4:
            return [self.embed_single_text(text, tokenizer, model, device, max_length, target_dim)
                   for text in texts]

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
                batch_size = self.estimate_optimal_batch_size(
                    tensor_shape=tensor_shape,
                    dtype=torch.float16 if device == "cuda" else torch.float32,
                    operation_type='embedding',
                    max_batch_size=batch_size
                )

            # Process in batches if needed
            if len(texts) > batch_size:
                return self._process_embedding_batches(
                    texts, tokenizer, model, device, batch_size,
                    handle_oom, max_length, target_dim
                )

            # Process as single batch
            return self._process_single_embedding_batch(
                batch_tokenized, target_dim, handle_oom
            )

        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            # Fall back to individual processing
            return [self.embed_single_text(text, tokenizer, model, device, max_length, target_dim)
                   for text in texts]

    def _process_embedding_batches(self, texts, tokenizer, model, device,
                                  batch_size, handle_oom, max_length, target_dim):
        """Process embeddings in multiple batches."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            batch_texts = texts[i:end_idx]

            # Tokenize this batch
            batch_inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(device)

            try:
                # Process this batch
                batch_results = self._extract_embeddings_from_batch(batch_inputs)
                batch_np = batch_results.cpu().numpy()

                # Apply target dimension if needed
                if target_dim is not None:
                    batch_np = self._resize_embeddings(batch_np, target_dim)

                all_embeddings.extend(batch_np)

                # Clean up
                del batch_inputs, batch_results
                if self.is_cuda:
                    torch.cuda.empty_cache()

            except Exception as batch_error:
                logger.error(f"Error processing batch {i}-{end_idx}: {batch_error}")
                if handle_oom and "out of memory" in str(batch_error).lower():
                    # Fall back to individual processing for this batch
                    batch_embeddings = [
                        self.embed_single_text(text, tokenizer, model, device, max_length, target_dim)
                        for text in batch_texts
                    ]
                    all_embeddings.extend(batch_embeddings)
                else:
                    raise

        return all_embeddings

    def _process_single_embedding_batch(self, batch_tokenized, target_dim, handle_oom):
        """Process a single embedding batch."""
        try:
            embeddings = self._extract_embeddings_from_batch(batch_tokenized)
            embeddings_np = embeddings.cpu().numpy()

            # Apply target dimension if needed
            if target_dim is not None:
                embeddings_np = self._resize_embeddings(embeddings_np, target_dim)

            return list(embeddings_np)

        except Exception as e:
            if handle_oom and "out of memory" in str(e).lower():
                logger.warning("OOM in single batch, falling back to individual processing")
                # This should not happen, but fallback anyway
                raise
            else:
                raise

    def _extract_embeddings_from_batch(self, batch_inputs):
        """Extract embeddings from a batch of tokenized inputs."""
        with torch.no_grad():
            outputs = batch_inputs.get('model')(**batch_inputs, output_hidden_states=True)

            # Strategy 1: Use last_hidden_state with attention masking
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
                attention_mask = batch_inputs.get('attention_mask',
                                                torch.ones_like(batch_inputs['input_ids']))

                # Masked average pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

            # Strategy 2: Use hidden_states if available
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                last_layer = outputs.hidden_states[-1]
                attention_mask = batch_inputs.get('attention_mask',
                                                torch.ones_like(batch_inputs['input_ids']))

                # Masked average pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_layer.size()).float()
                sum_embeddings = torch.sum(last_layer * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

            # Strategy 3: Use logits as fallback
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
                if len(logits.shape) == 3:  # [batch, seq_len, vocab]
                    embeddings = logits[:, -1, :]  # Use last token
                else:  # [batch, features]
                    embeddings = logits

                # Normalize
                norms = torch.norm(embeddings, dim=1, keepdim=True)
                embeddings = embeddings / (norms + 1e-8)

            else:
                raise ValueError("Could not extract embeddings from model outputs")

            return embeddings

    def _resize_embeddings(self, embeddings_np, target_dim):
        """Resize embeddings to target dimension."""
        if embeddings_np.shape[1] == target_dim:
            return embeddings_np

        resized_embeddings = []
        for embedding in embeddings_np:
            if len(embedding) > target_dim:
                # Truncate
                resized = embedding[:target_dim]
            else:
                # Pad with zeros
                resized = np.pad(embedding, (0, target_dim - len(embedding)))

            # Normalize
            norm = np.linalg.norm(resized)
            if norm > 1e-10:
                resized = resized / norm

            resized_embeddings.append(resized)

        return resized_embeddings

    def embed_single_text(self,
                         text: str,
                         tokenizer,
                         model,
                         device: str = None,
                         max_length: int = 512,
                         target_dim: Optional[int] = None) -> np.ndarray:
        """
        Generate embedding for a single text with model compatibility.

        Args:
            text: Text to embed
            tokenizer: Tokenizer to use
            model: Model to use
            device: Device override
            max_length: Maximum token length
            target_dim: Target embedding dimension

        Returns:
            Embedding array
        """
        device = device or self.device

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
                outputs = model(**inputs, output_hidden_states=True)

                # Extract embedding using same strategy as batch processing
                embedding = self._extract_single_embedding(inputs, outputs)

                # Convert to numpy
                embedding_np = embedding.cpu().numpy()
                if len(embedding_np.shape) > 1:
                    embedding_np = embedding_np[0]  # Remove batch dimension

                # Handle dimension adjustment
                if target_dim is not None and len(embedding_np) != target_dim:
                    if len(embedding_np) > target_dim:
                        embedding_np = embedding_np[:target_dim]
                    else:
                        embedding_np = np.pad(embedding_np, (0, target_dim - len(embedding_np)))

                # Normalize
                norm = np.linalg.norm(embedding_np)
                if norm > 1e-10:
                    embedding_np = embedding_np / norm

                return embedding_np

        except Exception as e:
            logger.error(f"Error in single text embedding: {e}")
            raise

    def _extract_single_embedding(self, inputs, outputs):
        """Extract embedding from single text model outputs."""
        # Use same strategy as batch processing but for single item
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
            attention_mask = inputs.get('attention_mask', torch.ones_like(inputs['input_ids']))

            # Masked average pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask

        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            last_layer = outputs.hidden_states[-1]
            attention_mask = inputs.get('attention_mask', torch.ones_like(inputs['input_ids']))

            # Masked average pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_layer.size()).float()
            sum_embeddings = torch.sum(last_layer * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask

        elif hasattr(outputs, 'logits'):
            logits = outputs.logits
            if len(logits.shape) == 3:
                embedding = logits[:, -1, :]  # Last token
            else:
                embedding = logits

            # Normalize
            norm = torch.norm(embedding, dim=1, keepdim=True)
            embedding = embedding / (norm + 1e-8)

        else:
            raise ValueError("Could not extract embedding from model outputs")

        return embedding

    def tensor_batch_processing(self,
                               tensor_op: Callable[[torch.Tensor], torch.Tensor],
                               input_tensor: torch.Tensor,
                               batch_dim: int = 0,
                               operation_type: str = 'inference',
                               batch_size: Optional[int] = None,
                               cleanup: bool = None,
                               adaptive: bool = None,
                               handle_oom: bool = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Process a large tensor in batches with adaptive sizing and error handling.

        Args:
            tensor_op: Function that processes a tensor batch
            input_tensor: Input tensor to process
            batch_dim: Dimension to batch along
            operation_type: Type of operation for settings
            batch_size: Batch size override
            cleanup: Cleanup override
            adaptive: Adaptive sizing override
            handle_oom: OOM handling override

        Returns:
            Processed tensor or list of tensors
        """
        # Get settings
        settings = self.get_batch_settings(operation_type)
        batch_size = batch_size or settings['batch_size']
        cleanup = cleanup if cleanup is not None else settings['cleanup']
        adaptive = adaptive if adaptive is not None else settings['adaptive']
        handle_oom = handle_oom if handle_oom is not None else settings.get('handle_oom', True)

        # Get total size along batch dimension
        total_size = input_tensor.shape[batch_dim]

        # Determine batch size if adaptive
        if adaptive:
            item_shape = list(input_tensor.shape)
            item_shape[batch_dim] = 1
            batch_size = self.estimate_optimal_batch_size(
                tuple(item_shape),
                input_tensor.dtype,
                operation_type,
                max_batch_size=batch_size
            )

        # Process in batches
        output_batches = []
        current_batch_size = batch_size
        start_idx = 0

        while start_idx < total_size:
            # Get end index for current batch
            end_idx = min(start_idx + current_batch_size, total_size)

            # Create indexing tuple
            idx = [slice(None)] * input_tensor.ndim
            idx[batch_dim] = slice(start_idx, end_idx)

            # Get batch
            batch = input_tensor[tuple(idx)]

            try:
                # Process batch
                processed_batch = tensor_op(batch)
                output_batches.append(processed_batch)

                # Move to next batch
                start_idx = end_idx

                # Clean up if requested
                if cleanup:
                    gc.collect()
                    if self.is_cuda:
                        torch.cuda.empty_cache()

            except RuntimeError as e:
                if handle_oom and "out of memory" in str(e).lower() and current_batch_size > 1:
                    # Reduce batch size and retry
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.warning(f"OOM error, reducing batch size to {current_batch_size}")

                    # Clean up and retry
                    del batch
                    gc.collect()
                    if self.is_cuda:
                        torch.cuda.empty_cache()
                    # Don't advance start_idx to retry
                else:
                    raise

        # Combine results
        try:
            result = torch.cat(output_batches, dim=batch_dim)
            return result
        except Exception as e:
            logger.warning(f"Could not concatenate results: {e}")
            return output_batches

    def validate_batch_processing_performance(self,
                                            test_function: Callable,
                                            test_data: Any,
                                            operation_type: str = 'embedding',
                                            batch_sizes: List[int] = None):
        """
        Validate and benchmark batch processing performance.

        Args:
            test_function: Function to test with various batch sizes
            test_data: Data to process in batches
            operation_type: Type of operation being tested
            batch_sizes: List of batch sizes to test

        Returns:
            Dictionary with performance metrics for each batch size
        """
        batch_sizes = batch_sizes or [1, 2, 4, 8, 16, 32]
        results = {}

        logger.info(f"Testing {len(batch_sizes)} batch sizes for {operation_type}: {batch_sizes}")

        for batch_size in batch_sizes:
            try:
                logger.info(f"Testing batch size: {batch_size}")

                # Clean start
                gc.collect()
                if self.is_cuda:
                    torch.cuda.empty_cache()
                    start_mem = torch.cuda.memory_allocated()

                # Test performance
                start_time = time.time()
                _ = test_function(test_data, batch_size)
                duration = time.time() - start_time

                # Get memory usage
                memory_used = 0
                if self.is_cuda:
                    end_mem = torch.cuda.memory_allocated()
                    peak_mem = torch.cuda.max_memory_allocated()
                    memory_used = peak_mem - start_mem

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
            logger.info(f"Optimal batch size for {operation_type}: {optimal_size}")

            # Update settings with optimal size
            self.update_batch_settings(operation_type, {'batch_size': optimal_size})

        # Store performance history
        self.performance_history[operation_type] = results

        return results

    def get_memory_stats(self) -> dict:
        """Get memory statistics for batch size optimization."""
        stats = {'cpu': {'available': 0, 'used': 0, 'percent': 0}}

        # CPU memory stats
        try:
            import psutil
            vm = psutil.virtual_memory()
            stats['cpu'] = {
                'available': vm.available,
                'used': vm.used,
                'percent': vm.percent
            }
        except ImportError:
            pass

        # GPU memory stats
        if self.is_cuda:
            device = torch.cuda.current_device()
            stats['gpu'] = {
                'total': torch.cuda.get_device_properties(device).total_memory,
                'allocated': torch.cuda.memory_allocated(device),
                'reserved': torch.cuda.memory_reserved(device),
                'percent': (torch.cuda.memory_allocated(device) /
                          torch.cuda.get_device_properties(device).total_memory) * 100
            }

        return stats

    def cleanup(self):
        """Clean up resources."""
        if self.is_cuda:
            torch.cuda.empty_cache()
        gc.collect()


# Global batch processor instance
_global_batch_processor = None

def get_batch_processor(device: str = "auto") -> BatchProcessor:
    """Get or create the global batch processor instance."""
    global _global_batch_processor
    if _global_batch_processor is None:
        _global_batch_processor = BatchProcessor(device=device)
    return _global_batch_processor

# Convenience functions for backward compatibility
def estimate_optimal_batch_size(tensor_shape: Tuple[int, ...],
                               dtype: torch.dtype = torch.float32,
                               target_memory_usage: float = 0.7,
                               min_batch_size: int = 1,
                               max_batch_size: int = 32) -> int:
    """Backward compatibility wrapper."""
    processor = get_batch_processor()
    return processor.estimate_optimal_batch_size(
        tensor_shape, dtype, 'embedding', min_batch_size, max_batch_size
    )

def batch_embed_texts(texts: List[str], tokenizer, model, device: str = "cuda",
                     batch_size: int = 32, adaptive: bool = True, handle_oom: bool = True,
                     max_length: int = 512, target_dim: Optional[int] = None) -> List[np.ndarray]:
    """Backward compatibility wrapper."""
    processor = get_batch_processor(device)
    return processor.batch_embed_texts(
        texts, tokenizer, model, device, batch_size, adaptive, handle_oom, max_length, target_dim
    )

def embed_single_text(text: str, tokenizer, model, device: str = "cuda",
                     max_length: int = 512, target_dim: Optional[int] = None) -> np.ndarray:
    """Backward compatibility wrapper."""
    processor = get_batch_processor(device)
    return processor.embed_single_text(text, tokenizer, model, device, max_length, target_dim)