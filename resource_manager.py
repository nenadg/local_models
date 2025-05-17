"""
Resource management utilities for local_models project.
Handles CUDA memory management and resource cleanup.
"""

import numpy as np
import gc
import os
import torch
from typing import Optional, Dict, Any, List, Union, Callable

class CUDAMemoryManager:
    """
    Manages CUDA memory to prevent memory leaks and optimize resource usage.
    """

    def __init__(self, device: str = None):
        """
        Initialize the memory manager for the specified device.

        Args:
            device: Device to manage ('cuda', 'cpu', or a specific cuda device)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_cuda = self.device.startswith("cuda")
        self.allocated_tensors = []

    def print_memory_stats(self, label: str = "Current"):
        """
        Print memory statistics for debugging.

        Args:
            label: Label for the stats printout
        """
        if not self.is_cuda:
            return

        # Get memory stats
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB

        #print(f"[{label}] CUDA Memory: {allocated:.2f}MB allocated, "
        #      f"{max_allocated:.2f}MB peak, {reserved:.2f}MB reserved")

    def clear_cache(self):
        """Clear CUDA cache and run garbage collection."""
        if self.is_cuda:
            # Release all PyTorch CUDA memory
            torch.cuda.empty_cache()

        # Run garbage collection
        gc.collect()

        # Print memory stats after clearing
        # self.print_memory_stats("After clearing cache")

    def optimize_for_inference(self):
        """Apply optimizations for inference mode."""
        if not self.is_cuda:
            return

        # Set inference optimization flags
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, 'cuda'):
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp = True

    def register_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Register a model for memory management.

        Args:
            model: PyTorch model to register

        Returns:
            The registered model (unchanged)
        """
        # Nothing special needed for now, just return the model
        return model

    def unload_model(self, model: torch.nn.Module):
        """
        Properly unload a model to free memory.

        Args:
            model: Model to unload
        """
        if model is None:
            return

        try:
            # Move model to CPU first
            model.cpu()

            # Delete model
            del model

            # Clear cache
            self.clear_cache()
        except Exception as e:
            print(f"Error unloading model: {e}")

    def optimize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Optimize a tensor for memory efficiency.

        Args:
            tensor: Tensor to optimize

        Returns:
            Optimized tensor
        """
        if not self.is_cuda or tensor is None:
            return tensor

        # Track tensor for potential cleanup
        self.allocated_tensors.append(tensor)

        # For large tensors, use more aggressive optimization
        if tensor.numel() > 1000000:  # More than ~1M elements
            # Use lower precision if possible
            if tensor.dtype == torch.float32:
                return tensor.to(dtype=torch.float16)

        return tensor

    def cleanup_tensors(self):
        """Clean up tracked tensors."""
        for tensor in self.allocated_tensors:
            if tensor is not None:
                try:
                    del tensor
                except:
                    pass

        self.allocated_tensors = []
        self.clear_cache()

    def __del__(self):
        """Cleanup on object destruction."""
        self.cleanup_tensors()
        self.clear_cache()


class VectorStoreManager:
    """
    Manages vector store resources to prevent memory leaks.
    """

    def __init__(self, max_stores: int = 5):
        """
        Initialize the vector store manager.

        Args:
            max_stores: Maximum number of stores to keep in memory
        """
        self.max_stores = max_stores
        self.active_stores = {}  # user_id -> (store, last_accessed_time)
        self.last_accessed = []  # List of user_ids in order of access

    def register_store(self, user_id: str, store: Any):
        """
        Register a vector store for a user.

        Args:
            user_id: User identifier
            store: Vector store object
        """
        import time

        # Record the store with current timestamp
        self.active_stores[user_id] = (store, time.time())

        # Update access list
        if user_id in self.last_accessed:
            self.last_accessed.remove(user_id)
        self.last_accessed.append(user_id)

        # Check if we need to release any stores
        self._cleanup_if_needed()

    def get_store(self, user_id: str) -> Any:
        """
        Get a store for a user, updating its access time.

        Args:
            user_id: User identifier

        Returns:
            The vector store or None if not found
        """
        import time

        if user_id not in self.active_stores:
            return None

        # Update access time
        store, _ = self.active_stores[user_id]
        self.active_stores[user_id] = (store, time.time())

        # Update access list
        if user_id in self.last_accessed:
            self.last_accessed.remove(user_id)
        self.last_accessed.append(user_id)

        return store

    def release_store(self, user_id: str):
        """
        Explicitly release a store for a user.

        Args:
            user_id: User identifier
        """
        if user_id not in self.active_stores:
            return

        store, _ = self.active_stores[user_id]

        # Call cleanup method if available
        if hasattr(store, 'cleanup') and callable(store.cleanup):
            try:
                store.cleanup()
            except Exception as e:
                print(f"Error cleaning up store: {e}")

        # Remove from tracking
        del self.active_stores[user_id]
        if user_id in self.last_accessed:
            self.last_accessed.remove(user_id)

    def _cleanup_if_needed(self):
        """Release least recently used stores if we exceed the maximum."""
        while len(self.active_stores) > self.max_stores:
            # Get oldest user
            oldest_user = self.last_accessed[0]
            # Release its store
            self.release_store(oldest_user)

    def cleanup_all(self):
        """Release all stores."""
        user_ids = list(self.active_stores.keys())
        for user_id in user_ids:
            self.release_store(user_id)


class ResourceManager:
    """
    Central resource manager for the application.
    Manages CUDA memory, vector stores, and other resources.
    """

    def __init__(self, device: str = None, max_vector_stores: int = 5):
        """
        Initialize the resource manager.

        Args:
            device: Device to use for CUDA management
            max_vector_stores: Maximum number of vector stores to keep in memory
        """
        self.cuda_manager = CUDAMemoryManager(device)
        self.vector_store_manager = VectorStoreManager(max_vector_stores)

        # Track performance metrics
        self.batch_performance = {}

        # Default parameters for batch processing
        self.default_batch_settings = {
            'embedding': {
                'batch_size': 8,
                'cleanup': True,
                'adaptive': True,
                'parallel': False,
                'max_workers': 4
            },
            'inference': {
                'batch_size': 4,
                'cleanup': True,
                'adaptive': True,
                'handle_oom': True
            }
        }

    def optimize_for_inference(self):
        """Apply inference optimizations."""
        self.cuda_manager.optimize_for_inference()

    def clear_cache(self):
        """Clear memory cache."""
        self.cuda_manager.clear_cache()

    def register_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Register a model for memory management."""
        return self.cuda_manager.register_model(model)

    def unload_model(self, model: torch.nn.Module):
        """Unload a model."""
        self.cuda_manager.unload_model(model)

    def register_vector_store(self, user_id: str, store: Any):
        """Register a vector store."""
        self.vector_store_manager.register_store(user_id, store)

    def get_vector_store(self, user_id: str) -> Any:
        """Get a vector store for a user."""
        return self.vector_store_manager.get_store(user_id)

    def release_vector_store(self, user_id: str):
        """Release a vector store."""
        self.vector_store_manager.release_store(user_id)

    def optimize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize a tensor for memory efficiency."""
        return self.cuda_manager.optimize_tensor(tensor)

    def cleanup(self):
        """Perform a full cleanup of all resources."""
        self.vector_store_manager.cleanup_all()
        self.cuda_manager.cleanup_tensors()
        self.cuda_manager.clear_cache()

    def get_batch_settings(self, operation_type: str) -> dict:
        """
        Get the appropriate batch settings for an operation type.

        Args:
            operation_type: Type of operation ('embedding' or 'inference')

        Returns:
            Dictionary of batch settings
        """
        if operation_type in self.default_batch_settings:
            return self.default_batch_settings[operation_type].copy()
        return {
            'batch_size': 8,
            'cleanup': True,
            'adaptive': True,
            'handle_oom': True
        }

    def update_batch_settings(self, operation_type: str, settings: dict):
        """
        Update batch settings for an operation type.

        Args:
            operation_type: Type of operation ('embedding' or 'inference')
            settings: Dictionary of settings to update
        """
        if operation_type in self.default_batch_settings:
            self.default_batch_settings[operation_type].update(settings)

    def get_memory_stats(self) -> dict:
        """
        Get detailed memory statistics for better batch size estimation.

        Returns:
            Dictionary of memory statistics
        """
        stats = {
            'cpu': {
                'available': 0,
                'used': 0,
                'percent': 0
            }
        }

        # Get CPU memory stats if psutil is available
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

        # Get GPU memory stats if CUDA is available
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            stats['gpu'] = {
                'total': torch.cuda.get_device_properties(device).total_memory,
                'allocated': torch.cuda.memory_allocated(device),
                'reserved': torch.cuda.memory_reserved(device),
                'percent': (torch.cuda.memory_allocated(device) /
                           torch.cuda.get_device_properties(device).total_memory) * 100
            }

        return stats

    def suggest_optimal_batch_size(self, tensor_shape: tuple, operation_type: str = 'inference') -> int:
        """
        Suggest an optimal batch size based on tensor shape and memory.

        Args:
            tensor_shape: Shape of a single tensor item
            operation_type: Type of operation ('embedding' or 'inference')

        Returns:
            Suggested batch size
        """
        from batch_utils import estimate_optimal_batch_size

        # Get appropriate target memory usage based on operation type
        if operation_type == 'embedding':
            target_usage = 0.6  # Embeddings typically need less memory
        else:
            target_usage = 0.5  # Inference needs more headroom

        # Use the utility function with appropriate parameters
        return estimate_optimal_batch_size(
            tensor_shape=tensor_shape,
            dtype=torch.float16 if self.is_cuda else torch.float32,
            target_memory_usage=target_usage
        )

    def batch_process_embeddings(self, texts: List[str], model, tokenizer, device="cuda") -> List[np.ndarray]:
        """
        Process embeddings in batches using the batch_utils functions.

        Args:
            texts: List of texts to embed
            model: Model to use
            tokenizer: Tokenizer to use
            device: Device to use

        Returns:
            List of embeddings
        """
        from batch_utils import batch_embed_texts

        # Get batch settings
        settings = self.get_batch_settings('embedding')

        # Process in batches
        return batch_embed_texts(
            texts=texts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            **settings
        )

    def batch_process_inference(self,
                              model: torch.nn.Module,
                              inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                              batch_dim: int = 0) -> torch.Tensor:
        """
        Process model inference in batches.

        Args:
            model: PyTorch model to use for inference
            inputs: Input tensor or dictionary of tensors
            batch_dim: Dimension to batch along

        Returns:
            Output tensor
        """
        from batch_utils import tensor_batch_processing

        # Get batch settings
        settings = self.get_batch_settings('inference')

        # Define inference function
        def inference_op(batch):
            with torch.no_grad():
                if isinstance(batch, dict):
                    return model(**batch)
                else:
                    return model(batch)

        # Process in batches
        if isinstance(inputs, dict):
            # For dictionary inputs, extract the main input tensor
            main_input = next(iter(inputs.values()))

            # Process each input separately
            results = {}
            for key, tensor in inputs.items():
                # Define operation for this input
                def process_input(batch):
                    # Create batch dictionary
                    batch_dict = {k: v for k, v in inputs.items()}
                    batch_dict[key] = batch
                    return inference_op(batch_dict)

                # Process this input in batches
                results[key] = tensor_batch_processing(
                    tensor_op=process_input,
                    input_tensor=tensor,
                    batch_dim=batch_dim,
                    **settings
                )

            return results
        else:
            # For tensor input, process directly
            return tensor_batch_processing(
                tensor_op=inference_op,
                input_tensor=inputs,
                batch_dim=batch_dim,
                **settings
            )

    def benchmark_batch_sizes(self,
                             test_function: Callable,
                             test_data: Any,
                             operation_type: str):
        """
        Benchmark different batch sizes and update settings.

        Args:
            test_function: Function to test
            test_data: Data to process
            operation_type: Type of operation ('embedding' or 'inference')
        """
        from batch_utils import validate_batch_processing_performance

        # Test performance with different batch sizes
        performance = validate_batch_processing_performance(
            test_function=test_function,
            test_data=test_data
        )

        # Store results
        self.batch_performance[operation_type] = performance

        # Find optimal batch size
        valid_sizes = [size for size, result in performance.items()
                     if 'error' not in result]

        if valid_sizes:
            optimal_size = max(valid_sizes,
                              key=lambda s: performance[s]['items_per_second'])

            # Update settings
            self.update_batch_settings(operation_type, {'batch_size': optimal_size})

            print(f"Updated {operation_type} batch size to {optimal_size}")
            return optimal_size

        return None

    def __del__(self):
        """Cleanup on object destruction."""
        self.cleanup()


def get_memory_usage(pid=None):
    """
    Get memory usage of the current process in MB.

    Args:
        pid: Process ID to check (defaults to current process)

    Returns:
        Memory usage in MB
    """
    pid = pid or os.getpid()
    try:
        import psutil
        process = psutil.Process(pid)
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB
    except ImportError:
        return 0  # psutil not available