"""
Batch processing utilities for TinyLlama Chat system.
Provides functions for efficient processing of large tensor operations.
"""

import torch
import gc
from typing import Callable, Any

def tensor_batch_processing(tensor_op: Callable[[torch.Tensor], torch.Tensor],
                          input_tensor: torch.Tensor,
                          batch_dim: int = 0,
                          batch_size: int = 8,
                          cleanup: bool = True) -> torch.Tensor:
    """
    Process a large tensor in batches along a specified dimension to manage memory efficiently.
    
    Args:
        tensor_op: Function that processes a tensor batch
        input_tensor: The input tensor to process
        batch_dim: Dimension to batch along
        batch_size: Size of batches
        cleanup: Whether to run cleanup between batches
        
    Returns:
        Processed tensor with results combined
    """
    # Get total size along batch dimension
    total_size = input_tensor.shape[batch_dim]
    
    # Create list for output batches
    output_batches = []
    
    # Process in batches
    for start_idx in range(0, total_size, batch_size):
        # Get end index for current batch
        end_idx = min(start_idx + batch_size, total_size)
        
        # Create indexing tuple to select along the batch dimension
        idx = [slice(None)] * input_tensor.ndim
        idx[batch_dim] = slice(start_idx, end_idx)
        
        # Get batch
        batch = input_tensor[tuple(idx)]
        
        # Process batch
        processed_batch = tensor_op(batch)
        
        # Save batch result
        output_batches.append(processed_batch)
        
        # Clean up memory if requested
        if cleanup:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Combine results
    try:
        # Try to concatenate along batch dimension
        return torch.cat(output_batches, dim=batch_dim)
    except:
        # If concatenation fails, just return the list of batches
        return output_batches