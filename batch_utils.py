"""
Batch processing utilities for TinyLlama Chat system.
Provides functions for efficient processing of large tensor operations.
"""

import torch
import gc
import numpy as np

from typing import Callable, Any,List

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

def batch_embedding_processing(embedding_function: Callable[[str], np.ndarray],
                             texts: List[str],
                             batch_size: int = 8,
                             cleanup: bool = True) -> List[np.ndarray]:
    """
    Process a list of texts to generate embeddings in batches for memory efficiency.

    Args:
        embedding_function: Function that processes a text to an embedding
        texts: List of texts to embed
        batch_size: Size of batches
        cleanup: Whether to run cleanup between batches

    Returns:
        List of embedding arrays
    """
    # Create list for output embeddings
    output_embeddings = []

    # Process in batches
    for start_idx in range(0, len(texts), batch_size):
        # Get end index for current batch
        end_idx = min(start_idx + batch_size, len(texts))

        # Get batch
        batch = texts[start_idx:end_idx]

        # Process each text in the batch
        batch_embeddings = []
        for text in batch:
            try:
                embedding = embedding_function(text)
                batch_embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding: {e}")
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

        # Clean up memory if requested
        if cleanup:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return output_embeddings