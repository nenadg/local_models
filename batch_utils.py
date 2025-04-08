"""
Batch processing utilities for memory-intensive operations.
Provides functions for efficient processing of large datasets with managed memory.
"""

import gc
import torch
import numpy as np
from typing import List, Callable, TypeVar, Iterator, Any, Dict, Tuple, Optional, Union

T = TypeVar('T')
R = TypeVar('R')

def process_in_batches(items: List[T], 
                     batch_size: int, 
                     process_fn: Callable[[List[T]], Any],
                     cleanup_fn: Callable = None) -> List[Any]:
    """
    Process a list of items in batches to manage memory usage.
    
    Args:
        items: List of items to process
        batch_size: Number of items to process in each batch
        process_fn: Function to process each batch
        cleanup_fn: Optional function to call between batches for cleanup
        
    Returns:
        List of results from all batches
    """
    results = []
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        # Get current batch
        batch = items[i:i + batch_size]
        
        # Process the batch
        batch_results = process_fn(batch)
        
        # Add to results
        if isinstance(batch_results, list):
            results.extend(batch_results)
        else:
            results.append(batch_results)
        
        # Run cleanup if provided
        if cleanup_fn is not None:
            cleanup_fn()
        else:
            # Basic cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return results

def batch_embedding_generation(texts: List[str], 
                              embedding_fn: Callable[[str], np.ndarray],
                              batch_size: int = 32,
                              show_progress: bool = False) -> np.ndarray:
    """
    Generate embeddings for a list of texts in batches to manage memory.
    
    Args:
        texts: List of texts to embed
        embedding_fn: Function to generate embeddings for a single text
        batch_size: Size of batches to process
        show_progress: Whether to display progress
        
    Returns:
        Array of embeddings for all texts
    """
    # Get embedding dimension from first item
    if not texts:
        return np.array([])
        
    # Get dimension from first embedding
    first_embedding = embedding_fn(texts[0])
    embedding_dim = first_embedding.shape[0]
    
    # Pre-allocate result array
    all_embeddings = np.zeros((len(texts), embedding_dim), dtype=np.float32)
    all_embeddings[0] = first_embedding  # Store first embedding
    
    # Process remaining texts in batches
    for i in range(1, len(texts), batch_size):
        if show_progress and i % (batch_size * 5) == 0:
            print(f"Processing embeddings: {i}/{len(texts)}")
            
        # Get current batch
        end_idx = min(i + batch_size, len(texts))
        batch = texts[i:end_idx]
        
        # Generate embeddings
        for j, text in enumerate(batch):
            embedding = embedding_fn(text)
            all_embeddings[i + j] = embedding
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_embeddings

def chunked_file_processing(file_path: str, 
                           process_fn: Callable[[bytes], Any],
                           chunk_size: int = 1024 * 1024) -> List[Any]:
    """
    Process a large file in chunks to avoid loading it all into memory.
    
    Args:
        file_path: Path to the file to process
        chunk_size: Size of chunks to read at once
        process_fn: Function to process each chunk
        
    Returns:
        List of results from processing each chunk
    """
    results = []
    
    with open(file_path, 'rb') as file:
        while True:
            # Read chunk
            chunk = file.read(chunk_size)
            
            # Exit if end of file
            if not chunk:
                break
                
            # Process chunk
            result = process_fn(chunk)
            results.append(result)
            
            # Clean up memory
            gc.collect()
    
    return results

def tensor_batch_processing(tensor_op: Callable[[torch.Tensor], torch.Tensor],
                          input_tensor: torch.Tensor,
                          batch_dim: int = 0,
                          batch_size: int = 8,
                          cleanup: bool = True) -> torch.Tensor:
    """
    Process a large tensor in batches along a specified dimension.
    
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