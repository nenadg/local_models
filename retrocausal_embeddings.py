"""
RetrocausalEmbeddingSystem for UnifiedMemoryManager
Provides a simpler, more elegant approach to semantic embeddings with
dimension interdependence and retrocausal feedback.
"""

import numpy as np
import torch
import faiss
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
import hashlib
from datetime import datetime

# Setup logger
logger = logging.getLogger("retrocausal_embeddings")

class RetrocausalEmbeddingSystem:
    """
    A more efficient embedding system that leverages dimension interdependence
    and feedback loops to create rich semantic relationships with less computation.
    
    This system replaces the traditional fractal embedding approach with a simpler
    but more powerful retrocausal embedding system where dimensions influence each other.
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        feedback_strength: float = 0.3,
        causal_dims: int = 32,
        contrast_factor: float = 0.4,
        seed: int = 42
    ):
        """
        Initialize the retrocausal embedding system.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            feedback_strength: Strength of dimensional feedback (0.0-1.0)
            causal_dims: Number of dimensions to use for causal feedback
            contrast_factor: Factor for enhancing semantic contrast
            seed: Random seed for reproducible dimension mapping
        """
        self.embedding_dim = embedding_dim
        self.feedback_strength = feedback_strength
        self.causal_dims = min(causal_dims, embedding_dim // 4)  # Limit to 1/4 of dims
        self.contrast_factor = contrast_factor
        
        # Initialize parameters with fixed seed for reproducibility
        rng = np.random.RandomState(seed)
        
        # Create dimension mappings for interdependence
        # Each dimension has influence on and is influenced by others
        self.influence_map = self._create_influence_map(rng)
        
        # Feedback matrices - simpler than full rotation matrices
        self.feedback_matrix = self._create_feedback_matrix(rng)
        
        # Seed the "future state" dimensions 
        self.future_seed = rng.normal(0, 0.01, (self.causal_dims,))
        
        # Cache for frequently accessed embeddings
        self.embedding_cache = {}
        
        logger.info(f"Initialized RetrocausalEmbeddingSystem with {embedding_dim} dimensions, "
                   f"{self.causal_dims} causal dimensions, and {feedback_strength} feedback strength")
    
    def _create_influence_map(self, rng: np.random.RandomState) -> Dict[int, List[int]]:
        """
        Create a map of which dimensions influence which other dimensions.
        
        Args:
            rng: Random number generator
            
        Returns:
            Dictionary mapping dimension indices to the indices they influence
        """
        influence_map = {}
        
        # Identify causal dimensions (the "future state" dimensions)
        causal_indices = set(rng.choice(
            self.embedding_dim, 
            size=self.causal_dims, 
            replace=False
        ).tolist())
        
        # For each dimension, determine which other dimensions it influences
        for i in range(self.embedding_dim):
            # Each dimension influences 3-7 other dimensions
            num_influences = rng.randint(3, 8)
            
            # Select dimensions to influence (excluding self)
            available_dims = [d for d in range(self.embedding_dim) if d != i]
            influence_indices = rng.choice(
                available_dims, 
                size=min(num_influences, len(available_dims)), 
                replace=False
            ).tolist()
            
            # Ensure causal dimensions have broader influence
            if i in causal_indices:
                # Add 2-3 more influences for causal dimensions
                extra_count = rng.randint(2, 4)
                remaining_dims = [d for d in available_dims if d not in influence_indices]
                if remaining_dims:
                    extra_indices = rng.choice(
                        remaining_dims,
                        size=min(extra_count, len(remaining_dims)),
                        replace=False
                    ).tolist()
                    influence_indices.extend(extra_indices)
            
            influence_map[i] = influence_indices
        
        # Store causal indices for later use
        self.causal_indices = list(causal_indices)
        
        return influence_map
    
    def _create_feedback_matrix(self, rng: np.random.RandomState) -> np.ndarray:
        """
        Create a sparse feedback matrix that defines how dimensions affect each other.
        
        Args:
            rng: Random number generator
            
        Returns:
            Feedback matrix as a sparse array
        """
        # Create a feedback matrix that's much sparser than a rotation matrix
        feedback = np.zeros((self.embedding_dim, self.embedding_dim))
        
        # Set feedback values based on influence map
        for source_dim, target_dims in self.influence_map.items():
            for target_dim in target_dims:
                # Vary feedback strength slightly
                factor = self.feedback_strength * (0.8 + 0.4 * rng.random())
                # Sign determines whether influence is positive or negative
                sign = 1 if rng.random() > 0.3 else -1
                feedback[source_dim, target_dim] = sign * factor
        
        # Normalize to avoid explosion/vanishing
        row_sums = np.sum(np.abs(feedback), axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        normalized_feedback = feedback / row_sums * self.feedback_strength
        
        return normalized_feedback
    
    def transform_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Apply retrocausal transformation to an embedding.
        
        This method creates a semantically richer embedding by allowing
        dimensions to influence each other based on the influence map.
        
        Args:
            embedding: Original embedding vector
            
        Returns:
            Transformed embedding with interdependent dimensions
        """
        # Ensure embedding is the right shape and type
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
            
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Expected embedding dimension {self.embedding_dim}, "
                           f"got {embedding.shape[0]}")
        
        # Create hash for caching
        embedding_hash = hashlib.md5(embedding.tobytes()).hexdigest()
        if embedding_hash in self.embedding_cache:
            return self.embedding_cache[embedding_hash]
        
        # Initialize with original embedding
        transformed = embedding.copy()
        
        # Extract "future state" dimensions (the causal dimensions)
        # These represent the synthesized semantic essence
        future_state = embedding[self.causal_indices]
        
        # Apply contrast enhancement to make features more distinct
        # This focuses embeddings on their most distinctive features
        if self.contrast_factor > 0:
            # Calculate mean and standard deviation
            embedding_mean = np.mean(transformed)
            embedding_std = np.std(transformed)
            if embedding_std > 0:
                # Apply contrast - values far from mean become more extreme
                deviation = transformed - embedding_mean
                transformed = embedding_mean + deviation * (1.0 + self.contrast_factor * 
                                                          np.abs(deviation) / embedding_std)
        
        # Apply interdimensional influence 
        # This is the key step that implements the retrocausal effect
        # where dimensions influence each other
        
        # First-order influence (direct)
        influence = np.zeros_like(transformed)
        for i in range(self.embedding_dim):
            # Target dimensions that are influenced by dimension i
            targets = self.influence_map.get(i, [])
            for target in targets:
                influence[target] += transformed[i] * self.feedback_matrix[i, target]
        
        # Apply the influence with nonlinearity to create richer patterns
        transformed += influence * np.tanh(influence)
        
        # Apply future state influence (the retrocausal component)
        # This allows the "future state" dimensions to influence all dimensions
        future_influence = np.zeros_like(transformed)
        for i, causal_idx in enumerate(self.causal_indices):
            # Calculate weighted combination of future state
            seed_val = self.future_seed[i]
            future_val = future_state[i]
            # This step introduces nonlinear combinations of dimensions
            # to create emergent patterns
            combined = np.tanh(seed_val * future_val)
            
            # Apply to influenced dimensions
            targets = self.influence_map.get(causal_idx, [])
            for target in targets:
                future_influence[target] += combined * self.feedback_matrix[causal_idx, target]
        
        # Apply future influence with scaled strength
        future_scale = 0.5 * self.feedback_strength
        transformed += future_influence * future_scale
        
        # Final nonlinearity to create richer patterns
        # The hyperbolic tangent keeps values controlled while preserving sign
        transformed = np.tanh(transformed)
        
        # Normalize the embedding to unit length (for cosine similarity)
        norm = np.linalg.norm(transformed)
        if norm > 1e-10:
            transformed = transformed / norm
        
        # Cache the result
        self.embedding_cache[embedding_hash] = transformed
        
        # Limit cache size
        if len(self.embedding_cache) > 1000:
            # Remove a random 20% of cache entries
            remove_count = len(self.embedding_cache) // 5
            keys_to_remove = list(self.embedding_cache.keys())[:remove_count]
            for key in keys_to_remove:
                del self.embedding_cache[key]
        
        return transformed
    
    def batch_transform_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform a batch of embeddings using retrocausal approach.
        
        Args:
            embeddings: Batch of embeddings with shape (batch_size, embedding_dim)
            
        Returns:
            Transformed embeddings with same shape
        """
        # Handle torch tensors
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
            
        # Validate input shape
        if len(embeddings.shape) != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected embeddings with shape (batch_size, {self.embedding_dim}), "
                           f"got {embeddings.shape}")
        
        batch_size = embeddings.shape[0]
        transformed = np.zeros_like(embeddings)
        
        # Process each embedding
        # TODO: Make this more efficient with vectorized operations
        for i in range(batch_size):
            transformed[i] = self.transform_embedding(embeddings[i])
        
        return transformed
    
    def search_similar(
        self, 
        query_embedding: np.ndarray,
        index: faiss.Index,
        items: List[Any],
        top_k: int = 5,
        min_similarity: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Search for similar items using the retrocausal embedding system.
        
        Args:
            query_embedding: Query embedding vector
            index: FAISS index containing embeddings
            items: List of items corresponding to the index
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of search results with similarity scores
        """
        # Transform the query embedding
        transformed_query = self.transform_embedding(query_embedding)
        
        # Convert to the right format for FAISS
        query_vector = np.array([transformed_query], dtype=np.float32)
        
        # Search the index
        search_k = min(index.ntotal, top_k * 2)  # Get more, then filter
        similarities, indices = index.search(query_vector, search_k)
        
        # Process results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and similarities[0][i] > min_similarity:
                # Get the item
                if idx < len(items):
                    item = items[idx]
                    
                    # Create result entry
                    result = {
                        "id": getattr(item, "id", str(idx)),
                        "content": getattr(item, "content", str(item)),
                        "similarity": float(similarities[0][i]),
                        "metadata": getattr(item, "metadata", {})
                    }
                    
                    results.append(result)
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Limit to top_k
        return results[:top_k]

    def create_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Create a FAISS index from transformed embeddings.
        
        Args:
            embeddings: Original embeddings to transform and index
            
        Returns:
            FAISS index containing transformed embeddings
        """
        # Transform all embeddings
        transformed_embeddings = self.batch_transform_embeddings(embeddings)
        
        # Create the index
        index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add the transformed embeddings
        index.add(transformed_embeddings.astype(np.float32))
        
        return index

# Integration with UnifiedMemoryManager

def integrate_retrocausal_embeddings(memory_manager, embedding_dim=None, **kwargs):
    """
    Integrate the RetrocausalEmbeddingSystem into an existing UnifiedMemoryManager.
    
    Args:
        memory_manager: The UnifiedMemoryManager instance
        embedding_dim: Optional dimension override (default: use memory_manager.embedding_dim)
        **kwargs: Additional parameters for RetrocausalEmbeddingSystem
        
    Returns:
        Updated memory_manager with retrocausal embeddings
    """
    # Get embedding dimension
    if embedding_dim is None:
        embedding_dim = memory_manager.embedding_dim
    
    # Create retrocausal embedding system
    retrocausal_system = RetrocausalEmbeddingSystem(
        embedding_dim=embedding_dim,
        **kwargs
    )
    
    # Store the original _generate_level_embedding method to allow toggling back
    if not hasattr(memory_manager, '_original_generate_level_embedding'):
        memory_manager._original_generate_level_embedding = memory_manager._generate_level_embedding
    
    # Store the retrocausal system on the memory manager
    memory_manager.retrocausal_system = retrocausal_system
    
    # Create a new _generate_level_embedding method that uses retrocausal embeddings
    def retrocausal_generate_level_embedding(self, base_embedding: np.ndarray, level: int) -> np.ndarray:
        """Generate embedding using retrocausal system."""
        if level == 0:
            # Level 0 is original embedding
            return base_embedding
        
        # For other levels, use retrocausal transformation
        return self.retrocausal_system.transform_embedding(base_embedding)
    
    # Replace the method
    memory_manager._generate_level_embedding = retrocausal_generate_level_embedding.__get__(memory_manager)
    
    # Create optimized methods for batch processing
    def retrocausal_generate_level_embeddings_batch(self, base_embeddings: np.ndarray, level: int) -> np.ndarray:
        """Generate level-specific embeddings for a batch of base embeddings."""
        if level == 0:
            return base_embeddings
        
        # Transform with retrocausal system
        return self.retrocausal_system.batch_transform_embeddings(base_embeddings)
    
    # Replace the batch method
    memory_manager._generate_level_embeddings_batch = retrocausal_generate_level_embeddings_batch.__get__(memory_manager)
    
    # Add a method to toggle retrocausal embeddings on/off
    def toggle_retrocausal_embeddings(self):
        """Toggle between retrocausal and original fractal embeddings."""
        if hasattr(self, '_using_retrocausal') and self._using_retrocausal:
            # Switch back to original method
            self._generate_level_embedding = self._original_generate_level_embedding
            self._using_retrocausal = False
            return False
        else:
            # Switch to retrocausal method
            self._generate_level_embedding = retrocausal_generate_level_embedding.__get__(self)
            self._using_retrocausal = True
            return True
    
    # Add the toggle method
    memory_manager.toggle_retrocausal_embeddings = toggle_retrocausal_embeddings.__get__(memory_manager)
    
    # Enable retrocausal embeddings by default
    memory_manager._using_retrocausal = True
    
    # Add diagnostics method
    def retrocausal_diagnostics(self):
        """Print diagnostics for the retrocausal embedding system."""
        if not hasattr(self, 'retrocausal_system'):
            print("Retrocausal embedding system not initialized")
            return
        
        print("\n======= RETROCAUSAL EMBEDDING DIAGNOSTICS =======")
        print(f"Embedding dimension: {self.retrocausal_system.embedding_dim}")
        print(f"Causal dimensions: {self.retrocausal_system.causal_dims}")
        print(f"Feedback strength: {self.retrocausal_system.feedback_strength}")
        print(f"Contrast factor: {self.retrocausal_system.contrast_factor}")
        print(f"Cache size: {len(self.retrocausal_system.embedding_cache)}")
        print(f"Currently active: {getattr(self, '_using_retrocausal', False)}")
        
        # Calculate average influence
        total_influences = sum(len(targets) for targets in self.retrocausal_system.influence_map.values())
        avg_influence = total_influences / len(self.retrocausal_system.influence_map)
        print(f"Average dimension influences: {avg_influence:.2f} other dimensions")
        
        print("===============================================\n")
    
    # Add the diagnostics method
    memory_manager.retrocausal_diagnostics = retrocausal_diagnostics.__get__(memory_manager)
    
    # Rebuild indices if necessary
    if hasattr(memory_manager, 'items') and len(memory_manager.items) > 0:
        print(f"[{datetime.now().strftime('%d/%m/%y %H:%M:%S')}] [Memory] Rebuilding indices with retrocausal embeddings...")
        
        # Save original embeddings
        original_embeddings = [item.embedding for item in memory_manager.items]
        
        # Generate new retrocausal embeddings for all items
        for item in memory_manager.items:
            # Generate new embedding
            item.embedding = retrocausal_system.transform_embedding(item.embedding)
        
        # Rebuild indices
        memory_manager._rebuild_index()
        
        print(f"[{datetime.now().strftime('%d/%m/%y %H:%M:%S')}] [Memory] Indices rebuilt successfully with retrocausal embeddings")
    
    return memory_manager

# Add command to main chat script to enable retrocausal embeddings
def add_retrocausal_command():
    """
    Add commands to TinyLlamaChat to control retrocausal embeddings.
    Returns the command strings to add to help text.
    """
    return [
        "!toggle-retrocausal - Toggle retrocausal embeddings on/off",
        "!retrocausal-stats - Show retrocausal embedding diagnostics",
        "!set-feedback: [0.0-1.0] - Set retrocausal feedback strength"
    ]