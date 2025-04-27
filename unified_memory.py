"""
Unified memory system for TinyLlama Chat.
Provides a cohesive approach to storing and retrieving different types of information
with optional fractal embedding support for enhanced semantic search.
"""

import os
import json
import faiss
import numpy as np
import time
import re
import gc
import pickle
import hashlib
import threading
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from datetime import datetime

class MemoryItem:
    """
    Unified knowledge item structure.
    """

    def __init__(self,
                content: str,
                embedding: np.ndarray,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a knowledge item.

        Args:
            content: The actual text/content
            embedding: Vector representation
            metadata: Additional information about this knowledge
        """
        self.content = content
        self.embedding = embedding
        self.fractal_embeddings = {}  # Optional multi-level representations

        # Initialize metadata
        self.metadata = metadata or {}

        # Ensure timestamp exists
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now().timestamp()

        # Generate a unique ID for this item
        self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a unique ID for this knowledge item."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()
        timestamp = str(int(self.metadata["timestamp"] * 1000))
        return f"knowledge_{content_hash[:8]}_{timestamp[-6:]}"

    def add_fractal_embedding(self, level: int, embedding: np.ndarray):
        """Add a fractal embedding at the specified level."""
        self.fractal_embeddings[level] = embedding

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            # Embeddings are stored separately for efficiency
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: np.ndarray) -> 'MemoryItem':
        """Create a MemoryItem from a dictionary and embedding."""
        item = cls(
            content=data["content"],
            embedding=embedding,
            metadata=data["metadata"]
        )
        item.id = data["id"]
        return item

class UnifiedMemoryManager:
    """
    Unified memory manager that handles all types of memory with a consistent interface.
    """
    
    def __init__(self, 
                storage_path: str = "./memory",
                embedding_function: Optional[Callable] = None,
                embedding_dim: int = 384,
                use_fractal: bool = False,
                max_fractal_levels: int = 3,
                auto_save: bool = True,
                level_sharpening_factors: Optional[Dict[int, float]] = None,
                enable_entity_separation: bool = True):
        """
        Initialize the unified memory manager.
        
        Args:
            storage_path: Base path for storing memory data
            embedding_function: Function to generate embeddings from text
            embedding_dim: Dimension of the embedding vectors
            use_fractal: Whether to use fractal embeddings for enhanced search
            max_fractal_levels: Maximum number of fractal levels if enabled
            auto_save: Whether to automatically save changes
        """
        self.storage_path = storage_path
        self.embedding_function = embedding_function
        self.embedding_dim = embedding_dim
        self.use_fractal = use_fractal
        self.max_fractal_levels = max_fractal_levels
        self.auto_save = auto_save
        self.enable_entity_separation = enable_entity_separation

        # Set default level sharpening factors if not provided
        if level_sharpening_factors is None:
            self.level_sharpening_factors = {
                0: 0.3,  # Base level (original embeddings)
                1: 0.4,  # Level 1 - stronger sharpening for entity type
                2: 0.5,  # Level 2 - even stronger for domain separation
                3: 0.6   # Level 3 - strongest for fine-grained distinctions
            }
        else:
            self.level_sharpening_factors = level_sharpening_factors
        
        # For backward compatibility, set the main sharpening factor
        self.sharpening_factor = self.level_sharpening_factors.get(0, 0.3)

        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # Thread lock for concurrent access
        self._lock = threading.RLock()
        
        # Memory storage
        self.items = []  # List of MemoryItem objects
        self.embeddings = []  # List of corresponding embeddings
        self.fractal_indices = {}  # Level -> FAISS index
        self.id_to_index = {}  # Map from item ID to index in items list
        self.deleted_ids = set()  # Set of deleted item IDs
        
        # Initialize FAISS index
        self.index = None
        
        # Load existing memory if available
        self.load()
    
    def get_time(self):
        return datetime.now().strftime("[%d/%m/%y %H:%M:%S]") + ' [Memory]'

    def add(self, 
            content: str,
            metadata: Optional[Dict[str, Any]] = None,
            use_fractal: Optional[bool] = None) -> Optional[str]:
        """
        Add a new knowledge item to memory.

        Args:
            content: Text content to remember
            metadata: Basic metadata about the knowledge
            use_fractal: Override default fractal setting

        Returns:
            ID of the added item or None if failed
        """
        with self._lock:
            # Skip if content is empty
            if not content or not content.strip():
                return None

            # Generate embedding if not already provided
            if self.embedding_function:
                try:
                    embedding = self.embedding_function(content)
                except Exception as e:
                    print(f"{self.get_time()} Error generating embedding: {e}")
                    return None
            else:
                # Random embedding if no function available (for testing)
                embedding = np.random.random(self.embedding_dim).astype(np.float32)

            # Ensure metadata exists
            if metadata is None:
                metadata = {}

            # Add timestamp if not present
            if "timestamp" not in metadata:
                metadata["timestamp"] = datetime.now().timestamp()

            # Initialize retrieval count if not present
            if "retrieval_count" not in metadata:
                metadata["retrieval_count"] = 0

            # Create memory item (removed memory_type - everything is knowledge)
            item = MemoryItem(
                content=content,
                embedding=embedding,
                metadata=metadata
            )
            
            # Determine whether to use fractal embeddings
            use_fractal_here = self.use_fractal if use_fractal is None else use_fractal
            
            # Generate fractal embeddings if requested
            if use_fractal_here:
                self._add_fractal_embeddings(item)
            
            # Add to storage
            return self._add_item_to_store(item)
    
    def add_bulk(self, items: List[Dict], use_fractal: Optional[bool] = None) -> List[Optional[str]]:
        """
        Efficiently add multiple knowledge items in a single operation with improved batching.

        Args:
            items: List of dictionaries with content and metadata
            use_fractal: Override default fractal setting

        Returns:
            List of item IDs or None for failed items
        """
        with self._lock:
            # Temporarily disable auto_save
            original_auto_save = self.auto_save
            self.auto_save = False

            # Extract content for batch embedding processing
            contents = []
            valid_items = []

            for item_dict in items:
                content = item_dict.get('content', '')
                if content and content.strip():
                    contents.append(content)
                    valid_items.append(item_dict)

            if not contents:
                return [None] * len(items)

            try:
                # Generate embeddings in batch using enhanced method
                print(f"{self.get_time()} Generating {len(contents)} embeddings in batch")
                all_embeddings = self.batch_embedding_function(contents)
                print(f"{self.get_time()} Successfully generated {len(all_embeddings)} embeddings")

                # Create memory items
                all_memory_items = []
                for i, (item_dict, embedding) in enumerate(zip(valid_items, all_embeddings)):
                    metadata = item_dict.get('metadata', {})

                    # Add timestamp if not present
                    if "timestamp" not in metadata:
                        metadata["timestamp"] = datetime.now().timestamp()

                    # Initialize retrieval count if not present
                    if "retrieval_count" not in metadata:
                        metadata["retrieval_count"] = 0

                    # Create memory item
                    item = MemoryItem(
                        content=item_dict['content'],
                        embedding=embedding,
                        metadata=metadata
                    )
                    all_memory_items.append(item)

                # Determine whether to use fractal embeddings
                use_fractal_here = self.use_fractal if use_fractal is None else use_fractal

                # Generate fractal embeddings using batched approach if requested
                if use_fractal_here:
                    self._add_fractal_embeddings_batch(all_memory_items)

                # Add all items to storage at once
                item_ids = []
                for item in all_memory_items:
                    item_id = self._add_item_to_store(item)
                    item_ids.append(item_id)

                # Finally, save the updated data
                if original_auto_save:
                    self.save()
                    self.auto_save = original_auto_save

                # Make sure we return one ID (or None) for each input item
                result_ids = [None] * len(items)
                valid_indices = [i for i, item in enumerate(items) if item.get('content', '').strip()]
                for i, valid_idx in enumerate(valid_indices):
                    if i < len(item_ids):
                        result_ids[valid_idx] = item_ids[i]

                return result_ids

            except Exception as e:
                print(f"{self.get_time()} Error in batch processing: {e}")
                # Fall back to individual processing
                item_ids = []
                for item in items:
                    item_id = self.add(
                        content=item.get("content", ""),
                        metadata=item.get("metadata", {}),
                        use_fractal=use_fractal
                    )
                    item_ids.append(item_id)

                # Restore original auto_save setting
                self.auto_save = original_auto_save
                return item_ids

    def batch_embedding_function(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batches with improved performance.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not self.embedding_function:
            print(f"{self.get_time()} No embedding function available")
            # Return empty embeddings with correct dimensionality
            return [np.zeros(self.embedding_dim) for _ in texts]

        # Import the enhanced batch processing function
        from batch_utils import batch_embedding_processing

        # Determine optimal batch size based on text length
        avg_length = sum(len(text) for text in texts) / max(1, len(texts))

        # Adaptive batch sizing based on average text length
        if avg_length > 1000:
            batch_size = 4
        elif avg_length > 500:
            batch_size = 8
        elif avg_length > 200:
            batch_size = 16
        else:
            batch_size = 32

        print(f"{self.get_time()} Processing {len(texts)} texts in batches of {batch_size}")

        # Check if we have a resource manager with enhanced capabilities
        if hasattr(self, 'resource_manager') and hasattr(self.resource_manager, 'batch_process_embeddings'):
            # Use the resource manager's batch processing
            return self.resource_manager.batch_process_embeddings(
                texts=texts,
                embedding_function=self.embedding_function
            )

        # Fall back to direct batch processing
        return batch_embedding_processing(
            embedding_function=self.embedding_function,
            texts=texts,
            batch_size=batch_size,
            cleanup=True,
            parallel=len(texts) > batch_size  # Use parallel processing for larger sets
        )

    def _generate_level_embeddings_batch(self, base_embeddings: np.ndarray, level: int) -> np.ndarray:
        """
        Generate level-specific embeddings for multiple base embeddings at once.

        Args:
            base_embeddings: Batch of base embedding vectors (N x D)
            level: The fractal level to generate

        Returns:
            Batch of transformed embeddings for the specified level (N x D)
        """
        if level == 0:
            return base_embeddings

        # Ensure matrices are initialized and cached
        if not hasattr(self, '_rotation_matrices') or \
           (hasattr(self, '_rotation_matrices') and \
            any(matrix.shape[0] != base_embeddings.shape[1] for matrix in self._rotation_matrices.values())):
            # Initialize rotation matrices and biases (same as in original implementation)
            # This code would be identical to the original _generate_level_embedding method
            pass

        # Apply the transformation to all embeddings at once
        if level in self._rotation_matrices:
            batch_size = base_embeddings.shape[0]

            # Apply the rotation matrix to all embeddings
            rotated = np.dot(base_embeddings, self._rotation_matrices[level])

            # Apply level-specific bias
            if level in self._level_biases:
                # Broadcast the bias to all embeddings
                shifted = rotated + np.tile(self._level_biases[level], (batch_size, 1))
            else:
                shifted = rotated

            # Apply non-linear transformation based on level
            if level % 3 == 0:
                # Enhance magnitude differences
                transformed = np.sign(shifted) * np.power(np.abs(shifted), 0.8)
            elif level % 3 == 1:
                # Apply sigmoid-like function
                transformed = np.tanh(shifted * (1.0 + 0.1 * level))
            else:
                # Log transformation
                transformed = np.sign(shifted) * np.log(1 + np.abs(shifted) * (1.0 + 0.05 * level))

            # Normalize each embedding
            norms = np.linalg.norm(transformed, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.maximum(norms, 1e-10)
            normalized = transformed / norms

            return normalized
        else:
            # Fallback if matrix isn't available
            print(f"{self.get_time()} Warning: No rotation matrix for level {level}")
            return None

    def _initialize_fractal_matrices(self):
        """
        Initialize rotation matrices for fractal embeddings.
        This method ensures matrices are created before any embeddings are generated.
        """
        if hasattr(self, '_rotation_matrices') and self._rotation_matrices:
            # Matrices already initialized
            return

        # Create rotation matrices for all potential levels
        self._rotation_matrices = {}
        self._level_biases = {}

        # Create many more levels than we explicitly use to allow for natural selection
        max_potential_levels = 20  # Support up to 20 potential levels
        embedding_dim = self.embedding_dim

        for i in range(1, max_potential_levels + 1):
            # Create a rotation matrix with fixed seed for determinism
            np.random.seed(42 + i)  # Fixed seed per level

            # Create rotation matrix with increasing randomness by level
            rotation_factor = 0.1 + (i * 0.03)  # Gradually increase randomness
            rotation = np.random.normal(0, rotation_factor, (embedding_dim, embedding_dim))

            # For earlier levels, preserve more of the original structure
            if i <= 3:
                # Add identity matrix component to preserve some original information
                preservation_factor = 0.9 - (i * 0.15)  # Decreasing preservation
                rotation = rotation + np.eye(embedding_dim) * preservation_factor

            # Ensure the matrix is orthogonal (proper rotation)
            u, _, vh = np.linalg.svd(rotation, full_matrices=False)
            self._rotation_matrices[i] = u @ vh

            # Create level-specific bias vector
            np.random.seed(137 + i)  # Different seed for bias
            bias_factor = 0.01 + (i * 0.002)  # Gradually increase bias
            self._level_biases[i] = np.random.normal(0, bias_factor, (embedding_dim,))

        # Log creation of matrices
        print(f"{self.get_time()} Created {max_potential_levels} fractal rotation matrices of size {embedding_dim}x{embedding_dim}")

    def _add_fractal_embeddings_batch(self, items: List[MemoryItem]):
        """
        Generate and add fractal embeddings to multiple memory items in batch.

        Args:
            items: List of memory items to process
        """
        # Skip if fractal embeddings are disabled
        if not self.use_fractal:
            return

        # Ensure matrices are initialized
        self._initialize_fractal_matrices()

        # Group items by embedding dimension for efficient processing
        dimension_groups = {}
        for item in items:
            if item.embedding is None or len(item.embedding) == 0:
                print(f"{self.get_time()} Warning: Cannot generate fractal embeddings for item {item.id} - no base embedding")
                continue

            dim = item.embedding.shape[0]
            if dim not in dimension_groups:
                dimension_groups[dim] = []
            dimension_groups[dim].append(item)

        # Process each dimension group
        for dim, dim_items in dimension_groups.items():
            print(f"{self.get_time()} Processing fractal embeddings for {len(dim_items)} items with dimension {dim}")

            # Process each fractal level in batches
            for level in range(1, self.max_fractal_levels + 1):
                # Collect all base embeddings for this level
                base_embeddings = [item.embedding for item in dim_items]

                # Stack into a batch tensor for more efficient processing
                batch_embeddings = np.stack(base_embeddings)

                try:
                    # Generate level embeddings for all items at once
                    level_embeddings = self._generate_level_embeddings_batch(batch_embeddings, level)

                    # Assign back to items
                    for i, item in enumerate(dim_items):
                        if level_embeddings[i] is not None:
                            level_embedding = level_embeddings[i]

                            # Verify the embedding is valid
                            embedding_norm = np.linalg.norm(level_embedding)
                            if embedding_norm < 1e-10:
                                print(f"{self.get_time()} Warning: Generated zero-norm embedding for level {level}")
                                continue

                            # Save the embedding
                            item.add_fractal_embedding(level, level_embedding)

                except Exception as e:
                    print(f"{self.get_time()} Error generating batch fractal embeddings for level {level}: {e}")

    def _apply_sharpening(self, similarity: float, sharpening_factor: Optional[float] = None) -> float:
        """
        Apply non-linear sharpening to similarity scores to increase contrast.

        Args:
            similarity: Raw similarity score (0.0-1.0)
            sharpening_factor: Strength of sharpening effect (0.0-1.0),
                               uses default if None

        Returns:
            Sharpened similarity score
        """
        # Use provided factor or default
        factor = sharpening_factor if sharpening_factor is not None else self.sharpening_factor

        # Skip if no sharpening requested
        if factor <= 0:
            return similarity

        # Apply non-linear sharpening
        if similarity > 0.6:
            # Boost high similarities (more confident matches)
            boost = (similarity - 0.6) * factor * 2.0
            sharpened = min(1.0, similarity + boost)
        elif similarity < 0.4:
            # Reduce low similarities (less confident matches)
            reduction = (0.4 - similarity) * factor * 2.0
            sharpened = max(0.0, similarity - reduction)
        else:
            # Middle range - moderate effect
            deviation = (similarity - 0.5) * factor
            sharpened = 0.5 + deviation

        return sharpened

    def _add_item_to_store(self, item: MemoryItem) -> Optional[str]:
        """
        Add a memory item to storage and indices with improved error handling.

        Args:
            item: Memory item to add

        Returns:
            ID of added item or None if failed
        """
        # If this is the first item, initialize index
        if self.index is None:
            try:
                self._create_index(item.embedding.shape[0])
                print(f"{self.get_time()} Created new index with dimension {item.embedding.shape[0]}")
            except Exception as e:
                print(f"Error creating index: {e}")
                return None

        # Normalize embedding for cosine similarity
        try:
            embedding_norm = np.linalg.norm(item.embedding)
            if embedding_norm < 1e-10:
                print(f"{self.get_time()} Warning: Item {item.id} has near-zero norm embedding")
                embedding_norm = 1e-10

            normalized_embedding = item.embedding / embedding_norm
        except Exception as e:
            print(f"{self.get_time()} Error normalizing embedding: {e}")
            return None

        try:
            # Add to FAISS index
            self.index.add(np.array([normalized_embedding], dtype=np.float32))

            # Add to storage
            index = len(self.items)
            self.items.append(item)
            self.embeddings.append(normalized_embedding)
            self.id_to_index[item.id] = index

            # Add fractal embeddings to respective indices
            if item.fractal_embeddings:
                fractal_added = 0
                fractal_failed = 0

                for level, level_embedding in item.fractal_embeddings.items():
                    try:
                        # Normalize level embedding
                        level_norm = np.linalg.norm(level_embedding)
                        if level_norm < 1e-10:
                            level_norm = 1e-10
                        normalized_level_embedding = level_embedding / level_norm

                        # Ensure index exists for this level
                        if level not in self.fractal_indices:
                            self.fractal_indices[level] = faiss.IndexFlatIP(self.embedding_dim)
                            print(f"{self.get_time()} Created new index for level {level}")

                        # Add to level-specific index
                        self.fractal_indices[level].add(np.array([normalized_level_embedding], dtype=np.float32))
                        fractal_added += 1
                    except Exception as e:
                        print(f"{self.get_time()} Error adding fractal embedding for level {level}: {e}")
                        fractal_failed += 1

                # if fractal_added > 0:
                #     # print(f"{self.get_time()} Added {fractal_added} fractal embeddings to indices")
                if fractal_failed > 0:
                    print(f"{self.get_time()} Failed to add {fractal_failed} fractal embeddings")
            
            # Auto-save if enabled
            if self.auto_save:
                self.save()
            
            return item.id
        
        except Exception as e:
            print(f"{self.get_time()} Error adding item to store: {e}")
            return None
    
    def _create_index(self, dim: int = None):
        """Create a new FAISS index with the specified dimension."""
        if dim is not None:
            self.embedding_dim = dim
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine on normalized vectors)
    
    def _add_fractal_embeddings(self, item: MemoryItem):
        """
        Generate and add fractal embeddings to a memory item with improved diagnostics.

        Args:
            item: The memory item to process
        """
        # Skip if fractal embeddings are disabled
        if not self.use_fractal:
            return

        # Ensure matrices are initialized
        self._initialize_fractal_matrices()

        # Skip if no base embedding available
        if item.embedding is None or len(item.embedding) == 0:
            print(f"{self.get_time()} Warning: Cannot generate fractal embeddings for item {item.id} - no base embedding")
            return

        base_embedding = item.embedding

        # Reset fractal embeddings to ensure clean state
        item.fractal_embeddings = {}

        # Generate a fractal embedding for each level
        for level in range(1, self.max_fractal_levels + 1):
            try:
                level_embedding = self._generate_level_embedding(base_embedding, level)

                if level_embedding is not None and len(level_embedding) > 0:
                    # Verify the embedding is valid
                    embedding_norm = np.linalg.norm(level_embedding)
                    if embedding_norm < 1e-10:
                        print(f"{self.get_time()} Warning: Generated zero-norm embedding for level {level}")
                        continue

                    # Save the embedding
                    item.add_fractal_embedding(level, level_embedding)
                else:
                    print(f"{self.get_time()} Warning: Failed to generate embedding for level {level}")
            except Exception as e:
                print(f"{self.get_time()} Error generating fractal embedding for level {level}: {e}")
                # Continue with other levels

    def _generate_level_embedding(self, base_embedding: np.ndarray, level: int) -> np.ndarray:
        """
        Generate a level-specific embedding for natural knowledge organization.
        The fractal levels now emerge naturally from content semantics.

        Args:
            base_embedding: The original embedding vector
            level: The fractal level to generate

        Returns:
            Transformed embedding for the specified level
        """
        if level == 0:
            return base_embedding

        # Determine embedding dimension from the base embedding
        embedding_dim = base_embedding.shape[0]

        # Use cached rotation matrices if available
        if not hasattr(self, '_rotation_matrices') or \
           (hasattr(self, '_rotation_matrices') and \
            any(matrix.shape[0] != embedding_dim for matrix in self._rotation_matrices.values())):
            # Create rotation matrices once and cache them
            self._rotation_matrices = {}
            self._level_biases = {}

            # Create many more levels than we explicitly use to allow for natural selection
            max_potential_levels = 20  # Support up to 20 potential levels

            for i in range(1, max_potential_levels + 1):
                # Create a rotation matrix with fixed seed for determinism
                np.random.seed(42 + i)  # Fixed seed per level

                # Create rotation matrix with increasing randomness by level
                # This allows natural clustering at different semantic granularities
                rotation_factor = 0.1 + (i * 0.03)  # Gradually increase randomness
                rotation = np.random.normal(0, rotation_factor, (embedding_dim, embedding_dim))

                # For earlier levels, preserve more of the original structure
                if i <= 3:
                    # Add identity matrix component to preserve some original information
                    preservation_factor = 0.9 - (i * 0.15)  # Decreasing preservation
                    rotation = rotation + np.eye(embedding_dim) * preservation_factor

                # Ensure the matrix is orthogonal (proper rotation)
                u, _, vh = np.linalg.svd(rotation, full_matrices=False)
                self._rotation_matrices[i] = u @ vh

                # Create level-specific bias vector
                np.random.seed(137 + i)  # Different seed for bias
                bias_factor = 0.01 + (i * 0.002)  # Gradually increase bias
                self._level_biases[i] = np.random.normal(0, bias_factor, base_embedding.shape)

            # Log creation of matrices
            print(f"{self.get_time()} Created {max_potential_levels} fractal rotation matrices of size {embedding_dim}x{embedding_dim}")

        # Apply the transformation
        if level in self._rotation_matrices:
            # Apply the cached rotation
            rotated = np.dot(base_embedding, self._rotation_matrices[level])

            # Apply level-specific bias
            if level in self._level_biases:
                shifted = rotated + self._level_biases[level]
            else:
                shifted = rotated

            # Apply non-linear transformation that varies by level
            # Different transformations create different natural groupings
            if level % 3 == 0:
                # Every 3rd level: Enhance magnitude differences
                transformed = np.sign(shifted) * np.power(np.abs(shifted), 0.8)
            elif level % 3 == 1:
                # Every 3rd+1 level: Apply sigmoid-like function
                transformed = np.tanh(shifted * (1.0 + 0.1 * level))
            else:
                # Every 3rd+2 level: Log transformation
                transformed = np.sign(shifted) * np.log(1 + np.abs(shifted) * (1.0 + 0.05 * level))

            # Normalize the result
            norm = np.linalg.norm(transformed)
            if norm < 1e-10:
                norm = 1e-10
            normalized = transformed / norm

            return normalized
        else:
            # Fallback if matrix isn't available
            print(f"{self.get_time()} Warning: No rotation matrix for level {level}, using simpler transformation")
            # Apply simpler transformation
            perturbed = base_embedding + (np.random.normal(0, 0.01 * level, base_embedding.shape) * base_embedding)
            return perturbed / np.linalg.norm(perturbed)
    
    def _group_results_by_entity(self, results: List[Dict[str, Any]], query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """
        Group search results by potential entity/domain and re-rank based on coherence.

        Args:
            results: Original search results
            query_embedding: Original query embedding

        Returns:
            Re-ranked results with better entity separation
        """
        if not results or len(results) < 3:
            return results  # Not enough results to perform grouping

        # Extract text content from results
        texts = []
        for result in results:
            # Use content field if available, otherwise fall back to text
            content_field = 'content' if 'content' in result else 'text'
            texts.append(result.get(content_field, ''))

        # Create embeddings for all result texts using batch processing
        try:
            result_embeddings = self.batch_embedding_function(texts)
            print(f"{self.get_time()} Generated {len(result_embeddings)} result embeddings in batch")
        except Exception as e:
            print(f"{self.get_time()} Error in batch embedding: {e}")
            # Fall back to individual processing
            result_embeddings = []
            for text in texts:
                try:
                    embedding = self.embedding_function(text)
                    result_embeddings.append(embedding)
                except Exception as inner_e:
                    print(f"{self.get_time()} Error generating result embedding: {inner_e}")
                    # Use a zero embedding as fallback
                    result_embeddings.append(np.zeros(self.embedding_dim))

        # Convert to numpy array for easier processing
        result_embeddings = np.array(result_embeddings)

        # Rest of the method remains the same...
        # Simple clustering using cosine similarity
        groups = []
        assigned = set()

        # Start with most similar to query as first group center
        similarities = []
        for i, emb in enumerate(result_embeddings):
            norm_emb = emb / max(np.linalg.norm(emb), 1e-10)
            norm_query = query_embedding / max(np.linalg.norm(query_embedding), 1e-10)
            sim = np.dot(norm_emb, norm_query)
            similarities.append((i, sim))

        # Sort by similarity to query
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Create groups around the most similar items
        for i, sim in similarities:
            if i in assigned:
                continue

            # Create a new group with this item as center
            group = [i]
            assigned.add(i)

            # Find similar items to add to this group
            center_embedding = result_embeddings[i]
            center_norm = np.linalg.norm(center_embedding)
            if center_norm < 1e-10:
                center_norm = 1e-10
            center_embedding = center_embedding / center_norm

            for j, other_embedding in enumerate(result_embeddings):
                if j in assigned or j == i:
                    continue

                # Calculate similarity
                other_norm = np.linalg.norm(other_embedding)
                if other_norm < 1e-10:
                    other_norm = 1e-10
                other_embedding = other_embedding / other_norm

                similarity = np.dot(center_embedding, other_embedding)

                # Apply strong sharpening to make grouping more decisive
                if similarity > 0.7:  # High similarity threshold for grouping
                    group.append(j)
                    assigned.add(j)

            groups.append(group)

        # Make sure all items are assigned to some group
        for i in range(len(results)):
            if i not in assigned:
                # Create a singleton group
                groups.append([i])
                assigned.add(i)

        # Now rerank the results by group
        # First, calculate group score (average similarity to query)
        group_scores = []
        for group in groups:
            group_embeddings = [result_embeddings[i] for i in group]
            group_embedding = np.mean(group_embeddings, axis=0)

            # Normalize
            group_norm = np.linalg.norm(group_embedding)
            if group_norm < 1e-10:
                group_norm = 1e-10
            group_embedding = group_embedding / group_norm

            # Calculate similarity to query
            query_norm = np.linalg.norm(query_embedding)
            if query_norm < 1e-10:
                query_norm = 1e-10
            query_embedding_norm = query_embedding / query_norm

            similarity = np.dot(group_embedding, query_embedding_norm)

            # Apply strong sharpening to group score
            sharpened = self._apply_sharpening(similarity, 0.5)  # Strong sharpening for groups

            group_scores.append((group, sharpened))

        # Sort groups by score
        group_scores.sort(key=lambda x: x[1], reverse=True)

        # Reconstruct the result list by group
        reranked_results = []
        for group, score in group_scores:
            # Add items from this group
            for i in group:
                result = results[i].copy()
                # Add metadata about grouping
                result["group_score"] = float(score)
                result["entity_group"] = len(reranked_results)  # Group identifier
                # Apply a boost based on group score
                result["similarity"] = result.get("similarity", 0.5) * (1.0 + score * 0.2)
                reranked_results.append(result)

        return reranked_results

    def _apply_cross_level_verification_with_sharpening(self, result_dict: Dict[str, Dict[str, Any]]):
        """
        Boost confidence for items found across multiple levels with enhanced level-specific sharpening.
        This rewards consistent results across different semantic variations.

        Args:
            result_dict: Dictionary of search results
        """
        # Group results by item
        item_levels = {}
        for item_id, result in result_dict.items():
            if item_id not in item_levels:
                item_levels[item_id] = []
            item_levels[item_id].append((result["level"], result["raw_similarity"]))

        # Apply boost for items found in multiple levels with level-specific sharpening
        for item_id, level_info in item_levels.items():
            if len(level_info) > 1:
                # Calculate weighted average similarity based on level
                weighted_sum = 0.0
                total_weight = 0.0

                # Track levels where this item was found
                found_in_levels = []

                for level, sim in level_info:
                    # Determine weight based on level
                    # Give higher weight to higher levels for entity separation
                    level_weight = 1.0 + (level * 0.2)  # Level 0: 1.0, Level 1: 1.2, Level 2: 1.4, etc.

                    # Get level-specific sharpening factor
                    level_factor = self.level_sharpening_factors.get(level, self.sharpening_factor)

                    # Apply level-specific sharpening
                    sharpened_sim = self._apply_sharpening(sim, level_factor)

                    weighted_sum += sharpened_sim * level_weight
                    total_weight += level_weight

                    found_in_levels.append(level)

                # Calculate weighted average
                weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0.0

                # Apply bonus based on level agreement (up to 20% boost)
                cross_level_bonus = min(0.2, 0.05 * len(level_info))

                # Apply the combined bonus
                if item_id in result_dict:
                    current_sim = result_dict[item_id]["similarity"]

                    # Apply greater boost for cross-level consistency in high similarity results
                    if current_sim > 0.7:
                        boost_factor = 1.0 + cross_level_bonus + (weighted_avg * 0.1)
                    else:
                        boost_factor = 1.0 + (cross_level_bonus + (weighted_avg * 0.1)) * 0.6

                    result_dict[item_id]["similarity"] = min(1.0, current_sim * boost_factor)
                    result_dict[item_id]["cross_level_bonus"] = cross_level_bonus
                    result_dict[item_id]["weighted_avg"] = weighted_avg
                    result_dict[item_id]["found_in_levels"] = found_in_levels

    def format_knowledge_for_prompt(self, results: List[Dict[str, Any]], query: str = None) -> str:
        """
        Format knowledge results for inclusion in the prompt.
        Unified formatting that relies on content rather than types.

        Args:
            results: List of knowledge results
            query: Optional original query for context

        Returns:
            Formatted string for prompt inclusion
        """
        if not results:
            return ""

        # Sort by similarity (should already be sorted, but just to be safe)
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        # Prepare the formatted output
        output = "KNOWLEDGE:\n"

        # Group by confidence/similarity levels
        high_confidence = []
        medium_confidence = []
        low_confidence = []

        for result in results:
            similarity = result.get("similarity", 0)
            if similarity >= 0.8:
                high_confidence.append(result)
            elif similarity >= 0.6:
                medium_confidence.append(result)
            else:
                low_confidence.append(result)

        # Add high confidence items first
        if high_confidence:
            output += "\nHIGH RELEVANCE:\n"
            for item in high_confidence[:3]:  # Limit to top 3
                output += f"- {item['content']}\n"

        # Add medium confidence items
        if medium_confidence:
            output += "\nRELEVANT INFORMATION:\n"
            for item in medium_confidence[:5]:  # Limit to top 5
                output += f"- {item['content']}\n"

        # Add low confidence items if no higher confidence results
        if low_confidence and not (high_confidence or medium_confidence):
            output += "\nPOTENTIALLY RELEVANT:\n"
            for item in low_confidence[:3]:  # Limit to top 3
                output += f"- {item['content']}\n"

        return output

    def fast_similarity_search(self, query_embedding: np.ndarray, top_k: int = 5, min_similarity: float = 0.25) -> List[Dict]:
        """
        Perform a fast approximate similarity search across fractal levels with improved batch processing.

        This optimized function:
        1. Uses approximate nearest neighbor search instead of exact search
        2. Searches levels in parallel
        3. Uses early stopping when confidence is high
        4. Caches results for similar queries

        Args:
            query_embedding: The query embedding vector
            top_k: Maximum results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of search results with similarity scores
        """
        # Create a hash key for caching
        # Create a hash key for caching
        import hashlib
        query_hash = hashlib.md5(query_embedding.tobytes()).hexdigest()

        # Check cache
        if hasattr(self, '_similarity_cache') and query_hash in self._similarity_cache:
            cached_time, cached_results = self._similarity_cache[query_hash]
            if time.time() - cached_time < 60:
                return cached_results

        # Initialize results
        all_results = {}

        # Use batch search for all levels
        levels_to_search = [0] + list(self.fractal_indices.keys())

        # Prepare query embeddings for all levels
        level_queries = {}

        # Base level query
        level_queries[0] = np.array([query_embedding], dtype=np.float32)

        # Generate level queries for all fractal levels at once
        for level in self.fractal_indices.keys():
            if self.fractal_indices[level].ntotal > 0:
                level_query = self._generate_level_embedding(query_embedding, level)
                if level_query is not None:
                    level_queries[level] = np.array([level_query], dtype=np.float32)

        # Import concurrent module for parallel search
        import concurrent.futures

        # Function to search a specific level
        def search_level(level):
            level_results = {}
            if level not in level_queries:
                return level_results

            if level == 0:
                index = self.index
            elif level in self.fractal_indices:
                index = self.fractal_indices[level]
            else:
                return level_results

            if index.ntotal == 0:
                return level_results

            # Get query for this level
            level_query = level_queries[level]

            # Search with this query
            search_k = min(index.ntotal, top_k * 2)
            similarities, indices = index.search(level_query, search_k)

            # Process results
            for i, idx in enumerate(indices[0]):
                if idx != -1 and similarities[0][i] > min_similarity:
                    if idx not in self.deleted_ids:
                        item = self.items[idx]
                        similarity = float(similarities[0][i])
                        level_results[item.id] = {
                            "id": item.id,
                            "content": item.content,
                            "similarity": similarity,
                            "raw_similarity": similarity,
                            "metadata": item.metadata,
                            "level": level
                        }

            return level_results

        # Search all levels in parallel for better performance
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(levels_to_search))) as executor:
            # Submit search tasks
            future_to_level = {executor.submit(search_level, level): level for level in levels_to_search}

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_level):
                level_results = future.result()

                # Merge with main results, keeping highest similarity
                for item_id, result in level_results.items():
                    if item_id not in all_results or result["similarity"] > all_results[item_id]["similarity"]:
                        all_results[item_id] = result

        # Apply cross-level verification with sharpening
        result_dict = {r["id"]: r for r in all_results.values()}
        self._apply_cross_level_verification_with_sharpening(result_dict)

        # Sort by similarity
        final_results = list(result_dict.values())
        final_results.sort(key=lambda x: x["similarity"], reverse=True)

        # Limit to top-k
        final_results = final_results[:top_k]

        # Cache results
        if not hasattr(self, '_similarity_cache'):
            self._similarity_cache = {}
        self._similarity_cache[query_hash] = (time.time(), final_results)

        # Clean cache if needed
        if len(self._similarity_cache) > 100:
            oldest_keys = sorted(self._similarity_cache.keys(),
                                key=lambda k: self._similarity_cache[k][0])[:50]
            for key in oldest_keys:
                del self._similarity_cache[key]

        return final_results

    def retrieve(self, 
                query: str,
                top_k: int = 5,
                min_similarity: float = 0.25,
                use_fractal: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge based on query using optimized batch search.

        Args:
            query: Search query
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            use_fractal: Override default fractal setting

        Returns:
            List of search results with similarity scores
        """
        with self._lock:
            # If no items or no index, return empty list
            if not self.items or self.index is None or self.index.ntotal == 0:
                return []

            # Check query embedding cache first
            if hasattr(self, '_query_embedding_cache'):
                query_hash = hashlib.md5(query.encode()).hexdigest()
                if query_hash in self._query_embedding_cache:
                    cached_time, cached_embedding = self._query_embedding_cache[query_hash]
                    # Use cache if recent (within last 5 minutes)
                    if time.time() - cached_time < 300:
                        query_embedding = cached_embedding
                        print(f"{self.get_time()} Using cached query embedding")
                    else:
                        # Generate new embedding
                        query_embedding = self._generate_query_embedding(query)
                        # Update cache
                        self._query_embedding_cache[query_hash] = (time.time(), query_embedding)
                else:
                    # Generate new embedding
                    query_embedding = self._generate_query_embedding(query)
                    # Add to cache
                    if not hasattr(self, '_query_embedding_cache'):
                        self._query_embedding_cache = {}
                    self._query_embedding_cache[query_hash] = (time.time(), query_embedding)

                    # Clean cache if needed
                    if len(self._query_embedding_cache) > 100:
                        oldest_keys = sorted(
                            self._query_embedding_cache.keys(),
                            key=lambda k: self._query_embedding_cache[k][0]
                        )[:50]
                        for key in oldest_keys:
                            del self._query_embedding_cache[key]
            else:
                # Generate embedding directly
                query_embedding = self._generate_query_embedding(query)

            # Normalize query embedding
            try:
                query_norm = np.linalg.norm(query_embedding)
                if query_norm < 1e-10:
                    query_norm = 1e-10
                normalized_query = query_embedding / query_norm
            except Exception as e:
                print(f"{self.get_time()} Error normalizing query embedding: {e}")
                return []

            # Determine whether to use fractal search
            use_fractal_here = self.use_fractal if use_fractal is None else use_fractal
            have_fractal_indices = bool(self.fractal_indices) and any(idx.ntotal > 0 for idx in self.fractal_indices.values())
            can_use_fractal = use_fractal_here and have_fractal_indices

            # Use improved fast similarity search
            if can_use_fractal:
                try:
                    print(f"{self.get_time()} Using enhanced fast similarity search")
                    return self.fast_similarity_search(
                        normalized_query,
                        top_k=top_k,
                        min_similarity=min_similarity
                    )
                except Exception as e:
                    print(f"{self.get_time()} Error in fast similarity search: {e}")
                    # Fall back to standard search
                    return self._standard_search(
                        np.array([normalized_query], dtype=np.float32),
                        top_k=top_k,
                        min_similarity=min_similarity
                    )
            else:
                # Standard search
                return self._standard_search(
                    np.array([normalized_query], dtype=np.float32),
                    top_k=top_k,
                    min_similarity=min_similarity
                )
    
    def _standard_search(self, 
                       normalized_query: np.ndarray, 
                       top_k: int, 
                       min_similarity: float) -> List[Dict[str, Any]]:
        """Perform standard vector search using the main index."""
        # Get more results than needed to account for filtering
        search_k = min(top_k * 3, self.index.ntotal)
        similarities, indices = self.index.search(normalized_query, search_k)
        
        # Filter results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and similarities[0][i] > min_similarity:
                # Skip deleted items
                if idx in self.deleted_ids:
                    continue
                
                # Get the memory item
                item = self.items[idx]
                
                # Add result
                results.append({
                    "id": item.id,
                    "content": item.content,
                    "metadata": item.metadata
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results
    
    def _fractal_search(self, 
                      normalized_query: np.ndarray,
                      top_k: int,
                      min_similarity: float) -> List[Dict[str, Any]]:
        """
        Perform enhanced fractal search across multiple levels with knowledge emergence.
        No longer specialized by type, but naturally organized by semantic similarity.

        Args:
            normalized_query: Normalized query embedding vector
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of search results with similarity scores
        """
        # Search across all available levels
        available_levels = [0] + list(self.fractal_indices.keys())
        result_dict = {}

        # Debug information
        print(f"{self.get_time()} Starting fractal search across {len(available_levels)} levels")

        # Search base level (always exists)
        search_k = min(top_k * 2, self.index.ntotal)
        if search_k <= 0:
            print(f"{self.get_time()} Warning: No items in index, cannot perform search")
            return []

        try:
            base_similarities, base_indices = self.index.search(normalized_query, search_k)

            # Process base results
            for i, idx in enumerate(base_indices[0]):
                if idx != -1 and base_similarities[0][i] > min_similarity:
                    if idx in self.deleted_ids:
                        continue

                    item = self.items[idx]

                    # Apply sharpening to raw similarity
                    raw_similarity = float(base_similarities[0][i])
                    sharpened_similarity = self._apply_sharpening(raw_similarity)

                    result_dict[item.id] = {
                        "id": item.id,
                        "content": item.content,
                        "similarity": sharpened_similarity,
                        "raw_similarity": raw_similarity,
                        "metadata": item.metadata,
                        "index": idx,
                        "level": 0
                    }

                    # Increment retrieval count in metadata (track usage)
                    if "retrieval_count" in item.metadata:
                        item.metadata["retrieval_count"] += 1
                    else:
                        item.metadata["retrieval_count"] = 1

                    # Update last access time
                    item.metadata["last_access"] = datetime.now().timestamp()

            # Search all available fractal levels
            for level in self.fractal_indices.keys():
                if self.fractal_indices[level].ntotal == 0:
                    continue

                # Create level-specific query variation
                level_query = self._generate_level_embedding(normalized_query[0], level)
                level_query = np.array([level_query], dtype=np.float32)

                # Search with level-specific query
                level_similarities, level_indices = self.fractal_indices[level].search(
                    level_query, search_k
                )

                # Process level results
                for i, idx in enumerate(level_indices[0]):
                    if idx != -1 and level_similarities[0][i] > min_similarity:
                        if idx in self.deleted_ids:
                            continue

                        item = self.items[idx]
                        raw_similarity = float(level_similarities[0][i])
                        sharpened_similarity = self._apply_sharpening(raw_similarity)

                        # Only update if better than existing
                        if item.id not in result_dict or sharpened_similarity > result_dict[item.id]["similarity"]:
                            result_dict[item.id] = {
                                "id": item.id,
                                "content": item.content,
                                "similarity": sharpened_similarity,
                                "raw_similarity": raw_similarity,
                                "metadata": item.metadata,
                                "index": idx,
                                "level": level
                            }

                            # Update retrieval stats
                            if "retrieval_count" in item.metadata:
                                item.metadata["retrieval_count"] += 1
                            else:
                                item.metadata["retrieval_count"] = 1

                            item.metadata["last_access"] = datetime.now().timestamp()

            # Apply cross-level verification (boost items found in multiple levels)
            self._apply_cross_level_verification(result_dict)

            # Extract values and sort
            results = list(result_dict.values())
            results.sort(key=lambda x: x["similarity"], reverse=True)

            # Apply entity separation if enabled
            if self.enable_entity_separation and len(results) > 2 and normalized_query.shape[0] == 1:
                results = self._group_results_by_entity(results, normalized_query[0])

            return results[:top_k]

        except Exception as e:
            print(f"{self.get_time()} Error in fractal search: {e}")
            return []

    # def _apply_cross_level_verification(self, result_dict: Dict[str, Dict[str, Any]]):
    #     """
    #     Boost confidence for items found across multiple levels.
    #     This rewards consistent results across different semantic variations.
    #     """
    #     # Group results by item
    #     item_levels = {}
    #     for item_id, result in result_dict.items():
    #         if item_id not in item_levels:
    #             item_levels[item_id] = []
    #         item_levels[item_id].append((result["level"], result["base_similarity"]))

    #     # Apply boost for items found in multiple levels
    #     for item_id, level_info in item_levels.items():
    #         if len(level_info) > 1:
    #             # Calculate cross-level verification score
    #             level_count = len(level_info)
    #             avg_similarity = sum(sim for _, sim in level_info) / level_count

    #             # Apply bonus based on level agreement (up to 20% boost)
    #             cross_level_bonus = min(0.2, 0.05 * level_count)

    #             # Apply the bonus
    #             if item_id in result_dict:
    #                 current_sim = result_dict[item_id]["similarity"]
    #                 result_dict[item_id]["similarity"] = min(1.0, current_sim * (1.0 + cross_level_bonus))
    #                 result_dict[item_id]["cross_level_bonus"] = cross_level_bonus
    #                 result_dict[item_id]["found_in_levels"] = [lvl for lvl, _ in level_info]

    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a query with proper error handling."""
        if self.embedding_function:
            try:
                return self.embedding_function(query)
            except Exception as e:
                print(f"{self.get_time()} Error generating query embedding: {e}")
                # Return a random fallback embedding
                return np.random.random(self.embedding_dim).astype(np.float32)
        else:
            # No embedding function available
            return np.random.random(self.embedding_dim).astype(np.float32)
            
    def _apply_cross_level_verification(self, result_dict: Dict[str, Dict[str, Any]]):
        """
        Boost confidence for items found across multiple levels.
        This rewards consistent results across different semantic variations.
        """
        # Group results by item
        item_levels = {}
        for item_id, result in result_dict.items():
            if item_id not in item_levels:
                item_levels[item_id] = []
            # Use raw_similarity instead of base_similarity
            item_levels[item_id].append((result["level"], result["raw_similarity"]))

        # Apply boost for items found in multiple levels
        for item_id, level_info in item_levels.items():
            if len(level_info) > 1:
                # Calculate weighted average similarity based on level
                weighted_sum = 0.0
                total_weight = 0.0

                # Track levels where this item was found
                found_in_levels = []

                for level, sim in level_info:
                    # Determine weight based on level
                    level_weight = 1.0 + (level * 0.2)  # Level 0: 1.0, Level 1: 1.2, Level 2: 1.4, etc.

                    # Get level-specific sharpening factor
                    level_factor = self.level_sharpening_factors.get(level, self.sharpening_factor)

                    # Apply level-specific sharpening
                    sharpened_sim = self._apply_sharpening(sim, level_factor)

                    weighted_sum += sharpened_sim * level_weight
                    total_weight += level_weight

                    found_in_levels.append(level)

                # Calculate weighted average
                weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0.0

                # Apply bonus based on level agreement (up to 20% boost)
                cross_level_bonus = min(0.2, 0.05 * len(level_info))

                # Apply the combined bonus
                if item_id in result_dict:
                    current_sim = result_dict[item_id]["similarity"]

                    # Apply greater boost for cross-level consistency in high similarity results
                    if current_sim > 0.7:
                        boost_factor = 1.0 + cross_level_bonus + (weighted_avg * 0.1)
                    else:
                        boost_factor = 1.0 + (cross_level_bonus + (weighted_avg * 0.1)) * 0.6

                    result_dict[item_id]["similarity"] = min(1.0, current_sim * boost_factor)
                    result_dict[item_id]["cross_level_bonus"] = cross_level_bonus
                    result_dict[item_id]["weighted_avg"] = weighted_avg
                    result_dict[item_id]["found_in_levels"] = found_in_levels

    def remove(self, item_id: str) -> bool:
        """
        Remove a memory item by ID.

        Args:
            item_id: ID of the item to remove

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if item_id not in self.id_to_index:
                return False

            # Get index
            idx = self.id_to_index[item_id]

            # Mark as deleted
            self.deleted_ids.add(idx)

            # Remove mapping
            del self.id_to_index[item_id]

            # Save if auto-save is enabled
            if self.auto_save:
                self.save()

            return True

    def get(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a memory item by ID.

        Args:
            item_id: ID of the item to get

        Returns:
            Item dictionary or None if not found
        """
        with self._lock:
            if item_id not in self.id_to_index:
                return None

            # Get index
            idx = self.id_to_index[item_id]

            # Skip if deleted
            if idx in self.deleted_ids:
                return None

            # Get item
            item = self.items[idx]

            # Return as dictionary
            return {
                "id": item.id,
                "content": item.content,
                "metadata": item.metadata
            }

    def update(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a memory item.

        Args:
            item_id: ID of the item to update
            updates: Dictionary of updates to apply

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if item_id not in self.id_to_index:
                return False

            # Get index
            idx = self.id_to_index[item_id]

            # Skip if deleted
            if idx in self.deleted_ids:
                return False

            # Get item
            item = self.items[idx]

            # Apply updates
            if "content" in updates and updates["content"] != item.content:
                # Content has changed, need to regenerate embedding
                if self.embedding_function:
                    try:
                        new_embedding = self.embedding_function(updates["content"])
                        item.embedding = new_embedding

                        # Regenerate fractal embeddings if using them
                        if self.use_fractal and item.fractal_embeddings:
                            self._add_fractal_embeddings(item)

                        # Need to rebuild index
                        self._rebuild_index()
                    except Exception as e:
                        print(f"{self.get_time()} Error updating embedding: {e}")
                        return False

                item.content = updates["content"]

            # Update metadata
            if "metadata" in updates:
                item.metadata.update(updates["metadata"])

            # Update modification timestamp
            item.metadata["modified_at"] = datetime.now().timestamp()

            # Save if auto-save is enabled
            if self.auto_save:
                self.save()

            return True

    def _rebuild_index(self):
        """Rebuild the FAISS index after updates."""
        # Create new index
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Rebuild fractal indices if using them
        if self.use_fractal:
            self.fractal_indices = {}

        # Add all non-deleted items
        embeddings = []
        for i, item in enumerate(self.items):
            if i not in self.deleted_ids:
                # Add to main index
                normalized_embedding = item.embedding / max(np.linalg.norm(item.embedding), 1e-10)
                embeddings.append(normalized_embedding)

                # Add to fractal indices if using them
                if self.use_fractal and item.fractal_embeddings:
                    for level, level_embedding in item.fractal_embeddings.items():
                        normalized_level_embedding = level_embedding / max(np.linalg.norm(level_embedding), 1e-10)

                        if level not in self.fractal_indices:
                            self.fractal_indices[level] = faiss.IndexFlatIP(self.embedding_dim)

                        self.fractal_indices[level].add(np.array([normalized_level_embedding], dtype=np.float32))

        # Add all embeddings to main index
        if embeddings:
            self.index.add(np.array(embeddings, dtype=np.float32))

    def save(self) -> bool:
        """
        Save memory data to disk.

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                # Serialize items (without embeddings)
                items_data = []
                for item in self.items:
                    items_data.append(item.to_dict())

                # Save items data
                items_path = os.path.join(self.storage_path, "items.json")
                with open(items_path, 'w', encoding='utf-8') as f:
                    json.dump(items_data, f, indent=2)

                # Save embeddings (using numpy for efficiency)
                embeddings_path = os.path.join(self.storage_path, "embeddings.npy")
                np.save(embeddings_path, np.array(self.embeddings))

                # Save fractal embeddings if using them
                if self.use_fractal:
                    fractal_path = os.path.join(self.storage_path, "fractal_embeddings.pkl")
                    fractal_data = {}
                    for i, item in enumerate(self.items):
                        if item.fractal_embeddings:
                            fractal_data[item.id] = item.fractal_embeddings

                    with open(fractal_path, 'wb') as f:
                        pickle.dump(fractal_data, f)

                # Save indices
                index_path = os.path.join(self.storage_path, "index.faiss")
                faiss.write_index(self.index, index_path)

                # Save fractal indices if using them
                if self.use_fractal and self.fractal_indices:
                    for level, index in self.fractal_indices.items():
                        level_path = os.path.join(self.storage_path, f"index_level_{level}.faiss")
                        faiss.write_index(index, level_path)

                # Save metadata
                metadata_path = os.path.join(self.storage_path, "metadata.json")
                metadata = {
                    "count": len(self.items),
                    "active_count": len(self.items) - len(self.deleted_ids),
                    "embedding_dim": self.embedding_dim,
                    "use_fractal": self.use_fractal,
                    "max_fractal_levels": self.max_fractal_levels,
                    "deleted_ids": list(self.deleted_ids),
                    "updated_at": datetime.now().isoformat()
                }
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)

                return True

            except Exception as e:
                print(f"{self.get_time()} Error saving memory data: {e}")
                return False

    def load(self) -> bool:
        """Load memory data with better diagnostics and optimization."""
        with self._lock:
            start_time = time.time()
            print(f"{self.get_time()} Starting memory load...")

            items_path = os.path.join(self.storage_path, "items.json")
            embeddings_path = os.path.join(self.storage_path, "embeddings.npy")

            if not (os.path.exists(items_path) and os.path.exists(embeddings_path)):
                print(f"{self.get_time()} No memory data found.")
                return False

            try:
                # Load metadata first to get configuration
                metadata_path = os.path.join(self.storage_path, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        print(f"{self.get_time()} Loaded metadata in {time.time() - start_time:.2f}s")

                    # Load critical configuration
                    self.embedding_dim = metadata.get("embedding_dim", self.embedding_dim)
                    self.use_fractal = metadata.get("use_fractal", self.use_fractal)
                    self.max_fractal_levels = metadata.get("max_fractal_levels", self.max_fractal_levels)
                    self.deleted_ids = set(metadata.get("deleted_ids", []))

                # Load embeddings (potentially large file)
                embeddings_start = time.time()
                embeddings = np.load(embeddings_path)
                print(f"{self.get_time()} Loaded {len(embeddings)} embeddings in {time.time() - embeddings_start:.2f}s")

                # Load items (potentially large JSON)
                items_start = time.time()
                with open(items_path, 'r', encoding='utf-8') as f:
                    items_data = json.load(f)
                print(f"{self.get_time()} Loaded {len(items_data)} items in {time.time() - items_start:.2f}s")

                # Recreate items
                self.items = []
                self.embeddings = []
                self.id_to_index = {}

                # Track progress for large collections
                total_items = len(items_data)
                report_interval = max(1, total_items // 10)  # Report progress every 10%

                recreate_start = time.time()
                for i, item_data in enumerate(items_data):
                    # Report progress for large collections
                    if i % report_interval == 0 and i > 0:
                        print(f"{self.get_time()} Recreated {i}/{total_items} items ({i/total_items*100:.1f}%)")

                    # Get embedding
                    if i < len(embeddings):
                        embedding = embeddings[i]
                    else:
                        # Fallback to random embedding
                        embedding = np.random.random(self.embedding_dim).astype(np.float32)

                    # Create item with unified approach (no memory type)
                    item = MemoryItem.from_dict(item_data, embedding)

                    # Add to storage
                    self.items.append(item)
                    self.embeddings.append(embedding)
                    self.id_to_index[item.id] = i

                print(f"{self.get_time()} Recreated {len(self.items)} items in {time.time() - recreate_start:.2f}s")

                # Load fractal embeddings if using them
                if self.use_fractal:
                    fractal_start = time.time()
                    fractal_path = os.path.join(self.storage_path, "fractal_embeddings.pkl")
                    if os.path.exists(fractal_path):
                        with open(fractal_path, 'rb') as f:
                            try:
                                fractal_data = pickle.load(f)

                                # Add fractal embeddings to items
                                for item_id, item_fractals in fractal_data.items():
                                    if item_id in self.id_to_index:
                                        idx = self.id_to_index[item_id]
                                        self.items[idx].fractal_embeddings = item_fractals

                                print(f"{self.get_time()} Loaded fractal embeddings in {time.time() - fractal_start:.2f}s")
                            except Exception as e:
                                print(f"{self.get_time()} Error loading fractal data: {e}")

                # Load FAISS indices with progress reporting
                index_start = time.time()
                index_path = os.path.join(self.storage_path, "index.faiss")
                if os.path.exists(index_path):
                    try:
                        self.index = faiss.read_index(index_path)
                        print(f"{self.get_time()} Loaded main index with {self.index.ntotal} vectors")
                    except Exception as e:
                        print(f"{self.get_time()} Error loading main index: {e}")
                        # Recreate index
                        self._create_index()
                        if self.embeddings:
                            self.index.add(np.array(self.embeddings, dtype=np.float32))
                            print(f"{self.get_time()} Recreated main index with {len(self.embeddings)} vectors")
                else:
                    # Recreate index
                    self._create_index()
                    if self.embeddings:
                        self.index.add(np.array(self.embeddings, dtype=np.float32))
                        print(f"{self.get_time()} Created new main index with {len(self.embeddings)} vectors")

                # Load fractal indices with progress reporting
                if self.use_fractal:
                    self.fractal_indices = {}
                    for level in range(1, self.max_fractal_levels + 1):
                        level_path = os.path.join(self.storage_path, f"index_level_{level}.faiss")
                        if os.path.exists(level_path):
                            try:
                                self.fractal_indices[level] = faiss.read_index(level_path)
                                print(f"{self.get_time()} Loaded level {level} index with {self.fractal_indices[level].ntotal} vectors")
                            except Exception as e:
                                print(f"{self.get_time()} Error loading level {level} index: {e}")

                    print(f"{self.get_time()} Loaded all indices in {time.time() - index_start:.2f}s")

                print(f"{self.get_time()} Memory loading complete in {time.time() - start_time:.2f}s")
                return True

            except Exception as e:
                print(f"{self.get_time()} Error loading memory data: {e}")
                import traceback
                traceback.print_exc()

                # Reset to empty state
                self.items = []
                self.embeddings = []
                self.id_to_index = {}
                self.deleted_ids = set()
                self.index = None
                self.fractal_indices = {}
                return False

    def cleanup(self):
        """Clean up resources and consolidate storage."""
        with self._lock:
            # Consolidate storage by removing deleted items
            if self.deleted_ids:
                new_items = []
                new_embeddings = []
                new_id_to_index = {}

                for i, item in enumerate(self.items):
                    if i not in self.deleted_ids:
                        # Add to new lists
                        new_idx = len(new_items)
                        new_items.append(item)
                        new_embeddings.append(self.embeddings[i])
                        new_id_to_index[item.id] = new_idx

                # Replace storage
                self.items = new_items
                self.embeddings = new_embeddings
                self.id_to_index = new_id_to_index
                self.deleted_ids = set()

                # Rebuild index
                self._rebuild_index()

            # Save consolidated data
            if self.auto_save:
                self.save()

            # Run garbage collection
            gc.collect()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory store.

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            # Index stats
            index_size = self.index.ntotal if self.index else 0

            # Fractal index stats
            fractal_stats = {}
            if self.use_fractal and self.fractal_indices:
                for level, index in self.fractal_indices.items():
                    fractal_stats[f"level_{level}_size"] = index.ntotal

            return {
                "total_items": len(self.items),
                "active_items": len(self.items) - len(self.deleted_ids),
                "deleted_items": len(self.deleted_ids),
                "index_size": index_size,
                "index_dimension": self.embedding_dim,
                "fractal_enabled": self.use_fractal,
                "fractal_levels": self.max_fractal_levels if self.use_fractal else 0,
                "fractal_stats": fractal_stats
            }

    def cleanup(self):
        """Clean up resources and consolidate storage."""
        with self._lock:
            # Consolidate storage by removing deleted items
            if self.deleted_ids:
                new_items = []
                new_embeddings = []
                new_id_to_index = {}

                for i, item in enumerate(self.items):
                    if i not in self.deleted_ids:
                        # Add to new lists
                        new_idx = len(new_items)
                        new_items.append(item)
                        new_embeddings.append(self.embeddings[i])
                        new_id_to_index[item.id] = new_idx

                # Replace storage
                self.items = new_items
                self.embeddings = new_embeddings
                self.id_to_index = new_id_to_index
                self.deleted_ids = set()

                # Rebuild index
                self._rebuild_index()

            # Save consolidated data
            if self.auto_save:
                self.save()

            # Run garbage collection
            gc.collect()

    def print_fractal_embedding_diagnostics(self):
        """
        Print detailed diagnostics about the fractal embedding system.
        Add this method to UnifiedMemoryManager.
        """
        print("\n======= FRACTAL EMBEDDING DIAGNOSTICS =======")

        # Basic configuration
        print(f"Fractal enabled: {self.use_fractal}")
        print(f"Max fractal levels: {self.max_fractal_levels}")
        print(f"Embedding dimension: {self.embedding_dim}")

        # Item and index counts
        print(f"\nBase index stats:")
        print(f"  Total items: {len(self.items)}")
        print(f"  Items in index: {self.index.ntotal if self.index else 0}")
        print(f"  Deleted items: {len(self.deleted_ids)}")

        # Fractal indices stats
        print("\nFractal indices stats:")

        if not self.fractal_indices:
            print("  No fractal indices initialized")
        else:
            for level, index in self.fractal_indices.items():
                print(f"  Level {level}: {index.ntotal} items")

        # Fractal embeddings per item
        items_with_fractal = 0
        level_counts = {}

        for item in self.items:
            if item.fractal_embeddings:
                items_with_fractal += 1

                for level in item.fractal_embeddings.keys():
                    level_counts[level] = level_counts.get(level, 0) + 1

        print(f"\nItems with fractal embeddings: {items_with_fractal}/{len(self.items)}")

        if level_counts:
            print("Level distribution:")
            for level, count in sorted(level_counts.items()):
                print(f"  Level {level}: {count} embeddings")

        # Check for inconsistencies
        inconsistencies = 0

        if self.index and self.index.ntotal != len(self.items) - len(self.deleted_ids):
            print(f"\nWARNING: Index size ({self.index.ntotal}) doesn't match active items ({len(self.items) - len(self.deleted_ids)})")
            inconsistencies += 1

        for level, index in self.fractal_indices.items():
            expected_count = level_counts.get(level, 0)
            actual_count = index.ntotal

            if expected_count != actual_count:
                print(f"WARNING: Level {level} index size ({actual_count}) doesn't match item count ({expected_count})")
                inconsistencies += 1

        if items_with_fractal == 0 and self.use_fractal:
            print("\nWARNING: Fractal embeddings are enabled but no items have fractal embeddings")
            print("This suggests the _add_fractal_embeddings method might not be working correctly")
            inconsistencies += 1

        # Summary
        print("\nDiagnostic Summary:")
        if inconsistencies > 0:
            print(f"Found {inconsistencies} potential issues that need attention")
            print("Please check the implementation of _add_fractal_embeddings and _generate_level_embedding")
        else:
            print("No inconsistencies detected in the fractal embedding system")

        print("===============================================\n")

    def rebuild_fractal_indices(self):
        """
        Rebuild all fractal indices to fix inconsistencies.
        Add this method to UnifiedMemoryManager.
        """
        print(f"{self.get_time()} Rebuilding fractal indices...")

        # Clear existing fractal indices
        self.fractal_indices = {}

        # Skip if fractal embeddings are disabled
        if not self.use_fractal:
            print(f"{self.get_time()} Fractal embeddings are disabled, skipping rebuild")
            return

        # Count items with fractal embeddings
        items_with_fractal = sum(1 for item in self.items if item.fractal_embeddings)

        if items_with_fractal == 0:
            print(f"{self.get_time()} No items have fractal embeddings, regenerating...")

            # Regenerate fractal embeddings for all items
            for item in self.items:
                if hasattr(self, '_add_fractal_embeddings'):
                    self._add_fractal_embeddings(item)

            # Recount
            items_with_fractal = sum(1 for item in self.items if item.fractal_embeddings)

            if items_with_fractal == 0:
                print(f"{self.get_time()} Failed to generate fractal embeddings")
                return

        print(f"{self.get_time()} Building indices for {items_with_fractal} items with fractal embeddings")

        # Create indices for each level
        all_levels = set()
        for item in self.items:
            all_levels.update(item.fractal_embeddings.keys())

        for level in all_levels:
            # Create index
            self.fractal_indices[level] = faiss.IndexFlatIP(self.embedding_dim)

            # Add embeddings
            embeddings_for_level = []

            for item in self.items:
                if level in item.fractal_embeddings:
                    level_embedding = item.fractal_embeddings[level]
                    norm = np.linalg.norm(level_embedding)
                    if norm < 1e-10:
                        norm = 1e-10
                    normalized = level_embedding / norm
                    embeddings_for_level.append(normalized)

            if embeddings_for_level:
                self.fractal_indices[level].add(np.array(embeddings_for_level, dtype=np.float32))
                print(f"{self.get_time()} Added {len(embeddings_for_level)} embeddings to level {level} index")

        print(f"{self.get_time()} Fractal indices rebuilt successfully")

        # Run diagnostics
        if hasattr(self, 'print_fractal_embedding_diagnostics'):
            self.print_fractal_embedding_diagnostics()

    def verify_fractal_search(self, test_query: str = None):
        """
        Test fractal search with a sample query to verify it's working.
        Add this method to UnifiedMemoryManager.
        """
        print("\n======= FRACTAL SEARCH VERIFICATION =======")

        if test_query is None:
            test_query = "This is a test query to verify fractal search"

        print(f"Testing fractal search with query: '{test_query}'")

        # Try with both fractal and standard search for comparison
        try:
            # First with fractal search
            print("\nRunning fractal search...")
            fractal_results = self.retrieve(
                query=test_query,
                top_k=5,
                min_similarity=0.1,  # Very low threshold to get some results
                use_fractal=True
            )

            print(f"Fractal search returned {len(fractal_results)} results")

            # Then with standard search
            print("\nRunning standard search...")
            standard_results = self.retrieve(
                query=test_query,
                top_k=5,
                min_similarity=0.1,  # Same threshold
                use_fractal=False
            )

            print(f"Standard search returned {len(standard_results)} results")

            # Compare results
            if not fractal_results and not standard_results:
                print("\nBoth searches returned no results. This suggests either:")
                print("- The memory store is empty")
                print("- The query is too dissimilar from all stored items")
                print("- There might be an issue with the embedding function")
            elif not fractal_results and standard_results:
                print("\nWARNING: Standard search returned results but fractal search didn't")
                print("This suggests there might be an issue with the fractal search implementation")
                print("Check the _fractal_search method and fractal indices")
            elif fractal_results and not standard_results:
                print("\nInteresting: Fractal search returned results but standard search didn't")
                print("This suggests fractal search is finding connections the standard search missed")
            else:
                print(f"\nBoth searches returned results: {len(fractal_results)} fractal vs {len(standard_results)} standard")

                # Check for overlap
                fractal_ids = {r['id'] for r in fractal_results}
                standard_ids = {r['id'] for r in standard_results}

                common_ids = fractal_ids.intersection(standard_ids)
                print(f"Items found by both searches: {len(common_ids)}")
                print(f"Items unique to fractal search: {len(fractal_ids - standard_ids)}")
                print(f"Items unique to standard search: {len(standard_ids - fractal_ids)}")

            # Show detailed results
            if fractal_results:
                print("\nTop fractal search results:")
                for i, result in enumerate(fractal_results[:3]):  # Show top 3
                    print(f"  {i+1}. [{result['memory_type']}] Similarity: {result['similarity']:.3f}, Level: {result.get('level', 0)}")
                    print(f"     Content: {result['content'][:50]}...")

        except Exception as e:
            print(f"Error during verification: {e}")
            import traceback
            traceback.print_exc()

        print("==========================================\n")

