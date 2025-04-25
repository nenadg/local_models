"""
Unified memory system for TinyLlama Chat.
Provides a cohesive approach to storing and retrieving different types of information
with optional fractal embedding support for enhanced semantic search.
"""

import os
import json
import faiss
import torch
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
    Standardized structure for all memory items with consistent metadata.
    """
    
    def __init__(self, 
                content: str, 
                embedding: np.ndarray,
                memory_type: str = "general",
                source: str = "conversation",
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a memory item.
        
        Args:
            content: The actual text/content
            embedding: Vector representation
            memory_type: Type of memory ("conversation", "command", "knowledge", etc.)
            source: Where this memory came from
            metadata: Additional metadata specific to the type
        """
        self.content = content
        self.embedding = embedding
        self.memory_type = memory_type
        self.fractal_embeddings = {}  # Optional multi-level representations
        
        # Initialize metadata
        self.metadata = metadata or {}
        self.metadata["source"] = source
        self.metadata["timestamp"] = datetime.now().timestamp()
        self.metadata["memory_type"] = memory_type
        
        # Generate a unique ID for this item
        self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for this memory item."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()
        timestamp = str(int(self.metadata["timestamp"] * 1000))
        return f"{self.memory_type}_{content_hash[:8]}_{timestamp[-6:]}"
    
    def add_fractal_embedding(self, level: int, embedding: np.ndarray):
        """Add a fractal embedding at the specified level."""
        self.fractal_embeddings[level] = embedding
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "metadata": self.metadata,
            # Embeddings are stored separately for efficiency
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: np.ndarray) -> 'MemoryItem':
        """Create a MemoryItem from a dictionary and embedding."""
        item = cls(
            content=data["content"],
            embedding=embedding,
            memory_type=data["memory_type"],
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
                auto_save: bool = True):
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
    
    def add(self, 
           content: str, 
           memory_type: str = "general",
           source: str = "conversation",
           metadata: Optional[Dict[str, Any]] = None,
           use_fractal: Optional[bool] = None) -> Optional[str]:
        """
        Add a new memory item.
        
        Args:
            content: Text content to remember
            memory_type: Type of memory
            source: Source of the memory
            metadata: Additional metadata
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
                    print(f"Error generating embedding: {e}")
                    return None
            else:
                # Random embedding if no function available (for testing)
                embedding = np.random.random(self.embedding_dim).astype(np.float32)
            
            # Create memory item
            item = MemoryItem(
                content=content,
                embedding=embedding,
                memory_type=memory_type,
                source=source,
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
        Efficiently add multiple memory items in a single operation.

        Args:
            items: List of dictionaries with content, memory_type, source, and metadata
            use_fractal: Override default fractal setting

        Returns:
            List of item IDs or None for failed items
        """
        with self._lock:
            # Temporarily disable auto_save
            original_auto_save = self.auto_save
            self.auto_save = False

            # If this is the first item, initialize index
            if self.index is None and len(items) > 0:
                # Find first item with content to estimate dimensionality
                for item in items:
                    content = item.get('content')
                    if content and content.strip():
                        # Generate embedding to get dimensionality
                        if self.embedding_function:
                            try:
                                embedding = self.embedding_function(content)
                                self._create_index(embedding.shape[0])
                                break
                            except Exception:
                                pass

                # If still no index, create with default dimensions
                if self.index is None:
                    self._create_index()

            # Generate all embeddings in parallel
            all_embeddings = []
            all_memory_items = []
            item_ids = []

            try:
                import concurrent.futures

                # Define embedding generator function
                def generate_embedding_for_item(item_dict):
                    content = item_dict.get('content', '')
                    if not content or not content.strip():
                        return None, None

                    memory_type = item_dict.get('memory_type', 'general')
                    source = item_dict.get('source', 'conversation')
                    metadata = item_dict.get('metadata', {})

                    # Generate embedding
                    if self.embedding_function:
                        try:
                            embedding = self.embedding_function(content)
                            # Create memory item
                            item = MemoryItem(
                                content=content,
                                embedding=embedding,
                                memory_type=memory_type,
                                source=source,
                                metadata=metadata
                            )
                            return item, embedding
                        except Exception:
                            pass

                    return None, None

                # Process items in parallel (up to 8 at a time)
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(generate_embedding_for_item, item) for item in items]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            item, embedding = future.result()
                            if item and embedding is not None:
                                all_memory_items.append(item)
                                all_embeddings.append(embedding)
                        except Exception:
                            # Skip failed items
                            item_ids.append(None)

                # Determine whether to use fractal embeddings
                use_fractal_here = self.use_fractal if use_fractal is None else use_fractal

                # Generate fractal embeddings for all items if requested (batched)
                if use_fractal_here:
                    for item in all_memory_items:
                        self._add_fractal_embeddings(item)

                # Add all items to storage at once
                if all_memory_items:
                    # Prepare normalized embeddings for the main index
                    normalized_embeddings = []
                    for embedding in all_embeddings:
                        norm = np.linalg.norm(embedding)
                        if norm < 1e-10:
                            norm = 1e-10
                        normalized_embeddings.append(embedding / norm)

                    # Add to main index
                    self.index.add(np.array(normalized_embeddings, dtype=np.float32))

                    # Add to storage
                    for i, item in enumerate(all_memory_items):
                        index = len(self.items)
                        self.items.append(item)
                        self.embeddings.append(normalized_embeddings[i])
                        self.id_to_index[item.id] = index
                        item_ids.append(item.id)

                    # Add fractal embeddings to respective indices
                    if use_fractal_here:
                        for level in range(1, self.max_fractal_levels + 1):
                            level_embeddings = []
                            items_with_level = []

                            for item in all_memory_items:
                                if level in item.fractal_embeddings:
                                    level_embedding = item.fractal_embeddings[level]
                                    norm = np.linalg.norm(level_embedding)
                                    if norm < 1e-10:
                                        norm = 1e-10
                                    level_embeddings.append(level_embedding / norm)
                                    items_with_level.append(item)

                            if level_embeddings:
                                # Ensure index exists for this level
                                if level not in self.fractal_indices:
                                    self.fractal_indices[level] = faiss.IndexFlatIP(self.embedding_dim)

                                # Add all embeddings at once
                                self.fractal_indices[level].add(np.array(level_embeddings, dtype=np.float32))

                # Finally, save the updated data
                if self.auto_save:
                    self.save()

                return item_ids

            finally:
                # Restore original auto_save setting
                self.auto_save = original_auto_save

    def _apply_sharpening(self, similarity: float, sharpening_factor: float) -> float:
        """
        Apply non-linear sharpening to similarity scores to increase contrast.

        Args:
            similarity: Raw similarity score (0.0-1.0)
            sharpening_factor: Strength of sharpening effect (0.0-1.0)

        Returns:
            Sharpened similarity score
        """
        # Skip if no sharpening requested
        if sharpening_factor <= 0:
            return similarity

        # Apply non-linear sharpening
        if similarity > 0.6:
            # Boost high similarities (more confident matches)
            boost = (similarity - 0.6) * sharpening_factor * 2.0
            sharpened = min(1.0, similarity + boost)
        elif similarity < 0.4:
            # Reduce low similarities (less confident matches)
            reduction = (0.4 - similarity) * sharpening_factor * 2.0
            sharpened = max(0.0, similarity - reduction)
        else:
            # Middle range - moderate effect
            deviation = (similarity - 0.5) * sharpening_factor
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
                print(f"Created new index with dimension {item.embedding.shape[0]}")
            except Exception as e:
                print(f"Error creating index: {e}")
                return None

        # Normalize embedding for cosine similarity
        try:
            embedding_norm = np.linalg.norm(item.embedding)
            if embedding_norm < 1e-10:
                print(f"Warning: Item {item.id} has near-zero norm embedding")
                embedding_norm = 1e-10

            normalized_embedding = item.embedding / embedding_norm
        except Exception as e:
            print(f"Error normalizing embedding: {e}")
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
                            print(f"Created new index for level {level}")

                        # Add to level-specific index
                        self.fractal_indices[level].add(np.array([normalized_level_embedding], dtype=np.float32))
                        fractal_added += 1
                    except Exception as e:
                        print(f"Error adding fractal embedding for level {level}: {e}")
                        fractal_failed += 1

                if fractal_added > 0:
                    print(f"Added {fractal_added} fractal embeddings to indices")
                if fractal_failed > 0:
                    print(f"Failed to add {fractal_failed} fractal embeddings")
            
            # Auto-save if enabled
            if self.auto_save:
                self.save()
            
            return item.id
        
        except Exception as e:
            print(f"Error adding item to store: {e}")
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

        # Skip if no base embedding available
        if item.embedding is None or len(item.embedding) == 0:
            print(f"Warning: Cannot generate fractal embeddings for item {item.id} - no base embedding")
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
                        print(f"Warning: Generated zero-norm embedding for level {level}")
                        continue

                    # Save the embedding
                    item.add_fractal_embedding(level, level_embedding)
                else:
                    print(f"Warning: Failed to generate embedding for level {level}")
            except Exception as e:
                print(f"Error generating fractal embedding for level {level}: {e}")
                # Continue with other levels
    
    # In UnifiedMemoryManager._generate_level_embedding
    def _generate_level_embedding(self, base_embedding: np.ndarray, level: int) -> np.ndarray:
        """
        Generate a level-specific embedding with improved fractal transformations.

        Args:
            base_embedding: The original embedding vector
            level: The fractal level to generate

        Returns:
            Transformed embedding for the specified level
        """
        if level == 0:
            return base_embedding

        # Use cached rotation matrices if available
        if not hasattr(self, '_rotation_matrices'):
            # Create rotation matrices once and cache them
            self._rotation_matrices = {}
            for i in range(1, self.max_fractal_levels + 1):
                # Create a rotation matrix with fixed seed for determinism
                np.random.seed(42 + i)  # Fixed seed per level
                rotation = np.random.normal(0, 0.1 * i, (self.embedding_dim, self.embedding_dim))
                # Ensure the matrix is orthogonal (proper rotation)
                u, _, vh = np.linalg.svd(rotation, full_matrices=False)
                self._rotation_matrices[i] = u @ vh

                # Log creation of matrix
                print(f"Created fractal rotation matrix for level {i}")

        # Apply the transformation with more pronounced level-dependent changes
        if level in self._rotation_matrices:
            # Apply the cached rotation
            rotated = np.dot(base_embedding, self._rotation_matrices[level])

            # Add a small level-dependent shift
            np.random.seed(137 + level)  # Different seed for shift
            shift = np.random.normal(0, 0.02 * level, rotated.shape)
            shifted = rotated + shift

            # Normalize the result
            normalized = shifted / np.linalg.norm(shifted)

            return normalized
        else:
            # Fallback if matrix isn't available
            print(f"Warning: No rotation matrix for level {level}, using simpler transformation")
            # Apply simpler transformation
            perturbed = base_embedding + (np.random.normal(0, 0.01 * level, base_embedding.shape) * base_embedding)
            return perturbed / np.linalg.norm(perturbed)
    
    def _apply_cross_level_verification_with_sharpening(self, result_dict: Dict[str, Dict[str, Any]], sharpening_factor: float):
        """
        Boost confidence for items found across multiple levels with enhanced sharpening.
        This rewards consistent results across different semantic variations.

        Args:
            result_dict: Dictionary of search results
            sharpening_factor: Base sharpening factor to apply
        """
        # Group results by item
        item_levels = {}
        for item_id, result in result_dict.items():
            if item_id not in item_levels:
                item_levels[item_id] = []
            item_levels[item_id].append((result["level"], result["raw_similarity"]))

        # Apply boost for items found in multiple levels with sharpening
        for item_id, level_info in item_levels.items():
            if len(level_info) > 1:
                # Calculate cross-level verification score
                level_count = len(level_info)
                avg_similarity = sum(sim for _, sim in level_info) / level_count

                # Apply sharpening to the average similarity
                sharpened_avg = self._apply_sharpening(avg_similarity, sharpening_factor)

                # Apply bonus based on level agreement (up to 20% boost)
                cross_level_bonus = min(0.2, 0.05 * level_count)

                # Apply an additional sharpening boost
                sharpening_boost = 0.1 * sharpened_avg * sharpening_factor

                # Apply the combined bonus
                if item_id in result_dict:
                    current_sim = result_dict[item_id]["similarity"]

                    # Apply greater boost for cross-level consistency in high similarity results
                    if current_sim > 0.7:
                        boost_factor = 1.0 + cross_level_bonus + sharpening_boost
                    else:
                        boost_factor = 1.0 + (cross_level_bonus + sharpening_boost) * 0.5

                    result_dict[item_id]["similarity"] = min(1.0, current_sim * boost_factor)
                    result_dict[item_id]["cross_level_bonus"] = cross_level_bonus
                    result_dict[item_id]["sharpening_boost"] = sharpening_boost
                    result_dict[item_id]["found_in_levels"] = [lvl for lvl, _ in level_info]
                    
    def retrieve(self, 
                query: str,
                memory_types: Optional[List[str]] = None,
                top_k: int = 5,
                min_similarity: float = 0.25,
                use_fractal: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on query with improved diagnostics and error handling.

        Args:
            query: Search query
            memory_types: Optional filter by memory types
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            use_fractal: Override default fractal setting

        Returns:
            List of search results with similarity scores
        """
        with self._lock:
            # If no items or no index, return empty list
            if not self.items or self.index is None:
                print("Cannot search: No items or index not initialized")
                return []

            if self.index.ntotal == 0:
                print("Cannot search: Index is empty")
                return []

            # Generate query embedding
            start_time = time.time()
            if self.embedding_function:
                try:
                    query_embedding = self.embedding_function(query)
                    embedding_time = time.time() - start_time
                    print(f"Generated query embedding in {embedding_time:.3f}s")
                except Exception as e:
                    print(f"Error generating query embedding: {e}")
                    return []
            else:
                # Random embedding for testing
                query_embedding = np.random.random(self.embedding_dim).astype(np.float32)
                print("Warning: Using random embedding (no embedding function available)")

            # Normalize query embedding
            try:
                query_norm = np.linalg.norm(query_embedding)
                if query_norm < 1e-10:
                    print("Warning: Query embedding has near-zero norm")
                    query_norm = 1e-10

                normalized_query = query_embedding / query_norm
                normalized_query = np.array([normalized_query], dtype=np.float32)
            except Exception as e:
                print(f"Error normalizing query embedding: {e}")
                return []

            # Determine whether to use fractal search
            use_fractal_here = self.use_fractal if use_fractal is None else use_fractal

            # Check if fractal indices exist if we want to use fractal search
            have_fractal_indices = bool(self.fractal_indices) and any(idx.ntotal > 0 for idx in self.fractal_indices.values())
            can_use_fractal = use_fractal_here and have_fractal_indices

            print(f"Search config: use_fractal={use_fractal_here}, have_indices={have_fractal_indices}, can_use={can_use_fractal}")

            # Use fractal search if enabled and indices are available
            search_start = time.time()
            if can_use_fractal:
                try:
                    results = self._fractal_search(
                        normalized_query,
                        top_k=top_k,
                        min_similarity=min_similarity
                    )
                    search_time = time.time() - search_start
                    print(f"Fractal search completed in {search_time:.3f}s, found {len(results)} results")
                except Exception as e:
                    print(f"Error in fractal search: {e}")
                    # Fall back to standard search
                    print("Falling back to standard search")
                    try:
                        results = self._standard_search(
                            normalized_query,
                            top_k=top_k,
                            min_similarity=min_similarity
                        )
                    except Exception as inner_e:
                        print(f"Standard search also failed: {inner_e}")
                        return []
            else:
                # Standard search
                try:
                    print("Using standard search")
                    results = self._standard_search(
                        normalized_query,
                        top_k=top_k,
                        min_similarity=min_similarity
                    )
                    search_time = time.time() - search_start
                    print(f"Standard search completed in {search_time:.3f}s, found {len(results)} results")
                except Exception as e:
                    print(f"Error in standard search: {e}")
                    return []

            # Filter by memory types if specified
            if memory_types:
                try:
                    # Process memory types to handle prefix matching (e.g., "command_*")
                    matched_results = []

                    for result in results:
                        memory_type = result.get("memory_type", "")

                        # Check for direct match
                        direct_match = memory_type in memory_types

                        # Check for prefix match
                        prefix_match = False
                        for allowed_type in memory_types:
                            if allowed_type.endswith('*') and memory_type.startswith(allowed_type[:-1]):
                                prefix_match = True
                                break

                        if direct_match or prefix_match:
                            matched_results.append(result)

                    filtered_count = len(results) - len(matched_results)
                    if filtered_count > 0:
                        print(f"Filtered {filtered_count} results by memory type")

                    results = matched_results
                except Exception as e:
                    print(f"Error filtering by memory type: {e}")
                    # Continue with unfiltered results
            
            # Return only requested number
            return results[:top_k]
    
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
                    "memory_type": item.memory_type,
                    "similarity": float(similarities[0][i]),
                    "metadata": item.metadata,
                    "index": idx
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results
    
    def _fractal_search(self, 
                      normalized_query: np.ndarray,
                      top_k: int,
                      min_similarity: float) -> List[Dict[str, Any]]:
        """
        Perform enhanced fractal search across multiple levels with proper sharpening.

        Args:
            normalized_query: Normalized query embedding vector
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of search results with similarity scores
        """
        # Get sharpening factor from instance variable
        sharpening_factor = getattr(self, 'sharpening_factor', 0.3)

        # Level weights (decreasing importance for higher levels)
        level_weights = [1.0, 0.8, 0.6, 0.4, 0.2]

        # Ensure list is long enough
        while len(level_weights) <= self.max_fractal_levels:
            level_weights.append(level_weights[-1] * 0.5)

        # Debug information
        print(f"Starting fractal search with {self.index.ntotal} items in base index")

        # Start with base index search
        search_k = min(top_k * 2, self.index.ntotal)

        # Add error handling for empty index
        if search_k <= 0:
            print("Warning: No items in index, cannot perform search")
            return []

        try:
            base_similarities, base_indices = self.index.search(normalized_query, search_k)

            # Debug information
            print(f"Base search returned {len(base_indices[0])} results")

            # Track results to avoid duplicates
            result_dict = {}

            # Process base results with sharpening
            for i, idx in enumerate(base_indices[0]):
                if idx != -1 and base_similarities[0][i] > min_similarity:
                    if idx in self.deleted_ids:
                        continue

                    item = self.items[idx]

                    # Apply sharpening to raw similarity
                    raw_similarity = float(base_similarities[0][i])
                    sharpened_similarity = self._apply_sharpening(raw_similarity, sharpening_factor)
                    # Apply level weight after sharpening
                    similarity = sharpened_similarity * level_weights[0]

                    result_dict[item.id] = {
                        "id": item.id,
                        "content": item.content,
                        "memory_type": item.memory_type,
                        "similarity": similarity,
                        "raw_similarity": raw_similarity,
                        "sharpened_similarity": sharpened_similarity,
                        "base_similarity": raw_similarity,  # For backward compatibility
                        "metadata": item.metadata,
                        "index": idx,
                        "level": 0,
                        "level_weight": level_weights[0]
                    }

            # Debug information
            print(f"Found {len(result_dict)} results in base level")

            # Search fractal indices if available
            for level in range(1, self.max_fractal_levels + 1):
                if level not in self.fractal_indices:
                    print(f"Skipping level {level} - no index available")
                    continue

                # Skip empty indices
                if self.fractal_indices[level].ntotal == 0:
                    print(f"Level {level} index is empty, skipping")
                    continue

                print(f"Searching level {level} index with {self.fractal_indices[level].ntotal} items")

                # Create level-specific query variation
                level_query = self._generate_level_embedding(normalized_query[0], level)
                level_query = np.array([level_query], dtype=np.float32)

                try:
                    # Search with level-specific query
                    similarities, indices = self.fractal_indices[level].search(
                        level_query,
                        search_k
                    )

                    print(f"Level {level} search returned {len(indices[0])} results")

                    # Get level weight
                    weight = level_weights[min(level, len(level_weights)-1)]

                    # Process results with sharpening
                    for i, idx in enumerate(indices[0]):
                        if idx != -1 and similarities[0][i] > min_similarity:
                            if idx in self.deleted_ids:
                                continue

                            # Apply sharpening to raw similarity
                            raw_similarity = float(similarities[0][i])

                            # Additional sharpening for higher levels to encourage diversity
                            level_adjusted_factor = sharpening_factor * (1.0 + (level * 0.05))
                            sharpened_similarity = self._apply_sharpening(raw_similarity, level_adjusted_factor)

                            # Apply level weight after sharpening
                            weighted_similarity = sharpened_similarity * weight

                            # Only consider if similarity passes threshold
                            if weighted_similarity < min_similarity:
                                continue

                            # Get item
                            item = self.items[idx]

                            # Update if this is better than existing or add if new
                            if item.id not in result_dict or weighted_similarity > result_dict[item.id]["similarity"]:
                                result_dict[item.id] = {
                                    "id": item.id,
                                    "content": item.content,
                                    "memory_type": item.memory_type,
                                    "similarity": weighted_similarity,
                                    "raw_similarity": raw_similarity,
                                    "sharpened_similarity": sharpened_similarity,
                                    "base_similarity": raw_similarity,  # For backward compatibility
                                    "metadata": item.metadata,
                                    "index": idx,
                                    "level": level,
                                    "level_weight": weight
                                }
                except Exception as e:
                    print(f"Error searching fractal level {level}: {e}")
                    # Continue with other levels rather than failing completely
        except Exception as e:
            print(f"Error in base search: {e}")
            return []

        # Apply cross-level verification with enhanced sharpening
        try:
            self._apply_cross_level_verification_with_sharpening(result_dict, sharpening_factor)
        except Exception as e:
            print(f"Error applying cross-level verification: {e}")
            # Continue even if verification fails

        # Extract values and sort
        results = list(result_dict.values())

        # Debug the results
        print(f"Total results after fractal search: {len(results)}")

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results

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
            item_levels[item_id].append((result["level"], result["base_similarity"]))

        # Apply boost for items found in multiple levels
        for item_id, level_info in item_levels.items():
            if len(level_info) > 1:
                # Calculate cross-level verification score
                level_count = len(level_info)
                avg_similarity = sum(sim for _, sim in level_info) / level_count

                # Apply bonus based on level agreement (up to 20% boost)
                cross_level_bonus = min(0.2, 0.05 * level_count)

                # Apply the bonus
                if item_id in result_dict:
                    current_sim = result_dict[item_id]["similarity"]
                    result_dict[item_id]["similarity"] = min(1.0, current_sim * (1.0 + cross_level_bonus))
                    result_dict[item_id]["cross_level_bonus"] = cross_level_bonus
                    result_dict[item_id]["found_in_levels"] = [lvl for lvl, _ in level_info]

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
                "memory_type": item.memory_type,
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
                        print(f"Error updating embedding: {e}")
                        return False

                item.content = updates["content"]

            # Update memory type if provided
            if "memory_type" in updates:
                item.memory_type = updates["memory_type"]

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
                print(f"Error saving memory data: {e}")
                return False

    def load(self) -> bool:
        """
        Load memory data from disk.

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            items_path = os.path.join(self.storage_path, "items.json")
            embeddings_path = os.path.join(self.storage_path, "embeddings.npy")
            metadata_path = os.path.join(self.storage_path, "metadata.json")

            if not (os.path.exists(items_path) and os.path.exists(embeddings_path)):
                return False

            try:
                # Load metadata first to get configuration
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    self.embedding_dim = metadata.get("embedding_dim", self.embedding_dim)
                    self.use_fractal = metadata.get("use_fractal", self.use_fractal)
                    self.max_fractal_levels = metadata.get("max_fractal_levels", self.max_fractal_levels)
                    self.deleted_ids = set(metadata.get("deleted_ids", []))

                # Load embeddings
                embeddings = np.load(embeddings_path)

                # Load items
                with open(items_path, 'r', encoding='utf-8') as f:
                    items_data = json.load(f)

                # Recreate items
                self.items = []
                self.embeddings = []
                self.id_to_index = {}

                for i, item_data in enumerate(items_data):
                    # Get embedding
                    if i < len(embeddings):
                        embedding = embeddings[i]
                    else:
                        # Should not happen, but fallback to random embedding
                        embedding = np.random.random(self.embedding_dim).astype(np.float32)

                    # Create item
                    item = MemoryItem.from_dict(item_data, embedding)

                    # Add to storage
                    self.items.append(item)
                    self.embeddings.append(embedding)
                    self.id_to_index[item.id] = i

                # Load fractal embeddings if using them
                if self.use_fractal:
                    fractal_path = os.path.join(self.storage_path, "fractal_embeddings.pkl")
                    if os.path.exists(fractal_path):
                        with open(fractal_path, 'rb') as f:
                            fractal_data = pickle.load(f)

                        # Add fractal embeddings to items
                        for item_id, item_fractals in fractal_data.items():
                            if item_id in self.id_to_index:
                                idx = self.id_to_index[item_id]
                                self.items[idx].fractal_embeddings = item_fractals

                # Load index
                index_path = os.path.join(self.storage_path, "index.faiss")
                if os.path.exists(index_path):
                    self.index = faiss.read_index(index_path)
                else:
                    # Recreate index
                    self._create_index()
                    if self.embeddings:
                        self.index.add(np.array(self.embeddings, dtype=np.float32))

                # Load fractal indices if using them
                if self.use_fractal:
                    self.fractal_indices = {}
                    for level in range(1, self.max_fractal_levels + 1):
                        level_path = os.path.join(self.storage_path, f"index_level_{level}.faiss")
                        if os.path.exists(level_path):
                            self.fractal_indices[level] = faiss.read_index(level_path)

                return True

            except Exception as e:
                print(f"Error loading memory data: {e}")
                # Reset to empty state
                self.items = []
                self.embeddings = []
                self.id_to_index = {}
                self.deleted_ids = set()
                self.index = None
                self.fractal_indices = {}
                return False

    def clear(self, memory_types: Optional[List[str]] = None) -> bool:
        """
        Clear memory data.

        Args:
            memory_types: Optional list of memory types to clear (None for all)

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if memory_types:
                # Mark items of specified types as deleted
                for i, item in enumerate(self.items):
                    if item.memory_type in memory_types:
                        self.deleted_ids.add(i)
                        if item.id in self.id_to_index:
                            del self.id_to_index[item.id]
            else:
                # Clear all data
                self.items = []
                self.embeddings = []
                self.id_to_index = {}
                self.deleted_ids = set()
                self.index = None
                self.fractal_indices = {}

            # Save if auto-save is enabled
            if self.auto_save:
                self.save()

            return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory store.

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            # Count items by type
            type_counts = {}
            for item in self.items:
                if item.memory_type not in type_counts:
                    type_counts[item.memory_type] = 0
                type_counts[item.memory_type] += 1

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
                "type_counts": type_counts,
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
        print("Rebuilding fractal indices...")

        # Clear existing fractal indices
        self.fractal_indices = {}

        # Skip if fractal embeddings are disabled
        if not self.use_fractal:
            print("Fractal embeddings are disabled, skipping rebuild")
            return

        # Count items with fractal embeddings
        items_with_fractal = sum(1 for item in self.items if item.fractal_embeddings)

        if items_with_fractal == 0:
            print("No items have fractal embeddings, regenerating...")

            # Regenerate fractal embeddings for all items
            for item in self.items:
                if hasattr(self, '_add_fractal_embeddings'):
                    self._add_fractal_embeddings(item)

            # Recount
            items_with_fractal = sum(1 for item in self.items if item.fractal_embeddings)

            if items_with_fractal == 0:
                print("Failed to generate fractal embeddings")
                return

        print(f"Building indices for {items_with_fractal} items with fractal embeddings")

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
                print(f"Added {len(embeddings_for_level)} embeddings to level {level} index")

        print("Fractal indices rebuilt successfully")

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

