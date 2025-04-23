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
    
    def _add_item_to_store(self, item: MemoryItem) -> str:
        """Add a memory item to storage and indices."""
        # If this is the first item, initialize index
        if self.index is None:
            self._create_index(item.embedding.shape[0])
        
        # Normalize embedding for cosine similarity
        normalized_embedding = item.embedding / max(np.linalg.norm(item.embedding), 1e-10)
        
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
                for level, level_embedding in item.fractal_embeddings.items():
                    # Normalize level embedding
                    normalized_level_embedding = level_embedding / max(np.linalg.norm(level_embedding), 1e-10)
                    
                    # Ensure index exists for this level
                    if level not in self.fractal_indices:
                        self.fractal_indices[level] = faiss.IndexFlatIP(self.embedding_dim)
                    
                    # Add to level-specific index
                    self.fractal_indices[level].add(np.array([normalized_level_embedding], dtype=np.float32))
            
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
        """Generate and add fractal embeddings to a memory item."""
        base_embedding = item.embedding
        
        # Generate a fractal embedding for each level
        for level in range(1, self.max_fractal_levels + 1):
            level_embedding = self._generate_level_embedding(base_embedding, level)
            item.add_fractal_embedding(level, level_embedding)
    
    def _generate_level_embedding(self, base_embedding: np.ndarray, level: int) -> np.ndarray:
        """
        Generate a level-specific embedding with controlled semantic variation.
        Each level introduces a specific type of variation to explore different
        areas of the semantic space while maintaining core meaning.
        """
        if level == 0:
            return base_embedding
        
        # Create a rotation matrix with controlled randomness
        # Higher levels have more rotation/variation
        rotation_strength = 0.05 * level  # 5% per level
        rotation_matrix = np.eye(base_embedding.shape[0])  # Identity matrix
        
        # Add controlled noise to rotation matrix
        noise = np.random.normal(0, rotation_strength, rotation_matrix.shape)
        rotation_matrix += noise
        
        # Ensure the rotation matrix is well-conditioned
        u, s, vh = np.linalg.svd(rotation_matrix, full_matrices=False)
        rotation_matrix = u @ vh  # Reconstruct with balanced singular values
        
        # Apply rotation to base embedding
        rotated_embedding = np.dot(base_embedding, rotation_matrix)
        
        # Add directional noise (more for higher levels)
        noise_direction = np.random.normal(0, 1, base_embedding.shape)
        noise_direction /= max(np.linalg.norm(noise_direction), 1e-10)  # Normalize
        
        # Scale noise by level
        noise_scale = 0.05 * level  # 5% per level
        
        # Combine base embedding (with decreasing weight) and noise (with increasing weight)
        level_embedding = (
            (1.0 - noise_scale) * rotated_embedding +  # Weighted rotated embedding
            noise_scale * noise_direction              # Weighted noise
        )
        
        # Normalize to maintain vector properties
        level_embedding /= max(np.linalg.norm(level_embedding), 1e-10)
        
        return level_embedding
    
    def retrieve(self, 
                query: str, 
                memory_types: Optional[List[str]] = None,
                top_k: int = 5, 
                min_similarity: float = 0.25,
                use_fractal: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on query.
        
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
            if not self.items or self.index is None or self.index.ntotal == 0:
                return []
            
            # Generate query embedding
            if self.embedding_function:
                try:
                    query_embedding = self.embedding_function(query)
                except Exception as e:
                    print(f"Error generating query embedding: {e}")
                    return []
            else:
                # Random embedding for testing
                query_embedding = np.random.random(self.embedding_dim).astype(np.float32)
            
            # Normalize query embedding
            normalized_query = query_embedding / max(np.linalg.norm(query_embedding), 1e-10)
            normalized_query = np.array([normalized_query], dtype=np.float32)
            
            # Determine whether to use fractal search
            use_fractal_here = self.use_fractal if use_fractal is None else use_fractal
            
            # Use fractal search if enabled and indices are available
            if use_fractal_here and self.fractal_indices:
                results = self._fractal_search(
                    normalized_query, 
                    top_k=top_k, 
                    min_similarity=min_similarity
                )
            else:
                # Standard search
                results = self._standard_search(
                    normalized_query, 
                    top_k=top_k, 
                    min_similarity=min_similarity
                )
            
            # Filter by memory types if specified
            if memory_types:
                results = [r for r in results if r.get("memory_type") in memory_types]
            
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
        Perform enhanced fractal search across multiple levels
        to find semantically related content.
        """
        # Level weights (decreasing importance for higher levels)
        level_weights = [1.0, 0.7, 0.5, 0.3]
        
        # Ensure list is long enough
        while len(level_weights) <= self.max_fractal_levels:
            level_weights.append(level_weights[-1] * 0.5)
        
        # Start with base index search
        search_k = min(top_k * 2, self.index.ntotal)
        base_similarities, base_indices = self.index.search(normalized_query, search_k)
        
        # Track results to avoid duplicates
        result_dict = {}
        
        # Process base results
        for i, idx in enumerate(base_indices[0]):
            if idx != -1 and base_similarities[0][i] > min_similarity:
                if idx in self.deleted_ids:
                    continue
                
                item = self.items[idx]
                similarity = float(base_similarities[0][i]) * level_weights[0]
                
                result_dict[item.id] = {
                    "id": item.id,
                    "content": item.content,
                    "memory_type": item.memory_type,
                    "similarity": similarity,
                    "base_similarity": float(base_similarities[0][i]),
                    "metadata": item.metadata,
                    "index": idx,
                    "level": 0,
                    "level_weight": level_weights[0]
                }
        
        # Search fractal indices
        for level in range(1, self.max_fractal_levels + 1):
            if level not in self.fractal_indices:
                continue
            
            # Create level-specific query variation
            level_query = self._generate_level_embedding(normalized_query[0], level)
            level_query = np.array([level_query], dtype=np.float32)

            try:
                # Search with level-specific query
                similarities, indices = self.fractal_indices[level].search(
                    level_query,
                    search_k
                )

                # Get level weight
                weight = level_weights[min(level, len(level_weights)-1)]

                # Process results
                for i, idx in enumerate(indices[0]):
                    if idx != -1 and similarities[0][i] > min_similarity:
                        if idx in self.deleted_ids:
                            continue

                        # Calculate weighted similarity
                        item = self.items[idx]
                        raw_similarity = float(similarities[0][i])
                        weighted_similarity = raw_similarity * weight

                        # Only consider if similarity passes threshold
                        if weighted_similarity < min_similarity:
                            continue

                        # Update if this is better than existing or add if new
                        if item.id not in result_dict or weighted_similarity > result_dict[item.id]["similarity"]:
                            result_dict[item.id] = {
                                "id": item.id,
                                "content": item.content,
                                "memory_type": item.memory_type,
                                "similarity": weighted_similarity,
                                "base_similarity": raw_similarity,
                                "metadata": item.metadata,
                                "index": idx,
                                "level": level,
                                "level_weight": weight
                            }
            except Exception as e:
                print(f"Error searching fractal level {level}: {e}")

        # Apply cross-level verification (boost confidence if found in multiple levels)
        self._apply_cross_level_verification(result_dict)

        # Extract values and sort
        results = list(result_dict.values())
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