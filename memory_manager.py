"""
Refactored vector store implementation with improved memory management,
error handling, and a cleaner class hierarchy.
"""
import re
import os
import json
import faiss
import numpy as np
import gc
import pickle
import weakref
import threading
import torch
from typing import List, Dict, Any, Optional, Tuple, Set, Callable, Union
from datetime import datetime

class VectorStore:
    """
    A unified vector store using FAISS for efficient similarity search.
    Enhanced with both pre-indexing embedding sharpening (clustering-based)
    and post-search similarity sharpening for improved memory retrieval.
    """
    def __init__(self, 
                 storage_path: str = "./memory/vector_store", 
                 embedding_function: Optional[Callable] = None,
                 embedding_dim: int = 384,
                 auto_save: bool = True,
                 fractal_enabled: bool = True,
                 max_fractal_levels: int = 3):
        """
        Initialize the vector store.
        
        Args:
            storage_path: Base path for storing index and metadata
            embedding_function: Function to generate embeddings from text
            embedding_dim: Dimension of the embedding vectors
            auto_save: Whether to automatically save changes

        Enhanced vector store with fractal capabilities

        Args:
            fractal_enabled: Enable multi-resolution embedding
            max_fractal_levels: Maximum number of fractal decomposition levels
        """
        self.storage_path = storage_path
        self.index_path = f"{storage_path}.index"
        self.data_path = f"{storage_path}.pkl"
        self.embedding_function = embedding_function
        self.embedding_dim = embedding_dim
        self.auto_save = auto_save
        self.fractal_enabled = fractal_enabled
        self.max_fractal_levels = max_fractal_levels
        self.fractal_indices = {}  # Indices for different fractal levels
        
        # Thread lock for concurrent access
        self._lock = threading.RLock()
        
        # Document storage
        self.documents = []
        self.metadata = []
        self.doc_hashes = set()
        self.deleted_ids = set()
        
        # Memory management
        self._embeddings_cache = {}
        self._max_cache_size = 100
        
        # Create storage directory if it doesn't exist
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        # Initialize FAISS index (will be created when first embedding is added)
        self.index = None
        
        # Register finalizer for cleanup on garbage collection
        self._finalizer = weakref.finalize(self, self._cleanup)
        
        # Load existing data if available
        self.load()
    
    def _create_index(self, dim: int = None):
        """Create a new FAISS index with the specified dimension"""
        if dim is not None:
            self.embedding_dim = dim
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine on normalized vectors)
    
    def _compute_hash(self, text: str) -> str:
        """
        Compute a hash that allows more unique documents
        """
        import hashlib
        import uuid

        # Generate a unique identifier for each document
        # This ensures each document gets a unique hash
        unique_id = str(uuid.uuid4())

        # Combine text with unique identifier to ensure uniqueness
        hash_input = f"{text}_{unique_id}"

        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def add(self, text: str, embedding: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        with self._lock:
            # Extensive logging
            #print(f"\n[FRACTAL DEBUG] Adding document: {text[:50]}...")
            #print(f"[FRACTAL DEBUG] Fractal Enabled: {hasattr(self, 'fractal_enabled') and self.fractal_enabled}")

            # Create document hash for deduplication
            doc_hash = self._compute_hash(text)

            # Skip if it's a duplicate
            if doc_hash in self.doc_hashes:
                #print(f"[FRACTAL DEBUG] Duplicate document, skipping: {text[:50]}")
                return False

            # Explicitly set fractal attributes if not present
            if not hasattr(self, 'fractal_enabled'):
                #print("[FRACTAL DEBUG] Setting fractal_enabled to False")
                self.fractal_enabled = False
            if not hasattr(self, 'max_fractal_levels'):
                #print("[FRACTAL DEBUG] Setting max_fractal_levels to 3")
                self.max_fractal_levels = 3
            if not hasattr(self, 'fractal_indices'):
                #print("[FRACTAL DEBUG] Initializing fractal_indices")
                self.fractal_indices = {}

            # If this is the first document, create the index
            if self.index is None:
                #print("[FRACTAL DEBUG] Creating initial index")
                self._create_index(embedding.shape[0])
            elif embedding.shape[0] != self.embedding_dim:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[0]}")

            # Add to cache with a copy to avoid memory issues
            self._embeddings_cache[doc_hash] = embedding.copy()
            self._trim_cache()

            # Normalize embedding for cosine similarity
            normalized_embedding = embedding / np.linalg.norm(embedding)

            try:
                # Add to FAISS index
                self.index.add(np.array([normalized_embedding], dtype=np.float32))

                # Add to document storage
                self.documents.append(text)
                self.metadata.append(metadata or {})
                self.doc_hashes.add(doc_hash)

                # Generate and add fractal embeddings if enabled
                if self.fractal_enabled:
                    #print("[FRACTAL DEBUG] Generating fractal embeddings")
                    try:
                        fractal_embeddings = self._generate_fractal_embeddings(normalized_embedding)

                        # Add embeddings to different level indices
                        for level, level_embedding in fractal_embeddings.items():
                            # Skip base level (already added to main index)
                            if level == 0:
                                continue

                            #print(f"[FRACTAL DEBUG] Adding embedding for level {level}")

                            # Ensure index exists for this level
                            if level not in self.fractal_indices:
                                #print(f"[FRACTAL DEBUG] Creating index for level {level}")
                                self.fractal_indices[level] = faiss.IndexFlatIP(self.embedding_dim)

                            # Add to level-specific index
                            self.fractal_indices[level].add(np.array([level_embedding], dtype=np.float32))

                        #print("[FRACTAL DEBUG] Fractal embeddings added successfully")
                    except Exception as fractal_err:
                        print(f"[FRACTAL DEBUG] Error generating fractal embeddings: {fractal_err}")
                else:
                    print("[FRACTAL DEBUG] Fractal embeddings not enabled")

                # Save if auto-save is enabled
                if self.auto_save:
                    self.save()

                # Periodically run garbage collection after additions
                if len(self.documents) % 50 == 0:
                    gc.collect()

                return True
            except Exception as e:
                print(f"[FRACTAL DEBUG] Error adding document to vector store: {e}")
                return False

    def search(self,
               query_embedding: np.ndarray,
               top_k: int = 10,
               min_similarity: float = 0.25,
               apply_sharpening: bool = False,
               sharpening_factor: float = 0.3,
               fractal_level: Optional[int] = None,
               multi_level_search: bool = True) -> List[Dict[str, Any]]:
        """
        Enhanced search method with intelligent fractal embedding integration
        """
        # Normalize query embedding
        normalized_query = query_embedding / np.linalg.norm(query_embedding)

        # If fractal search is disabled
        if not hasattr(self, 'fractal_enabled') or not self.fractal_enabled or not multi_level_search:
            # Standard search
            similarities, indices = self.index.search(
                np.array([normalized_query], dtype=np.float32),
                top_k
            )

            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and similarities[0][i] > min_similarity:
                    results.append({
                        'text': self.documents[idx],
                        'similarity': float(similarities[0][i]),
                        'metadata': self.metadata[idx],
                        'index': idx,
                        'level': 0
                    })

            return results[:top_k]

        # Fractal search implementation
        results = []

        # Determine search levels
        if fractal_level is not None:
            search_levels = [fractal_level]
        else:
            search_levels = list(range(min(self.max_fractal_levels + 1, len(self.fractal_indices))))

        # Search across levels with adaptive weighting
        level_weights = [1.0, 0.7, 0.5, 0.3]  # Decreasing weight for each level

        # Perform search on base index first
        base_similarities, base_indices = self.index.search(
            np.array([normalized_query], dtype=np.float32),
            top_k
        )

        # Add base results
        for i, idx in enumerate(base_indices[0]):
            if idx != -1 and base_similarities[0][i] > min_similarity:
                results.append({
                    'text': self.documents[idx],
                    'similarity': float(base_similarities[0][i]),
                    'metadata': self.metadata[idx],
                    'index': idx,
                    'level': 0
                })

        # Search fractal indices
        for level in search_levels:
            if level == 0 or level not in self.fractal_indices:
                continue

            # Create level-specific query variation
            level_query = normalized_query * (1 + 0.1 * level)

            # Search with level-specific query
            similarities, indices = self.fractal_indices[level].search(
                np.array([level_query], dtype=np.float32),
                top_k
            )

            # Process results with level-based weighting
            weight = level_weights[min(level, len(level_weights)-1)]

            for i, idx in enumerate(indices[0]):
                if idx != -1:
                    adjusted_similarity = float(similarities[0][i]) * weight

                    # Only add if above minimum similarity
                    if adjusted_similarity > min_similarity:
                        results.append({
                            'text': self.documents[idx],
                            'similarity': adjusted_similarity,
                            'metadata': self.metadata[idx],
                            'index': idx,
                            'level': level
                        })

        # Deduplicate and sort results
        unique_results = {}
        for result in results:
            key = result['text']
            if key not in unique_results or result['similarity'] > unique_results[key]['similarity']:
                unique_results[key] = result

        # Sort and return top results
        sorted_results = sorted(
            unique_results.values(),
            key=lambda x: x['similarity'],
            reverse=True
        )

        return sorted_results[:top_k]

    def enhanced_fractal_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_similarity: float = 0.25,
        apply_sharpening: bool = False,
        sharpening_factor: float = 0.3,
        multi_level_search: bool = True,
        level_weights: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Enhanced fractal similarity search with improved weighting and level merging.
        Incorporates sophisticated cross-level verification for better results.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            apply_sharpening: Whether to apply similarity sharpening
            sharpening_factor: Factor to control sharpening strength
            multi_level_search: Whether to search across multiple fractal levels
            level_weights: Custom weights for different fractal levels (defaults to [1.0, 0.7, 0.5, 0.3])

        Returns:
            List of search results with similarity scores
        """
        # Normalize query embedding
        normalized_query = query_embedding / np.linalg.norm(query_embedding)

        # If fractal search is disabled
        if not hasattr(self, 'fractal_enabled') or not self.fractal_enabled or not multi_level_search:
            # Standard search
            similarities, indices = self.index.search(
                np.array([normalized_query], dtype=np.float32),
                top_k
            )

            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and similarities[0][i] > min_similarity:
                    results.append({
                        'text': self.documents[idx],
                        'similarity': float(similarities[0][i]),
                        'metadata': self.metadata[idx],
                        'index': idx,
                        'level': 0
                    })

            return results[:top_k]

        # Setup for fractal search
        results = []

        # Default level weights if not provided
        if level_weights is None:
            level_weights = [1.0, 0.7, 0.5, 0.3]

        # Make sure we have enough weights for all levels
        while len(level_weights) < self.max_fractal_levels + 1:
            level_weights.append(level_weights[-1] * 0.5)  # Exponential decay for higher levels

        # Determine search levels
        search_levels = list(range(min(self.max_fractal_levels + 1, len(self.fractal_indices) + 1)))

        # Create level-specific query variations
        level_queries = {}
        for level in search_levels:
            # Base level uses original query
            if level == 0:
                level_queries[level] = normalized_query
            else:
                # Generate level-specific query embedding with controlled semantic variation
                level_query = self._generate_level_query_embedding(normalized_query, level)
                level_queries[level] = level_query

        # Perform search on base index first
        base_similarities, base_indices = self.index.search(
            np.array([normalized_query], dtype=np.float32),
            top_k
        )

        # Create results dict for deduplication tracking
        result_dict = {}

        # Add base results
        for i, idx in enumerate(base_indices[0]):
            if idx != -1 and base_similarities[0][i] > min_similarity:
                similarity = float(base_similarities[0][i]) * level_weights[0]

                # Create unique key for this document
                doc_key = self.documents[idx]

                result_dict[doc_key] = {
                    'text': self.documents[idx],
                    'similarity': similarity,
                    'base_similarity': float(base_similarities[0][i]),
                    'metadata': self.metadata[idx],
                    'index': idx,
                    'level': 0,
                    'level_weight': level_weights[0]
                }

        # Search fractal indices
        for level in search_levels:
            if level == 0 or level not in self.fractal_indices:
                continue

            # Get level-specific query
            level_query = level_queries[level]

            # Search with level-specific query
            try:
                similarities, indices = self.fractal_indices[level].search(
                    np.array([level_query], dtype=np.float32),
                    top_k
                )

                # Get level weight (with bounds checking)
                weight = level_weights[min(level, len(level_weights)-1)]

                # Process results
                for i, idx in enumerate(indices[0]):
                    if idx != -1 and similarities[0][i] > min_similarity:
                        # Calculate weighted similarity
                        raw_similarity = float(similarities[0][i])
                        weighted_similarity = raw_similarity * weight

                        # Only proceed if similarity passes threshold
                        if weighted_similarity < min_similarity:
                            continue

                        # Create unique key for this document
                        doc_key = self.documents[idx]

                        # Update if this is better than existing or add if new
                        if doc_key not in result_dict or weighted_similarity > result_dict[doc_key]['similarity']:
                            result_dict[doc_key] = {
                                'text': self.documents[idx],
                                'similarity': weighted_similarity,
                                'base_similarity': raw_similarity,
                                'metadata': self.metadata[idx],
                                'index': idx,
                                'level': level,
                                'level_weight': weight
                            }
            except Exception as e:
                print(f"Error searching fractal level {level}: {e}")

        # Apply cross-level verification (boost confidence if found in multiple levels)
        if len(search_levels) > 1:
            # Group by document text
            doc_to_levels = {}
            for doc_key, result in result_dict.items():
                if doc_key not in doc_to_levels:
                    doc_to_levels[doc_key] = []
                doc_to_levels[doc_key].append((result['level'], result['base_similarity']))

            # Apply boost for documents found in multiple levels
            for doc_key, level_info in doc_to_levels.items():
                if len(level_info) > 1:
                    # Calculate cross-level verification score
                    level_count = len(level_info)
                    avg_similarity = sum(sim for _, sim in level_info) / level_count

                    # Apply bonus based on level agreement (up to 20% boost)
                    cross_level_bonus = min(0.2, 0.05 * level_count)

                    # Apply the bonus
                    if doc_key in result_dict:
                        current_sim = result_dict[doc_key]['similarity']
                        result_dict[doc_key]['similarity'] = min(1.0, current_sim * (1.0 + cross_level_bonus))
                        result_dict[doc_key]['cross_level_bonus'] = cross_level_bonus
                        result_dict[doc_key]['found_in_levels'] = [lvl for lvl, _ in level_info]

        # Extract values from result_dict
        results = list(result_dict.values())

        # Apply post-search sharpening if requested
        if apply_sharpening and results:
            for result in results:
                similarity = result['similarity']

                # Check if this is a correction (should be prioritized)
                is_correction = result.get('metadata', {}).get('is_correction', False)

                # Apply different sharpening based on similarity ranges and correction status
                if is_correction:
                    # Always boost corrections significantly
                    boost_factor = 0.3 + (sharpening_factor * 0.2)
                    sharpened_similarity = similarity + boost_factor
                elif similarity > 0.7:  # Very high similarity
                    # Logarithmic boost (diminishing returns for very high similarities)
                    boost = np.log1p(similarity) * sharpening_factor * 0.3
                    sharpened_similarity = similarity + boost
                elif similarity > 0.5:  # Medium-high similarity
                    # Linear boost
                    boost = (similarity - 0.5) * sharpening_factor * 0.5
                    sharpened_similarity = similarity + boost
                elif similarity > 0.35:  # Medium similarity
                    # Neutral zone - minimal change
                    sharpened_similarity = similarity
                else:  # Low similarity
                    # Decrease low similarities more aggressively with higher sharpening factors
                    reduction = (0.35 - similarity) * sharpening_factor * 0.6
                    sharpened_similarity = similarity - reduction

                # Store updated similarity (clamped to valid range)
                result['original_similarity'] = similarity
                result['similarity'] = min(1.0, max(0.0, sharpened_similarity))

        # Sort by similarity (might have changed after sharpening)
        results.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        # Return only requested number after all processing
        return results[:top_k]

    def _generate_level_query_embedding(self, base_query: np.ndarray, level: int) -> np.ndarray:
        """
        Generate a level-specific query embedding with controlled semantic drift.

        Each level introduces a specific type of variation to probe different
        areas of the semantic space while maintaining core query intent.

        Args:
            base_query: The normalized base query embedding
            level: The fractal level to generate a query for

        Returns:
            Level-specific query embedding
        """
        if level == 0:
            return base_query

        # Create a rotation matrix with controlled randomness
        # Higher levels have more rotation/variation
        rotation_strength = 0.05 * level  # 5% per level
        rotation_matrix = np.eye(base_query.shape[0])  # Identity matrix

        # Add controlled noise to rotation matrix
        noise = np.random.normal(0, rotation_strength, rotation_matrix.shape)
        rotation_matrix += noise

        # Ensure the rotation matrix is well-conditioned
        u, s, vh = np.linalg.svd(rotation_matrix, full_matrices=False)
        rotation_matrix = u @ vh  # Reconstruct with balanced singular values

        # Apply rotation to base query
        rotated_query = np.dot(base_query, rotation_matrix)

        # Add directional noise (more for higher levels)
        noise_direction = np.random.normal(0, 1, base_query.shape)
        noise_direction /= np.linalg.norm(noise_direction)  # Normalize

        # Scale noise by level
        noise_scale = 0.05 * level  # 5% per level

        # Combine base query (with decreasing weight) and noise (with increasing weight)
        level_query = (
            (1.0 - noise_scale) * rotated_query +  # Weighted rotated query
            noise_scale * noise_direction           # Weighted noise
        )

        # Normalize to maintain vector properties
        return level_query / np.linalg.norm(level_query)

    def _generate_level_query_embedding(self, base_query: np.ndarray, level: int) -> np.ndarray:
        """
        Generate a level-specific query embedding
        """
        if level == 0:
            return base_query

        # Create a rotation matrix
        rotation_matrix = np.random.normal(0, 0.05 * level, (base_query.shape[0], base_query.shape[0]))

        # Rotate and slightly perturb the query
        level_query = np.dot(base_query, rotation_matrix)

        # Add some controlled noise
        noise = np.random.normal(0, 0.1 / level, base_query.shape)
        level_query = level_query + noise

        # Normalize to maintain vector properties
        return level_query / np.linalg.norm(level_query)

    def _generate_fractal_embeddings(self, embedding: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Improved fractal embedding generation with controlled semantic divergence
        """
        fractal_embeddings = {0: embedding}  # Base level

        for level in range(1, self.max_fractal_levels + 1):
            # Create more meaningful semantic variation
            # Use Principal Component Analysis (PCA) for controlled variation

            # Create variation through multiple mechanisms
            # 1. Directional noise with decreasing intensity
            noise_direction = np.random.normal(0, 1, embedding.shape)
            noise_direction /= np.linalg.norm(noise_direction)

            # Decrease noise intensity with level
            noise_scale = 0.1 / (level ** 1.5)

            # 2. Slight rotation matrix
            rotation_matrix = np.eye(embedding.shape[0])
            rotation_matrix += np.random.normal(0, 0.05, rotation_matrix.shape)

            # Combine variations
            variation = (
                0.7 * embedding +  # Preserve core semantics
                0.2 * np.dot(embedding, rotation_matrix) +  # Rotate slightly
                0.1 * (noise_direction * noise_scale)  # Add controlled noise
            )

            # Normalize to maintain vector properties
            variation /= np.linalg.norm(variation)

            fractal_embeddings[level] = variation

        return fractal_embeddings

    # def _compute_deviation(self, base: np.ndarray, variation: np.ndarray) -> float:
    #     """
    #     More sophisticated deviation calculation
    #     Measures semantic distance while preserving meaningful relationships
    #     """
    #     # Cosine similarity with a twist
    #     cosine_sim = np.dot(base, variation)

    #     # Angular distance provides more intuitive deviation measurement
    #     deviation = np.arccos(np.clip(cosine_sim, -1.0, 1.0)) / np.pi

    #     return deviation

    def _compute_deviation(self, base: np.ndarray, variation: np.ndarray) -> float:
        """
        Compute semantic deviation between two embeddings with better metrics.
        Measures semantic distance while preserving meaningful relationships.

        Args:
            base: Base embedding vector
            variation: Variation embedding vector

        Returns:
            Deviation score (0-1 range)
        """
        # Cosine similarity
        cos_sim = np.dot(base, variation) / (np.linalg.norm(base) * np.linalg.norm(variation))
        cos_sim = np.clip(cos_sim, -1.0, 1.0)  # Ensure valid range

        # Angular distance provides more intuitive deviation measurement
        angular_distance = np.arccos(cos_sim) / np.pi

        return angular_distance

    def _trim_cache(self):
        """Trim the embeddings cache if it grows too large"""
        if len(self._embeddings_cache) > self._max_cache_size:
            # Remove oldest items
            items_to_remove = len(self._embeddings_cache) - self._max_cache_size
            keys_to_remove = list(self._embeddings_cache.keys())[:items_to_remove]
            
            for key in keys_to_remove:
                del self._embeddings_cache[key]
    
    def remove(self, index: int) -> bool:
        """
        Remove a document at the specified index.
        
        Args:
            index: The index of the document to remove
            
        Returns:
            Success boolean
        """
        with self._lock:
            if index < 0 or index >= len(self.documents):
                return False
                
            # Mark document as removed
            doc_hash = self._compute_hash(self.documents[index])
            
            # Record deleted ID
            self.deleted_ids.add(index)
            
            # Set to empty string instead of removing to maintain indices
            self.documents[index] = ""
            self.metadata[index] = {"deleted": True}
            
            # Remove from cache and hash set
            if doc_hash in self._embeddings_cache:
                del self._embeddings_cache[doc_hash]
            self.doc_hashes.discard(doc_hash)
            
            # Save if auto-save is enabled
            if self.auto_save:
                self.save()
            
            return True

    def search_default(self,
               query_embedding: np.ndarray, 
               top_k: int = 10, 
               min_similarity: float = 0.25,
               apply_sharpening: bool = False,
               sharpening_factor: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search for similar documents using FAISS index with optional sharpening.
        
        Args:
            query_embedding: Search query embedding
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            apply_sharpening: Whether to apply similarity sharpening
            sharpening_factor: Factor to control sharpening strength
            
        Returns:
            List of result dictionaries
        """
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            
            # Normalize query embedding
            normalized_query = query_embedding / np.linalg.norm(query_embedding)
            normalized_query = np.array([normalized_query], dtype=np.float32)
            
            # Get more results than needed to account for filtering
            search_k = min(top_k * 3, self.index.ntotal)
            similarities, indices = self.index.search(normalized_query, search_k)
            
            # Filter results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and similarities[0][i] > min_similarity:
                    # Skip deleted documents
                    if idx in self.deleted_ids:
                        continue
                    
                    # Include full metadata and document content
                    results.append({
                        'text': self.documents[idx],
                        'similarity': float(similarities[0][i]),
                        'metadata': self.metadata[idx],
                        'index': idx
                    })
            
            # Apply sharpening if requested
            if apply_sharpening and results:
                results = self._apply_sharpening(results, sharpening_factor)
            
            # Sort by similarity (might have changed after sharpening)
            results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            # Return only requested number
            return results[:top_k]
    
    def _apply_sharpening(self, results: List[Dict[str, Any]], sharpening_factor: float) -> List[Dict[str, Any]]:
        """
        Apply sharpening to search results to improve relevance.
        
        Args:
            results: Search results to sharpen
            sharpening_factor: Factor to control sharpening strength
            
        Returns:
            Sharpened results
        """
        for result in results:
            # Store the original similarity for comparison
            similarity = result['similarity']
            result['original_similarity'] = similarity
            
            # Check if this is a correction (should be prioritized)
            is_correction = result.get('metadata', {}).get('is_correction', False)
            
            # Apply different sharpening based on similarity ranges and correction status
            if is_correction:
                # Always boost corrections significantly
                boost_factor = 0.3 + (sharpening_factor * 0.2)
                sharpened_similarity = similarity + boost_factor
            elif similarity > 0.7:  # Very high similarity
                # Logarithmic boost (diminishing returns for very high similarities)
                boost = np.log1p(similarity) * sharpening_factor * 0.3
                sharpened_similarity = similarity + boost
            elif similarity > 0.5:  # Medium-high similarity
                # Linear boost
                boost = (similarity - 0.5) * sharpening_factor * 0.5
                sharpened_similarity = similarity + boost
            elif similarity > 0.35:  # Medium similarity
                # Neutral zone - minimal change
                sharpened_similarity = similarity
            else:  # Low similarity
                # Decrease low similarities more aggressively with higher sharpening factors
                reduction = (0.35 - similarity) * sharpening_factor * 0.6
                sharpened_similarity = similarity - reduction
            
            # Store updated similarity (clamped to valid range)
            result['similarity'] = min(1.0, max(0.0, sharpened_similarity))
        
        return results
    
    def consolidate_memories(self, similarity_threshold: float = 0.85, min_cluster_size: int = 2) -> bool:
        """
        Consolidate similar memories to reduce redundancy.
        
        Args:
            similarity_threshold: Threshold for considering documents similar
            min_cluster_size: Minimum size for a cluster to be considered
            
        Returns:
            Boolean indicating whether any changes were made
        """
        with self._lock:
            # Skip if no index or not enough documents
            if self.index is None or self.index.ntotal < min_cluster_size:
                return False
                
            print(f"Consolidating memories... ({self.index.ntotal} documents)")
            
            try:
                # Find duplicate or very similar documents by content hash
                content_groups = {}
                for idx, doc in enumerate(self.documents):
                    # Skip deleted documents
                    if idx in self.deleted_ids or not doc:
                        continue
                        
                    # Create a simpler hash for content comparison
                    content_hash = self._compute_hash(doc[:100])  # Just use first 100 chars
                    if content_hash not in content_groups:
                        content_groups[content_hash] = []
                    content_groups[content_hash].append(idx)
                
                # Find groups with multiple documents (potential duplicates)
                duplicate_groups = [indices for content_hash, indices in content_groups.items()
                                  if len(indices) >= min_cluster_size]
                
                if not duplicate_groups:
                    print("No duplicate groups found. Nothing to consolidate.")
                    return False
                
                print(f"Found {len(duplicate_groups)} groups of similar documents")
                
                # Track which indices to keep (start with all)
                indices_to_keep = set(range(len(self.documents)))
                indices_to_keep -= self.deleted_ids  # Remove already deleted
                
                # Process each group
                for group in duplicate_groups:
                    # Keep the first document in each group, remove others
                    keep_idx = group[0]
                    for remove_idx in group[1:]:
                        # Merge metadata from removed document to kept document
                        kept_meta = self.metadata[keep_idx]
                        removed_meta = self.metadata[remove_idx]
                        
                        # Update sources list
                        if 'sources' not in kept_meta:
                            kept_meta['sources'] = []
                        
                        # Add source query if available
                        if 'source_query' in removed_meta and removed_meta['source_query'] not in kept_meta.get('sources', []):
                            kept_meta.setdefault('sources', []).append(removed_meta['source_query'])
                        
                        # Add other sources if available
                        if 'sources' in removed_meta:
                            for source in removed_meta['sources']:
                                if source not in kept_meta['sources']:
                                    kept_meta['sources'].append(source)
                        
                        # Remove the duplicate from indices to keep
                        if remove_idx in indices_to_keep:
                            indices_to_keep.remove(remove_idx)
                
                # Check if we found any duplicates to remove
                if len(indices_to_keep) < len(self.documents) - len(self.deleted_ids):
                    return self.rebuild_index(list(indices_to_keep))
                
                return False
                
            except Exception as e:
                print(f"Error during memory consolidation: {e}")
                return False
    
    def rebuild_index(self, valid_indices: Optional[List[int]] = None) -> bool:
        """
        Completely rebuild the index to reclaim memory and remove deleted documents.
        
        Args:
            valid_indices: Optional list of indices to keep
            
        Returns:
            Success boolean
        """
        with self._lock:
            if not self.documents or self.index is None:
                return False
            
            try:
                # Determine which indices to keep
                if valid_indices is None:
                    valid_indices = [i for i in range(len(self.documents)) 
                                   if i not in self.deleted_ids and self.documents[i]]
                
                if not valid_indices:
                    return False
                
                # Create new filtered lists
                new_docs = [self.documents[i] for i in valid_indices]
                new_metadata = [self.metadata[i] for i in valid_indices]
                new_hashes = {self._compute_hash(doc) for doc in new_docs if doc}
                
                # Generate embeddings for valid documents
                if self.embedding_function:
                    # Create embeddings for all valid documents
                    new_embeddings = np.zeros((len(new_docs), self.embedding_dim), dtype=np.float32)
                    for i, doc in enumerate(new_docs):
                        # Skip empty documents
                        if not doc:
                            continue
                            
                        # Use the provided embedding function
                        embedding = self.embedding_function(doc)
                        
                        # Skip if embedding generation failed
                        if embedding is None or embedding.size != self.embedding_dim:
                            continue
                            
                        new_embeddings[i] = embedding / np.linalg.norm(embedding)  # Normalize
                    
                    # Create new index
                    new_index = faiss.IndexFlatIP(self.embedding_dim)
                    if len(new_embeddings) > 0:
                        new_index.add(new_embeddings.astype(np.float32))
                    
                    # Update instance variables
                    self.index = new_index
                    self.documents = new_docs
                    self.metadata = new_metadata
                    self.doc_hashes = new_hashes
                    self.deleted_ids = set()
                    
                    # Clear cache
                    self._embeddings_cache.clear()
                    
                    # Save changes
                    if self.auto_save:
                        self.save()
                    
                    # Run garbage collection
                    gc.collect()
                    
                    print(f"Rebuilt index with {len(new_docs)} documents")
                    return True
                else:
                    print("Cannot rebuild index - no embedding function available")
                    return False
            
            except Exception as e:
                print(f"Error rebuilding index: {e}")
                return False
    
    def save(self) -> bool:
        """
        Save vector store to disk.
        
        Returns:
            Success boolean
        """
        with self._lock:
            try:
                # Save index if it exists
                if self.index is not None:
                    # Create temp file first to avoid corruption
                    temp_index_path = f"{self.index_path}.tmp"
                    faiss.write_index(self.index, temp_index_path)
                    
                    # Safely rename
                    if os.path.exists(temp_index_path):
                        if os.path.exists(self.index_path):
                            os.remove(self.index_path)
                        os.rename(temp_index_path, self.index_path)
                
                # Save documents and metadata
                data = {
                    'documents': self.documents,
                    'metadata': self.metadata,
                    'doc_hashes': list(self.doc_hashes),
                    'deleted_ids': list(self.deleted_ids),
                    'embedding_dim': self.embedding_dim,
                    'updated_at': datetime.now().isoformat()
                }
                
                # Save to temp file first
                temp_data_path = f"{self.data_path}.tmp"
                with open(temp_data_path, 'wb') as f:
                    pickle.dump(data, f)
                    
                # Safely rename
                if os.path.exists(temp_data_path):
                    if os.path.exists(self.data_path):
                        os.remove(self.data_path)
                    os.rename(temp_data_path, self.data_path)
                
                return True
                
            except Exception as e:
                print(f"Error saving vector store: {e}")
                # Attempt to clean up temp files
                for path in [f"{self.index_path}.tmp", f"{self.data_path}.tmp"]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass
                return False
    
    def load(self) -> bool:
        """
        Load vector store from disk.
        
        Returns:
            Success boolean
        """
        with self._lock:
            try:
                # Check if files exist
                if not (os.path.exists(self.index_path) and os.path.exists(self.data_path)):
                    print("No existing vector store found, initializing new store")
                    return False
                    
                # Load documents and metadata
                with open(self.data_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.documents = data.get('documents', [])
                self.metadata = data.get('metadata', [])
                self.doc_hashes = set(data.get('doc_hashes', []))
                self.deleted_ids = set(data.get('deleted_ids', []))
                self.embedding_dim = data.get('embedding_dim', self.embedding_dim)
                
                # Load index if there are documents
                if self.documents and os.path.getsize(self.index_path) > 0:
                    self.index = faiss.read_index(self.index_path)
                    
                    # Validate index size matches document count
                    expected_count = len(self.documents) - len(self.deleted_ids)
                    if self.index.ntotal != expected_count:
                        print(f"Warning: Index size ({self.index.ntotal}) doesn't match document count "
                              f"({expected_count})")
                        print("This might lead to issues. Consider running rebuild_index() to fix.")
                
                return True
                
            except Exception as e:
                print(f"Error loading vector store: {e}")
                # Initialize new store if loading fails
                self.documents = []
                self.metadata = []
                self.doc_hashes = set()
                self.deleted_ids = set()
                self.index = None
                return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            active_docs = len(self.documents) - len(self.deleted_ids)
            index_size = self.index.ntotal if self.index else 0
            
            return {
                'total_documents': len(self.documents),
                'active_documents': active_docs,
                'deleted_documents': len(self.deleted_ids),
                'index_size': index_size,
                'index_dimension': self.embedding_dim,
                'storage_path': self.storage_path,
            }
    
    def cleanup(self):
        """Explicitly clean up resources."""
        with self._lock:
            # Clear caches
            self._embeddings_cache.clear()
            
            # Run garbage collection
            gc.collect()
    
    def _cleanup(self):
        """Cleanup method called by finalizer."""
        self.cleanup()

    def sharpen_embeddings(self, embeddings: np.ndarray, sharpening_factor: float = 0.3,
                      min_cluster_size: int = 3) -> np.ndarray:
        """
        Sharpen embeddings by pushing them closer to their cluster centroids.
        Similar to how image sharpening enhances edges between regions.
        """
        if len(embeddings) < min_cluster_size * 2:
            # Not enough vectors to perform meaningful clustering
            return embeddings

        # Compute similarity matrix - ensure all positive or zero values
        similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                # Compute cosine similarity
                similarity_matrix[i, j] = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                # Ensure non-negative values (though cosine similarity should be in [-1,1])
                similarity_matrix[i, j] = max(0.0, similarity_matrix[i, j])

        # Convert similarities to distances (1 - similarity)
        distances = 1 - similarity_matrix
        # Ensure distances are non-negative
        distances = np.maximum(0, distances)

        # Try to cluster embeddings using DBSCAN
        try:
            from sklearn.cluster import DBSCAN

            # Cluster embeddings using DBSCAN
            clustering = DBSCAN(eps=0.2, min_samples=min_cluster_size, metric='precomputed')
            labels = clustering.fit_predict(distances)

            # Get unique cluster labels (ignoring noise points with label -1)
            unique_clusters = set(label for label in labels if label != -1)

            if not unique_clusters:
                # No valid clusters found
                return embeddings

            # Compute cluster centroids
            centroids = {}
            for cluster in unique_clusters:
                # Get indices of vectors in this cluster
                cluster_indices = np.where(labels == cluster)[0]
                # Calculate centroid
                cluster_vectors = embeddings[cluster_indices]
                centroid = np.mean(cluster_vectors, axis=0)
                # Normalize centroid
                centroid = centroid / np.linalg.norm(centroid)
                centroids[cluster] = centroid

            # Sharpen embeddings by moving them toward their centroids
            sharpened_embeddings = embeddings.copy()

            for i, label in enumerate(labels):
                if label != -1:  # Skip noise points
                    centroid = centroids[label]
                    # Vector pointing from embedding to centroid
                    direction = centroid - embeddings[i]
                    # Move embedding toward centroid by sharpening_factor
                    sharpened_embeddings[i] = embeddings[i] + (direction * sharpening_factor)
                    # Renormalize
                    sharpened_embeddings[i] = sharpened_embeddings[i] / np.linalg.norm(sharpened_embeddings[i])

            return sharpened_embeddings

        except ImportError:
            # Fall back to simpler method if sklearn is not available
            print("Warning: sklearn not available, using original embeddings")
            return embeddings
        except Exception as e:
            print(f"Error during embedding sharpening: {e}")
            return embeddings

    def add_with_sharpening(self, text: str, embedding: np.ndarray, metadata: Dict[str, Any] = None,
                           pre_sharpen: bool = False, sharpen_factor: float = 0.3) -> bool:
        """
        Add a document with optional embedding pre-sharpening.

        Args:
            text: Text to add
            embedding: Embedding vector
            metadata: Additional metadata
            pre_sharpen: Whether to apply clustering-based pre-sharpening
            sharpen_factor: Sharpening factor to use

        Returns:
            Success boolean
        """
        with self._lock:
            if pre_sharpen and hasattr(self, 'documents') and len(self.documents) >= 10:
                # Extract existing embeddings for sharpening context
                # We need at least 10 existing embeddings for meaningful clustering
                existing_embeddings = []
                for doc_hash in list(self._embeddings_cache.keys())[:100]:  # Use up to 100 for efficiency
                    if doc_hash in self._embeddings_cache:
                        existing_embeddings.append(self._embeddings_cache[doc_hash])

                if len(existing_embeddings) >= 10:
                    # We have enough context for sharpening
                    existing_embeddings = np.array(existing_embeddings)
                    # Add new embedding to the array
                    combined = np.vstack([existing_embeddings, embedding[np.newaxis, :]])
                    # Apply sharpening to all embeddings
                    sharpened = self.sharpen_embeddings(combined, sharpening_factor=sharpen_factor)
                    # Extract just the sharpened version of the new embedding
                    embedding = sharpened[-1]

            # Call the regular add method with potentially sharpened embedding
            return self.add(text, embedding, metadata)

    def diagnostics_fractal_embeddings(self, sample_size=10):
        """
        Diagnostic method to verify fractal embedding generation.

        Args:
            sample_size: Number of random embeddings to test

        Returns:
            Dict with diagnostic information about fractal embeddings
        """
        diagnostics = {
            "base_embeddings": [],
            "fractal_variations": [],
            "variation_stats": {
                "mean_cosine_similarities": [],
                "variance_across_levels": []
            }
        }

        # Generate sample embeddings and their fractal variations
        for _ in range(sample_size):
            # Random high-dimensional embedding
            base_embedding = np.random.random(self.embedding_dim)
            base_embedding /= np.linalg.norm(base_embedding)

            # Generate fractal variations
            fractal_embeddings = self._generate_fractal_embeddings(base_embedding)

            # Store base and variations
            diagnostics["base_embeddings"].append(base_embedding)
            diagnostics["fractal_variations"].append(fractal_embeddings)

            # Calculate cosine similarities between base and variations
            level_similarities = []
            for level, var_embedding in fractal_embeddings.items():
                if level == 0:
                    continue  # Skip base level
                similarity = np.dot(base_embedding, var_embedding)
                level_similarities.append(similarity)

            # Store stats
            diagnostics["variation_stats"]["mean_cosine_similarities"].append(
                np.mean(level_similarities)
            )
            diagnostics["variation_stats"]["variance_across_levels"].append(
                np.var(level_similarities)
            )

        # Compute overall statistics
        diagnostics["summary"] = {
            "total_embeddings": sample_size,
            "avg_mean_similarity": np.mean(diagnostics["variation_stats"]["mean_cosine_similarities"]),
            "avg_variation_variance": np.mean(diagnostics["variation_stats"]["variance_across_levels"])
        }

        return diagnostics

    def print_fractal_embedding_diagnostics(self):
        """
        Enhanced diagnostic method with more meaningful deviation analysis.
        Provides a structured analysis of fractal embedding behavior.
        """
        if not hasattr(self, 'fractal_enabled') or not self.fractal_enabled:
            print("Fractal embeddings are not enabled.")
            return

        # Check if fractal indices exist
        if not hasattr(self, 'fractal_indices') or not self.fractal_indices:
            print("No fractal indices found. Ensure documents have been added.")
            return

        print("\n=== Fractal Embedding Diagnostics ===")
        print(f"Fractal Levels: {len(self.fractal_indices)}")

        # Generate test embedding
        test_embedding = np.random.random(self.embedding_dim)
        test_embedding /= np.linalg.norm(test_embedding)

        # Generate fractal embeddings
        test_fractal_embeddings = self._generate_fractal_embeddings(test_embedding)

        # Analyze deviation across levels
        deviations = []
        for level in range(1, len(test_fractal_embeddings)):
            base = test_fractal_embeddings[0]
            variation = test_fractal_embeddings[level]
            deviation = self._compute_deviation(base, variation)
            deviations.append(deviation)

        # Print level information
        for level, index in self.fractal_indices.items():
            print(f"Level {level}:")
            print(f"  Total Embeddings: {index.ntotal}")

        # Print deviation progression
        print("\nDeviation Progression (Test Embeddings):")
        for i, dev in enumerate(deviations, 1):
            print(f"Level {i}: {dev:.4f}")

        # Check if deviation is increasing naturally
        is_increasing = all(
            deviations[i] > deviations[i-1]
            for i in range(1, len(deviations))
        )

        print(f"\nDeviation Progression: {'Optimal' if is_increasing else 'May Need Tuning'}")

        # Provide more detailed interpretation
        if is_increasing:
            print("✅ Fractal embeddings show a promising deviation pattern")
        else:
            print("⚠️ Fractal embeddings may need further refinement")

    def visualize_fractal_embeddings(self):
        """
        Visualize fractal embedding variations using PCA.
        Requires matplotlib and scikit-learn.
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
        except ImportError:
            print("Visualization requires matplotlib and scikit-learn. Please install them.")
            return

        diagnostics = self.diagnostics_fractal_embeddings()

        plt.figure(figsize=(12, 6))

        # PCA for dimensionality reduction
        pca = PCA(n_components=2)

        for i, (base_emb, variations) in enumerate(
            zip(diagnostics["base_embeddings"], diagnostics["fractal_variations"])
        ):
            # Prepare data for PCA
            all_embeddings = [base_emb] + list(variations.values())
            pca_result = pca.fit_transform(all_embeddings)

            # Plot
            plt.scatter(
                pca_result[:, 0],
                pca_result[:, 1],
                label=f'Embedding Set {i+1}',
                alpha=0.7
            )

        plt.title("Fractal Embedding Variations (PCA)")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.legend()
        plt.tight_layout()

        # Save the figure to a file
        plt.savefig("fractal_embeddings_visualization.png")
        print("Visualization saved to 'fractal_embeddings_visualization.png'")

        # Show the figure if in an interactive environment
        try:
            plt.show()
        except:
            pass


class MemoryManager:
    """
    Unified memory manager for vector storage and retrieval.
    This refactored class combines functionality from EnhancedMemoryManager
    and EnhancedMemoryManagerWithSharpening into a single, more maintainable class.
    
    Key features:
    - Efficient vector storage with FAISS
    - Automatic memory extraction from conversations
    - Vector space sharpening for better retrieval
    - Memory consolidation to reduce redundancy
    - Support for multiple users
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        memory_dir: str = "./memory",
        device: str = None,
        auto_memorize: bool = True,
        sharpening_enabled: bool = True,
        sharpening_factor: float = 0.3,
        fractal_enabled: bool = False,
        max_fractal_levels: int = 3
    ):
        """
        Initialize the memory manager.
        
        Args:
            model_name: Name of the embedding model to use
            memory_dir: Directory for storing memory files
            device: Device to use for embedding generation
            auto_memorize: Whether to automatically memorize conversations
            sharpening_enabled: Whether to enable vector space sharpening
            sharpening_factor: Factor to control sharpening strength
        """
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        
        self.user_stores = {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.auto_memorize = auto_memorize
        self.sharpening_enabled = sharpening_enabled
        self.sharpening_factor = max(0.0, min(1.0, sharpening_factor))  # Ensure valid range
        self.fractal_enabled = fractal_enabled
        self.max_fractal_levels = max_fractal_levels

        # Load embedding model
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.embedding_dim = self.model.config.hidden_size
            print(f"Loaded embedding model: {model_name}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            # Fallback to random embeddings
            self.tokenizer = None
            self.model = None
            self.embedding_dim = 384  # Common dimension
        
        # Schedule for periodic maintenance
        self.last_consolidation = datetime.now()
        self.consolidation_interval = 60 * 60  # 1 hour in seconds
    
    def _get_user_store(self, user_id: str) -> VectorStore:
        """
        Get or create a vector store for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Vector store for the user
        """
        if user_id not in self.user_stores:
            store_path = os.path.join(self.memory_dir, f"{user_id}_store")
            self.user_stores[user_id] = VectorStore(
                storage_path=store_path,
                embedding_function=self.generate_embedding,
                embedding_dim=self.embedding_dim,
                fractal_enabled=self.fractal_enabled,
                max_fractal_levels=self.max_fractal_levels
            )
        return self.user_stores[user_id]


    def toggle_auto_memorize(self) -> bool:
        """
        Toggle automatic memorization on/off.

        Returns:
            New auto_memorize state
        """
        self.auto_memorize = not self.auto_memorize
        return self.auto_memorize

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a text snippet.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if self.model is None:
            # Fallback to random embeddings
            return np.random.random(self.embedding_dim)
        
        try:
            # Tokenize and embed the text
            import torch
            inputs = self.tokenizer(
                text, 
                return_tensors="pt",
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use mean pooling to get a single vector
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.random.random(self.embedding_dim)
    
    def extract_key_information(self, query: str, response: str) -> List[str]:
        """
        Extract key information from conversation to remember.
        
        Args:
            query: User query
            response: Assistant response
            
        Returns:
            List of key information snippets
        """
        import re
        # Extract factual statements (sentences with potential facts)
        combined_text = f"{query} {response}"
        sentences = re.split(r'[.!?]', combined_text)
        key_info = []
        
        # Process each sentence with lower thresholds
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            words = sentence.lower().split()
            if not words:
                continue
                
            # Simple criteria: sentences with entities or numbers
            contains_entity = bool(re.search(r'\b[A-Z][a-z]+\b', sentence))
            contains_numbers = bool(re.search(r'\b\d+\b', sentence))
            
            # Either has capitalized words or numbers or is long enough
            if contains_entity or contains_numbers or len(words) > 5:
                clean_sentence = re.sub(r'\s+', ' ', sentence)
                key_info.append(clean_sentence)
        
        return key_info[:10]  # Return up to 10 memories
    
    def _check_maintenance(self, user_id: str):
        """
        Check if memory maintenance (consolidation) is due.

        Args:
            user_id: The user ID for the store to check
        """
        # Skip if the memory manager doesn't implement consolidation functionality
        if not hasattr(self, 'consolidate_memories'):
            return

        # Check if we have access to the last consolidation time
        if hasattr(self, 'last_consolidation') and hasattr(self, 'consolidation_interval'):
            now = datetime.now()
            time_diff = (now - self.last_consolidation).total_seconds()

            if time_diff > self.consolidation_interval:
                # Call consolidate_memories with user_id if method accepts it
                try:
                    self.consolidate_memories(user_id)
                    self.last_consolidation = now
                except TypeError:
                    # If it doesn't accept user_id, try without it
                    try:
                        self.consolidate_memories()
                        self.last_consolidation = now
                    except Exception as e:
                        print(f"Error during memory consolidation: {e}")
        else:
            # If we don't have access to timing attributes, try direct consolidation
            # (will happen on every query, but better than never consolidating)
            try:
                # Try with and without user_id
                try:
                    self.consolidate_memories(user_id)
                except TypeError:
                    self.consolidate_memories()
            except Exception:
                # If consolidation fails, silently continue
                pass

    def add_memory(self, 
                  user_id: str, 
                  query: str, 
                  response: str,
                  memory_type: str = "general", 
                  attributes: Dict = None) -> int:
        """
        Add a new memory with enhanced metadata tagging.
        
        Args:
            user_id: User identifier
            query: Query text
            response: Response text
            memory_type: Type of memory
            attributes: Additional attributes
            
        Returns:
            Number of memories added
        """
        if not self.auto_memorize:
            return 0
            
        # Check if maintenance is due
        self._check_maintenance(user_id)
        
        attributes = attributes or {}
        
        # Extract key information
        key_info = self.extract_key_information(query, response)
        
        # Detect corrections
        is_correction = False
        correction_indicators = [
            "incorrect", "wrong", "not true", "false", "mistake",
            "that's not", "but that's", "but that is", "actually",
            "in fact", "not correct"
        ]
        
        query_lower = query.lower()
        if any(indicator in query_lower for indicator in correction_indicators):
            is_correction = True
            memory_type = "correction"
        
        # Store each key piece of information
        memories_added = 0
        for info in key_info:
            embedding = self.generate_embedding(info)
            
            # Create enhanced metadata
            if memory_type == "correction" or is_correction:
                # Extract key terms from the correction
                terms = set(query_lower.split())
                stop_words = {'a', 'an', 'the', 'but', 'and', 'or', 'is', 'are', 'was', 'were', 
                             'to', 'for', 'in', 'on', 'at', 'by'}
                keywords = ' '.join([term for term in terms if term not in stop_words and len(term) > 2])
                
                metadata = {
                    'source_query': query,
                    'source_response': response[:100] + "..." if len(response) > 100 else response,
                    'timestamp': datetime.now().isoformat(),
                    'type': memory_type,
                    'is_correction': True,
                    'correction_keywords': keywords,
                    'priority': 10,  # Higher priority for corrections
                    'attributes': attributes
                }
            else:
                metadata = {
                    'source_query': query,
                    'source_response': response[:100] + "..." if len(response) > 100 else response,
                    'timestamp': datetime.now().isoformat(),
                    'type': memory_type,
                    'priority': 1,  # Normal priority
                    'attributes': attributes
                }
            
            # Add to the store
            store = self._get_user_store(user_id)
            added = store.add(text=info, embedding=embedding, metadata=metadata)
            
            if added:
                memories_added += 1
        
        return memories_added
    
    def retrieve_relevant_memories(self, 
                                  user_id: str, 
                                  query: str, 
                                  top_k: int = 8, 
                                  apply_sharpening: bool = None) -> str:
        """
        Retrieve relevant memories for a query.
        
        Args:
            user_id: User identifier
            query: Query text
            top_k: Maximum number of memories to retrieve
            apply_sharpening: Whether to apply sharpening
            
        Returns:
            Formatted string of relevant memories
        """
        # Use class setting if not explicitly specified
        if apply_sharpening is None:
            apply_sharpening = self.sharpening_enabled
            
        # Get the store
        store = self._get_user_store(user_id)
        
        # Generate embedding for the query
        query_embedding = self.generate_embedding(query)
        
        # Search with optional sharpening
        # results = store.search(
        #     query_embedding,
        #     top_k=top_k*2,  # Get more results for better filtering
        #     min_similarity=0.25,
        #     apply_sharpening=apply_sharpening,
        #     sharpening_factor=self.sharpening_factor
        # )

        results = self.enhanced_fractal_search(
            query_embedding,
            top_k=retrieve_count*2,
            min_similarity=0.25,
            apply_sharpening=apply_sharpening,
            sharpening_factor=sharpening_factor,
            multi_level_search=True
        )


        # If no results, return empty string
        if not results:
            return ""
            
        # Group results into clear categories
        corrections = []
        factual_info = []
        general_info = []
        
        for result in results[:top_k]:
            if result.get('metadata', {}).get('is_correction', False):
                corrections.append(result)
            elif any(term in result['text'].lower() for term in 
                    ['definition', 'fact', 'rule', 'alphabet', 'order']):
                factual_info.append(result)
            else:
                general_info.append(result)
                
        # Compose the memory text with clear sections
        memory_text = ""
        
        if corrections:
            memory_text += "IMPORTANT CORRECTIONS (You MUST apply these):\n"
            for result in corrections:
                memory_text += f"- {result['text']}\n"
            memory_text += "\n"
            
        if factual_info:
            memory_text += "FACTUAL INFORMATION:\n"
            for result in factual_info:
                memory_text += f"- {result['text']}\n"
            memory_text += "\n"
            
        if general_info and (not corrections or not factual_info):
            memory_text += "OTHER RELEVANT INFORMATION:\n"
            for result in general_info:
                memory_text += f"- {result['text']}\n"
                
        return memory_text

    def add_memory_with_sharpening(self,
                  user_id: str,
                  query: str,
                  response: str,
                  memory_type: str = "general",
                  attributes: Dict = None,
                  pre_sharpen: bool = False) -> int:
        """
        Add a new memory with enhanced metadata tagging and optional embedding pre-sharpening.

        Args:
            user_id: User identifier
            query: Query text
            response: Response text
            memory_type: Type of memory
            attributes: Additional attributes
            pre_sharpen: Whether to apply clustering-based pre-sharpening

        Returns:
            Number of memories added
        """
        if not self.auto_memorize:
            return 0

        # Check if maintenance is due
        self._check_maintenance(user_id)

        attributes = attributes or {}

        # Extract key information
        key_info = self.extract_key_information(query, response)

        # Detect corrections
        is_correction = False
        correction_indicators = [
            "incorrect", "wrong", "not true", "false", "mistake",
            "that's not", "but that's", "but that is", "actually",
            "in fact", "not correct"
        ]

        query_lower = query.lower()
        if any(indicator in query_lower for indicator in correction_indicators):
            is_correction = True
            memory_type = "correction"

        # Store each key piece of information
        memories_added = 0
        for info in key_info:
            embedding = self.generate_embedding(info)

            # Create enhanced metadata
            if memory_type == "correction" or is_correction:
                # Extract key terms from the correction
                terms = set(query_lower.split())
                stop_words = {'a', 'an', 'the', 'but', 'and', 'or', 'is', 'are', 'was', 'were',
                             'to', 'for', 'in', 'on', 'at', 'by'}
                keywords = ' '.join([term for term in terms if term not in stop_words and len(term) > 2])

                metadata = {
                    'source_query': query,
                    'source_response': response[:100] + "..." if len(response) > 100 else response,
                    'timestamp': datetime.now().isoformat(),
                    'type': memory_type,
                    'is_correction': True,
                    'correction_keywords': keywords,
                    'priority': 10,  # Higher priority for corrections
                    'attributes': attributes
                }
            else:
                metadata = {
                    'source_query': query,
                    'source_response': response[:100] + "..." if len(response) > 100 else response,
                    'timestamp': datetime.now().isoformat(),
                    'type': memory_type,
                    'priority': 1,  # Normal priority
                    'attributes': attributes
                }

            # Add to the store with optional pre-sharpening
            store = self._get_user_store(user_id)
            if pre_sharpen and hasattr(store, 'add_with_sharpening'):
                added = store.add_with_sharpening(
                    text=info,
                    embedding=embedding,
                    metadata=metadata,
                    pre_sharpen=True,
                    sharpen_factor=self.sharpening_factor
                )
            else:
                added = store.add(text=info, embedding=embedding, metadata=metadata)

            if added:
                memories_added += 1

        return memories_added