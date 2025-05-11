"""
Unified memory system for TinyLlama Chat.
Provides a cohesive approach to storing and retrieving different types of information
with improved integration and simplified architecture.
"""
import re
import os
import json
import faiss
import numpy as np
import time
import hashlib
import threading
import pickle
import gc
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable, Union, TypeVar, Generic

T = TypeVar('T')

class MemoryItem:
    """
    Unified knowledge item structure with simplified design.
    """

    def __init__(self,
                content: str,
                embedding: np.ndarray,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a memory item.

        Args:
            content: The actual text/content
            embedding: Vector representation
            metadata: Additional information about this item
        """
        self.content = content
        self.embedding = embedding  # Base embedding
        self.additional_embeddings = {}  # For enhanced embeddings (previously "fractal")

        # Initialize metadata
        self.metadata = metadata or {}

        # Ensure timestamp exists
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now().timestamp()

        # Ensure source exists
        if "source" not in self.metadata:
            self.metadata["source"] = "unknown"

        # Ensure retrieval count exists
        if "retrieval_count" not in self.metadata:
            self.metadata["retrieval_count"] = 0

        # Generate ID based on content and timestamp
        self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a unique ID for this item."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()
        timestamp = str(int(self.metadata["timestamp"] * 1000))
        return f"item_{content_hash[:8]}_{timestamp[-6:]}"

    def add_enhanced_embedding(self, level: int, embedding: np.ndarray):
        """Add an enhanced embedding at the specified level."""
        self.additional_embeddings[level] = embedding

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            # Embeddings stored separately
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: np.ndarray) -> 'MemoryItem':
        """Create a MemoryItem from a dictionary and embedding."""
        item = cls(
            content=data["content"],
            embedding=embedding,
            metadata=data["metadata"]
        )
        item.id = data.get("id", item.id)  # Use stored ID if available
        return item

class MemoryManager:
    """
    Unified memory manager with simplified architecture and improved integration.
    Fixes issues with standard and enhanced search functions.
    """

    def __init__(self,
                storage_path: str = "./memory",
                embedding_dim: int = None,  # Make this optional (384 default)
                enable_enhanced_embeddings: bool = True,
                max_enhancement_levels: int = 3,
                auto_save: bool = True,
                similarity_enhancement_factor: float = 0.3):
        """
        Initialize the memory manager.

        Args:
            storage_path: Base path for storing memory data
            embedding_dim: Dimension of the embedding vectors
            enable_enhanced_embeddings: Whether to use enhanced embeddings (previously "fractal")
            max_enhancement_levels: Maximum number of enhancement levels if enabled
            auto_save: Whether to automatically save changes
            similarity_enhancement_factor: Factor for non-linear similarity enhancement (0.0-1.0)
        """
        self.storage_path = storage_path
        self.embedding_dim = embedding_dim
        self.enable_enhanced_embeddings = enable_enhanced_embeddings
        self.max_enhancement_levels = max_enhancement_levels
        self.auto_save = auto_save
        self.similarity_enhancement_factor = similarity_enhancement_factor

        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)

        # Thread lock for concurrent access
        self._lock = threading.RLock()

        # Memory storage
        self.items = []  # List of MemoryItem objects
        self.embeddings = []  # List of corresponding base embeddings
        self.enhanced_indices = {}  # Level -> FAISS index
        self.id_to_index = {}  # Map from item ID to index in items list
        self.deleted_ids = set()  # Set of deleted item IDs

        # Initialize FAISS index
        self.index = None

        # Embedding function to be set externally
        self.embedding_function = None
        self.batch_embedding_function = None

        # Load existing memory if available
        self.load()

        # Initialize rotational matrices for enhanced embeddings
        if enable_enhanced_embeddings:
           self._initialize_enhancement_matrices()

    def get_time(self) -> str:
        """Get formatted timestamp for logging."""
        return datetime.now().strftime("[%d/%m/%y %H:%M:%S]") + ' [Memory]'

    def display_progress_bar(self, iteration, total, prefix='', suffix='', length=50, fill='█', print_end='\r'):
        """
        Display a progress bar in the console.

        Args:
            iteration: Current iteration (Int)
            total: Total iterations (Int)
            prefix: Prefix string (Str)
            suffix: Suffix string (Str)
            length: Bar length (Int)
            fill: Bar fill character (Str)
            print_end: End character (e.g. '\r', '\n') (Str)
        """
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
        # Print a new line when done
        if iteration >= total:
            print()

    def set_embedding_function(self,
                              function: Callable[[str], np.ndarray],
                              batch_function: Optional[Callable[[List[str]], List[np.ndarray]]] = None):
        """
        Set embedding function for generating vector representations.

        Args:
            function: Function to embed single text
            batch_function: Function to embed multiple texts (for efficiency)
        """
        self.embedding_function = function
        self.batch_embedding_function = batch_function

    def update_embedding_dimension(self, new_dim):
        """
        Update the embedding dimension and migrate all components.

        Args:
            new_dim: New embedding dimension
        """
        if self.embedding_dim == new_dim:
            return

        print(f"{self.get_time()} Updating embedding dimension: {self.embedding_dim} → {new_dim}")

        # Store old dimension
        old_dim = self.embedding_dim

        # Update dimension
        self.embedding_dim = new_dim

        # Resize embeddings
        self._resize_all_embeddings(old_dim, new_dim)

        # Reinitialize rotation matrices
        self._initialize_enhancement_matrices()

        # Rebuild FAISS indices
        self._rebuild_index()
        self._rebuild_enhanced_indices()

        print(f"{self.get_time()} Migration to dimension {new_dim} complete")

    def add(self,
            content: str,
            metadata: Optional[Dict[str, Any]] = None,
            use_enhanced_embeddings: Optional[bool] = None) -> Optional[str]:
        """
        Add a new item to memory.

        Args:
            content: Text content to remember
            metadata: Metadata about the item
            use_enhanced_embeddings: Override default enhanced embedding setting

        Returns:
            ID of the added item or None if failed
        """
        with self._lock:
            # Skip if content is empty
            if not content or not content.strip():
                return None

            # Generate embedding if embedding function is available
            if self.embedding_function:
                try:
                    embedding = self.embedding_function(content)
                except Exception as e:
                    print(f"{self.get_time()} Error generating embedding: {e}")
                    return None
            else:
                # No embedding function available
                print(f"{self.get_time()} No embedding function set, cannot add item")
                return None

            # Ensure metadata exists
            if metadata is None:
                metadata = {}

            # Create memory item
            item = MemoryItem(
                content=content,
                embedding=embedding,
                metadata=metadata
            )

            # Determine whether to use enhanced embeddings
            use_enhanced = self.enable_enhanced_embeddings if use_enhanced_embeddings is None else use_enhanced_embeddings

            # Generate enhanced embeddings if requested
            if use_enhanced:
                self._add_enhanced_embeddings(item)

            # Add to storage
            return self._add_item_to_store(item)

    def _resize_all_embeddings(self, old_dim, new_dim):
        """
        Resize all embeddings to match the new dimension.

        Args:
            old_dim: Original dimension
            new_dim: Target dimension
        """
        print(f"{self.get_time()} Resizing all embeddings from {old_dim}D to {new_dim}D")

        # Resize base embeddings
        for i, item in enumerate(self.items):
            if len(item.embedding) != new_dim:
                # Resize embedding
                if len(item.embedding) > new_dim:
                    # Truncate
                    new_embedding = item.embedding[:new_dim]
                else:
                    # Pad with zeros
                    new_embedding = np.pad(item.embedding, (0, new_dim - len(item.embedding)))

                # Normalize
                norm = np.linalg.norm(new_embedding)
                if norm > 1e-10:
                    new_embedding = new_embedding / norm

                # Update embedding
                item.embedding = new_embedding
                self.embeddings[i] = new_embedding

        # Resize enhanced embeddings
        for item in self.items:
            if hasattr(item, 'additional_embeddings') and item.additional_embeddings:
                resized_embeddings = {}
                for level, embedding in item.additional_embeddings.items():
                    if len(embedding) != new_dim:
                        # Resize embedding
                        if len(embedding) > new_dim:
                            # Truncate
                            new_embedding = embedding[:new_dim]
                        else:
                            # Pad with zeros
                            new_embedding = np.pad(embedding, (0, new_dim - len(embedding)))

                        # Normalize
                        norm = np.linalg.norm(new_embedding)
                        if norm > 1e-10:
                            new_embedding = new_embedding / norm

                        resized_embeddings[level] = new_embedding
                    else:
                        resized_embeddings[level] = embedding

                # Update enhanced embeddings
                item.additional_embeddings = resized_embeddings

        print(f"{self.get_time()} All embeddings resized to {new_dim}D")

    def _initialize_enhancement_matrices(self):
        """
        Initialize matrices for enhanced embeddings with proper dimension detection and progress bar.
        """
        # Use the actual embedding dimension from the first item or from config
        if hasattr(self, 'embedding_dim'):
            embedding_dim = self.embedding_dim
        elif self.items and hasattr(self.items[0], 'embedding'):
            embedding_dim = self.items[0].embedding.shape[0]
        else:
            # Default dimension if we can't determine it yet
            embedding_dim = 2048  # TinyLlama's embedding dimension is 2048

        print(f"{self.get_time()} Initializing enhancement matrices for dimension {embedding_dim}")

        if hasattr(self, '_rotation_matrices') and self._rotation_matrices:
            # Check if matrices match the current dimension
            first_matrix = next(iter(self._rotation_matrices.values())) if self._rotation_matrices else None
            if first_matrix is not None and first_matrix.shape[0] == embedding_dim:
                # Matrices already initialized with correct dimension
                return
            # Otherwise we'll recreate them with the correct dimension

        # Create rotation matrices for all potential levels
        self._rotation_matrices = {}
        self._level_biases = {}

        # Support up to 20 potential levels
        max_potential_levels = 4

        print(f"{self.get_time()} Creating {max_potential_levels} enhancement matrices...")

        # Create matrices with fixed random seed for determinism
        for i in range(1, max_potential_levels + 1):
            # Display progress bar
            self.display_progress_bar(i, max_potential_levels,
                               prefix=f'{self.get_time()} Progress:',
                               suffix=f'Matrix {i}/{max_potential_levels}',
                               length=40)

            # Set fixed seed per level
            np.random.seed(42 + i * 10)  # Increased seed difference

            # Create rotation matrix with increasing randomness by level
            # FIXED: Adjust rotation factor to increase diversity between levels
            rotation_factor = 0.15 + (i * 0.05)  # Increased base factor and increment
            rotation = np.random.normal(0, rotation_factor, (embedding_dim, embedding_dim))

            # For earlier levels, preserve more of the original structure
            if i <= 2:  # Changed from 3 to 2
                # Add identity matrix component to preserve original information
                preservation_factor = 0.8 - (i * 0.2)  # More aggressive reduction
                rotation = rotation + np.eye(embedding_dim) * preservation_factor

            # Ensure the matrix is orthogonal
            u, _, vh = np.linalg.svd(rotation, full_matrices=False)
            self._rotation_matrices[i] = u @ vh

            # Create level-specific bias vector with increased variation
            np.random.seed(137 + i * 10)  # Increased seed difference
            bias_factor = 0.02 + (i * 0.01)  # Doubled impact
            self._level_biases[i] = np.random.normal(0, bias_factor, (embedding_dim,))

        print(f"{self.get_time()} Enhancement matrices initialization complete.")

    def _load_enhanced_embeddings(self):
        """Load enhanced embeddings if available."""
        if not self.enable_enhanced_embeddings:
            return

        enhanced_path = os.path.join(self.storage_path, "enhanced_embeddings.pkl")
        if not os.path.exists(enhanced_path):
            print(f"{self.get_time()} No enhanced embeddings found")
            return

        try:
            print(f"{self.get_time()} Loading enhanced embeddings...")
            with open(enhanced_path, 'rb') as f:
                enhanced_data = pickle.load(f)

            # Initialize enhancement matrices with current dimension
            self._initialize_enhancement_matrices()

            # Process enhanced embeddings
            loaded_count = 0

            for item_id, item_enhanced in enhanced_data.items():
                if item_id in self.id_to_index:
                    idx = self.id_to_index[item_id]
                    self.items[idx].additional_embeddings = item_enhanced
                    loaded_count += 1

            print(f"{self.get_time()} Loaded {loaded_count} enhanced embeddings.")

            # Rebuild enhanced indices if needed
            if loaded_count > 0:
                self._rebuild_enhanced_indices()

        except Exception as e:
            print(f"{self.get_time()} Error loading enhanced embeddings: {e}")
            traceback.print_exc()

    def _add_enhanced_embeddings(self, item: MemoryItem):
        """
        Generate and add enhanced embeddings to a memory item with dimension validation.
        """
        # Skip if enhanced embeddings are disabled
        if not self.enable_enhanced_embeddings:
            return

        # Ensure matrices are initialized with current embedding dimension
        if not hasattr(self, '_rotation_matrices') or not self._rotation_matrices:
            self._initialize_enhancement_matrices()
        else:
            # Check if dimensions match
            first_matrix = next(iter(self._rotation_matrices.values()))
            if first_matrix.shape[0] != self.embedding_dim:
                print(f"{self.get_time()} Rotation matrix dimension ({first_matrix.shape[0]}) does not match current embedding dimension ({self.embedding_dim}), reinitializing...")
                self._initialize_enhancement_matrices()

        # Skip if no base embedding available
        if item.embedding is None or len(item.embedding) == 0:
            print(f"{self.get_time()} Warning: Cannot generate enhanced embeddings - no base embedding")
            return

        # Check embedding dimension
        if len(item.embedding) != self.embedding_dim:
            print(f"{self.get_time()} Warning: Embedding dimension mismatch: {len(item.embedding)} vs {self.embedding_dim}")
            # Try to resize embedding to match
            if len(item.embedding) > self.embedding_dim:
                item.embedding = item.embedding[:self.embedding_dim]
            else:
                # Pad with zeros
                item.embedding = np.pad(item.embedding, (0, self.embedding_dim - len(item.embedding)))

        base_embedding = item.embedding

        # Generate enhanced embedding for each level
        for level in range(1, self.max_enhancement_levels + 1):
            try:
                level_embedding = self._generate_level_embedding(base_embedding, level)

                if level_embedding is not None and len(level_embedding) > 0:
                    # Verify the embedding is valid
                    embedding_norm = np.linalg.norm(level_embedding)
                    if embedding_norm < 1e-10:
                        continue

                    # Save the embedding
                    item.add_enhanced_embedding(level, level_embedding)
            except Exception as e:
                print(f"{self.get_time()} Error generating enhanced embedding: {e}")

    def _generate_level_embedding(self, base_embedding: np.ndarray, level: int) -> np.ndarray:
        """
        Generate a level-specific embedding.

        Args:
            base_embedding: The original embedding vector
            level: The enhancement level to generate

        Returns:
            Transformed embedding for the specified level
        """
        if level == 0:
            return base_embedding

        # Check if rotation matrices exist with correct dimension
        if not hasattr(self, '_rotation_matrices') or level not in self._rotation_matrices:
            # Initialize matrices if needed
            self._initialize_enhancement_matrices()

            # Check again after initialization
            if level not in self._rotation_matrices:
                return None

        # Check dimension match
        matrix_dim = self._rotation_matrices[level].shape[0]
        if matrix_dim != len(base_embedding):
            print(f"{self.get_time()} Dimension mismatch in _generate_level_embedding: {matrix_dim} vs {len(base_embedding)}")
            return None

        # Apply the transformation
        try:
            rotated = np.dot(base_embedding, self._rotation_matrices[level])

            # Apply level-specific bias if available
            if level in self._level_biases:
                shifted = rotated + self._level_biases[level]
            else:
                shifted = rotated

            # Apply non-linear transformation that varies by level
            if level % 3 == 0:
                # Enhance magnitude differences
                transformed = np.sign(shifted) * np.power(np.abs(shifted), 0.8)
            elif level % 3 == 1:
                # Apply sigmoid-like function
                transformed = np.tanh(shifted * (1.0 + 0.1 * level))
            else:
                # Log transformation
                transformed = np.sign(shifted) * np.log(1 + np.abs(shifted) * (1.0 + 0.05 * level))

            # Normalize the result
            norm = np.linalg.norm(transformed)
            if norm < 1e-10:
                norm = 1e-10
            normalized = transformed / norm

            return normalized
        except Exception as e:
            print(f"{self.get_time()} Error in _generate_level_embedding: {e}")
            return None

    def _add_item_to_store(self, item: MemoryItem) -> Optional[str]:
        """
        Add a memory item to storage and indices.

        Args:
            item: Memory item to add

        Returns:
            ID of added item or None if failed
        """
        # If this is the first item, initialize index
        if self.index is None:
            try:
                self._create_index(item.embedding.shape[0])
            except Exception as e:
                print(f"{self.get_time()} Error creating index: {e}")
                return None

        # Check embedding dimension matches index
        if len(item.embedding) != self.embedding_dim:
            print(f"{self.get_time()} Embedding dimension mismatch: {len(item.embedding)} vs {self.embedding_dim}")

            # Resize embedding to match index
            if len(item.embedding) > self.embedding_dim:
                item.embedding = item.embedding[:self.embedding_dim]
            else:
                item.embedding = np.pad(item.embedding, (0, self.embedding_dim - len(item.embedding)))

            print(f"{self.get_time()} Resized embedding to match index dimension")

        # Normalize embedding for cosine similarity
        try:
            embedding_norm = np.linalg.norm(item.embedding)
            if embedding_norm < 1e-10:
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

            # Add enhanced embeddings to respective indices
            if item.additional_embeddings:
                for level, level_embedding in item.additional_embeddings.items():
                    try:
                        # Normalize level embedding
                        level_norm = np.linalg.norm(level_embedding)
                        if level_norm < 1e-10:
                            level_norm = 1e-10
                        normalized_level_embedding = level_embedding / level_norm

                        # Ensure index exists for this level
                        if level not in self.enhanced_indices:
                            self.enhanced_indices[level] = faiss.IndexFlatIP(self.embedding_dim)

                        # Add to level-specific index
                        self.enhanced_indices[level].add(np.array([normalized_level_embedding], dtype=np.float32))
                    except Exception as e:
                        print(f"{self.get_time()} Error adding enhanced embedding: {e}")

            # Auto-save if enabled
            if self.auto_save:
                self.save()

            return item.id

        except Exception as e:
            print(f"{self.get_time()} Error adding item to store: {e}")
            traceback.print_exc()
            return None

    def _create_index(self, dim: int = None):
        """Create a new FAISS index with the specified dimension."""
        # Update dimension if provided
        if dim is not None:
            self.embedding_dim = dim

        # Free old index if it exists
        if hasattr(self, 'index') and self.index is not None:
            del self.index

        # Create new index with current dimension
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine on normalized vectors)

        print(f"{self.get_time()} Created new FAISS index with dimension {self.embedding_dim}")

    def _enhance_similarity(self, similarity: float) -> float:
        """
        Apply non-linear enhancement to similarity scores.

        Args:
            similarity: Raw similarity score (0.0-1.0)

        Returns:
            Enhanced similarity score
        """
        # Skip if no enhancement requested
        if self.similarity_enhancement_factor <= 0:
            return similarity

        # Apply non-linear enhancement
        if similarity > 0.50:
            # Boost high similarities (more confident matches)
            boost = (similarity - 0.6) * self.similarity_enhancement_factor * 2.0
            enhanced = min(1.0, similarity + boost)
        elif similarity < 0.50:
            # Reduce low similarities (less confident matches)
            reduction = (0.4 - similarity) * self.similarity_enhancement_factor * 2.0
            enhanced = max(0.0, similarity - reduction)
        else:
            # Middle range - moderate effect
            deviation = (similarity - 0.5) * self.similarity_enhancement_factor
            enhanced = 0.5 + deviation

        return enhanced

    def retrieve(self,
                query: str,
                top_k: int = 5,
                min_similarity: float = 0.25,
                use_enhanced_embeddings: Optional[bool] = None,
                metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant items based on semantic similarity with batch processing optimization.

        Args:
            query: The search query
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            use_enhanced_embeddings: Override default enhanced embedding setting
            metadata_filter: Filter results by metadata fields

        Returns:
            List of matching items with similarity scores
        """
        with self._lock:
            # Basic validation
            if not query or not query.strip():
                return []

            # Check if we have any memories to search
            if not self.items or self.index is None or self.index.ntotal == 0:
                return []

            # Check if embedding function is available
            if not self.embedding_function:
                print(f"{self.get_time()} No embedding function set")
                return []

            # Generate query embedding with potential batch processing optimization
            try:
                query_embedding = self.embedding_function(query)

                # Check if we got a valid embedding
                if query_embedding is None or len(query_embedding) == 0:
                    return []

                # Normalize query embedding
                query_norm = np.linalg.norm(query_embedding)
                if query_norm < 1e-10:
                    query_norm = 1e-10
                normalized_query = query_embedding / query_norm
            except Exception as e:
                print(f"{self.get_time()} Error generating query embedding: {e}")
                return []

            # Determine search approach
            use_enhanced = self.enable_enhanced_embeddings
            if use_enhanced_embeddings is not None:
                use_enhanced = use_enhanced_embeddings

            have_enhanced_indices = bool(self.enhanced_indices) and any(
                idx.ntotal > 0 for idx in self.enhanced_indices.values())

            can_use_enhanced = use_enhanced and have_enhanced_indices

            # Perform search
            # if can_use_enhanced:
            results = self._enhanced_search(
                normalized_query,
                top_k=top_k,
                min_similarity=min_similarity
            )
            # else:
            #     results = self._standard_search(
            #         normalized_query,
            #         top_k=top_k,
            #         min_similarity=min_similarity
            #     )

            # Apply metadata filtering if provided
            # if metadata_filter:
            #     filtered_results = []
            #     for result in results:
            #         if all(result.get('metadata', {}).get(k) == v for k, v in metadata_filter.items()):
            #             filtered_results.append(result)
            #     results = filtered_results

            # Update retrieval metadata
            for result in results:
                item_id = result.get('id')
                if item_id in self.id_to_index:
                    idx = self.id_to_index[item_id]
                    if idx not in self.deleted_ids:
                        item = self.items[idx]
                        # Update retrieval count
                        item.metadata["retrieval_count"] = item.metadata.get("retrieval_count", 0) + 1
                        # Update last access time
                        item.metadata["last_access"] = datetime.now().timestamp()

            # Apply deduplication
            # results = self._deduplicate_results(results)

            return results

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate results based on content similarity.

        Args:
            results: List of retrieval results

        Returns:
            Deduplicated list
        """
        if not results:
            return []

        # Sort by similarity first
        sorted_results = sorted(results, key=lambda x: x.get("similarity", 0), reverse=True)

        # Track seen content hashes
        seen_hashes = set()
        deduplicated = []

        for result in sorted_results:
            content = result.get("content", "").strip().lower()

            # Create a content hash for comparison
            content_hash = hashlib.md5(content.encode()).hexdigest()

            # For very similar content, use a simplified hash
            simple_hash = hashlib.md5(re.sub(r'\W+', ' ', content).encode()).hexdigest()

            # Check if we've seen this content before
            if content_hash not in seen_hashes and simple_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                seen_hashes.add(simple_hash)
                deduplicated.append(result)

        return deduplicated

    def _standard_search(self, query_embedding: np.ndarray, top_k: int, min_similarity: float) -> List[Dict[str, Any]]:
        """
        Perform standard vector search using the main index.

        Args:
            query_embedding: Normalized query embedding
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of matching items with similarity scores
        """
        try:
            # Validate input
            if query_embedding is None or not isinstance(query_embedding, np.ndarray):
                print(f"{self.get_time()} Invalid query embedding in _standard_search")
                return []

            if self.index is None or self.index.ntotal == 0:
                print(f"{self.get_time()} FAISS index is empty or not initialized")
                return []

            # Ensure query has correct dimension
            if len(query_embedding) != self.embedding_dim:
                print(f"{self.get_time()} Query dimension mismatch: {len(query_embedding)} vs {self.embedding_dim}")
                if len(query_embedding) > self.embedding_dim:
                    query_embedding = query_embedding[:self.embedding_dim]
                else:
                    query_embedding = np.pad(query_embedding, (0, self.embedding_dim - len(query_embedding)))

                # Renormalize
                query_norm = np.linalg.norm(query_embedding)
                if query_norm > 1e-10:
                    query_embedding = query_embedding / query_norm

            # Get more results than needed to account for filtering
            search_k = min(top_k * 3, self.index.ntotal)  # Increased from 2x to 3x

            # Prepare query for FAISS - ensure it's float32
            query_array = np.array([query_embedding], dtype=np.float32)

            # Perform search
            similarities, indices = self.index.search(query_array, search_k)

            # Process results with validation
            results = []
            for i, idx in enumerate(indices[0]):
                # Skip invalid indices
                if idx == -1:
                    continue

                # Skip low similarity results
                if similarities[0][i] < min_similarity:
                    continue

                # Validate index bounds
                if idx >= len(self.items):
                    print(f"{self.get_time()} Index out of bounds: {idx} >= {len(self.items)}")
                    continue

                # Skip deleted items
                if idx in self.deleted_ids:
                    continue

                # Get the memory item
                item = self.items[idx]
                similarity = float(similarities[0][i])

                # Apply similarity enhancement
                # enhanced_similarity = self._enhance_similarity(similarity)

                # Add to results
                results.append({
                    "id": item.id,
                    "content": item.content,
                    "similarity": similarity,
                    "raw_similarity": similarity,
                    "metadata": item.metadata
                })

            # Sort by similarity
            results.sort(key=lambda x: x["similarity"], reverse=True)

            # Limit to top-k
            return results[:top_k]

        except Exception as e:
            print(f"{self.get_time()} Error in standard search: {e}")
            traceback.print_exc()
            return []

    def _rebuild_enhanced_indices(self):
        """Rebuild FAISS indices for enhanced embeddings."""
        if not self.enable_enhanced_embeddings:
            return

        # Reset existing indices
        self.enhanced_indices = {}

        # Collect levels from all items
        levels = set()
        for item in self.items:
            if hasattr(item, 'additional_embeddings'):
                levels.update(item.additional_embeddings.keys())

        if not levels:
            return

        print(f"{self.get_time()} Rebuilding enhanced indices for levels: {sorted(levels)}")

        # Create index for each level with progress bar
        for idx, level in enumerate(sorted(levels)):
            # Display progress
            self.display_progress_bar(idx+1, len(levels),
                               prefix=f'{self.get_time()} Building indices:',
                               suffix=f'Level {level}/{len(levels)}',
                               length=40)

            # Create new index with current dimension
            self.enhanced_indices[level] = faiss.IndexFlatIP(self.embedding_dim)

            # Collect embeddings for this level
            level_embeddings = []
            valid_indices = []

            for i, item in enumerate(self.items):
                if hasattr(item, 'additional_embeddings') and level in item.additional_embeddings:
                    embedding = item.additional_embeddings[level]

                    # Ensure correct dimension
                    if len(embedding) != self.embedding_dim:
                        continue

                    # Normalize
                    norm = np.linalg.norm(embedding)
                    if norm > 1e-10:
                        embedding = embedding / norm

                    level_embeddings.append(embedding)
                    valid_indices.append(i)

            # Add to index if we have any
            if level_embeddings:
                try:
                    self.enhanced_indices[level].add(np.array(level_embeddings, dtype=np.float32))
                except Exception as e:
                    print(f"{self.get_time()} Error creating level {level} index: {e}")

        # Print newline after progress bar
        print(f"{self.get_time()} Enhanced indices rebuilt for {len(levels)} levels")

    def _enhanced_search(self, query_embedding: np.ndarray, top_k: int, min_similarity: float) -> List[Dict[str, Any]]:
        """
        Perform enhanced search across multiple embedding levels with proper error handling.

        Args:
            query_embedding: Normalized query embedding
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of matching items with similarity scores
        """
        try:
            # Validate inputs
            if query_embedding is None or not isinstance(query_embedding, np.ndarray):
                print(f"{self.get_time()} Invalid query embedding in _enhanced_search")
                return []

            if not self.enhanced_indices:
                print(f"{self.get_time()} No enhanced indices available")
                return self._standard_search(query_embedding, top_k, min_similarity)

            # Ensure query has correct dimension
            if len(query_embedding) != self.embedding_dim:
                print(f"{self.get_time()} Query dimension mismatch: {len(query_embedding)} vs {self.embedding_dim}")
                # Resize query embedding
                if len(query_embedding) > self.embedding_dim:
                    query_embedding = query_embedding[:self.embedding_dim]
                else:
                    query_embedding = np.pad(query_embedding, (0, self.embedding_dim - len(query_embedding)))
                # Renormalize
                norm = np.linalg.norm(query_embedding)
                if norm > 1e-10:
                    query_embedding = query_embedding / norm

            # Track results by item ID for merging across levels
            result_dict = {}

            # Start with base level search to ensure we always have results
            base_results = [] # self._standard_search(query_embedding, top_k * 2, min_similarity)

            # Initialize result dictionary with base results
            for result in base_results:
                item_id = result.get("id")
                if item_id:  # Only add if ID exists
                    result_dict[item_id] = result

            # Check if enhanced indices exist and have correct dimension
            valid_enhanced_indices = {}
            for level, index in self.enhanced_indices.items():
                if index.d == self.embedding_dim and index.ntotal > 0:  # Added index size check
                    valid_enhanced_indices[level] = index

            # Skip enhanced search if no valid indices
            if not valid_enhanced_indices:
                print(f"{self.get_time()} No valid enhanced indices found")
                return list(result_dict.values())

            # Search across all valid enhanced levels and merge results
            # Process each enhancement level with improved search logic
            for level in sorted(valid_enhanced_indices.keys()):
                try:
                    # Get index for this level and verify it has entries
                    level_index = valid_enhanced_indices[level]
                    if level_index.ntotal == 0:
                        continue

                    # Generate level-specific query with consolidated validation
                    level_query = self._generate_level_embedding(query_embedding, level)
                    if level_query is None or len(level_query) != self.embedding_dim:
                        print(f"{self.get_time()} Invalid level {level} query embedding - skipping")
                        continue

                    # Normalize the query vector - use a higher epsilon for stability
                    query_norm = np.linalg.norm(level_query)
                    if query_norm < 1e-6:  # More forgiving threshold
                        print(f"{self.get_time()} Level {level} query has near-zero norm - skipping")
                        continue

                    normalized_query = level_query / query_norm

                    # Calculate dynamic search depth based on index size
                    # More aggressive for higher levels to capture broader matches
                    depth_factor = 1.0 + (level * 0.5)  # Increase search depth for higher levels
                    index_size = level_index.ntotal
                    search_k = min(int(top_k * depth_factor), index_size)

                    # Perform the search with proper error handling
                    query_array = np.array([normalized_query], dtype=np.float32)
                    try:
                        similarities, indices = level_index.search(query_array, search_k)
                    except Exception as e:
                        print(f"{self.get_time()} FAISS search failed for level {level}: {e}")
                        continue

                    # Track how many results we found at this level
                    found_at_this_level = 0

                    # Use more aggressive similarity thresholding for higher levels
                    # This allows more diverse results at higher levels
                    level_min_similarity = min_similarity * (0.8 ** level)  # Decrease threshold for higher levels

                    # Process search results with proper bounds checking
                    for i, idx in enumerate(indices[0]):
                        # Skip invalid indices
                        if idx == -1:
                            continue

                        # Validate similarity with level-specific threshold
                        similarity = float(similarities[0][i])
                        if similarity <= level_min_similarity:
                            continue

                        # Validate index bounds
                        if idx < 0 or idx >= len(self.items):
                            continue

                        # Skip deleted items
                        if idx in self.deleted_ids:
                            continue

                        # Get the item
                        item = self.items[idx]

                        # Calculate level-adjusted similarity
                        # For higher levels, we boost the similarity of novel results
                        # to promote diversity in the final result set
                        base_similarity = similarity
                        # enhanced_similarity = self._enhance_similarity(base_similarity)

                        # Apply level-specific enhancement
                        # if level == 1:
                        #     # Level 1: emphasize recall with aggressive enhancement
                        #     enhanced_similarity = self._enhance_similarity(min(1.0, base_similarity * 1.15))
                        # elif level == 2:
                        #     # Level 2: emphasize precision with base enhancement
                        #     enhanced_similarity = self._enhance_similarity(base_similarity * 0.95)
                        # elif level == 3:
                        #     # Level 3: moderate enhancement
                        #     enhanced_similarity = self._enhance_similarity(base_similarity)
                        # else:
                        #     # Other levels: standard enhancement
                        #     enhanced_similarity = self._enhance_similarity(base_similarity * 0.5)

                        # Add diversity bonus for novel content
                        # if item.id not in result_dict:
                        #     # New items get a small novelty bonus
                        #     novelty_bonus = 0.001 * level
                        #     enhanced_similarity = min(1.0, enhanced_similarity + novelty_bonus)

                        # Add to result_dict if better than existing or not found yet
                        if item.id not in result_dict: # or enhanced_similarity > result_dict[item.id]["similarity"]:
                            # When replacing a previous level result, preserve the best raw similarity
                            old_raw = result_dict.get(item.id, {}).get("raw_similarity", 0)
                            best_raw = max(old_raw, similarity)

                            # Create result with improved metadata for debugging
                            result_dict[item.id] = {
                                "id": item.id,
                                "content": item.content,
                                "similarity": similarity,
                                "raw_similarity": best_raw,
                                "metadata": item.metadata,
                                "level": level,
                                "level_metrics": {
                                    "original_similarity": similarity,
                                    "enhancement": 1 # enhanced_similarity - similarity
                                }
                            }

                            found_at_this_level += 1

                    # Log level results for debugging
                    if found_at_this_level > 0:
                        print(f"{self.get_time()} Level {level} found {found_at_this_level} items")

                except Exception as e:
                    print(f"{self.get_time()} Error processing level {level}: {e}")
                    traceback.print_exc()
                    # Continue to next level

            # print("\nRESULT DICT", result_dict)
            print("\n")

            # Apply cross-level verification to boost confidence in items found in multiple levels
            result_dict = self._apply_cross_level_verification(result_dict)

            # Extract final results
            results = list(result_dict.values())

            # Sort by similarity
            # results.sort(key=lambda x: x["similarity"], reverse=True)

            # Apply diversity promotion
            results = self._promote_diversity(results, top_k)

            # print ("\nRESULTS", results)
            # Limit to top-k
            return results[:top_k]

        except Exception as e:
            print(f"{self.get_time()} Error in enhanced search: {e}")
            traceback.print_exc()
            # Fall back to standard search
            return self._standard_search(query_embedding, top_k, min_similarity)

    def _promote_diversity(self, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Promote diversity in search results to avoid redundancy.

        Args:
            results: Sorted list of search results
            top_k: Desired number of results

        Returns:
            List of diverse results
        """
        if len(results) <= top_k:
            return results

        # Always keep the top result
        diverse_results = [results[0]]

        # Calculate content embeddings for remaining results
        remaining = results[1:]

        # Select additional results that maximize diversity
        while len(diverse_results) < top_k and remaining:
            # Find the result that has the highest minimum distance to selected results
            best_idx = -1
            best_min_distance = -1

            for idx, candidate in enumerate(remaining):
                # Calculate minimum distance to already selected results
                min_distance = float('inf')
                candidate_content = candidate.get("content", "")

                for selected in diverse_results:
                    selected_content = selected.get("content", "")

                    # Simple text distance (could use embeddings for better results)
                    distance = self._text_distance(candidate_content, selected_content)
                    min_distance = min(min_distance, distance)

                # If this candidate is more diverse, select it
                if min_distance > best_min_distance:
                    best_idx = idx
                    best_min_distance = min_distance

            # Add the most diverse result
            if best_idx >= 0:
                diverse_results.append(remaining[best_idx])
                remaining.pop(best_idx)
            else:
                # Fall back to the next best result by similarity
                diverse_results.append(remaining[0])
                remaining.pop(0)

        return diverse_results

    def _text_distance(self, text1: str, text2: str) -> float:
        """
        Calculate a simple distance metric between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Distance score (higher means more different)
        """
        # Convert to lowercase
        text1 = text1.lower()
        text2 = text2.lower()

        # Get word sets
        words1 = set(re.findall(r'\b\w+\b', text1))
        words2 = set(re.findall(r'\b\w+\b', text2))

        # Calculate Jaccard distance
        if not words1 and not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        # Higher value means more different
        return 1.0 - (intersection / union)

    def _apply_cross_level_verification(self, result_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Boost confidence for items found across multiple levels.

        Args:
            result_dict: Dictionary of search results by item ID

        Returns:
            Updated result_dict with verification applied
        """
        try:
            # Group results by item
            item_levels = {}
            for item_id, result in result_dict.items():
                if "level" in result:  # Check if level is specified
                    if item_id not in item_levels:
                        item_levels[item_id] = []
                    item_levels[item_id].append((result.get("level", 0), result["raw_similarity"]))

            # Apply boost for items found in multiple levels
            for item_id, level_info in item_levels.items():
                if len(level_info) > 1:
                    # Calculate weighted average based on level
                    weighted_sum = 0.0
                    total_weight = 0.0

                    for level, sim in level_info:
                        # Higher weight for higher levels
                        level_weight = 1.0 + (level * 0.2)
                        weighted_sum += sim * level_weight
                        total_weight += level_weight

                    # Calculate weighted average
                    weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0.0

                    # Apply bonus based on level agreement (up to 20% boost)
                    cross_level_bonus = min(0.2, 0.05 * len(level_info))

                    # Apply the combined boost
                    if item_id in result_dict:
                        current_sim = result_dict[item_id]["similarity"]

                        # Greater boost for high-confidence matches
                        if current_sim > 0.7:
                            boost_factor = 1.0 + cross_level_bonus + (weighted_avg * 0.1)
                        else:
                            boost_factor = 1.0 + (cross_level_bonus + (weighted_avg * 0.1)) * 0.6

                        # Apply the boost with a cap at 1.0
                        result_dict[item_id]["similarity"] = min(1.0, current_sim * boost_factor)
                        result_dict[item_id]["cross_level_bonus"] = cross_level_bonus
                        result_dict[item_id]["found_in_levels"] = [lvl for lvl, _ in level_info]
        except Exception as e:
            print(f"{self.get_time()} Error in cross-level verification: {e}")
            # Continue without applying verification rather than failing

        # Return the modified dictionary
        return result_dict

    def remove(self, item_id: str) -> bool:
        """
        Remove an item by ID.

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
        Get an item by ID.

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
        Update an item.

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

            # Track if we need to rebuild
            need_rebuild = False

            # Apply content update
            if "content" in updates and updates["content"] != item.content:
                # Content has changed, regenerate embedding
                if self.embedding_function:
                    try:
                        new_embedding = self.embedding_function(updates["content"])
                        item.embedding = new_embedding

                        # Regenerate enhanced embeddings
                        if self.enable_enhanced_embeddings:
                            # Clear existing enhanced embeddings
                            item.additional_embeddings = {}
                            self._add_enhanced_embeddings(item)

                        # Need to rebuild index
                        need_rebuild = True
                    except Exception as e:
                        print(f"{self.get_time()} Error updating embedding: {e}")
                        return False

                item.content = updates["content"]

            # Update metadata
            if "metadata" in updates:
                item.metadata.update(updates["metadata"])

            # Update modification timestamp
            item.metadata["modified_at"] = datetime.now().timestamp()

            # Rebuild index if needed
            if need_rebuild:
                self._rebuild_index()

            # Save if auto-save is enabled
            if self.auto_save:
                self.save()

            return True

    def _rebuild_index(self):
        """Rebuild the FAISS index with current embeddings."""
        if not self.embeddings:
            print(f"{self.get_time()} No embeddings to index")
            self._create_index()
            return

        print(f"{self.get_time()} Rebuilding index for {len(self.embeddings)} items...")

        # Create new index
        self._create_index()

        # Add embeddings in batches to avoid memory issues
        batch_size = 1000
        total_batches = (len(self.embeddings) + batch_size - 1) // batch_size

        for batch_idx, i in enumerate(range(0, len(self.embeddings), batch_size)):
            # Get batch
            batch = self.embeddings[i:i+batch_size]

            # Skip empty batches
            if not batch:
                continue

            # Display progress
            self.display_progress_bar(batch_idx+1, total_batches,
                               prefix=f'{self.get_time()} Indexing:',
                               suffix=f'Batch {batch_idx+1}/{total_batches}',
                               length=40)

            try:
                self.index.add(np.array(batch, dtype=np.float32))
            except Exception as e:
                print(f"{self.get_time()} Error adding batch to index: {e}")

        print(f"{self.get_time()} Index rebuilt with {self.index.ntotal} vectors")

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

                # Save enhanced embeddings
                if self.enable_enhanced_embeddings:
                    enhanced_path = os.path.join(self.storage_path, "enhanced_embeddings.pkl")
                    enhanced_data = {}
                    for i, item in enumerate(self.items):
                        if item.additional_embeddings:
                            enhanced_data[item.id] = item.additional_embeddings

                    with open(enhanced_path, 'wb') as f:
                        pickle.dump(enhanced_data, f)

                # Save indices
                index_path = os.path.join(self.storage_path, "index.faiss")
                faiss.write_index(self.index, index_path)

                # Save enhanced indices
                if self.enhanced_indices:
                    for level, index in self.enhanced_indices.items():
                        level_path = os.path.join(self.storage_path, f"index_level_{level}.faiss")
                        faiss.write_index(index, level_path)

                # Save metadata
                metadata_path = os.path.join(self.storage_path, "metadata.json")
                metadata = {
                    "count": len(self.items),
                    "active_count": len(self.items) - len(self.deleted_ids),
                    "embedding_dim": self.embedding_dim,
                    "enable_enhanced_embeddings": self.enable_enhanced_embeddings,
                    "max_enhancement_levels": self.max_enhancement_levels,
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
        """
        Load memory data from disk using a cache-first approach with background loading.

        Returns:
            Success status (True if initial cache loading succeeds)
        """
        with self._lock:
            start_time = time.time()
            print(f"{self.get_time()} Starting full memory load...")

            items_path = os.path.join(self.storage_path, "items.json")
            embeddings_path = os.path.join(self.storage_path, "embeddings.npy")

            if not (os.path.exists(items_path) and os.path.exists(embeddings_path)):
                print(f"{self.get_time()} No memory data found.")
                return False

            try:
                # Load items data
                with open(items_path, 'r', encoding='utf-8') as f:
                    items_data = json.load(f)

                print(f"{self.get_time()} Found {len(items_data)} items in storage")

                # Load embeddings
                try:
                    embeddings = np.load(embeddings_path)
                    print(f"{self.get_time()} Loaded {len(embeddings)} embeddings")

                except Exception as e:
                    print(f"{self.get_time()} Error loading embeddings: {e}")
                    return False

                # Initialize storage
                self.items = []
                self.embeddings = []
                self.id_to_index = {}

                # Load metadata to get embedding dimension and deleted IDs
                metadata_path = os.path.join(self.storage_path, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)

                        # Set embedding dimension from metadata if available
                        if 'embedding_dim' in metadata and metadata['embedding_dim'] is not None:
                            self.embedding_dim = metadata['embedding_dim']

                        # Load deleted IDs
                        if 'deleted_ids' in metadata:
                            self.deleted_ids = set(metadata['deleted_ids'])
                    except Exception as e:
                        print(f"{self.get_time()} Error loading metadata: {e}")

                # Process items with dimension detection and handling
                for i, item_data in enumerate(items_data):
                    try:
                        # Detect embedding dimension from first item if not set
                        if self.embedding_dim is None and i == 0 and i < len(embeddings):
                            self.embedding_dim = len(embeddings[i])
                            print(f"{self.get_time()} Detected embedding dimension: {self.embedding_dim}")

                        # Get embedding with dimension handling
                        if i < len(embeddings):
                            embedding = embeddings[i]

                            # Check dimension and resize if needed
                            if self.embedding_dim is not None and len(embedding) != self.embedding_dim:
                                if len(embedding) > self.embedding_dim:
                                    # Truncate
                                    embedding = embedding[:self.embedding_dim]
                                else:
                                    # Pad with zeros
                                    embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))

                                # Normalize after resizing
                                norm = np.linalg.norm(embedding)
                                if norm > 1e-10:
                                    embedding = embedding / norm
                        else:
                            # Create random embedding if no embedding available
                            print(f"{self.get_time()} No embedding for item {i}, creating random one")
                            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
                            embedding = embedding / np.linalg.norm(embedding)

                        # Create memory item
                        item = MemoryItem.from_dict(item_data, embedding)

                        # Add to storage
                        self.items.append(item)
                        self.embeddings.append(embedding)
                        self.id_to_index[item.id] = len(self.items) - 1

                    except Exception as e:
                        print(f"{self.get_time()} Error processing item {i}: {e}")

                # Create or load FAISS index
                self._rebuild_index()

                # Load enhanced embeddings if available
                self._load_enhanced_embeddings()

                print(f"{self.get_time()} Memory load complete: {len(self.items)} items loaded in {time.time() - start_time:.2f}s")
                return True

            except Exception as e:
                print(f"{self.get_time()} Error in full memory load: {e}")
                traceback.print_exc()
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

            # Enhanced index stats
            enhanced_stats = {}
            if self.enable_enhanced_embeddings and self.enhanced_indices:
                for level, index in self.enhanced_indices.items():
                    enhanced_stats[f"level_{level}_size"] = index.ntotal

            return {
                "total_items": len(self.items),
                "active_items": len(self.items) - len(self.deleted_ids),
                "deleted_items": len(self.deleted_ids),
                "index_size": index_size,
                "index_dimension": self.embedding_dim,
                "enhanced_enabled": self.enable_enhanced_embeddings,
                "enhancement_levels": self.max_enhancement_levels,
                "enhanced_stats": enhanced_stats
            }