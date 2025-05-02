"""
Unified memory system for TinyLlama Chat.
Provides a cohesive approach to storing and retrieving different types of information
with improved integration and simplified architecture.
"""

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

        # Cache for embeddings
        self._embedding_cache = {}
        self._embedding_cache_capacity = 1000

        # Cache for similarity search results
        self._similarity_cache = {}
        self._similarity_cache_capacity = 100

        # Load existing memory if available
        self.load()

        # Initialize rotational matrices for enhanced embeddings
        # if enable_enhanced_embeddings:
        #    self._initialize_enhancement_matrices()

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

    def get_time(self) -> str:
        """Get formatted timestamp for logging."""
        return datetime.now().strftime("[%d/%m/%y %H:%M:%S]") + ' [Memory]'

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

    def add_bulk(self, items: List[Dict[str, Any]], use_enhanced_embeddings: Optional[bool] = None) -> List[Optional[str]]:
        """
        Efficiently add multiple items in a single operation using batch processing.

        Args:
            items: List of dictionaries with content and metadata
            use_enhanced_embeddings: Override default enhanced embedding setting

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

            # Generate embeddings
            if not self.batch_embedding_function:
                # Fall back to individual embedding
                result_ids = []
                for item in items:
                    item_id = self.add(
                        content=item.get("content", ""),
                        metadata=item.get("metadata", {}),
                        use_enhanced_embeddings=use_enhanced_embeddings
                    )
                    result_ids.append(item_id)

                # Restore original auto_save setting
                self.auto_save = original_auto_save

                # Save if needed
                if original_auto_save:
                    self.save()

                return result_ids

            try:
                # Check if we can use tensor batch processing
                try:
                    from batch_utils import tensor_batch_processing
                    batch_utils_available = True
                except ImportError:
                    batch_utils_available = False

                # Generate embeddings in batch
                if batch_utils_available and len(contents) > 50:
                    print(f"{self.get_time()} Using tensor_batch_processing for {len(contents)} embeddings")

                    # Define a function that processes a batch of texts to embeddings
                    # This works with a tensor of already-computed embeddings
                    def process_embedding_batch(emb_batch):
                        return emb_batch

                    # First, get all embeddings normally
                    all_embeddings = self.batch_embedding_function(contents)

                    # Convert to tensor
                    embeddings_tensor = torch.tensor(np.array(all_embeddings), dtype=torch.float32)

                    # Process in optimized batches
                    processed_embeddings = tensor_batch_processing(
                        tensor_op=process_embedding_batch,
                        input_tensor=embeddings_tensor,
                        batch_dim=0,
                        batch_size=64,
                        cleanup=True,
                        adaptive=True
                    )

                    # Convert back to list of numpy arrays
                    if isinstance(processed_embeddings, torch.Tensor):
                        all_embeddings = list(processed_embeddings.numpy())
                    else:
                        # Handle case where it returned a list of batches
                        merged_embeddings = []
                        for batch in processed_embeddings:
                            merged_embeddings.extend(batch.numpy())
                        all_embeddings = merged_embeddings
                else:
                    # Standard approach
                    all_embeddings = self.batch_embedding_function(contents)

                # Create memory items
                all_memory_items = []
                for i, (item_dict, embedding) in enumerate(zip(valid_items, all_embeddings)):
                    metadata = item_dict.get('metadata', {})

                    # Create memory item
                    item = MemoryItem(
                        content=item_dict['content'],
                        embedding=embedding,
                        metadata=metadata
                    )
                    all_memory_items.append(item)

                # Determine whether to use enhanced embeddings
                use_enhanced = self.enable_enhanced_embeddings if use_enhanced_embeddings is None else use_enhanced_embeddings

                # Generate enhanced embeddings
                if use_enhanced:
                    for item in all_memory_items:
                        self._add_enhanced_embeddings(item)

                # Add all items to storage
                item_ids = []
                for item in all_memory_items:
                    item_id = self._add_item_to_store(item)
                    item_ids.append(item_id)

                # Save the updated data
                if original_auto_save:
                    self.save()
                    self.auto_save = original_auto_save

                # Map results back to input
                result_ids = [None] * len(items)
                valid_indices = [i for i, item in enumerate(items) if item.get('content', '').strip()]
                for i, valid_idx in enumerate(valid_indices):
                    if i < len(item_ids):
                        result_ids[valid_idx] = item_ids[i]

                return result_ids

            except Exception as e:
                print(f"{self.get_time()} Error in batch processing: {e}")

                # Fall back to individual processing
                result_ids = []
                for item in items:
                    item_id = self.add(
                        content=item.get("content", ""),
                        metadata=item.get("metadata", {}),
                        use_enhanced_embeddings=use_enhanced_embeddings
                    )
                    result_ids.append(item_id)

                # Restore original auto_save setting
                self.auto_save = original_auto_save

                # Save if needed
                if original_auto_save:
                    self.save()

                return result_ids

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
        Initialize matrices for enhanced embeddings with proper dimension detection.
        """
        # Use the actual embedding dimension from the first item or from config
        if hasattr(self, 'embedding_dim'):
            embedding_dim = self.embedding_dim
        elif self.items and hasattr(self.items[0], 'embedding'):
            embedding_dim = self.items[0].embedding.shape[0]
        else:
            # Default dimension if we can't determine it yet
            embedding_dim = 2048  # TinyLlama's embedding dimension

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
        max_potential_levels = 20

        # Create matrices with fixed random seed for determinism
        for i in range(1, max_potential_levels + 1):
            # Set fixed seed per level
            np.random.seed(42 + i)

            # Create rotation matrix with increasing randomness by level
            rotation_factor = 0.1 + (i * 0.03)
            rotation = np.random.normal(0, rotation_factor, (embedding_dim, embedding_dim))

            # For earlier levels, preserve more of the original structure
            if i <= 3:
                # Add identity matrix component to preserve original information
                preservation_factor = 0.9 - (i * 0.15)
                rotation = rotation + np.eye(embedding_dim) * preservation_factor

            # Ensure the matrix is orthogonal
            u, _, vh = np.linalg.svd(rotation, full_matrices=False)
            self._rotation_matrices[i] = u @ vh

            # Create level-specific bias vector
            np.random.seed(137 + i)
            bias_factor = 0.01 + (i * 0.002)
            self._level_biases[i] = np.random.normal(0, bias_factor, (embedding_dim,))

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

            # Check for dimension mismatch
            need_migration = False
            old_dim = None

            # Check the first item to determine dimension
            if enhanced_data:
                first_id = next(iter(enhanced_data))
                first_item = enhanced_data[first_id]
                if first_item:
                    first_level = next(iter(first_item))
                    if first_level in first_item:
                        level_embedding = first_item[first_level]
                        old_dim = level_embedding.shape[0]
                        if old_dim != self.embedding_dim:
                            print(f"{self.get_time()} Enhanced embedding dimension mismatch: {old_dim} vs {self.embedding_dim}")
                            need_migration = True

            # Process enhanced embeddings
            loaded_count = 0
            migrated_count = 0
            for item_id, item_enhanced in enhanced_data.items():
                if item_id in self.id_to_index:
                    idx = self.id_to_index[item_id]

                    # Check if item needs migration
                    if need_migration:
                        migrated_embeddings = {}
                        for level, embedding in item_enhanced.items():
                            # Migrate embedding dimension
                            if len(embedding) != self.embedding_dim:
                                if len(embedding) > self.embedding_dim:
                                    # Truncate
                                    new_embedding = embedding[:self.embedding_dim]
                                else:
                                    # Pad
                                    new_embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))

                                # Normalize
                                norm = np.linalg.norm(new_embedding)
                                if norm > 1e-10:
                                    new_embedding = new_embedding / norm

                                migrated_embeddings[level] = new_embedding
                                migrated_count += 1
                            else:
                                migrated_embeddings[level] = embedding

                        # Save migrated embeddings
                        self.items[idx].additional_embeddings = migrated_embeddings
                    else:
                        # No migration needed
                        self.items[idx].additional_embeddings = item_enhanced

                    loaded_count += 1

            print(f"{self.get_time()} Loaded {loaded_count} enhanced embeddings, migrated {migrated_count}")

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

        # Apply the transformation using cached matrices
        if level in self._rotation_matrices:
            # Apply the cached rotation
            rotated = np.dot(base_embedding, self._rotation_matrices[level])

            # Apply level-specific bias
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
        else:
            # Fallback if matrix isn't available
            perturbed = base_embedding + (np.random.normal(0, 0.01 * level, base_embedding.shape) * base_embedding)
            return perturbed / np.linalg.norm(perturbed)

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
        if similarity > 0.6:
            # Boost high similarities (more confident matches)
            boost = (similarity - 0.6) * self.similarity_enhancement_factor * 2.0
            enhanced = min(1.0, similarity + boost)
        elif similarity < 0.4:
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

            # Check cache first for exactly matching query
            cache_key = f"query_{hashlib.md5(query.encode()).hexdigest()}"
            if hasattr(self, '_retrieval_cache') and cache_key in self._retrieval_cache:
                # Apply metadata filter to cached results
                if metadata_filter:
                    filtered_results = []
                    for result in self._retrieval_cache[cache_key]:
                        if all(result.get('metadata', {}).get(k) == v for k, v in metadata_filter.items()):
                            filtered_results.append(result)
                    return filtered_results[:top_k]
                return self._retrieval_cache[cache_key][:top_k]

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
            if can_use_enhanced:
                results = self._enhanced_search(
                    normalized_query,
                    top_k=top_k,
                    min_similarity=min_similarity
                )
            else:
                results = self._standard_search(
                    normalized_query,
                    top_k=top_k,
                    min_similarity=min_similarity
                )

            # Apply metadata filtering if provided
            if metadata_filter:
                filtered_results = []
                for result in results:
                    if all(result.get('metadata', {}).get(k) == v for k, v in metadata_filter.items()):
                        filtered_results.append(result)
                results = filtered_results

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

            # Cache the results
            if not hasattr(self, '_retrieval_cache'):
                self._retrieval_cache = {}
                self._retrieval_cache_capacity = 100

            self._retrieval_cache[cache_key] = results

            # Manage cache size
            if len(self._retrieval_cache) > self._retrieval_cache_capacity:
                # Remove oldest 20%
                remove_count = int(self._retrieval_cache_capacity * 0.2)
                oldest_keys = list(self._retrieval_cache.keys())[:remove_count]
                for k in oldest_keys:
                    del self._retrieval_cache[k]

            return results

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
        # Get more results than needed to account for filtering
        search_k = min(top_k * 2, self.index.ntotal)

        # Prepare query for FAISS
        query_array = np.array([query_embedding], dtype=np.float32)

        # Perform search
        similarities, indices = self.index.search(query_array, search_k)

        # Process results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and similarities[0][i] > min_similarity:
                # Skip deleted items
                if idx in self.deleted_ids:
                    continue

                # Get the memory item
                item = self.items[idx]
                similarity = float(similarities[0][i])

                # Apply similarity enhancement
                enhanced_similarity = self._enhance_similarity(similarity)

                # Add to results
                results.append({
                    "id": item.id,
                    "content": item.content,
                    "similarity": enhanced_similarity,
                    "raw_similarity": similarity,
                    "metadata": item.metadata
                })

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # Limit to top-k
        return results[:top_k]

    def _rebuild_enhanced_indices(self):
        """Rebuild FAISS indices for enhanced embeddings."""
        if not self.enable_enhanced_embeddings:
            return

        # Initialize new indices
        self.enhanced_indices = {}

        # Collect levels from all items
        levels = set()
        for item in self.items:
            levels.update(item.additional_embeddings.keys())

        # Create index for each level
        for level in levels:
            print(f"{self.get_time()} Building index for enhancement level {level}")

            # Create new index
            self.enhanced_indices[level] = faiss.IndexFlatIP(self.embedding_dim)

            # Collect embeddings for this level
            level_embeddings = []
            for item in self.items:
                if level in item.additional_embeddings:
                    embedding = item.additional_embeddings[level]
                    # Normalize
                    norm = np.linalg.norm(embedding)
                    if norm > 1e-10:
                        embedding = embedding / norm
                    level_embeddings.append(embedding)

            # Add to index if we have any
            if level_embeddings:
                self.enhanced_indices[level].add(np.array(level_embeddings, dtype=np.float32))

            print(f"{self.get_time()} Level {level} index created with {self.enhanced_indices[level].ntotal} vectors")

    def _enhanced_search(self, query_embedding: np.ndarray, top_k: int, min_similarity: float) -> List[Dict[str, Any]]:
        """
        Perform enhanced search across multiple embedding levels.

        Args:
            query_embedding: Normalized query embedding
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of matching items with similarity scores
        """
        # Search base level
        base_results = self._standard_search(query_embedding, top_k, min_similarity)

        # Track all results by item ID
        result_dict = {item["id"]: item for item in base_results}

        # Search all available enhanced levels
        for level in sorted(self.enhanced_indices.keys()):
            if self.enhanced_indices[level].ntotal == 0:
                continue

            # Create level-specific query
            level_query = self._generate_level_embedding(query_embedding, level)
            if level_query is None:
                continue

            # Normalize query
            query_norm = np.linalg.norm(level_query)
            if query_norm < 1e-10:
                query_norm = 1e-10
            normalized_query = level_query / query_norm

            # Search with level-specific query
            query_array = np.array([normalized_query], dtype=np.float32)
            search_k = min(top_k * 2, self.enhanced_indices[level].ntotal)
            similarities, indices = self.enhanced_indices[level].search(query_array, search_k)

            # Process level results
            for i, idx in enumerate(indices[0]):
                if idx != -1 and similarities[0][i] > min_similarity:
                    if idx in self.deleted_ids:
                        continue

                    item = self.items[idx]
                    similarity = float(similarities[0][i])
                    enhanced_similarity = self._enhance_similarity(similarity)

                    # Only update if better than existing or not found yet
                    if item.id not in result_dict or enhanced_similarity > result_dict[item.id]["similarity"]:
                        result_dict[item.id] = {
                            "id": item.id,
                            "content": item.content,
                            "similarity": enhanced_similarity,
                            "raw_similarity": similarity,
                            "metadata": item.metadata,
                            "level": level
                        }

        # Apply cross-level verification
        self._apply_cross_level_verification(result_dict)

        # Extract final results
        results = list(result_dict.values())
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # Limit to top-k
        return results[:top_k]

    def _apply_cross_level_verification(self, result_dict: Dict[str, Dict[str, Any]]):
        """
        Boost confidence for items found across multiple levels.

        Args:
            result_dict: Dictionary of search results
        """
        # Group results by item
        item_levels = {}
        for item_id, result in result_dict.items():
            if result.get("level") is not None:  # Check if level is specified
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

    def format_for_context(self, results: List[Dict[str, Any]], query: str = None) -> str:
        """
        Format retrieved items for inclusion in the context.

        Args:
            results: List of retrieval results
            query: Optional query that triggered the retrieval

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        # Sort by similarity (should already be sorted)
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        # Calculate adaptive thresholds
        similarities = [r.get("similarity", 0) for r in results]
        max_sim = max(similarities) if similarities else 0

        high_threshold = max(0.7, max_sim * 0.9)  # At least 90% of max
        medium_threshold = max(0.5, max_sim * 0.7)  # At least 70% of max

        # Group by confidence levels
        high_confidence = []
        medium_confidence = []
        low_confidence = []

        for result in results:
            similarity = result.get("similarity", 0)
            if similarity >= high_threshold:
                high_confidence.append(result)
            elif similarity >= medium_threshold:
                medium_confidence.append(result)
            else:
                low_confidence.append(result)

        # Build formatted output
        output = "MEMORY CONTEXT:\n"

        # Add high confidence items
        if high_confidence:
            output += "\nHIGH RELEVANCE INFORMATION:\n"
            for i, item in enumerate(high_confidence[:3]):  # Top 3
                content = item['content'].strip()
                item_id = item.get('id', '')[-6:]  # Short ID
                output += f"- [{item_id}] {content}\n"

        # Add medium confidence items
        if medium_confidence:
            output += "\nRELEVANT INFORMATION:\n"
            for i, item in enumerate(medium_confidence[:4]):  # Top 4
                content = item['content'].strip()
                item_id = item.get('id', '')[-6:]
                output += f"- [{item_id}] {content}\n"

        # Add low confidence items if needed
        if low_confidence and len(high_confidence) + len(medium_confidence) < 3:
            output += "\nPOTENTIALLY RELEVANT:\n"
            for i, item in enumerate(low_confidence[:2]):  # Top 2
                content = item['content'].strip()
                item_id = item.get('id', '')[-6:]
                output += f"- [{item_id}] {content}\n"

        # Add a reminder about using the memory
        output += "\nUse the information above to help answer the query if relevant.\n"

        return output

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
        for i in range(0, len(self.embeddings), batch_size):
            batch = self.embeddings[i:i+batch_size]
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
            print(f"{self.get_time()} Starting memory load...")

            # First load metadata to determine configuration
            metadata_path = os.path.join(self.storage_path, "metadata.json")
            old_embedding_dim = None
            need_migration = False

            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    old_embedding_dim = metadata.get('embedding_dim')

                    # Check if dimension migration is needed
                    if old_embedding_dim and old_embedding_dim != self.embedding_dim:
                        print(f"{self.get_time()} Embedding dimension changed: {old_embedding_dim} → {self.embedding_dim}")
                        print(f"{self.get_time()} Will perform dimension migration during load")
                        need_migration = True
                except Exception as e:
                    print(f"{self.get_time()} Error reading metadata: {e}")

            # Check for cache with migration support
            cache_path = os.path.join(self.storage_path, "quickstart_cache.pkl")

            # Try loading from cache
            if os.path.exists(cache_path) and not need_migration:  # Skip cache if migration needed
                try:
                    print(f"{self.get_time()} Found quickstart cache, loading...")
                    with open(cache_path, 'rb') as f:
                        cache_data = pickle.load(f)

                    # Check if cache is using current embedding dimension
                    cached_dim = cache_data.get('embedding_dim')
                    if cached_dim != self.embedding_dim:
                        print(f"{self.get_time()} Cache embedding dimension mismatch: {cached_dim} vs {self.embedding_dim}")
                        print(f"{self.get_time()} Falling back to full load with migration")
                        return self._full_load(need_migration=True, old_embedding_dim=cached_dim)

                    # Load items from cache if available
                    if 'lite_items' in cache_data:
                        print(f"{self.get_time()} Loading {len(cache_data['lite_items'])} items from cache")

                        # Reset storage
                        self.items = []
                        self.embeddings = []

                        # Load ID mapping
                        self.id_to_index = cache_data.get('id_to_index', {})
                        self.deleted_ids = set(cache_data.get('deleted_ids', []))

                        # We'll load full embeddings in the background
                        # For now, create placeholder items with random temp embeddings
                        for lite_item in cache_data['lite_items']:
                            # Create temporary embedding
                            temp_embedding = np.random.randn(self.embedding_dim).astype(np.float32)
                            temp_embedding = temp_embedding / np.linalg.norm(temp_embedding)

                            # Create memory item
                            item = MemoryItem(
                                content=lite_item['content'],
                                embedding=temp_embedding,
                                metadata=lite_item['metadata']
                            )
                            # Ensure ID matches original
                            item.id = lite_item['id']

                            # Add to storage
                            self.items.append(item)
                            self.embeddings.append(temp_embedding)

                        # Try to load the separate index file
                        if cache_data.get('has_index', False):
                            index_path = os.path.join(self.storage_path, "cache_index.faiss")
                            if os.path.exists(index_path):
                                try:
                                    self.index = faiss.read_index(index_path)
                                    print(f"{self.get_time()} Loaded FAISS index with {self.index.ntotal} vectors")
                                except Exception as e:
                                    print(f"{self.get_time()} Error loading cached index: {e}")
                                    # Create empty index
                                    self._create_index()
                            else:
                                # Create empty index
                                self._create_index()
                        else:
                            # Create empty index
                            self._create_index()

                        # Cache load complete
                        cache_load_time = time.time() - start_time
                        item_count = len(self.items)
                        print(f"{self.get_time()} Quick cache loaded {item_count} item metadata in {cache_load_time:.2f}s")

                        # Start background loading for complete data
                        thread = threading.Thread(
                            target=self._background_full_load,
                            args=(cache_data.get('timestamp', 0),),
                            daemon=True
                        )
                        thread.start()

                        return True

                except Exception as e:
                    print(f"{self.get_time()} Error loading from cache: {e}, falling back to full load")
                    # Fall back to full load

        # Proceed with full load with migration support
        return self._full_load(need_migration=need_migration, old_embedding_dim=old_embedding_dim)

    def _background_full_load(self, cache_timestamp: float):
        """
        Perform a full load in the background to update cache.

        Args:
            cache_timestamp: Timestamp when cache was created
        """
        try:
            print(f"{self.get_time()} Starting background full load...")
            start_time = time.time()

            # Check if data files are newer than cache
            items_path = os.path.join(self.storage_path, "items.json")
            embeddings_path = os.path.join(self.storage_path, "embeddings.npy")

            items_mtime = os.path.getmtime(items_path) if os.path.exists(items_path) else 0
            embeddings_mtime = os.path.getmtime(embeddings_path) if os.path.exists(embeddings_path) else 0

            if max(items_mtime, embeddings_mtime) <= cache_timestamp:
                print(f"{self.get_time()} Cache is up to date, skipping background load")
                return

            # Perform full load in background
            print(f"{self.get_time()} Cache is outdated, performing full background load")
            with self._lock:
                # Do the full load but keep track of current items for safety
                prev_item_count = len(self.items)
                success = self._full_load()

                if success:
                    new_item_count = len(self.items)
                    diff = new_item_count - prev_item_count

                    # Update the cache for next time
                    self._create_quickstart_cache()

                    load_time = time.time() - start_time
                    print(f"{self.get_time()} Background full load complete in {load_time:.2f}s, "
                          f"added {diff} new items")
                else:
                    print(f"{self.get_time()} Background full load failed")

        except Exception as e:
            print(f"{self.get_time()} Error in background load: {e}")

    def _create_quickstart_cache(self):
        """Create a quickstart cache for faster loading next time."""
        try:
            print(f"{self.get_time()} Trying to create quickstart cache")
            cache_path = os.path.join(self.storage_path, "quickstart_cache.pkl")

            # Create basic data dict
            cache_data = {
                'timestamp': time.time(),
                'embedding_dim': self.embedding_dim,
                'item_count': len(self.items),
                'deleted_count': len(self.deleted_ids)
            }

            # Don't try to pickle the FAISS index - it's not reliably serializable
            # Instead, save a separate FAISS index file
            if self.index and self.index.ntotal > 0:
                index_path = os.path.join(self.storage_path, "cache_index.faiss")
                try:
                    faiss.write_index(self.index, index_path)
                    cache_data['has_index'] = True
                    print(f"{self.get_time()} Saved FAISS index to {index_path}")
                except Exception as e:
                    print(f"{self.get_time()} Warning: Could not save FAISS index: {e}")
                    cache_data['has_index'] = False
            else:
                print(f"{self.get_time()} WARNING: No index to cache")
                cache_data['has_index'] = False

            # Add items (but without embeddings to save space)
            try:
                lite_items = []
                for item in self.items:
                    # Create a lightweight version without embeddings
                    lite_item = {
                        'id': item.id,
                        'content': item.content,
                        'metadata': item.metadata
                    }
                    lite_items.append(lite_item)

                cache_data['lite_items'] = lite_items
                cache_data['id_to_index'] = self.id_to_index
                cache_data['deleted_ids'] = list(self.deleted_ids)
            except Exception as e:
                print(f"{self.get_time()} Error creating lightweight items: {e}")
                # Continue anyway - we'll still save basic metadata

            # Save cache
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            cache_size = os.path.getsize(cache_path) / (1024 * 1024)  # Size in MB
            print(f"{self.get_time()} Quickstart cache created ({cache_size:.1f} MB)")
            return True

        except Exception as e:
            print(f"{self.get_time()} Error creating quickstart cache: {e}")
            traceback.print_exc()
            return False

    def _migrate_embeddings(self, embeddings, old_dim, new_dim):
        """
        Migrate embeddings from old dimension to new dimension.

        Args:
            embeddings: Array of embeddings
            old_dim: Original dimension
            new_dim: Target dimension

        Returns:
            Migrated embeddings
        """
        print(f"{self.get_time()} Migrating {len(embeddings)} embeddings from {old_dim}D to {new_dim}D")

        migrated_embeddings = []

        for i, emb in enumerate(embeddings):
            # Create new embedding with adjusted dimension
            if new_dim > old_dim:
                # Padding approach
                new_emb = np.pad(emb, (0, new_dim - old_dim))
            else:
                # Truncation approach
                new_emb = emb[:new_dim]

            # Normalize after resizing
            norm = np.linalg.norm(new_emb)
            if norm > 1e-10:
                new_emb = new_emb / norm

            migrated_embeddings.append(new_emb)

            # Show progress for large migrations
            if (i+1) % 1000 == 0 or i+1 == len(embeddings):
                print(f"{self.get_time()} Migrated {i+1}/{len(embeddings)} embeddings")

        return np.array(migrated_embeddings)

    def _full_load(self, need_migration=False, old_embedding_dim=None) -> bool:
        """
        Perform a full load of all memory data with migration support.

        Args:
            need_migration: Whether dimension migration is needed
            old_embedding_dim: Old embedding dimension if migration needed

        Returns:
            Success status
        """
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

            # Load embeddings with migration support
            try:
                embeddings = np.load(embeddings_path)
                print(f"{self.get_time()} Loaded {len(embeddings)} embeddings")

                # Check if dimensions match
                if len(embeddings) > 0:
                    actual_dim = embeddings[0].shape[0]
                    if actual_dim != self.embedding_dim:
                        print(f"{self.get_time()} Embedding dimension mismatch: {actual_dim} vs {self.embedding_dim}")
                        if need_migration:
                            print(f"{self.get_time()} Performing dimension migration...")
                            embeddings = self._migrate_embeddings(embeddings, actual_dim, self.embedding_dim)
                        else:
                            print(f"{self.get_time()} WARNING: Dimension mismatch but migration not requested")
                            # Continue anyway - we'll resize embeddings as we load them
            except Exception as e:
                print(f"{self.get_time()} Error loading embeddings: {e}")
                return False

            # Initialize storage
            self.items = []
            self.embeddings = []
            self.id_to_index = {}

            # Process items with dimension handling
            for i, item_data in enumerate(items_data):
                try:
                    # Get embedding with dimension handling
                    if i < len(embeddings):
                        embedding = embeddings[i]

                        # Check dimension and resize if needed
                        if len(embedding) != self.embedding_dim:
                            if len(embedding) > self.embedding_dim:
                                # Truncate
                                embedding = embedding[:self.embedding_dim]
                            else:
                                # Pad with zeros
                                embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
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

    def _load_indices_optimized(self, batch_utils_available: bool = False):
        """Load FAISS indices with optimization for large collections."""
        # Load FAISS index
        index_path = os.path.join(self.storage_path, "index.faiss")

        if os.path.exists(index_path):
            try:
                # Try to use memory-mapped IO for larger indices
                if os.path.getsize(index_path) > 100 * 1024 * 1024:  # > 100MB
                    self.index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
                else:
                    self.index = faiss.read_index(index_path)
            except Exception as e:
                print(f"{self.get_time()} Error loading main index: {e}")
                # Recreate index with batch processing
                self._create_index()

                if self.embeddings and batch_utils_available and len(self.embeddings) > 1000:
                    # Use batch processing for adding to index
                    self._add_to_index_with_batching()
                elif self.embeddings:
                    # Standard approach for smaller collections
                    self.index.add(np.array(self.embeddings, dtype=np.float32))
        else:
            # Create new index
            self._create_index()
            if self.embeddings:
                if batch_utils_available and len(self.embeddings) > 1000:
                    self._add_to_index_with_batching()
                else:
                    self.index.add(np.array(self.embeddings, dtype=np.float32))

        # Load enhanced indices with optimization
        if self.enable_enhanced_embeddings:
            self.enhanced_indices = {}
            for level in range(1, self.max_enhancement_levels + 1):
                level_path = os.path.join(self.storage_path, f"index_level_{level}.faiss")
                if os.path.exists(level_path):
                    try:
                        # Use memory mapping for large indices
                        if os.path.getsize(level_path) > 100 * 1024 * 1024:  # > 100MB
                            self.enhanced_indices[level] = faiss.read_index(level_path, faiss.IO_FLAG_MMAP)
                        else:
                            self.enhanced_indices[level] = faiss.read_index(level_path)
                    except Exception as e:
                        print(f"{self.get_time()} Error loading level {level} index: {e}")

    def _add_to_index_with_batching(self):
        """Add embeddings to FAISS index with batch processing."""
        try:
            from batch_utils import tensor_batch_processing

            # Convert embeddings to tensor
            embeddings_tensor = torch.tensor(np.array(self.embeddings), dtype=torch.float32)

            # Define batch operation
            def add_batch_to_index(batch):
                # Normalize batch for cosine similarity
                norms = torch.norm(batch, dim=1, keepdim=True)
                normalized_batch = batch / torch.clamp(norms, min=1e-10)

                # Add to index
                self.index.add(normalized_batch.numpy())

                # Return batch for consistency with tensor_batch_processing
                return batch

            # Process in batches
            tensor_batch_processing(
                tensor_op=add_batch_to_index,
                input_tensor=embeddings_tensor,
                batch_dim=0,
                batch_size=256,  # Process 256 embeddings at a time
                cleanup=True,
                adaptive=True
            )

        except ImportError:
            # Fall back to standard approach
            self.index.add(np.array(self.embeddings, dtype=np.float32))
        except Exception as e:
            print(f"{self.get_time()} Error in batch adding to index: {e}")
            # Fall back to standard approach
            try:
                self.index.add(np.array(self.embeddings, dtype=np.float32))
            except Exception as err:
                print(f"{self.get_time()} Fatal error adding to index: {err}")

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

            # Create quickstart cache for next boot
            self._create_quickstart_cache()

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