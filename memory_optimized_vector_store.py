"""
Memory-optimized vector store implementation that addresses potential memory leaks.
Extends the IndexedVectorStore with proper resource management.
"""

import os
import numpy as np
import gc
import faiss
import pickle
import weakref
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from datetime import datetime
from enhanced_memory_store import IndexedVectorStore
from batch_utils import process_in_batches, batch_embedding_generation


class MemoryOptimizedVectorStore(IndexedVectorStore):
    """
    Memory-optimized version of IndexedVectorStore that prevents memory leaks.
    """
    
    def __init__(self, storage_path: str = "./memory/vector_store", embedding_function: Optional[Callable] = None):
        """Initialize with proper resource tracking."""
        # Initialize parent class
        super().__init__(storage_path, embedding_function)
        
        # Track large objects for potential cleanup
        self._embeddings_cache = {}  # document_hash -> embedding
        self._max_cache_size = 100  # Maximum embeddings to keep in cache
        self._deleted_ids = set()  # Keep track of deleted indices
        
        # Register finalizer for cleanup on garbage collection
        self._finalizer = weakref.finalize(self, self._cleanup)
    
    def add(self, text: str, embedding: np.ndarray, metadata: Dict[str, Any] = None):
        """Add a document with memory-optimized tracking."""
        # Create document hash
        doc_hash = self._compute_hash(text)
        
        # Skip if duplicate
        if doc_hash in self.doc_hashes:
            return False
        
        # Verify embedding dimension
        if self.index is None:
            self._create_index(embedding.shape[0])
        elif embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[0]}")
        
        # Add to cache with weak reference to avoid memory leak
        self._embeddings_cache[doc_hash] = embedding.copy()
        
        # Trim cache if needed
        self._trim_cache()
        
        # Continue with normal addition
        result = super().add(text, embedding, metadata)
        
        # Periodically run garbage collection after additions
        if len(self.documents) % 50 == 0:
            gc.collect()
            
        return result
    
    def _trim_cache(self):
        """Trim the embeddings cache if it grows too large."""
        if len(self._embeddings_cache) > self._max_cache_size:
            # Remove oldest items
            items_to_remove = len(self._embeddings_cache) - self._max_cache_size
            keys_to_remove = list(self._embeddings_cache.keys())[:items_to_remove]
            
            for key in keys_to_remove:
                del self._embeddings_cache[key]
    
    def remove(self, index: int) -> bool:
        """
        Remove a document at the specified index.
        This is safer than direct deletion as it tracks removed indices.
        
        Args:
            index: The index of the document to remove
            
        Returns:
            Success boolean
        """
        if index < 0 or index >= len(self.documents):
            return False
            
        # Mark document as removed
        doc_hash = self._compute_hash(self.documents[index])
        
        # Record deleted ID
        self._deleted_ids.add(index)
        
        # Remove from documents and metadata
        self.documents[index] = ""  # Empty string instead of removing to maintain indices
        self.metadata[index] = {"deleted": True}
        
        # Remove from cache
        if doc_hash in self._embeddings_cache:
            del self._embeddings_cache[doc_hash]
            
        # Remove from doc_hashes
        self.doc_hashes.discard(doc_hash)
        
        # Note: We don't remove from the FAISS index here because it would
        # require rebuilding the entire index. Instead, we filter results
        # in the search method.
        
        # Save the changes
        self.save()
        return True
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10, min_similarity: float = 0.25) -> List[Dict[str, Any]]:
        """Search with filtering of deleted documents."""
        # Get results from parent method
        results = super().search(query_embedding, top_k=top_k+len(self._deleted_ids), min_similarity=min_similarity)
        
        # Filter out deleted documents
        filtered_results = [
            result for result in results
            if result.get('index') not in self._deleted_ids
        ]
        
        # Return only requested number
        return filtered_results[:top_k]
    
    def consolidate_memories(self, similarity_threshold: float = 0.85, min_cluster_size: int = 2):
        """Overridden consolidation with proper memory management."""
        result = super().consolidate_memories(similarity_threshold, min_cluster_size)
        
        # Clear cache after consolidation
        self._embeddings_cache.clear()
        self._deleted_ids.clear()
        
        # Force garbage collection
        gc.collect()
        
        return result
    
    def rebuild_index(self):
        """
        Completely rebuild the index to reclaim memory and remove deleted documents.
        This is an expensive operation but helps with memory fragmentation.
        """
        if not self.documents or self.index is None:
            return False
            
        # Filter out deleted documents
        valid_indices = [i for i in range(len(self.documents)) if i not in self._deleted_ids]
        
        if not valid_indices:
            return False
            
        # Create new filtered lists
        new_docs = [self.documents[i] for i in valid_indices]
        new_metadata = [self.metadata[i] for i in valid_indices]
        new_hashes = {self._compute_hash(doc) for doc in new_docs}
        
        # Generate embeddings for valid documents
        if self.embedding_function:
            try:
                # Use batch processing for embeddings
                new_embeddings = batch_embedding_generation(
                    new_docs,
                    self.embedding_function,
                    batch_size=50,
                    show_progress=True
                )

                # Normalize embeddings
                norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
                normalized_embeddings = new_embeddings / norms

                # Create new index
                new_index = faiss.IndexFlatIP(self.embedding_dim)
                if len(normalized_embeddings) > 0:
                    new_index.add(normalized_embeddings.astype(np.float32))
                
                # Update instance variables
                self.index = new_index
                self.documents = new_docs
                self.metadata = new_metadata
                self.doc_hashes = new_hashes
                self._deleted_ids.clear()
                
                # Save changes
                self.save()
                
                # Clear cache and run garbage collection
                self._embeddings_cache.clear()
                gc.collect()
                
                return True
            
            except Exception as e:
                print(f"Error rebuilding index: {e}")
                return False
        
        return False
    
    def save(self):
        """Memory-optimized save with error handling."""
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
                'embedding_dim': self.embedding_dim,
                'deleted_ids': list(self._deleted_ids),
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
                
        except Exception as e:
            print(f"Error saving vector store: {e}")
            # Attempt to clean up temp files
            for path in [f"{self.index_path}.tmp", f"{self.data_path}.tmp"]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
    
    def load(self):
        """Memory-optimized load with better error handling."""
        try:
            # Check if files exist
            if not (os.path.exists(self.index_path) and os.path.exists(self.data_path)):
                print("No existing vector store found, initializing new store")
                return
                
            # Load documents and metadata
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data.get('documents', [])
            self.metadata = data.get('metadata', [])
            self.doc_hashes = set(data.get('doc_hashes', []))
            self.embedding_dim = data.get('embedding_dim', 384)
            self._deleted_ids = set(data.get('deleted_ids', []))
            
            # Load index if there are documents
            if self.documents and os.path.getsize(self.index_path) > 0:
                self.index = faiss.read_index(self.index_path)
                
                # Validate index size matches document count
                if self.index.ntotal != len(self.documents) - len(self._deleted_ids):
                    print(f"Warning: Index size ({self.index.ntotal}) doesn't match document count "
                          f"({len(self.documents) - len(self._deleted_ids)})")
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            # Initialize new store if loading fails
            self.documents = []
            self.metadata = []
            self.doc_hashes = set()
            self._deleted_ids = set()
            self.index = None
    
    def cleanup(self):
        """Explicitly clean up resources."""
        # Clear caches
        self._embeddings_cache.clear()
        
        # Run garbage collection
        gc.collect()
    
    def _cleanup(self):
        """Cleanup method called by finalizer."""
        self.cleanup()
        print("Vector store resources released")