import numpy as np
from typing import List, Dict, Tuple, Optional

class TopicShiftDetector:
    """
    Detects significant shifts in conversation topics to manage context window more efficiently.
    """
    
    def __init__(self, 
                embedding_function=None,
                similarity_threshold: float = 0.35,
                memory_manager=None):
        """
        Initialize the topic shift detector.
        
        Args:
            embedding_function: Function to generate embeddings for topic comparison
            similarity_threshold: Threshold below which topics are considered different
            memory_manager: Optional memory manager to reuse embedding functionality
        """
        self.embedding_function = embedding_function
        self.similarity_threshold = similarity_threshold
        self.memory_manager = memory_manager
        
        # Keep track of recent topics
        self.recent_topics = []
        self.max_topics = 3  # Keep only the most recent topics
        
        # Enable debug logging
        self.debug = False
    
    def compute_similarity(self, embedding1, embedding2):
        """Compute cosine similarity between two embeddings."""
        # Normalize embeddings for cosine similarity
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
            
        normalized_emb1 = embedding1 / norm1
        normalized_emb2 = embedding2 / norm2
        
        # Compute cosine similarity
        return np.dot(normalized_emb1, normalized_emb2)
    
    def is_topic_shift(self, query: str) -> Tuple[bool, float]:
        """
        Determine if a new query represents a significant topic shift.
        
        Args:
            query: The new user query
            
        Returns:
            Tuple of (is_shift, best_similarity)
        """
        self.continuation_phrases = [
            "please continue",
            "continue anyway",
            "speculate anyway",
            "continue",
            "go on"
        ]

        query_lower = query.lower()

        # Check for continuation phrases
        for phrase in self.continuation_phrases:
            if phrase in query_lower:
                return False, 1.0

        # If no previous topics, not a shift
        if not self.recent_topics:
            self._add_topic(query)
            return False, 1.0
            
        # Use embedding function (direct or from memory manager)
        embedding_fn = self.embedding_function
        if embedding_fn is None and self.memory_manager and hasattr(self.memory_manager, 'embedding_function'):
            embedding_fn = self.memory_manager.embedding_function
            
        # Skip if no embedding function available
        if embedding_fn is None:
            return False, 1.0
            
        # Generate embedding for current query
        try:
            query_embedding = embedding_fn(query)
        except Exception as e:
            if self.debug:
                print(f"Error generating query embedding: {e}")
            return False, 1.0
            
        # Find best similarity with recent topics
        best_similarity = 0.0
        for topic, embedding in self.recent_topics:
            similarity = self.compute_similarity(query_embedding, embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                
        # Add current topic to recent topics
        self._add_topic(query, query_embedding)
        
        # Consider it a shift if similarity is below threshold
        is_shift = best_similarity < self.similarity_threshold
        
        if self.debug and is_shift:
            print(f"Topic shift detected! Similarity: {best_similarity:.3f}")
            
        return is_shift, best_similarity
        
    def _add_topic(self, query: str, embedding=None):
        """Add a topic to recent topics list with optional pre-computed embedding."""
        if embedding is None:
            # Use embedding function if available
            embedding_fn = self.embedding_function
            if embedding_fn is None and self.memory_manager and hasattr(self.memory_manager, 'embedding_function'):
                embedding_fn = self.memory_manager.embedding_function
                
            # Generate embedding if possible
            if embedding_fn is not None:
                try:
                    embedding = embedding_fn(query)
                except Exception:
                    # Use empty embedding if generation fails
                    embedding = np.zeros(384)  # Default embedding size
            else:
                # No embedding function available
                embedding = np.zeros(384)
        
        # Add to recent topics
        self.recent_topics.append((query, embedding))
        
        # Limit to max topics
        if len(self.recent_topics) > self.max_topics:
            self.recent_topics.pop(0)
    
    def reset(self):
        """Reset the topic history."""
        self.recent_topics = []