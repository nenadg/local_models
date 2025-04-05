import os
import numpy as np
from typing import List, Dict, Any, Tuple
from enhanced_memory_store import EnhancedMemoryManager, IndexedVectorStore
from sharpened_vector_space import SharpenedVectorStore  # Import the new sharpening utilities

class EnhancedMemoryManagerWithSharpening(EnhancedMemoryManager):
    """
    Extension of EnhancedMemoryManager with vector space sharpening capabilities
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        memory_dir: str = "./memory",
        device: str = None,
        auto_memorize: bool = True,
        sharpening_enabled: bool = True,
        sharpening_factor: float = 0.3
    ):
        # Initialize parent class
        super().__init__(
            model_name=model_name,
            memory_dir=memory_dir,
            device=device,
            auto_memorize=auto_memorize
        )
        
        # Initialize sharpening parameters
        self.sharpening_enabled = sharpening_enabled
        self.sharpening_factor = sharpening_factor
        
    def _get_user_store(self, user_id: str) -> IndexedVectorStore:
        """Get or create a vector store for a user with sharpening support"""
        if user_id not in self.user_stores:
            store_path = os.path.join(self.memory_dir, f"{user_id}_store")
            store = super()._get_user_store(user_id)

            # Extend store with sharpening capabilities
            sharpening_store = SharpenedVectorStore()

            # Enhance the store's search method with sharpening
            original_search = store.search
            store.search = sharpening_store.update_search_with_sharpening(
                original_search,
                sharpening_factor=self.sharpening_factor
            )
            
            # Add sharpening methods directly to store
            store.sharpen_embeddings = sharpening_store.sharpen_embeddings
            
            self.user_stores[user_id] = store
            
        return self.user_stores[user_id]
        
    def retrieve_relevant_memories(self, user_id: str, query: str, top_k: int = 8, apply_sharpening: bool = None) -> str:
        """
        Retrieve relevant memories with optional sharpening.
        
        Args:
            user_id: The user ID
            query: The query string
            top_k: Number of memories to retrieve
            apply_sharpening: Whether to apply sharpening (defaults to self.sharpening_enabled)
            
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
        
        # Search with sharpening
        results = store.search(
            query_embedding, 
            top_k=top_k*2, 
            min_similarity=0.25,
            apply_sharpening=apply_sharpening
        )
        
        # Format results as before...
        # This is just a partial implementation - the rest of the method 
        # would be copied from the original with minor adjustments for sharpening
        
        # Process the results...
        if not results:
            return ""
            
        # Additional code for processing results and creating the memories text
        # would go here, following the logic in the original method
        
        # For demonstration, add information about sharpening
        memories_text = "Based on our previous conversations:\n\n"
        for result in results[:top_k]:
            original = result.get('original_similarity', result['similarity'])
            sharpened = result['similarity']
            improvement = ""
            if 'original_similarity' in result:
                change = ((sharpened - original) / original) * 100
                if abs(change) > 1:
                    improvement = f" (relevance {'increased' if change > 0 else 'decreased'} by {abs(change):.1f}%)"
            
            memories_text += f"- {result['text']}{improvement}\n"
            
        return memories_text
        
    def toggle_sharpening(self) -> bool:
        """Toggle vector space sharpening on/off"""
        self.sharpening_enabled = not self.sharpening_enabled
        return self.sharpening_enabled
        
    def set_sharpening_factor(self, factor: float) -> None:
        """
        Set the sharpening factor (0-1) and update existing stores

        Args:
            factor: New sharpening factor between 0 and 1
        """
        self.sharpening_factor = max(0.0, min(1.0, factor))

        # Update all existing stores with the new sharpening factor
        for user_id, store in self.user_stores.items():
            # Get the original search method (before our enhancement)
            if hasattr(store, 'original_search'):
                original_search = store.original_search
            else:
                # Store the original search method for future reference
                original_search = store.search
                store.original_search = original_search

            # Create a new sharpening store with updated factor
            sharpening_store = SharpenedVectorStore()

            # Replace the search method with the updated one
            store.search = sharpening_store.update_search_with_sharpening(
                original_search,
                sharpening_factor=self.sharpening_factor
            )