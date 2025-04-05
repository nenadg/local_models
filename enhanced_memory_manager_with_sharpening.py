import os
import numpy as np
from typing import List, Dict, Any, Tuple
from enhanced_memory_store import EnhancedMemoryManager, IndexedVectorStore

from sharpened_vector_space import SharpenedVectorStore

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

        # Create the sharpening utility
        self.sharpening_util = SharpenedVectorStore()

    def _get_user_store(self, user_id: str) -> IndexedVectorStore:
        """Get or create a vector store for a user with sharpening support"""
        if user_id not in self.user_stores:
            # Get the base store from parent class
            store = super()._get_user_store(user_id)

            # Store the original search method for future reference
            if not hasattr(store, 'original_search'):
                store.original_search = store.search

            # Enhance the store's search method with sharpening
            store.search = self.sharpening_util.update_search_with_sharpening(
                store.original_search,
                sharpening_factor=self.sharpening_factor
            )

            # Store sharpening capability reference
            store.sharpen_embeddings = self.sharpening_util.sharpen_embeddings

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
            elif any(term in result['text'].lower() for term in ['definition', 'fact', 'rule', 'alphabet', 'order']):
                factual_info.append(result)
            else:
                general_info.append(result)

        # Compose the memory text with clear sections
        memory_text = ""

        if corrections:
            memory_text += "IMPORTANT CORRECTIONS (You MUST apply these):\n"
            for i, result in enumerate(corrections):
                improvement = ""
                if 'original_similarity' in result and apply_sharpening:
                    original = result.get('original_similarity', 0)
                    sharpened = result.get('similarity', 0)
                    if original > 0:  # Avoid division by zero
                        change = ((sharpened - original) / original) * 100
                        if abs(change) > 1:
                            improvement = f" (relevance {'increased' if change > 0 else 'decreased'} by {abs(change):.1f}%)"
                memory_text += f"- {result['text']}{improvement}\n"
            memory_text += "\n"

        if factual_info:
            memory_text += "FACTUAL INFORMATION:\n"
            for i, result in enumerate(factual_info):
                improvement = ""
                if 'original_similarity' in result and apply_sharpening:
                    original = result.get('original_similarity', 0)
                    sharpened = result.get('similarity', 0)
                    if original > 0:  # Avoid division by zero
                        change = ((sharpened - original) / original) * 100
                        if abs(change) > 1:
                            improvement = f" (relevance {'increased' if change > 0 else 'decreased'} by {abs(change):.1f}%)"
                memory_text += f"- {result['text']}{improvement}\n"
            memory_text += "\n"

        if general_info and (not corrections or not factual_info):
            memory_text += "OTHER RELEVANT INFORMATION:\n"
            for i, result in enumerate(general_info):
                improvement = ""
                if 'original_similarity' in result and apply_sharpening:
                    original = result.get('original_similarity', 0)
                    sharpened = result.get('similarity', 0)
                    if original > 0:  # Avoid division by zero
                        change = ((sharpened - original) / original) * 100
                        if abs(change) > 1:
                            improvement = f" (relevance {'increased' if change > 0 else 'decreased'} by {abs(change):.1f}%)"
                memory_text += f"- {result['text']}{improvement}\n"

        return memory_text

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
        factor = max(0.0, min(1.0, factor))

        if factor == self.sharpening_factor:
            return  # No change needed

        self.sharpening_factor = factor

        # Update all existing stores with the new sharpening factor
        for user_id, store in self.user_stores.items():
            # Make sure we have the original search method
            if not hasattr(store, 'original_search'):
                store.original_search = store.search

            # Replace the search method with an updated one using the new factor
            store.search = self.sharpening_util.update_search_with_sharpening(
                store.original_search,
                sharpening_factor=self.sharpening_factor
            )