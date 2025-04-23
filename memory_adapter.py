"""
Adapter to support migrating from the old memory systems to the unified memory system.
Provides backward compatibility while leveraging the new unified architecture.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Union
from unified_memory import UnifiedMemoryManager, MemoryItem

class MemoryAdapter:
    """
    Adapter class to support transition from old memory system to new unified memory system.
    Implements the key interfaces from the original MemoryManager class while using
    the new UnifiedMemoryManager under the hood.
    """
    
    def __init__(self, 
                memory_dir: str = "./memory",
                device: str = None,
                auto_memorize: bool = True,
                sharpening_enabled: bool = True,
                sharpening_factor: float = 0.3,
                fractal_enabled: bool = True,
                max_fractal_levels: int = 3):
        """
        Initialize the memory adapter with backward compatibility.
        
        Args:
            memory_dir: Base directory for memory storage
            device: Device for embedding generation (ignored, kept for compatibility)
            auto_memorize: Whether to automatically memorize conversations
            sharpening_enabled: Whether to enable vector space sharpening
            sharpening_factor: Factor to control sharpening strength
            fractal_enabled: Whether to enable fractal embeddings
            max_fractal_levels: Maximum number of fractal levels
        """
        self.memory_dir = memory_dir
        self.auto_memorize = auto_memorize
        self.sharpening_enabled = sharpening_enabled
        self.sharpening_factor = sharpening_factor
        self.fractal_enabled = fractal_enabled
        self.max_fractal_levels = max_fractal_levels
        
        # Create base directory
        os.makedirs(memory_dir, exist_ok=True)
        
        # Store user memory managers
        self.user_stores = {}
        
        # Store embedding function (to be set later)
        self.embedding_function = None
        self.embedding_dim = 384  # Default
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for text. This should be implemented by the chat system.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if self.embedding_function:
            return self.embedding_function(text)
        else:
            # Fallback to random embeddings if no function is provided
            return np.random.random(self.embedding_dim).astype(np.float32)
    
    def _get_user_store(self, user_id: str) -> UnifiedMemoryManager:
        """
        Get or create a memory store for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            UnifiedMemoryManager for the user
        """
        if user_id not in self.user_stores:
            # Create storage path
            user_path = os.path.join(self.memory_dir, user_id)
            
            # Create memory manager
            self.user_stores[user_id] = UnifiedMemoryManager(
                storage_path=user_path,
                embedding_function=self.generate_embedding,
                embedding_dim=self.embedding_dim,
                use_fractal=self.fractal_enabled,
                max_fractal_levels=self.max_fractal_levels,
                auto_save=True
            )
        
        return self.user_stores[user_id]
    
    def add_memory(self, 
                  user_id: str,
                  query: str,
                  response: str,
                  memory_type: str = "general",
                  attributes: Dict = None,
                  pre_sharpen: bool = None) -> int:
        """
        Add a new memory with backward compatibility.
        
        Args:
            user_id: User identifier
            query: Query text
            response: Response text
            memory_type: Type of memory
            attributes: Additional attributes
            pre_sharpen: Whether to apply pre-sharpening
            
        Returns:
            Number of memories added
        """
        if not self.auto_memorize:
            return 0
        
        # Default pre_sharpen to sharpening_enabled if not specified
        if pre_sharpen is None:
            pre_sharpen = self.sharpening_enabled
        
        attributes = attributes or {}
        
        # Extract key information
        key_info = self.extract_key_information(query, response)
        
        # Store each key piece of information
        memories_added = 0
        
        for info in key_info:
            # Create metadata with compatibility fields
            metadata = {
                'source_query': query,
                'source_response': response[:100] + "..." if len(response) > 100 else response,
                'timestamp': self._get_timestamp(),
                'memory_type': memory_type
            }
            
            # Add additional attributes
            metadata.update(attributes)
            
            # Add to the store
            store = self._get_user_store(user_id)
            
            item_id = store.add(
                content=info,
                memory_type=memory_type,
                source="conversation",
                metadata=metadata,
                use_fractal=self.fractal_enabled if pre_sharpen else False
            )
            
            if item_id:
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
        
        # Search with appropriate parameters
        results = store.retrieve(
            query=query,
            top_k=top_k*2,  # Get more results for better filtering
            min_similarity=0.25,
            use_fractal=self.fractal_enabled if apply_sharpening else False
        )
        
        # Format the results into memory text
        memory_text = self._format_memory_results(results, top_k)
        
        return memory_text
    
    def _format_memory_results(self, results: List[Dict[str, Any]], top_k: int) -> str:
        """
        Format search results into categorized memory text.
        
        Args:
            results: Search results from memory store
            top_k: Maximum number of results to include
            
        Returns:
            Formatted memory text
        """
        # If no results, return empty string
        if not results:
            return ""
        
        # Group results into clear categories
        corrections = []
        factual_info = []
        general_info = []
        
        for result in results[:top_k]:
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            
            # Check memory type and content for categorization
            if metadata.get('is_correction', False):
                corrections.append(result)
            elif any(term in content.lower() for term in 
                    ['definition', 'fact', 'rule', 'alphabet', 'order']):
                factual_info.append(result)
            else:
                general_info.append(result)
        
        # Compose the memory text with clear sections
        memory_text = ""
        
        if corrections:
            memory_text += "IMPORTANT CORRECTIONS (You MUST apply these):\n"
            for result in corrections:
                memory_text += f"- {result['content']}\n"
            memory_text += "\n"
        
        if factual_info:
            memory_text += "FACTUAL INFORMATION:\n"
            for result in factual_info:
                memory_text += f"- {result['content']}\n"
            memory_text += "\n"
        
        if general_info and (not corrections or not factual_info):
            memory_text += "OTHER RELEVANT INFORMATION:\n"
            for result in general_info:
                memory_text += f"- {result['content']}\n"
        
        return memory_text
    
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
    
    def toggle_auto_memorize(self) -> bool:
        """
        Toggle automatic memorization on/off.
        
        Returns:
            New auto_memorize state
        """
        self.auto_memorize = not self.auto_memorize
        return self.auto_memorize
    
    def toggle_sharpening(self) -> bool:
        """
        Toggle vector space sharpening on/off.
        
        Returns:
            New sharpening state
        """
        self.sharpening_enabled = not self.sharpening_enabled
        return self.sharpening_enabled
    
    def set_sharpening_factor(self, factor: float) -> None:
        """
        Set the sharpening factor for memory retrieval.
        
        Args:
            factor: Sharpening factor (0.0-1.0)
        """
        # Ensure valid range
        self.sharpening_factor = max(0.0, min(1.0, factor))
    
    def cleanup(self):
        """Clean up resources and save data."""
        for user_id, store in self.user_stores.items():
            try:
                # Clean up and save
                store.cleanup()
            except Exception as e:
                print(f"Error cleaning up memory for user {user_id}: {e}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def initialize_knowledge_system(self):
        """
        Initialize the knowledge system (stub for compatibility).
        In the new architecture, knowledge is unified with other memory types.
        
        Returns:
            True (always succeeds in the new system)
        """
        # In new system, no separate initialization is needed
        return True