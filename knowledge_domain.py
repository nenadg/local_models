"""
Knowledge domain container for TinyLlama Chat system.
Manages domain-specific vector stores and metadata for isolated knowledge management.
"""

import os
import json
import shutil
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from datetime import datetime

class KnowledgeDomain:
    """
    Container for domain-specific knowledge management.
    
    A KnowledgeDomain maintains isolated vector stores for specific knowledge domains,
    allowing for modular loading/unloading of knowledge and domain-specific configurations.
    It provides an interface to domain-specific vector stores and manages metadata.
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        base_directory: str = "./knowledge_domains",
        vector_store=None,
        embedding_function: Optional[Callable] = None,
        embedding_dim: int = 384,
        enable_fractal: bool = True,
        max_fractal_levels: int = 3
    ):
        """
        Initialize a knowledge domain.
        
        Args:
            name: Domain name (must be unique)
            description: Optional domain description
            base_directory: Base directory for domain storage
            vector_store: Optional existing vector store to use
            embedding_function: Function to generate embeddings
            embedding_dim: Dimension of embeddings
            enable_fractal: Whether to enable fractal embeddings
            max_fractal_levels: Maximum number of fractal levels
        """
        self.name = name
        self.description = description
        self.base_directory = base_directory
        self.embedding_function = embedding_function
        self.embedding_dim = embedding_dim
        
        # Create safe domain ID from name
        self.domain_id = self._create_safe_id(name)
        
        # Setup domain-specific directories
        self.domain_directory = os.path.join(base_directory, self.domain_id)
        self.vector_store_path = os.path.join(self.domain_directory, "vector_store")
        self.metadata_path = os.path.join(self.domain_directory, "metadata.json")
        self.knowledge_path = os.path.join(self.domain_directory, "knowledge.json")
        
        # Create directories if they don't exist
        os.makedirs(self.domain_directory, exist_ok=True)
        
        # Initialize empty metadata
        self.metadata = {
            "domain_id": self.domain_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "embedding_dim": embedding_dim,
            "version": "1.0.0",
            "stats": {
                "total_items": 0,
                "facts": 0,
                "definitions": 0,
                "procedures": 0,
                "relationships": 0,
                "mappings": 0
            },
            "fractal_enabled": enable_fractal,
            "max_fractal_levels": max_fractal_levels
        }
        
        # Initialize vector store if provided, else create new one
        if vector_store:
            self.vector_store = vector_store
        else:
            # Import VectorStore from the project
            try:
                from memory_manager import VectorStore
                self.vector_store = VectorStore(
                    storage_path=self.vector_store_path,
                    embedding_function=embedding_function,
                    embedding_dim=embedding_dim,
                    fractal_enabled=enable_fractal,
                    max_fractal_levels=max_fractal_levels
                )
            except ImportError:
                print("Warning: VectorStore not found. Vector storage functionality disabled.")
                self.vector_store = None
        
        # Try to load existing metadata if available
        self._load_metadata()
    
    def _create_safe_id(self, name: str) -> str:
        """
        Create a safe domain ID from name.
        
        Args:
            name: Domain name
            
        Returns:
            Safe domain ID
        """
        # Replace spaces and special characters
        safe_id = name.lower().replace(" ", "_")
        safe_id = "".join(c for c in safe_id if c.isalnum() or c == "_")
        
        # Add hash for uniqueness if needed
        if len(safe_id) < 3:
            name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
            safe_id = f"{safe_id}_{name_hash}"
        
        return safe_id
    
    def _load_metadata(self) -> bool:
        """
        Load domain metadata from file.
        
        Returns:
            Success boolean
        """
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    loaded_metadata = json.load(f)
                    # Update metadata with loaded values, preserving structure
                    for key, value in loaded_metadata.items():
                        if key in self.metadata:
                            self.metadata[key] = value
                return True
            except Exception as e:
                print(f"Error loading domain metadata: {e}")
        return False
    
    def _save_metadata(self) -> bool:
        """
        Save domain metadata to file.
        
        Returns:
            Success boolean
        """
        try:
            # Update the last_updated timestamp
            self.metadata["last_updated"] = datetime.now().isoformat()
            
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving domain metadata: {e}")
            return False
    
    def add_knowledge(self, knowledge_items: List[Dict[str, Any]]) -> int:
        """
        Add knowledge items to the domain.
        
        Args:
            knowledge_items: List of knowledge items
            
        Returns:
            Number of items successfully added
        """
        if not self.vector_store:
            print("Cannot add knowledge: Vector store not available")
            return 0
        
        items_added = 0
        
        for item in knowledge_items:
            # Add domain identifier to the item
            if "metadata" not in item:
                item["metadata"] = {}
            item["metadata"]["domain_id"] = self.domain_id
            
            # Convert item to text for embedding
            item_text = self._knowledge_item_to_text(item)
            
            # Create embedding
            if self.embedding_function:
                embedding = self.embedding_function(item_text)
            else:
                # Fallback if no embedding function available
                print("Warning: No embedding function available, using random embedding")
                embedding = np.random.random(self.embedding_dim)
            
            # Add to vector store
            success = self.vector_store.add(
                text=item_text,
                embedding=embedding,
                metadata={
                    "knowledge_item": item,
                    "domain_id": self.domain_id,
                    "item_type": item.get("type", "unknown"),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            if success:
                items_added += 1
                
                # Update statistics
                self.metadata["stats"]["total_items"] += 1
                item_type = item.get("type", "unknown")
                if item_type in self.metadata["stats"]:
                    self.metadata["stats"][item_type] += 1
        
        # Save updated metadata
        self._save_metadata()
        
        # Also save the complete knowledge items for backup/export
        self._append_to_knowledge_file(knowledge_items)
        
        return items_added
    
    def _knowledge_item_to_text(self, item: Dict[str, Any]) -> str:
        """
        Convert a knowledge item to text for embedding.
        
        Args:
            item: Knowledge item
            
        Returns:
            Text representation
        """
        item_type = item.get("type", "unknown")
        content = item.get("content", {})
        
        if item_type == "fact":
            return f"{content.get('subject', '')} {content.get('predicate', 'is')} {content.get('object', '')}. {content.get('context', '')}"
        
        elif item_type == "definition":
            return f"{content.get('term', '')} is defined as {content.get('definition', '')}. {content.get('context', '')}"
        
        elif item_type == "procedure":
            steps = content.get('steps', [])
            steps_text = ". ".join(steps)
            return f"{content.get('title', 'Procedure')}: {steps_text}. {content.get('context', '')}"
        
        elif item_type == "relationship":
            return f"{content.get('entity1', '')} relates to {content.get('entity2', '')} as {content.get('relationship_type', 'related')}. {content.get('context', '')}"
        
        elif item_type == "mapping":
            pairs = content.get('pairs', [])
            pairs_text = ", ".join([f"{pair.get('from', '')} maps to {pair.get('to', '')}" for pair in pairs])
            return f"Mapping {content.get('title', '')}: {pairs_text}. {content.get('context', '')}"
        
        else:
            # Fallback for unknown types
            return json.dumps(item)
    
    def _append_to_knowledge_file(self, knowledge_items: List[Dict[str, Any]]) -> bool:
        """
        Append knowledge items to the knowledge file.
        
        Args:
            knowledge_items: List of knowledge items
            
        Returns:
            Success boolean
        """
        try:
            # Load existing items if available
            existing_items = []
            if os.path.exists(self.knowledge_path):
                with open(self.knowledge_path, 'r', encoding='utf-8') as f:
                    existing_items = json.load(f)
            
            # Append new items
            all_items = existing_items + knowledge_items
            
            # Save back to file
            with open(self.knowledge_path, 'w', encoding='utf-8') as f:
                json.dump(all_items, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error appending to knowledge file: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.25) -> List[Dict[str, Any]]:
        """
        Search for knowledge in the domain.
        
        Args:
            query: Search query
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of matching knowledge items
        """
        if not self.vector_store or not self.embedding_function:
            print("Cannot search: Vector store or embedding function not available")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_function(query)
        
        # Use vector store's fractal search if available
        results = []
        if hasattr(self.vector_store, 'enhanced_fractal_search') and self.metadata.get("fractal_enabled", False):
            # Use enhanced fractal search
            search_results = self.vector_store.enhanced_fractal_search(
                query_embedding=query_embedding,
                top_k=top_k,
                min_similarity=min_similarity,
                apply_sharpening=True,
                multi_level_search=True
            )
        else:
            # Fall back to standard search
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                min_similarity=min_similarity
            )
        
        # Extract knowledge items from results
        for result in search_results:
            metadata = result.get('metadata', {})
            knowledge_item = metadata.get('knowledge_item', None)
            
            if knowledge_item:
                # Add search metadata
                knowledge_item['search_metadata'] = {
                    'similarity': result.get('similarity', 0.0),
                    'original_text': result.get('text', '')
                }
                
                results.append(knowledge_item)
            else:
                # If knowledge_item not found, create a simple one from the text
                simple_item = {
                    'type': metadata.get('item_type', 'unknown'),
                    'content': {
                        'text': result.get('text', '')
                    },
                    'metadata': {
                        'domain_id': self.domain_id,
                        'source': 'vector_store',
                        'confidence': 0.7
                    },
                    'search_metadata': {
                        'similarity': result.get('similarity', 0.0)
                    }
                }
                
                results.append(simple_item)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the domain.
        
        Returns:
            Dictionary of domain statistics
        """
        # Get vector store stats if available
        vector_stats = {}
        if self.vector_store and hasattr(self.vector_store, 'get_stats'):
            vector_stats = self.vector_store.get_stats()
        
        # Combine with metadata stats
        stats = {
            "domain_id": self.domain_id,
            "name": self.metadata["name"],
            "total_items": self.metadata["stats"]["total_items"],
            "knowledge_types": {
                k: v for k, v in self.metadata["stats"].items() if k != "total_items"
            },
            "vector_store": vector_stats,
            "last_updated": self.metadata["last_updated"]
        }
        
        return stats
    
    def save(self) -> bool:
        """
        Save the domain (metadata and vector store).
        
        Returns:
            Success boolean
        """
        success = True
        
        # Save metadata
        if not self._save_metadata():
            success = False
        
        # Save vector store if available
        if self.vector_store and hasattr(self.vector_store, 'save'):
            if not self.vector_store.save():
                success = False
        
        return success
    
    def export_knowledge(self, output_path: str = None) -> str:
        """
        Export all knowledge from the domain to a file.
        
        Args:
            output_path: Optional output file path
            
        Returns:
            Path to the exported file
        """
        if not output_path:
            output_path = os.path.join(self.domain_directory, f"export_{self.domain_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        try:
            # If knowledge file exists, just copy it
            if os.path.exists(self.knowledge_path):
                shutil.copy(self.knowledge_path, output_path)
            else:
                # Otherwise, extract all items from vector store
                all_items = []
                
                # Get all documents and metadata
                if self.vector_store:
                    for i in range(len(self.vector_store.documents)):
                        metadata = self.vector_store.metadata[i] if i < len(self.vector_store.metadata) else {}
                        knowledge_item = metadata.get('knowledge_item', None)
                        
                        if knowledge_item:
                            all_items.append(knowledge_item)
                
                # Save to output file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(all_items, f, indent=2)
            
            return output_path
        except Exception as e:
            print(f"Error exporting knowledge: {e}")
            return ""
    
    def import_knowledge(self, input_path: str) -> int:
        """
        Import knowledge from a file.
        
        Args:
            input_path: Path to the input file
            
        Returns:
            Number of items imported
        """
        try:
            # Load knowledge items from file
            with open(input_path, 'r', encoding='utf-8') as f:
                knowledge_items = json.load(f)
            
            # Add to domain
            return self.add_knowledge(knowledge_items)
        except Exception as e:
            print(f"Error importing knowledge: {e}")
            return 0
    
    def clear(self) -> bool:
        """
        Clear all knowledge from the domain.
        
        Returns:
            Success boolean
        """
        success = True
        
        # Reset vector store if available
        if self.vector_store:
            try:
                # If rebuild_index is available, use it with empty list
                if hasattr(self.vector_store, 'rebuild_index'):
                    self.vector_store.rebuild_index([])
                else:
                    # Otherwise, try to reinitialize
                    from memory_manager import VectorStore
                    self.vector_store = VectorStore(
                        storage_path=self.vector_store_path,
                        embedding_function=self.embedding_function,
                        embedding_dim=self.embedding_dim,
                        fractal_enabled=self.metadata.get("fractal_enabled", True),
                        max_fractal_levels=self.metadata.get("max_fractal_levels", 3)
                    )
            except Exception as e:
                print(f"Error clearing vector store: {e}")
                success = False
        
        # Reset statistics
        self.metadata["stats"] = {
            "total_items": 0,
            "facts": 0,
            "definitions": 0,
            "procedures": 0,
            "relationships": 0,
            "mappings": 0
        }
        
        # Update metadata
        self._save_metadata()
        
        # Clear knowledge file
        if os.path.exists(self.knowledge_path):
            try:
                with open(self.knowledge_path, 'w', encoding='utf-8') as f:
                    json.dump([], f)
            except Exception as e:
                print(f"Error clearing knowledge file: {e}")
                success = False
        
        return success
    
    def delete(self) -> bool:
        """
        Delete the entire domain.
        
        Returns:
            Success boolean
        """
        try:
            # Delete the domain directory and all contents
            if os.path.exists(self.domain_directory):
                shutil.rmtree(self.domain_directory)
            return True
        except Exception as e:
            print(f"Error deleting domain: {e}")
            return False