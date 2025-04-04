import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
import re

class SimpleVectorStore:
    """
    A simple vector store for saving and retrieving text embeddings.
    Uses cosine similarity for retrieval.
    """
    def __init__(self, storage_path: str = "./memory/vector_store"):
        self.storage_path = storage_path
        self.vectors = []
        self.documents = []
        self.metadata = []
        
        # Create storage directory if it doesn't exist
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        
        # Load existing data if available
        self.load()
    
    def add(self, text: str, embedding: np.ndarray, metadata: Dict[str, Any] = None):
        """Add a document to the vector store"""
        self.vectors.append(embedding)
        self.documents.append(text)
        self.metadata.append(metadata or {})
        self.save()
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity"""
        if not self.vectors:
            return []
        
        # Convert list to numpy array for efficient computation
        vectors_array = np.array(self.vectors)
        
        # Compute cosine similarity
        similarities = self._cosine_similarity(query_embedding, vectors_array)
        
        # Get top k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.6:  # Only include reasonably similar results
                results.append({
                    'text': self.documents[idx],
                    'similarity': float(similarities[idx]),
                    'metadata': self.metadata[idx]
                })
        
        return results
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between vectors"""
        if len(b.shape) == 1:
            # Single vector
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        else:
            # Multiple vectors
            return np.dot(b, a) / (np.linalg.norm(b, axis=1) * np.linalg.norm(a))
    
    def save(self):
        """Save vector store to disk"""
        data = {
            'vectors': [v.tolist() for v in self.vectors],
            'documents': self.documents,
            'metadata': self.metadata,
            'updated_at': datetime.now().isoformat()
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f)
    
    def load(self):
        """Load vector store from disk"""
        if not os.path.exists(self.storage_path):
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.vectors = [np.array(v) for v in data.get('vectors', [])]
            self.documents = data.get('documents', [])
            self.metadata = data.get('metadata', [])
        except Exception as e:
            print(f"Error loading vector store: {e}")


class MemoryManager:
    """
    Manages long-term memory for LLM conversations.
    Extracts and retrieves relevant information from past conversations.
    """
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        memory_dir: str = "./memory",
        device: str = None
    ):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        
        self.user_stores = {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # For very small models, we can load the embedding model
        # For production, would use a smaller model or API
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.embedding_dim = self.model.config.hidden_size
            print(f"Loaded embedding model: {model_name}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            # Fallback to random embeddings for demo
            self.tokenizer = None
            self.model = None
            self.embedding_dim = 384  # Common dimension
    
    def _get_user_store(self, user_id: str) -> SimpleVectorStore:
        """Get or create a vector store for a user"""
        if user_id not in self.user_stores:
            store_path = os.path.join(self.memory_dir, f"{user_id}_store.json")
            self.user_stores[user_id] = SimpleVectorStore(store_path)
        return self.user_stores[user_id]
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for a text snippet"""
        if self.model is None:
            # Fallback to random embeddings for demo
            return np.random.random(self.embedding_dim)
        
        # Tokenize and embed the text
        inputs = self.tokenizer(text, return_tensors="pt", 
                               truncation=True, max_length=512, 
                               padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use mean pooling to get a single vector
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        return embedding
    
    def extract_key_information(self, query: str, response: str) -> List[str]:
        """Extract key information from conversation to remember"""
        # In a production system, you'd use an LLM to extract key information
        # For this minimal implementation, we'll use a simple rule-based approach
        
        # Extract factual statements (sentences with potential facts)
        combined_text = f"{query} {response}"
        sentences = re.split(r'[.!?]', combined_text)
        key_info = []
        
        # Simple heuristics to identify potential facts
        fact_starters = [
            "is", "was", "are", "were", "has", "have", "had",
            "can", "will", "should", "must", "might", "may",
            "the", "a", "an", "in", "born", "created", "developed",
            "contains", "consists", "includes"
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15:  # Skip very short sentences
                words = sentence.lower().split()
                if words and words[0] in fact_starters:
                    key_info.append(sentence)
        
        # Limit the number of extracted facts
        return key_info[:3]
    
    def add_memory(self, user_id: str, query: str, response: str):
        """Add a new memory from a conversation"""
        # Get user's vector store
        store = self._get_user_store(user_id)
        
        # Extract key information
        key_info = self.extract_key_information(query, response)
        
        # Store each key piece of information
        for info in key_info:
            embedding = self.generate_embedding(info)
            store.add(
                text=info,
                embedding=embedding,
                metadata={
                    'source_query': query,
                    'timestamp': datetime.now().isoformat()
                }
            )
    
    def retrieve_relevant_memories(self, user_id: str, query: str, top_k: int = 3) -> str:
        """Retrieve relevant memories for a query"""
        # Get user's vector store
        store = self._get_user_store(user_id)
        
        # Generate embedding for query
        query_embedding = self.generate_embedding(query)
        
        # Search for similar memories
        results = store.search(query_embedding, top_k=top_k)
        
        # Format results
        if not results:
            return ""
        
        memories_text = "Based on our previous conversations, I remember:\n"
        for i, result in enumerate(results):
            memories_text += f"- {result['text']}\n"
        
        return memories_text

    def save_conversation(self, user_id: str, conversation: List[Dict[str, Any]]):
        """Save an entire conversation with enhanced memory extraction"""
        if len(conversation) < 2:
            return  # Skip empty conversations
        
        # Pair user queries with assistant responses
        for i in range(1, len(conversation), 2):
            if i + 1 < len(conversation):
                query = conversation[i].get('content', '')
                response = conversation[i+1].get('content', '')
                
                if query and response:
                    self.add_memory(user_id, query, response)