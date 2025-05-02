import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_memory import MemoryManager


def test_memory_add_retrieval():
    """Test adding and retrieving items from memory"""
    
    # Initialize memory manager
    memory_dir = "./test_memory"
    os.makedirs(memory_dir, exist_ok=True)
    memory = MemoryManager(storage_path=memory_dir, embedding_dim=384)
    
    # Define a simple embedding function for testing
    def simple_embedding(text):
        # Create deterministic but unique embedding based on text
        hash_val = sum(ord(c) for c in text)
        np.random.seed(hash_val)
        embedding = np.random.randn(384)
        return embedding / np.linalg.norm(embedding)
    
    # Set the embedding function
    memory.set_embedding_function(simple_embedding)
    
    # Add test items
    print("Adding test items to memory...")
    items = [
        {"content": "The capital of France is Paris.", "metadata": {"source": "test", "category": "geography"}},
        {"content": "Python is a programming language.", "metadata": {"source": "test", "category": "programming"}},
        {"content": "Water boils at 100 degrees Celsius at sea level.", "metadata": {"source": "test", "category": "science"}},
        {"content": "TinyLlama is a small language model.", "metadata": {"source": "test", "category": "ai"}},
        {"content": "The Earth revolves around the Sun.", "metadata": {"source": "test", "category": "astronomy"}}
    ]
    
    item_ids = []
    for item in items:
        item_id = memory.add(content=item["content"], metadata=item["metadata"])
        item_ids.append(item_id)
        print(f"Added item with ID: {item_id}")
    
    # Test basic retrieval
    print("\nTesting basic retrieval...")
    query = "Tell me about planets"
    results = memory.retrieve(query, top_k=2)
    print(f"Query: '{query}'")
    for result in results:
        print(f"- [{result['similarity']:.2f}] {result['content']}")
    
    # Test category filtering
    print("\nTesting metadata filtering...")
    query = "What is Python?"
    results = memory.retrieve(query, top_k=1, metadata_filter={"category": "programming"})
    print(f"Query with filter: '{query}', category='programming'")
    for result in results:
        print(f"- [{result['similarity']:.2f}] {result['content']}")
    
    # Test getting item by ID
    print("\nTesting get by ID...")
    if item_ids:
        item = memory.get(item_ids[0])
        print(f"Retrieved item: {item}")
    
    # Test updating item
    print("\nTesting update...")
    if item_ids:
        updated = memory.update(item_ids[0], {"metadata": {"updated": True}})
        print(f"Update success: {updated}")
        item = memory.get(item_ids[0])
        print(f"Updated item: {item}")
    
    # Test getting statistics
    print("\nMemory statistics:")
    stats = memory.get_stats()
    for key, value in stats.items():
        if not isinstance(value, dict):
            print(f"- {key}: {value}")
    
    # Clean up
    memory.cleanup()
    print("\nTest complete")

if __name__ == "__main__":
    test_memory_add_retrieval()