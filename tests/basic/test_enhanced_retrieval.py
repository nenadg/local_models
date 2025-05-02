import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_memory import MemoryManager

def test_enhanced_retrieval():
    """Test how enhanced embeddings improve retrieval quality"""

    # Initialize memory managers - one with and one without enhanced embeddings
    memory_dir = "./test_memory"
    os.makedirs(memory_dir, exist_ok=True)

    # Standard memory
    std_memory = MemoryManager(
        storage_path=f"{memory_dir}/standard",
        embedding_dim=384,
        enable_enhanced_embeddings=False
    )

    # Enhanced memory
    enh_memory = MemoryManager(
        storage_path=f"{memory_dir}/enhanced",
        embedding_dim=384,
        enable_enhanced_embeddings=True,
        max_enhancement_levels=3,
        similarity_enhancement_factor=0.4
    )

    # Define a topic-based embedding function
    def contextual_embedding(text):
        # Create deterministic but contextual embedding based on text
        seed_val = sum(ord(c) for c in text[:10])
        np.random.seed(seed_val)

        # Extract topic bias from keywords
        topics = {
            "computer": [1.0, 0.2, 0.1, 0.0, 0.1],
            "science": [0.2, 1.0, 0.3, 0.2, 0.1],
            "history": [0.1, 0.3, 1.0, 0.2, 0.1],
            "art": [0.0, 0.2, 0.2, 1.0, 0.2],
            "music": [0.1, 0.1, 0.1, 0.2, 1.0],
        }

        # Find topic bias
        bias = np.zeros(5)
        for topic, vec in topics.items():
            if topic.lower() in text.lower():
                bias += np.array(vec) * 0.5

        # Create base random embedding
        embedding = np.random.randn(384)

        # Inject topic bias into first 5 dimensions
        embedding[:5] = embedding[:5] * 0.3 + bias

        # Normalize
        return embedding / np.linalg.norm(embedding)

    # Set embedding functions
    std_memory.set_embedding_function(contextual_embedding)
    enh_memory.set_embedding_function(contextual_embedding)

    # Test data across multiple topics with subtle relationships
    test_items = [
        # Computer Science items
        {"content": "Python is a high-level programming language.", "metadata": {"topic": "computer"}},
        {"content": "Machine learning algorithms can identify patterns in data.", "metadata": {"topic": "computer"}},
        {"content": "Object-oriented programming uses classes and objects.", "metadata": {"topic": "computer"}},
        {"content": "Database systems store and organize data for efficient retrieval.", "metadata": {"topic": "computer"}},

        # Science items
        {"content": "The theory of relativity was developed by Albert Einstein.", "metadata": {"topic": "science"}},
        {"content": "Quantum mechanics explains behavior of matter at atomic scales.", "metadata": {"topic": "science"}},
        {"content": "DNA contains the genetic instructions for development and function.", "metadata": {"topic": "science"}},
        {"content": "Chemical reactions involve the formation or breaking of bonds.", "metadata": {"topic": "science"}},

        # History items
        {"content": "The Roman Empire was one of the largest empires of the ancient world.", "metadata": {"topic": "history"}},
        {"content": "The Industrial Revolution began in Great Britain in the 18th century.", "metadata": {"topic": "history"}},
        {"content": "World War II ended in 1945 with the surrender of Japan.", "metadata": {"topic": "history"}},
        {"content": "The Renaissance was a period of cultural, artistic and scientific revival.", "metadata": {"topic": "history"}},

        # Art items
        {"content": "The Mona Lisa was painted by Leonardo da Vinci.", "metadata": {"topic": "art"}},
        {"content": "Impressionism is characterized by small, visible brushstrokes.", "metadata": {"topic": "art"}},
        {"content": "Cubism was pioneered by Pablo Picasso and Georges Braque.", "metadata": {"topic": "art"}},
        {"content": "The Sistine Chapel ceiling was painted by Michelangelo.", "metadata": {"topic": "art"}},

        # Music items
        {"content": "Ludwig van Beethoven was a German composer and pianist.", "metadata": {"topic": "music"}},
        {"content": "Jazz originated in New Orleans in the late 19th century.", "metadata": {"topic": "music"}},
        {"content": "A symphony orchestra typically contains four sections of instruments.", "metadata": {"topic": "music"}},
        {"content": "The Beatles were an English rock band formed in Liverpool.", "metadata": {"topic": "music"}},

        # Cross-topic items (harder to categorize)
        {"content": "Computer algorithms can compose music similar to classical composers.", "metadata": {"topic": "computer_music"}},
        {"content": "Digital art combines traditional art techniques with computer technology.", "metadata": {"topic": "art_computer"}},
        {"content": "Historical data science analyzes digital archives to discover patterns.", "metadata": {"topic": "history_science"}},
        {"content": "The mathematics of music involves patterns, rhythms, and frequencies.", "metadata": {"topic": "science_music"}},
    ]

    # Add all items to both memory systems
    print("Adding test items to both memory systems...")
    for item in test_items:
        std_id = std_memory.add(content=item["content"], metadata=item["metadata"])
        enh_id = enh_memory.add(content=item["content"], metadata=item["metadata"])

    print(f"Added {len(test_items)} items to memory")

    # Test queries - each one has different nuances that test retrieval quality
    test_queries = [
        "How does programming work?",
        "Tell me about algorithms",
        "What are some scientific theories?",
        "Who were important historical figures?",
        "Tell me about famous paintings",
        "What types of music exist?",
        "How is technology used in art?",
        "What's the relationship between science and music?",
        "How has computing changed historical research?",
        "Can computers be creative with music?",
    ]

    # Compare results from both memory systems
    print("\nComparing retrieval results:")
    for query in test_queries:
        print(f"\nQuery: '{query}'")

        # Get results from standard memory
        std_results = std_memory.retrieve(query, top_k=3)

        print("Standard Memory Results:")
        for i, result in enumerate(std_results):
            print(f"  {i+1}. [{result['similarity']:.3f}] {result['content']}")

        # Get results from enhanced memory
        enh_results = enh_memory.retrieve(query, top_k=3)

        print("Enhanced Memory Results:")
        for i, result in enumerate(enh_results):
            similarity = result['similarity']
            level = result.get('level', 'Base')
            if level != 'Base':
                level_info = f" (Level {level})"
            else:
                level_info = ""

            print(f"  {i+1}. [{similarity:.3f}{level_info}] {result['content']}")

    # Clean up
    std_memory.cleanup()
    enh_memory.cleanup()
    print("\nTest complete")

if __name__ == "__main__":
    test_enhanced_retrieval()