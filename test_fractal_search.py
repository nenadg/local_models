from memory_manager import MemoryManager
import re

def test_fractal_search_effectiveness():
    """
    Demonstrate fractal search capabilities
    """
    # Setup memory manager with fractal enabled
    memory_manager = MemoryManager(
        memory_dir="./memory",
        fractal_enabled=True,
        max_fractal_levels=3
    )

    # Explicitly set fractal_enabled for the store
    store = memory_manager._get_user_store("test_user")
    store.fractal_enabled = True
    store.max_fractal_levels = 3

    # Ensure completely unique texts with intentional variations
    unique_texts = [
        "Machine learning explores artificial intelligence techniques",
        "AI technologies transform modern computational approaches",
        "Neural networks model complex cognitive processing systems",
        "Deep learning algorithms analyze massive interconnected datasets",
        "Bosnian Convertible Mark represents Bosnia's national currency",
        "The BAM monetary unit was established in late 1990s",
        "Euro exchange rates significantly impact BAM valuation",
        "International financial markets monitor Bosnian monetary performance",
        #"Fractal geometry",
        "rounding in mathematics"
    ]

    # Add texts with embeddings
    for text in unique_texts:
        embedding = memory_manager.generate_embedding(text)
        result = store.add(text, embedding)
        print(f"[TEST] Added document: {text[:50]}... Result: {result}")

    # Print diagnostics after adding documents
    print("\nFractal Diagnostic After Adding Documents:")
    store.print_fractal_embedding_diagnostics()

    # Test search with multiple queries
    test_queries = [
        # "machine learning techniques",
        "bosnian currency to eur peg",
        # "ai computational approaches",
        # "Fractal geometry",
        "what is the result of 124+5215.5 without rounding"
    ]

    for query in test_queries:
        print(f"\nSearch Query: '{query}'")
        query_embedding = memory_manager.generate_embedding(query)

        # Perform base and fractal searches
        base_results = store.enhanced_fractal_search(query_embedding, top_k=5, multi_level_search=False)
        fractal_results = store.enhanced_fractal_search(query_embedding, top_k=5, multi_level_search=True)

        print("Base Search Results:")
        for r in base_results:
            print(f"- {r['text'][:50]} (Similarity: {r['similarity']:.4f})")

        print("\nFractal Search Results:")
        for r in fractal_results:
            print(f"- {r['text'][:50]} (Similarity: {r['similarity']:.4f}, Level: {r.get('level', 0)})")


if __name__ == "__main__":
    test_fractal_search_effectiveness()