"""
Enhanced Memory System Test

This script tests the memory-enhanced chat system capabilities, including:
1. Memory addition and retrieval
2. Enhanced (formerly "fractal") embedding visualization
3. Enhanced retrieval quality testing
4. MCP content handling

Usage:
    python memory_test.py --test [all|memory|enhanced|visualization|mcp]
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the unified memory system
try:
    from unified_memory import MemoryManager
    from mcp_handler import MCPHandler
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)

def test_memory_add_retrieval(output_dir="./test_output"):
    """Test basic memory add and retrieval functionality"""
    print("\n========== TESTING MEMORY ADD/RETRIEVAL ==========")
    
    # Create test directory
    memory_dir = os.path.join(output_dir, "test_memory")
    os.makedirs(memory_dir, exist_ok=True)
    
    # Initialize memory manager
    memory = MemoryManager(
        storage_path=memory_dir,
        embedding_dim=384,
        enable_enhanced_embeddings=True,
        max_enhancement_levels=3,
        auto_save=True
    )
    
    # Define a simple embedding function for testing
    def simple_embedding(text):
        # Create deterministic but unique embedding based on text
        hash_val = sum(ord(c) for c in text)
        np.random.seed(hash_val)
        embedding = np.random.randn(384)
        return embedding / np.linalg.norm(embedding)
    
    # Batch version for efficiency testing
    def batch_embedding(texts):
        return [simple_embedding(text) for text in texts]
    
    # Set the embedding function
    memory.set_embedding_function(simple_embedding, batch_embedding)
    
    # Add test items
    print("\nAdding test items to memory...")
    items = [
        {"content": "The capital of France is Paris.", "metadata": {"source": "test", "category": "geography"}},
        {"content": "Python is a programming language.", "metadata": {"source": "test", "category": "programming"}},
        {"content": "Water boils at 100 degrees Celsius at sea level.", "metadata": {"source": "test", "category": "science"}},
        {"content": "Machine learning models can recognize patterns in data.", "metadata": {"source": "test", "category": "ai"}},
        {"content": "The Earth completes one orbit around the Sun in 365.25 days.", "metadata": {"source": "test", "category": "astronomy"}},
        {"content": "Shakespeare wrote Romeo and Juliet.", "metadata": {"source": "test", "category": "literature"}},
        {"content": "Neural networks are inspired by the human brain.", "metadata": {"source": "test", "category": "ai"}},
        {"content": "JavaScript is used for web development.", "metadata": {"source": "test", "category": "programming"}},
    ]
    
    # Add items individually
    item_ids = []
    for item in items[:4]:  # Add first 4 items individually
        item_id = memory.add(content=item["content"], metadata=item["metadata"])
        item_ids.append(item_id)
        print(f"Added item with ID: {item_id}")
    
    # Add items in bulk
    bulk_items = items[4:]  # Add remaining items in bulk
    bulk_item_dicts = []
    for item in bulk_items:
        bulk_item_dicts.append({
            "content": item["content"],
            "metadata": item["metadata"]
        })
    
    bulk_ids = memory.add_bulk(bulk_item_dicts)
    item_ids.extend(bulk_ids)
    print(f"Added {len(bulk_ids)} items in bulk: {bulk_ids}")
    
    # Test basic retrieval
    print("\nTesting basic retrieval...")
    query = "Tell me about planets"
    results = memory.retrieve(query, top_k=2)
    print(f"Query: '{query}'")
    for result in results:
        print(f"- [{result['similarity']:.3f}] {result['content']}")
    
    # Test category filtering
    print("\nTesting metadata filtering...")
    results = memory.retrieve("What is programming?", top_k=2, metadata_filter={"category": "programming"})
    print(f"Query with filter (category='programming'):")
    for result in results:
        print(f"- [{result['similarity']:.3f}] {result['content']}")
    
    # Test getting item by ID
    print("\nTesting get by ID...")
    if item_ids:
        item = memory.get(item_ids[0])
        print(f"Retrieved item: {item}")
    
    # Test updating item
    print("\nTesting update...")
    if item_ids:
        updated = memory.update(item_ids[0], {"metadata": {"updated": True, "importance": "high"}})
        print(f"Update success: {updated}")
        item = memory.get(item_ids[0])
        print(f"Updated item: {item}")
    
    # Test format for context
    print("\nTesting format for context...")
    query = "Tell me about science"
    results = memory.retrieve(query, top_k=3)
    context = memory.format_for_context(results, query)
    print(f"Formatted context:\n{context}")
    
    # Test removing an item
    print("\nTesting remove...")
    if item_ids:
        removed = memory.remove(item_ids[0])
        print(f"Removal success: {removed}")
        item = memory.get(item_ids[0])
        print(f"Item after removal: {item}")
    
    # Get memory statistics
    print("\nMemory statistics:")
    stats = memory.get_stats()
    for key, value in stats.items():
        if not isinstance(value, dict):
            print(f"- {key}: {value}")
    
    # Final cleanup
    memory.cleanup()
    print("\nMemory add/retrieval test complete.")
    return memory


def test_enhanced_retrieval(output_dir="./test_output"):
    """Test enhanced (fractal) embedding retrieval performance"""
    print("\n========== TESTING ENHANCED RETRIEVAL ==========")
    
    # Create test directories
    standard_dir = os.path.join(output_dir, "standard_memory")
    enhanced_dir = os.path.join(output_dir, "enhanced_memory")
    os.makedirs(standard_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)
    
    # Initialize memory managers - one with and one without enhanced embeddings
    std_memory = MemoryManager(
        storage_path=standard_dir,
        embedding_dim=384,
        enable_enhanced_embeddings=False
    )
    
    enh_memory = MemoryManager(
        storage_path=enhanced_dir,
        embedding_dim=384,
        enable_enhanced_embeddings=True,
        max_enhancement_levels=3,
        similarity_enhancement_factor=0.4
    )
    
    # Define domain-sensitive embedding function
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
    
    # Batch version for efficiency
    def batch_contextual_embedding(texts):
        return [contextual_embedding(text) for text in texts]
    
    # Set embedding functions
    std_memory.set_embedding_function(contextual_embedding, batch_contextual_embedding)
    enh_memory.set_embedding_function(contextual_embedding, batch_contextual_embedding)
    
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
    
    # Add test items to both memory systems
    print("\nAdding test items to both memory systems...")
    
    # Prepare items for bulk add
    bulk_items = []
    for item in test_items:
        bulk_items.append({
            "content": item["content"],
            "metadata": item["metadata"]
        })
    
    # Add all items to both memory systems
    std_ids = std_memory.add_bulk(bulk_items)
    enh_ids = enh_memory.add_bulk(bulk_items)
    
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
    
    # Initialize results storage for visualization
    comparison_data = {
        'queries': [],
        'standard_similarities': [],
        'enhanced_similarities': [],
        'similarity_gains': []
    }
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Get results from standard memory
        std_results = std_memory.retrieve(query, top_k=3)
        
        print("Standard Memory Results:")
        std_top_similarity = 0
        for i, result in enumerate(std_results):
            similarity = result['similarity']
            if i == 0:
                std_top_similarity = similarity
            print(f"  {i+1}. [{similarity:.3f}] {result['content']}")
        
        # Get results from enhanced memory
        enh_results = enh_memory.retrieve(query, top_k=3)
        
        print("Enhanced Memory Results:")
        enh_top_similarity = 0
        for i, result in enumerate(enh_results):
            similarity = result['similarity']
            if i == 0:
                enh_top_similarity = similarity
            level = result.get('level', 'Base')
            if level != 'Base':
                level_info = f" (Level {level})"
            else:
                level_info = ""
                
            print(f"  {i+1}. [{similarity:.3f}{level_info}] {result['content']}")
        
        # Store comparison data for visualization
        comparison_data['queries'].append(query.split()[0] + "...")  # Shortened for display
        comparison_data['standard_similarities'].append(std_top_similarity)
        comparison_data['enhanced_similarities'].append(enh_top_similarity)
        if std_top_similarity > 0:
            similarity_gain = (enh_top_similarity - std_top_similarity) / std_top_similarity * 100
        else:
            similarity_gain = 0 if enh_top_similarity == 0 else 100  # 100% improvement if went from 0 to something
        comparison_data['similarity_gains'].append(similarity_gain)
    
    # Create a comparison visualization
    plt.figure(figsize=(12, 6))
    
    # Bar chart showing similarity scores for both methods
    x = np.arange(len(test_queries))
    width = 0.35
    
    plt.bar(x - width/2, comparison_data['standard_similarities'], width, label='Standard')
    plt.bar(x + width/2, comparison_data['enhanced_similarities'], width, label='Enhanced')
    
    plt.xlabel('Query')
    plt.ylabel('Top Result Similarity')
    plt.title('Enhanced vs Standard Embedding Retrieval Performance')
    plt.xticks(x, comparison_data['queries'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    comparison_path = os.path.join(viz_dir, f"retrieval_comparison_{timestr}.png")
    plt.savefig(comparison_path, dpi=300)
    print(f"\nSaved comparison visualization to {comparison_path}")
    
    # Create a second plot showing similarity gains
    plt.figure(figsize=(12, 6))
    colors = ['green' if gain > 0 else 'red' for gain in comparison_data['similarity_gains']]
    plt.bar(x, comparison_data['similarity_gains'], color=colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.xlabel('Query')
    plt.ylabel('Similarity Gain (%)')
    plt.title('Enhanced Embedding Similarity Gain Over Standard')
    plt.xticks(x, comparison_data['queries'], rotation=45, ha='right')
    plt.tight_layout()
    
    gain_path = os.path.join(viz_dir, f"similarity_gain_{timestr}.png")
    plt.savefig(gain_path, dpi=300)
    print(f"Saved similarity gain visualization to {gain_path}")
    
    # Close plots
    plt.close('all')
    
    # Clean up
    std_memory.cleanup()
    enh_memory.cleanup()
    print("\nEnhanced retrieval test complete.")


def visualize_enhanced_embeddings(memory, output_dir="./test_output"):
    """Create simplified visualizations of enhanced embeddings"""
    print("\n========== CREATING ENHANCED EMBEDDING VISUALIZATIONS ==========")
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Get embedding data
    print("\nExtracting embeddings for visualization...")
    
    # Check if memory contains items
    stats = memory.get_stats()
    if stats['active_items'] == 0:
        print("Memory store is empty.")
        return
    
    # Extract embeddings and metadata
    base_embeddings = []
    enhanced_embeddings = {1: [], 2: [], 3: []}
    metadata = []
    
    # Get available items
    max_items = min(100, stats['active_items'])
    item_indices = list(range(len(memory.items)))
    
    # Filter out deleted items
    active_indices = [i for i in item_indices if i not in memory.deleted_ids]
    
    # Limit to max_items
    if len(active_indices) > max_items:
        np.random.seed(42)  # For reproducibility
        selected_indices = np.random.choice(active_indices, size=max_items, replace=False)
    else:
        selected_indices = active_indices
    
    # Extract data
    for idx in selected_indices:
        item = memory.items[idx]
        
        # Extract base embedding
        base_embeddings.append(item.embedding)
        
        # Extract enhanced embeddings
        for level in range(1, 4):
            if hasattr(item, 'additional_embeddings') and level in item.additional_embeddings:
                enhanced_embeddings[level].append(item.additional_embeddings[level])
            else:
                # Use zeros as placeholder
                enhanced_embeddings[level].append(np.zeros_like(item.embedding))
        
        # Extract metadata
        meta = {
            'id': item.id,
            'content_preview': item.content[:50],
            'category': item.metadata.get('category', 'unknown'),
            'retrieval_count': item.metadata.get('retrieval_count', 0)
        }
        metadata.append(meta)
    
    # Verify we have data
    if not base_embeddings:
        print("No embeddings found for visualization.")
        return
    
    # Convert to numpy arrays
    base_embeddings = np.array(base_embeddings)
    for level in enhanced_embeddings:
        enhanced_embeddings[level] = np.array(enhanced_embeddings[level])
    
    print(f"Extracted embeddings for {len(base_embeddings)} items")
    
    # Create visualizations
    try:
        from sklearn.decomposition import PCA
        
        # Apply PCA to reduce dimensions for visualization
        print("\nApplying dimensionality reduction...")
        pca = PCA(n_components=2)
        
        # Get categories for coloring
        categories = [meta.get('category', 'unknown') for meta in metadata]
        unique_categories = sorted(set(categories))
        category_to_color = {cat: i for i, cat in enumerate(unique_categories)}
        
        # Create base embedding visualization
        plt.figure(figsize=(10, 8))
        reduced = pca.fit_transform(base_embeddings)
        
        plt.scatter(reduced[:, 0], reduced[:, 1], c=[category_to_color[cat] for cat in categories], 
                   cmap='tab10', alpha=0.8, s=100)
        
        # Add annotations for some points
        for i, meta in enumerate(metadata):
            if i % 5 == 0:  # Annotate every 5th point to avoid clutter
                plt.annotate(meta['category'][:3], (reduced[i, 0], reduced[i, 1]), 
                            fontsize=8, ha='center', va='center')
        
        plt.title('Base Embeddings PCA Visualization')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        # Add colorbar legend
        from matplotlib.colors import ListedColormap
        n_categories = len(unique_categories)
        cmap = plt.cm.get_cmap('tab10', n_categories)
        
        # Create a mapping for categories
        cbar = plt.colorbar(ticks=np.linspace(0, n_categories-1, n_categories)/n_categories + 0.5/n_categories)
        cbar.set_ticklabels(unique_categories)
        cbar.set_label('Category')
        
        # Save figure
        timestr = time.strftime("%Y%m%d-%H%M%S")
        base_path = os.path.join(viz_dir, f"base_embeddings_{timestr}.png")
        plt.savefig(base_path, dpi=300)
        print(f"Saved base embeddings visualization to {base_path}")
        
        # Create enhanced embedding visualizations
        for level in [1, 2, 3]:
            # Filter out zero embeddings
            non_zero = np.any(enhanced_embeddings[level] != 0, axis=1)
            
            if not np.any(non_zero):
                print(f"No level {level} embeddings found.")
                continue
                
            # Extract non-zero embeddings and corresponding metadata
            level_embeddings = enhanced_embeddings[level][non_zero]
            level_categories = [categories[i] for i, nz in enumerate(non_zero) if nz]
            
            if len(level_embeddings) < 2:
                print(f"Not enough level {level} embeddings for visualization.")
                continue
                
            # Apply PCA
            plt.figure(figsize=(10, 8))
            level_reduced = pca.fit_transform(level_embeddings)
            
            plt.scatter(level_reduced[:, 0], level_reduced[:, 1], 
                       c=[category_to_color[cat] for cat in level_categories], 
                       cmap='tab10', alpha=0.8, s=100)
            
            # Add annotations
            for i, cat in enumerate(level_categories):
                if i % 5 == 0:
                    plt.annotate(cat[:3], (level_reduced[i, 0], level_reduced[i, 1]), 
                                fontsize=8, ha='center', va='center')
            
            plt.title(f'Level {level} Enhanced Embeddings PCA Visualization')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            
            # Add colorbar
            cbar = plt.colorbar(ticks=np.linspace(0, n_categories-1, n_categories)/n_categories + 0.5/n_categories)
            cbar.set_ticklabels(unique_categories)
            cbar.set_label('Category')
            
            # Save figure
            level_path = os.path.join(viz_dir, f"level_{level}_embeddings_{timestr}.png")
            plt.savefig(level_path, dpi=300)
            print(f"Saved level {level} embeddings visualization to {level_path}")
        
        # Create a combined visualization showing transformation across levels
        print("\nCreating combined visualization of embedding transformations...")
        
        # Select a few sample points for clarity (one from each category if possible)
        max_samples = min(5, len(unique_categories))
        samples = []
        seen_categories = set()
        
        for i, cat in enumerate(categories):
            if cat not in seen_categories and len(seen_categories) < max_samples:
                samples.append(i)
                seen_categories.add(cat)
        
        # If we don't have enough samples, add more from any category
        while len(samples) < max_samples and len(samples) < len(categories):
            for i, _ in enumerate(categories):
                if i not in samples:
                    samples.append(i)
                    break
        
        # Create a multi-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Process each level
        for level_idx, level in enumerate([1, 2, 3]):
            ax = axes[level_idx]
            
            # Get base embeddings for selected samples
            sample_base = np.array([base_embeddings[i] for i in samples])
            sample_categories = [categories[i] for i in samples]
            
            # Get enhanced embeddings for selected samples
            sample_enhanced = []
            valid_samples = []
            valid_categories = []
            
            for i, idx in enumerate(samples):
                if level in memory.items[idx].additional_embeddings:
                    sample_enhanced.append(memory.items[idx].additional_embeddings[level])
                    valid_samples.append(idx)
                    valid_categories.append(sample_categories[i])
            
            # Skip if no valid enhanced embeddings
            if not sample_enhanced:
                ax.text(0.5, 0.5, f"No Level {level}\nEmbeddings", ha='center', va='center', fontsize=14)
                ax.set_title(f"Level {level} Transformation")
                ax.axis('off')
                continue
                
            # Convert to numpy array
            sample_enhanced = np.array(sample_enhanced)
            
            # Apply PCA to both sets together to maintain same space
            combined = np.vstack([sample_base, sample_enhanced])
            pca_combined = PCA(n_components=2)
            reduced_combined = pca_combined.fit_transform(combined)
            
            # Split back into base and enhanced
            reduced_base = reduced_combined[:len(sample_base)]
            reduced_enhanced = reduced_combined[len(sample_base):]
            
            # Plot base points
            for i, cat in enumerate(sample_categories):
                color = plt.cm.tab10(category_to_color[cat] / len(unique_categories))
                ax.scatter(reduced_base[i, 0], reduced_base[i, 1], color=color, s=100, alpha=0.7, marker='o', label=f"{cat} (Base)")
            
            # Plot enhanced points
            for i, cat in enumerate(valid_categories):
                color = plt.cm.tab10(category_to_color[cat] / len(unique_categories))
                ax.scatter(reduced_enhanced[i, 0], reduced_enhanced[i, 1], color=color, s=100, alpha=0.7, marker='s', label=f"{cat} (Level {level})")
                
                # Draw connection line
                ax.plot([reduced_base[i, 0], reduced_enhanced[i, 0]], 
                       [reduced_base[i, 1], reduced_enhanced[i, 1]], 
                       color=color, alpha=0.5, linestyle='--')
                
                # Add arrow to show direction
                ax.annotate("", xy=(reduced_enhanced[i, 0], reduced_enhanced[i, 1]), 
                           xytext=(reduced_base[i, 0], reduced_base[i, 1]),
                           arrowprops=dict(arrowstyle="->", color=color, alpha=0.7))
            
            ax.set_title(f"Level {level} Transformation")
            
            # Add legend only to the last plot
            if level_idx == 2:
                # Custom legend for categories (deduped)
                handles = []
                labels = []
                seen = set()
                
                for cat in unique_categories:
                    if cat in seen_categories and cat not in seen:
                        color = plt.cm.tab10(category_to_color[cat] / len(unique_categories))
                        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10))
                        labels.append(cat)
                        seen.add(cat)
                
                # Add to the bottom plot
                ax.legend(handles, labels, title="Categories", loc='upper right')
        
        # Add overall title
        plt.suptitle("Enhanced Embedding Transformations Across Levels", fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        transform_path = os.path.join(viz_dir, f"embedding_transformations_{timestr}.png")
        plt.savefig(transform_path, dpi=300)
        print(f"Saved transformation visualization to {transform_path}")
        
        # Close all figures
        plt.close('all')
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nEnhanced embedding visualization complete.")


def test_mcp_functionality(output_dir="./test_output"):
    """Test MCP functionality for content generation and file handling"""
    print("\n========== TESTING MCP FUNCTIONALITY ==========")
    
    # Create test directory
    mcp_dir = os.path.join(output_dir, "mcp_test")
    os.makedirs(mcp_dir, exist_ok=True)
    
    # Initialize MCP handler
    mcp = MCPHandler(output_dir=mcp_dir, allow_shell_commands=True)
    
    print("\nTesting MCP Handler functionality...")
    
    # Test basic user command extraction
    user_input = "Create a Python script that calculates the Fibonacci sequence @{fibonacci.py}"
    print(f"\nOriginal input: '{user_input}'")
    cleaned, commands = mcp.extract_mcp_from_user_input(user_input)
    print(f"Cleaned input: '{cleaned}'")
    print(f"Extracted commands: {commands}")
    
    # Test MCP block extraction from model response
    test_response = """Here's a Python script that calculates Fibonacci numbers:

>>> FILE: fibonacci.py
def fibonacci(n):
    # Calculate the Fibonacci sequence up to n terms.
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib

if __name__ == "__main__":
    n = 10
    result = fibonacci(n)
    print(f"Fibonacci sequence up to {n} terms: {result}")
<<<

You can run this script with `python fibonacci.py`."""
    
    print("\nTest response:")
    print("-" * 40)
    print(test_response)
    print("-" * 40)
    
    cleaned, mcp_blocks = mcp.extract_mcp_blocks(test_response)
    
    print("\nCleaned output:")
    print("-" * 40)
    print(cleaned)
    print("-" * 40)
    
    print("Extracted MCP blocks:")
    for filename, content in mcp_blocks.items():
        print(f"\nFile: {filename}")
        print("-" * 20)
        print(content)
        print("-" * 20)
    
    # Test saving to files
    print("\nTesting saving MCP blocks to files...")
    
    results = mcp.save_mcp_blocks(mcp_blocks)
    
    for filename, success in results.items():
        full_path = os.path.join(mcp_dir, filename)
        if success and os.path.exists(full_path):
            with open(full_path, 'r') as f:
                content = f.read()
            print(f"Successfully saved {filename}:")
            print(f"Content (first 50 chars): {content[:50]}...")
        else:
            print(f"Failed to save {filename}")
    
    # Test streaming token handling
    print("\nTesting streaming token handling...")
    
    # Simulate a stream of tokens
    tokens = [
        "Here's", " a", " simple", " HTML", " page", ":", "\n\n",
        ">", ">", ">", " ", "F", "I", "L", "E", ":", " ", "t", "e", "s", "t", ".", "h", "t", "m", "l", "\n",
        "<", "!", "D", "O", "C", "T", "Y", "P", "E", " ", "h", "t", "m", "l", ">", "\n",
        "<", "h", "t", "m", "l", ">", "\n",
        "<", "h", "e", "a", "d", ">", "\n",
        " ", " ", "<", "t", "i", "t", "l", "e", ">", "T", "e", "s", "t", " ", "P", "a", "g", "e", "<", "/", "t", "i", "t", "l", "e", ">", "\n",
        "<", "/", "h", "e", "a", "d", ">", "\n",
        "<", "b", "o", "d", "y", ">", "\n",
        " ", " ", "<", "h", "1", ">", "H", "e", "l", "l", "o", " ", "W", "o", "r", "l", "d", "<", "/", "h", "1", ">", "\n",
        "<", "/", "b", "o", "d", "y", ">", "\n",
        "<", "/", "h", "t", "m", "l", ">", "\n",
        "<", "<", "<", "\n\n",
        "You", " can", " open", " this", " file", " in", " a", " browser", "."
    ]
    
    complete_text = ""
    displayed_text = ""
    mcp_buffer = ""
    
    print("\nSimulating token streaming (only showing displayed tokens):")
    print("-" * 40)
    for token in tokens:
        complete_text += token
        display_token, mcp_buffer = mcp.process_streaming_token(token, mcp_buffer)
        if display_token:
            displayed_text += display_token
            print(display_token, end="", flush=True)
    
    print("\n" + "-" * 40)
    
    # Finalize streaming to process any MCP blocks
    finalized = mcp.finalize_streaming(complete_text)
    
    print("\nFinalized text:")
    print("-" * 40)
    print(finalized)
    print("-" * 40)
    
    # Check if the streamed file was created
    streamed_file_path = os.path.join(mcp_dir, "test.html")
    if os.path.exists(streamed_file_path):
        with open(streamed_file_path, 'r') as f:
            content = f.read()
        print(f"\nStreamed file content:")
        print("-" * 20)
        print(content)
        print("-" * 20)
    
    # Test help text
    print("\nMCP Help Text:")
    print(mcp.get_help_text())
    
    print("\nMCP test complete.")


def main():
    """Main function to run all tests"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the memory-enhanced chat system")
    parser.add_argument("--test", type=str, default="all", 
                      choices=["all", "memory", "enhanced", "visualization", "mcp"],
                      help="Which test to run")
    parser.add_argument("--output-dir", type=str, default="./test_output",
                      help="Directory for test output")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print start time
    start_time = time.time()
    print(f"Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    memory = None
    
    try:
        # Run selected tests
        if args.test in ["all", "memory"]:
            memory = test_memory_add_retrieval(args.output_dir)
            
        if args.test in ["all", "enhanced"]:
            test_enhanced_retrieval(args.output_dir)
            
        if args.test in ["all", "visualization"]:
            # If memory isn't created yet, create it for visualization
            if memory is None:
                memory = test_memory_add_retrieval(args.output_dir)
            
            # Create visualizations
            visualize_enhanced_embeddings(memory, args.output_dir)
            
        if args.test in ["all", "mcp"]:
            test_mcp_functionality(args.output_dir)
        
    except Exception as e:
        print(f"Error during tests: {e}")
        import traceback
        traceback.print_exc()
    
    # Print completion time
    elapsed = time.time() - start_time
    print(f"\nTests completed in {elapsed:.2f} seconds")
    print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output files are in {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()