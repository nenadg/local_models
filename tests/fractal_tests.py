"""
Fractal Memory Testing Module for TinyLlama Chat System

This module provides comprehensive testing functions for:
1. Adding and retrieving from fractal memory
2. Visualizing fractal embeddings
3. MCP integration with fractal memory

Usage:
    python fractal_tests.py [--test all|add|retrieve|visualize|mcp]
"""

import os
import argparse
import random
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

import sys
# insert the project root (one level up) at the front of sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



# Import system components based on availability
try:
    from unified_memory import UnifiedMemoryManager
    from mcp_handler import MCPHandler
    from local_ai import TinyLlamaChat
except ImportError:
    print("Warning: Some system components couldn't be imported.")
    print("Make sure you're running this from the project root directory.")


class FractalMemoryTester:
    """Test class for fractal memory functionality in TinyLlama Chat system."""
    
    def __init__(self, memory_dir="./test_memory", fractal_enabled=True, max_fractal_levels=3):
        """Initialize the tester with specified parameters."""
        self.memory_dir = memory_dir
        self.fractal_enabled = fractal_enabled
        self.max_fractal_levels = max_fractal_levels
        
        # Create test directories
        os.makedirs(memory_dir, exist_ok=True)
        
        # Initialize a simple embedding function for testing
        self.embedding_dim = 384
        
        # Initialize memory manager
        self.memory_manager = UnifiedMemoryManager(
            storage_path=memory_dir,
            embedding_function=self.simple_embedding_function,
            embedding_dim=self.embedding_dim,
            use_fractal=fractal_enabled,
            max_fractal_levels=max_fractal_levels,
            auto_save=True,
            enable_entity_separation=True
        )
        
        # Initialize MCP handler for MCP tests
        self.output_dir = "./test_output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.mcp_handler = MCPHandler(output_dir=self.output_dir, allow_shell_commands=False)
        
        print(f"[INIT] Test environment ready with fractal_enabled={fractal_enabled}")
        print(f"[INIT] Memory directory: {memory_dir}")
        print(f"[INIT] MCP output directory: {self.output_dir}")
    
    def get_time(self):
        """Get a formatted timestamp for logging."""
        import datetime
        return datetime.datetime.now().strftime("[%d/%m/%y %H:%M:%S]")
    
    def simple_embedding_function(self, text):
        """
        Create a deterministic but varied embedding vector based on text content.
        This provides more realistic embeddings for testing.
        """
        # Use hash of text for deterministic randomness
        import hashlib
        text_hash = hashlib.md5(text.encode()).digest()
        seed = int.from_bytes(text_hash[:4], byteorder='little')
        np.random.seed(seed)
        
        # Base embedding with controlled randomness
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        
        # Add some content-specific patterns
        # Different text categories get different embedding patterns
        if "math" in text.lower() or any(op in text for op in ["+", "-", "*", "/"]):
            # Math-related texts
            embedding[:50] += 0.5
        elif "code" in text.lower() or "function" in text.lower():
            # Code-related texts
            embedding[50:100] += 0.5
        elif "history" in text.lower() or "ancient" in text.lower():
            # History-related texts
            embedding[100:150] += 0.5
        elif "science" in text.lower() or "physics" in text.lower():
            # Science-related texts
            embedding[150:200] += 0.5
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def test_add_to_fractal_memory(self, num_items=50, verbose=True):
        """
        Test adding items to fractal memory.
        
        Args:
            num_items: Number of test items to add
            verbose: Whether to print detailed progress
            
        Returns:
            Dictionary of test results
        """
        print(f"\n{self.get_time()} Starting fractal memory add test with {num_items} items...")
        
        # Sample content categories for testing
        categories = [
            "math", "science", "history", "literature", 
            "code", "geography", "art", "music"
        ]
        
        # Create test items
        test_items = []
        for i in range(num_items):
            # Choose a category
            category = random.choice(categories)
            
            # Create content with category-specific patterns
            if category == "math":
                content = f"Math fact #{i}: The {random.randint(1, 20)}th Fibonacci number is {random.randint(1, 1000)}."
            elif category == "science":
                content = f"Science fact #{i}: The atomic number of {random.choice(['hydrogen', 'helium', 'lithium', 'oxygen'])} is {random.randint(1, 118)}."
            elif category == "history":
                content = f"History fact #{i}: The {random.choice(['Roman', 'Greek', 'Egyptian', 'Chinese'])} civilization began around {random.randint(1, 5000)} BCE."
            elif category == "code":
                content = f"Code snippet #{i}: function calculate{i}() {{ return {random.randint(1, 100)}; }}"
            else:
                content = f"{category.capitalize()} fact #{i}: This is test item {i} in the {category} category."
            
            # Add metadata
            metadata = {
                "category": category,
                "test_id": i,
                "complexity": random.random()
            }
            
            test_items.append({
                "content": content,
                "metadata": metadata
            })
        
        # Add items to memory
        start_time = time.time()
        item_ids = []
        
        for i, item in enumerate(test_items):
            item_id = self.memory_manager.add(
                content=item["content"],
                metadata=item["metadata"],
                use_fractal=self.fractal_enabled
            )
            item_ids.append(item_id)
            
            if verbose and (i+1) % 10 == 0:
                print(f"  - Added {i+1}/{num_items} items...")
        
        # Get time took
        elapsed_time = time.time() - start_time
        
        # Verify items were added by checking memory stats
        stats = self.memory_manager.get_stats()
        success_rate = len([id for id in item_ids if id is not None]) / num_items
        
        # Print test results
        print(f"{self.get_time()} Test completed in {elapsed_time:.2f} seconds.")
        print(f"  - Items added: {stats['active_items']}/{num_items}")
        print(f"  - Success rate: {success_rate * 100:.1f}%")
        
        if self.fractal_enabled:
            # Display fractal stats
            fractal_stats = stats.get('fractal_stats', {})
            if fractal_stats:
                print("  - Fractal levels distribution:")
                for level, count in sorted([(k.split('_')[1], v) for k, v in fractal_stats.items() if k.startswith('level_')]):
                    print(f"    > Level {level}: {count} items")
        
        # Return test results
        return {
            "items_added": len([id for id in item_ids if id is not None]),
            "total_items": num_items,
            "success_rate": success_rate,
            "elapsed_time": elapsed_time,
            "item_ids": item_ids,
            "stats": stats
        }
    
    def test_retrieve_from_fractal_memory(self, num_queries=20, verbose=True):
        """
        Test retrieving items from fractal memory.
        
        Args:
            num_queries: Number of test queries to run
            verbose: Whether to print detailed progress
            
        Returns:
            Dictionary of test results
        """
        print(f"\n{self.get_time()} Starting fractal memory retrieval test with {num_queries} queries...")
        
        # Check if memory contains items
        stats = self.memory_manager.get_stats()
        if stats['active_items'] == 0:
            print("  - Error: No items in memory. Run test_add_to_fractal_memory first.")
            return {"error": "No items in memory"}
        
        # Sample query categories for testing
        categories = [
            "math", "science", "history", "literature", 
            "code", "geography", "art", "music"
        ]
        
        # Create test queries
        test_queries = []
        for i in range(num_queries):
            # Choose a category
            category = random.choice(categories)
            
            # Create query with category-specific patterns
            if category == "math":
                query = f"Tell me about the {random.randint(1, 20)}th Fibonacci number."
            elif category == "science":
                query = f"What is the atomic number of {random.choice(['hydrogen', 'helium', 'lithium', 'oxygen'])}?"
            elif category == "history":
                query = f"When did the {random.choice(['Roman', 'Greek', 'Egyptian', 'Chinese'])} civilization begin?"
            elif category == "code":
                query = f"Show me a function that calculates something."
            else:
                query = f"Tell me a fact about {category}."
            
            test_queries.append({
                "query": query,
                "category": category,
                "test_id": i
            })
        
        # Run queries
        start_time = time.time()
        query_results = []
        
        for i, query_info in enumerate(test_queries):
            query = query_info["query"]
            
            # Retrieve items with fractal search
            results = self.memory_manager.retrieve(
                query=query,
                top_k=5,
                min_similarity=0.1,  # Low threshold to ensure results
                use_fractal=self.fractal_enabled
            )
            
            query_results.append({
                "query": query,
                "category": query_info["category"],
                "results": results,
                "result_count": len(results)
            })
            
            if verbose and (i+1) % 5 == 0:
                print(f"  - Processed {i+1}/{num_queries} queries...")
        
        # Get time took
        elapsed_time = time.time() - start_time
        
        # Calculate statistics
        avg_results = sum(res["result_count"] for res in query_results) / max(1, len(query_results))
        zero_results = sum(1 for res in query_results if res["result_count"] == 0)
        
        # Get cross-level stats if available
        cross_level_matches = 0
        for query_result in query_results:
            for result in query_result.get("results", []):
                if "found_in_levels" in result and len(result["found_in_levels"]) > 1:
                    cross_level_matches += 1
        
        # Print test results
        print(f"{self.get_time()} Retrieval test completed in {elapsed_time:.2f} seconds.")
        print(f"  - Average results per query: {avg_results:.1f}")
        print(f"  - Queries with zero results: {zero_results}/{num_queries} ({zero_results/num_queries*100:.1f}%)")
        
        if self.fractal_enabled:
            print(f"  - Cross-level matches found: {cross_level_matches}")
        
        # Return test results
        return {
            "queries_processed": num_queries,
            "avg_results": avg_results,
            "zero_results": zero_results,
            "elapsed_time": elapsed_time,
            "cross_level_matches": cross_level_matches,
            "query_results": query_results
        }
    
    def test_visualize_fractal_memory(self, max_items=500, interactive=False):
        """
        Visualize fractal embeddings using dimensionality reduction techniques.
        
        Args:
            max_items: Maximum number of items to visualize
            interactive: Whether to display interactive plots
            
        Returns:
            Dictionary of visualization results
        """
        print(f"\n{self.get_time()} Visualizing fractal memory embeddings...")
        
        # Check if memory contains items
        stats = self.memory_manager.get_stats()
        if stats['active_items'] == 0:
            print("  - Error: No items in memory. Run test_add_to_fractal_memory first.")
            return {"error": "No items in memory"}
        
        # Get embeddings and metadata for all levels
        base_embeddings = []
        fractal_embeddings = {level: [] for level in range(1, self.max_fractal_levels + 1)}
        metadata_categories = []
        
        # Limit to max_items
        item_count = min(stats['active_items'], max_items)
        print(f"  - Collecting embeddings for {item_count} items...")
        
        # Extract embeddings for visualization
        for i, item in enumerate(self.memory_manager.items[:item_count]):
            if i in self.memory_manager.deleted_ids:
                continue
                
            # Store base embedding
            base_embeddings.append(item.embedding)
            
            # Store category for coloring
            category = item.metadata.get('category', 'unknown')
            metadata_categories.append(category)
            
            # Store fractal embeddings for each level
            for level in range(1, self.max_fractal_levels + 1):
                if level in item.fractal_embeddings:
                    fractal_embeddings[level].append(item.fractal_embeddings[level])
                else:
                    # Placeholder for missing embeddings
                    fractal_embeddings[level].append(np.zeros(self.embedding_dim))
        
        # Convert to numpy arrays
        base_embeddings = np.array(base_embeddings)
        for level in fractal_embeddings:
            fractal_embeddings[level] = np.array(fractal_embeddings[level])
        
        # Create unique categories and color mapping
        unique_categories = sorted(set(metadata_categories))
        category_to_color = {cat: i for i, cat in enumerate(unique_categories)}
        color_indices = [category_to_color.get(cat, 0) for cat in metadata_categories]
        
        # Initialize plots
        n_levels = 1 + len([level for level in fractal_embeddings if len(fractal_embeddings[level]) > 0])
        fig, axes = plt.subplots(n_levels, 2, figsize=(15, 5 * n_levels))
        
        # If only one level, make axes 2D
        if n_levels == 1:
            axes = np.array([axes])
        
        # Create colormap
        cmap = plt.cm.tab10
        
        # Prepare title and output path
        timestr = time.strftime("%Y%m%d-%H%M%S")
        output_dir = "./visualizations"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"fractal_viz_{timestr}.png")
        
        # Process each level
        level_idx = 0
        
        # First visualize base embeddings
        print(f"  - Visualizing base embeddings...")
        
        # PCA for base embeddings
        if len(base_embeddings) > 1:  # Need at least 2 samples for PCA
            pca = PCA(n_components=2)
            base_pca = pca.fit_transform(base_embeddings)
            axes[level_idx, 0].scatter(base_pca[:, 0], base_pca[:, 1], c=color_indices, cmap=cmap, alpha=0.7)
            axes[level_idx, 0].set_title(f"PCA: Base Embeddings")
            axes[level_idx, 0].set_xlabel(f"PCA-1 (var: {pca.explained_variance_ratio_[0]:.2f})")
            axes[level_idx, 0].set_ylabel(f"PCA-2 (var: {pca.explained_variance_ratio_[1]:.2f})")
        else:
            axes[level_idx, 0].text(0.5, 0.5, "Not enough samples for PCA", 
                                    ha='center', va='center', fontsize=12)
            
        # t-SNE for base embeddings
        if len(base_embeddings) > 1:  # Need at least 2 samples for t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(base_embeddings)-1))
            base_tsne = tsne.fit_transform(base_embeddings)
            axes[level_idx, 1].scatter(base_tsne[:, 0], base_tsne[:, 1], c=color_indices, cmap=cmap, alpha=0.7)
            axes[level_idx, 1].set_title(f"t-SNE: Base Embeddings")
        else:
            axes[level_idx, 1].text(0.5, 0.5, "Not enough samples for t-SNE", 
                                     ha='center', va='center', fontsize=12)
            
        level_idx += 1
        
        # Then visualize fractal levels
        for level in range(1, self.max_fractal_levels + 1):
            if level not in fractal_embeddings or len(fractal_embeddings[level]) == 0:
                continue
                
            print(f"  - Visualizing level {level} embeddings...")
            level_data = fractal_embeddings[level]
            
            # Skip if all zeros (placeholder embeddings)
            if np.all(level_data == 0):
                continue
                
            # Filter out zero embeddings
            non_zero_indices = ~np.all(level_data == 0, axis=1)
            if sum(non_zero_indices) < 2:  # Need at least 2 non-zero embeddings
                continue
                
            filtered_data = level_data[non_zero_indices]
            filtered_colors = np.array(color_indices)[non_zero_indices]
            
            # PCA for this level
            pca = PCA(n_components=2)
            level_pca = pca.fit_transform(filtered_data)
            axes[level_idx, 0].scatter(level_pca[:, 0], level_pca[:, 1], c=filtered_colors, cmap=cmap, alpha=0.7)
            axes[level_idx, 0].set_title(f"PCA: Level {level} Embeddings")
            axes[level_idx, 0].set_xlabel(f"PCA-1 (var: {pca.explained_variance_ratio_[0]:.2f})")
            axes[level_idx, 0].set_ylabel(f"PCA-2 (var: {pca.explained_variance_ratio_[1]:.2f})")
            
            # t-SNE for this level
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(filtered_data)-1))
            level_tsne = tsne.fit_transform(filtered_data)
            axes[level_idx, 1].scatter(level_tsne[:, 0], level_tsne[:, 1], c=filtered_colors, cmap=cmap, alpha=0.7)
            axes[level_idx, 1].set_title(f"t-SNE: Level {level} Embeddings")
            
            level_idx += 1
        
        # Add legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=cmap(category_to_color[cat]/len(unique_categories)), 
                              markersize=10, label=cat) 
                  for cat in unique_categories]
        fig.legend(handles=handles, title="Categories", loc="lower center", 
                   bbox_to_anchor=(0.5, 0), ncol=min(5, len(unique_categories)))
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1 + 0.02*min(5, len(unique_categories)))
        
        # Save the visualization
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"{self.get_time()} Visualization saved to {output_path}")
        
        if interactive:
            plt.show()
        else:
            plt.close()
        
        # Return visualization results
        return {
            "visualization_path": output_path,
            "categories": unique_categories,
            "item_count": len(base_embeddings),
            "levels_visualized": level_idx
        }
    
    def test_mcp_with_fractal_memory(self, verbose=True):
        """
        Test MCP integration with fractal memory.
        
        Args:
            verbose: Whether to print detailed progress
            
        Returns:
            Dictionary of test results
        """
        print(f"\n{self.get_time()} Testing MCP integration with fractal memory...")
        
        # Test MCP file generation
        test_output = """
>>> FILE: code_example.py
def calculate_fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# Calculate the 10th Fibonacci number
result = calculate_fibonacci(10)
print(f"The 10th Fibonacci number is {result}")
<<<

>>> FILE: math_fact.md
# Mathematical Facts

The Fibonacci sequence is defined by the recurrence relation:
F(n) = F(n-1) + F(n-2)

With initial conditions:
F(0) = 0, F(1) = 1

The sequence begins: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...
<<<
"""
        
        # Extract MCP blocks
        cleaned_content, mcp_blocks = self.mcp_handler.extract_mcp_blocks(test_output)
        
        # Save MCP blocks to files
        save_results = self.mcp_handler.save_mcp_blocks(mcp_blocks)
        
        if verbose:
            print(f"  - Extracted {len(mcp_blocks)} MCP blocks:")
            for filename in mcp_blocks:
                print(f"    > {filename}: {save_results.get(filename, False)}")
        
        # Test adding MCP content to memory
        add_results = {}
        for filename, content in mcp_blocks.items():
            # Create metadata
            metadata = {
                "source": "mcp",
                "filename": filename,
                "filetype": os.path.splitext(filename)[1][1:],
                "timestamp": time.time()
            }
            
            # Add to memory
            item_id = self.memory_manager.add(
                content=content,
                metadata=metadata,
                use_fractal=self.fractal_enabled
            )
            
            add_results[filename] = item_id is not None
            
            if verbose:
                print(f"  - Added to memory: {filename}: {add_results[filename]}")
        
        # Test retrieving MCP content
        retrieve_results = {}
        for filename, content in mcp_blocks.items():
            # Create a query based on the file type
            filetype = os.path.splitext(filename)[1][1:]
            if filetype == "py":
                query = "Python code for calculating Fibonacci numbers"
            elif filetype == "md":
                query = "Mathematical facts about the Fibonacci sequence"
            else:
                query = f"Content from {filename}"
            
            # Retrieve
            results = self.memory_manager.retrieve(
                query=query,
                top_k=3,
                min_similarity=0.1
            )
            
            # Check if we found the file
            found = False
            for result in results:
                if filename in result.get("metadata", {}).get("filename", ""):
                    found = True
                    break
            
            retrieve_results[filename] = {
                "found": found,
                "results_count": len(results)
            }
            
            if verbose:
                print(f"  - Retrieved {filename}: {found} (from {len(results)} results)")
        
        # Return test results
        return {
            "mcp_blocks": len(mcp_blocks),
            "save_results": save_results,
            "add_results": add_results,
            "retrieve_results": retrieve_results
        }

    def run_all_tests(self):
        """Run all test functions and return comprehensive results."""
        print(f"\n{self.get_time()} Running all fractal memory tests...")
        
        results = {}
        
        # Run tests
        results["add"] = self.test_add_to_fractal_memory(num_items=50)
        results["retrieve"] = self.test_retrieve_from_fractal_memory(num_queries=20)
        results["visualize"] = self.test_visualize_fractal_memory(max_items=200)
        results["mcp"] = self.test_mcp_with_fractal_memory()
        
        # Print summary
        print(f"\n{self.get_time()} All tests completed!")
        print(f"  - Items added: {results['add']['items_added']}/{results['add']['total_items']}")
        print(f"  - Average results per query: {results['retrieve']['avg_results']:.1f}")
        print(f"  - Visualization saved to: {results['visualize'].get('visualization_path', 'N/A')}")
        print(f"  - MCP blocks processed: {results['mcp']['mcp_blocks']}")
        
        return results
    
    def cleanup(self):
        """Clean up test resources."""
        print(f"\n{self.get_time()} Cleaning up test resources...")
        
        # Clean up memory manager
        if hasattr(self, "memory_manager") and hasattr(self.memory_manager, "cleanup"):
            self.memory_manager.cleanup()
            
        print(f"{self.get_time()} Cleanup complete!")


def plot_3d_fractal_visualization(memory_manager, max_items=200, level=1):
    """
    Create a 3D visualization of fractal embeddings with interactive rotation.
    
    Args:
        memory_manager: The memory manager instance
        max_items: Maximum number of items to visualize
        level: Fractal level to visualize (0 for base embeddings)
        
    Returns:
        Plot figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    print(f"Creating 3D visualization of fractal embeddings (level {level})...")
    
    # Check if memory contains items
    stats = memory_manager.get_stats()
    if stats['active_items'] == 0:
        print("  - Error: No items in memory.")
        return None
    
    # Get embeddings and metadata
    embeddings = []
    metadata_categories = []
    
    # Limit to max_items
    item_count = min(stats['active_items'], max_items)
    print(f"  - Collecting embeddings for {item_count} items...")
    
    # Extract embeddings for visualization
    for i, item in enumerate(memory_manager.items[:item_count]):
        if i in memory_manager.deleted_ids:
            continue
            
        # Get appropriate embedding based on level
        if level == 0:
            # Base embeddings
            embeddings.append(item.embedding)
        elif level in item.fractal_embeddings:
            # Fractal embeddings
            embeddings.append(item.fractal_embeddings[level])
        else:
            # Skip items without the requested embedding level
            continue
            
        # Store category for coloring
        category = item.metadata.get('category', 'unknown')
        metadata_categories.append(category)
    
    # Convert to numpy array
    embeddings = np.array(embeddings)
    
    # Create unique categories and color mapping
    unique_categories = sorted(set(metadata_categories))
    category_to_color = {cat: i for i, cat in enumerate(unique_categories)}
    color_indices = [category_to_color.get(cat, 0) for cat in metadata_categories]
    
    # Create colormap
    cmap = plt.cm.tab10
    
    # Perform PCA
    pca = PCA(n_components=3)
    points_3d = pca.fit_transform(embeddings)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    sc = ax.scatter(
        points_3d[:, 0], 
        points_3d[:, 1], 
        points_3d[:, 2],
        c=color_indices,
        cmap=cmap,
        s=50,
        alpha=0.7
    )
    
    # Label axes
    ax.set_xlabel(f'PCA-1 (var: {pca.explained_variance_ratio_[0]:.2f})')
    ax.set_ylabel(f'PCA-2 (var: {pca.explained_variance_ratio_[1]:.2f})')
    ax.set_zlabel(f'PCA-3 (var: {pca.explained_variance_ratio_[2]:.2f})')
    
    # Set title
    level_name = "Base" if level == 0 else f"Level {level}"
    ax.set_title(f'3D PCA Visualization of {level_name} Embeddings')
    
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=cmap(category_to_color[cat]/len(unique_categories)), 
                          markersize=10, label=cat) 
              for cat in unique_categories]
    ax.legend(handles=handles, title="Categories", loc="upper right")
    
    # Add explained variance info
    var_text = f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}"
    ax.text2D(0.05, 0.95, var_text, transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = "./visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"fractal_3d_viz_{level}_{timestr}.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    return fig


def main():
    """Main function to run tests based on command line arguments."""
    parser = argparse.ArgumentParser(description="Test fractal memory functionality")
    parser.add_argument("--test", type=str, default="all", 
                        choices=["all", "add", "retrieve", "visualize", "mcp", "3d"],
                        help="Which test to run")
    parser.add_argument("--items", type=int, default=50,
                        help="Number of items to add in test")
    parser.add_argument("--queries", type=int, default=20,
                        help="Number of queries to run in test")
    parser.add_argument("--fractal", type=bool, default=True,
                        help="Whether to enable fractal embeddings")
    parser.add_argument("--levels", type=int, default=3,
                        help="Number of fractal levels")
    parser.add_argument("--interactive", action="store_true",
                        help="Show interactive plots")
    parser.add_argument("--memory-dir", type=str, default="./test_memory",
                        help="Directory to store test memory")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = FractalMemoryTester(
        memory_dir=args.memory_dir,
        fractal_enabled=args.fractal,
        max_fractal_levels=args.levels
    )
    
    try:
        # Run selected test
        if args.test == "all":
            results = tester.run_all_tests()
        elif args.test == "add":
            results = tester.test_add_to_fractal_memory(num_items=args.items)
        elif args.test == "retrieve":
            results = tester.test_retrieve_from_fractal_memory(num_queries=args.queries)
        elif args.test == "visualize":
            results = tester.test_visualize_fractal_memory(max_items=args.items, interactive=args.interactive)
        elif args.test == "mcp":
            results = tester.test_mcp_with_fractal_memory()
        elif args.test == "3d":
            # First add some items if memory is empty
            stats = tester.memory_manager.get_stats()
            if stats['active_items'] == 0:
                tester.test_add_to_fractal_memory(num_items=args.items)
            
            # Create 3D visualization
            fig = plot_3d_fractal_visualization(
                tester.memory_manager, 
                max_items=args.items, 
                level=1  # Visualize level 1 by default
            )
            
            if args.interactive and fig is not None:
                plt.show()
    finally:
        # Clean up
        tester.cleanup()


if __name__ == "__main__":
    main()