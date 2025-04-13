"""
Visualization utilities for fractal embeddings in the local_models project.
Provides functionality for visualizing high-dimensional embeddings using dimensionality reduction.
"""

import numpy as np
import matplotlib
# Set the backend before importing pyplot
# Use 'Agg' for non-interactive environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Dict, List, Any, Optional, Tuple
import os

def visualize_fractal_embeddings(diagnostics: Dict[str, Any], title: str = "Fractal Embedding Variations",
                                save_path: Optional[str] = None):
    """
    Visualize fractal embedding variations using PCA to reduce dimensions.

    Args:
        diagnostics: Dictionary containing diagnostic information about fractal embeddings
                    (as returned by VectorStore.diagnostics_fractal_embeddings)
        title: Title for the visualization plot
        save_path: Optional path to save the visualization (e.g., "fractal_viz.png")
                   If None, the visualization will be saved to a default path

    Returns:
        Path to the saved visualization
    """
    plt.figure(figsize=(12, 6))

    # PCA for dimensionality reduction
    pca = PCA(n_components=2)

    for i, (base_emb, variations) in enumerate(
        zip(diagnostics["base_embeddings"], diagnostics["fractal_variations"])
    ):
        # Prepare data for PCA
        all_embeddings = [base_emb] + [v for k, v in variations.items()]
        all_embeddings_array = np.array(all_embeddings)
        pca_result = pca.fit_transform(all_embeddings_array)

        # Plot base embedding
        plt.scatter(
            pca_result[0, 0],
            pca_result[0, 1],
            marker='o',
            s=100,
            label=f'Base {i+1}' if i == 0 else None,
            color=f'C{i}',
            edgecolors='black'
        )

        # Plot variations with connecting lines to base
        for j, variation_coords in enumerate(pca_result[1:], 1):
            plt.scatter(
                variation_coords[0],
                variation_coords[1],
                marker='x',
                s=50,
                alpha=0.7,
                color=f'C{i}'
            )

            # Draw line from base to variation
            plt.plot(
                [pca_result[0, 0], variation_coords[0]],
                [pca_result[0, 1], variation_coords[1]],
                'k-',
                alpha=0.3,
                linewidth=0.5
            )

    plt.title(title)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Set default save path if none provided
    if not save_path:
        os.makedirs("./visualizations", exist_ok=True)
        save_path = f"./visualizations/fractal_embeddings_{int(np.random.rand() * 10000)}.png"

    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to {save_path}")
    plt.close()

    return save_path

def visualize_embedding_similarities(similarities: List[float], labels: List[str] = None,
                                    title: str = "Embedding Similarities", save_path: Optional[str] = None):
    """
    Visualize similarities between a query embedding and multiple result embeddings.

    Args:
        similarities: List of similarity scores (typically cosine similarities)
        labels: Optional list of labels for each similarity score
        title: Title for the visualization plot
        save_path: Optional path to save the visualization

    Returns:
        Path to the saved visualization
    """
    plt.figure(figsize=(10, 6))

    # Default labels if none provided
    if not labels:
        labels = [f"Embedding {i+1}" for i in range(len(similarities))]

    # Sort by similarity for better visualization
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_similarities = [similarities[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    # Limit to top 15 for readability
    if len(sorted_similarities) > 15:
        sorted_similarities = sorted_similarities[:15]
        sorted_labels = sorted_labels[:15]

    # Create bar plot
    bars = plt.barh(range(len(sorted_similarities)), sorted_similarities, height=0.6)

    # Color bars by similarity
    for i, bar in enumerate(bars):
        # Generate color from red to green based on similarity
        color = plt.cm.RdYlGn(sorted_similarities[i])
        bar.set_color(color)

    plt.yticks(range(len(sorted_labels)), sorted_labels)
    plt.xlim(0, 1)
    plt.xlabel("Similarity Score")
    plt.title(title)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    # Set default save path if none provided
    if not save_path:
        os.makedirs("./visualizations", exist_ok=True)
        save_path = f"./visualizations/embedding_similarities_{int(np.random.rand() * 10000)}.png"

    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to {save_path}")
    plt.close()

    return save_path

def visualize_multi_level_search(results: List[Dict[str, Any]], levels: List[int] = None,
                                title: str = "Multi-Level Search Results", save_path: Optional[str] = None):
    """
    Visualize search results from multiple fractal levels.

    Args:
        results: List of search result dictionaries, each containing 'similarity' and 'level' keys
        levels: Optional list of levels to visualize
        title: Title for the visualization plot
        save_path: Optional path to save the visualization

    Returns:
        Path to the saved visualization
    """
    # Group results by level
    level_results = {}
    for result in results:
        level = result.get('level', 0)
        if level not in level_results:
            level_results[level] = []
        level_results[level].append(result)

    # Filter levels if specified
    if levels:
        level_results = {k: v for k, v in level_results.items() if k in levels}

    # Skip if no results
    if not level_results:
        print("No results to visualize")
        return None

    # Set up plot
    plt.figure(figsize=(12, 6))

    # Colors for each level
    colors = plt.cm.viridis(np.linspace(0, 1, len(level_results)))

    # Plot each level's results
    for i, (level, level_data) in enumerate(sorted(level_results.items())):
        similarities = [r['similarity'] for r in level_data]
        x_values = np.arange(len(similarities)) + (i * 0.2)  # Offset for each level

        plt.bar(x_values, similarities, width=0.15, color=colors[i], alpha=0.7,
                label=f"Level {level}")

    plt.xlabel("Result Index")
    plt.ylabel("Similarity Score")
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Set default save path if none provided
    if not save_path:
        os.makedirs("./visualizations", exist_ok=True)
        save_path = f"./visualizations/multi_level_search_{int(np.random.rand() * 10000)}.png"

    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to {save_path}")
    plt.close()

    return save_path

def visualize_embedding_clusters(embeddings: np.ndarray, labels: Optional[List[int]] = None,
                               title: str = "Embedding Clusters", save_path: Optional[str] = None):
    """
    Visualize embedding clusters using PCA.

    Args:
        embeddings: 2D array of embeddings
        labels: Optional list of cluster labels for each embedding
        title: Title for the visualization plot
        save_path: Optional path to save the visualization

    Returns:
        Path to the saved visualization
    """
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    # If no labels provided, visualize without clustering
    if labels is None:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
    else:
        # Get unique labels, treating -1 as noise
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        # Plot each cluster with different color
        for label in unique_labels:
            cluster_points = reduced_embeddings[np.array(labels) == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}",
                       alpha=0.7)

        # Plot noise points if any
        noise_points = reduced_embeddings[np.array(labels) == -1]
        if len(noise_points) > 0:
            plt.scatter(noise_points[:, 0], noise_points[:, 1], color='gray',
                       alpha=0.5, marker='x', label='Noise')

    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Set default save path if none provided
    if not save_path:
        os.makedirs("./visualizations", exist_ok=True)
        save_path = f"./visualizations/embedding_clusters_{int(np.random.rand() * 10000)}.png"

    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to {save_path}")
    plt.close()

    return save_path

def test_visualizations():
    """Test the visualization functions with sample data"""
    print("Testing visualizations - images will be saved to ./visualizations/")
    os.makedirs("./visualizations", exist_ok=True)

    # Generate sample data for fractal embeddings
    num_samples = 5
    embedding_dim = 64

    # Create sample diagnostics
    diagnostics = {
        "base_embeddings": [],
        "fractal_variations": []
    }

    # Generate random embeddings and variations
    for _ in range(num_samples):
        base_embedding = np.random.random(embedding_dim)
        base_embedding /= np.linalg.norm(base_embedding)

        # Create variations
        variations = {
            0: base_embedding,
            1: base_embedding + np.random.normal(0, 0.1, embedding_dim),
            2: base_embedding + np.random.normal(0, 0.2, embedding_dim)
        }

        # Normalize variations
        for k in variations:
            if k > 0:
                variations[k] /= np.linalg.norm(variations[k])

        diagnostics["base_embeddings"].append(base_embedding)
        diagnostics["fractal_variations"].append(variations)

    # Visualize fractal embeddings
    fractal_path = visualize_fractal_embeddings(
        diagnostics,
        save_path="./visualizations/fractal_test.png"
    )
    print(f"Fractal embeddings visualization saved to: {fractal_path}")

    # Test similarity visualization
    similarities = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35]
    labels = [f"Document {i+1}" for i in range(len(similarities))]
    similarity_path = visualize_embedding_similarities(
        similarities,
        labels,
        save_path="./visualizations/similarities_test.png"
    )
    print(f"Embedding similarities visualization saved to: {similarity_path}")

    # Test multi-level search visualization
    results = []
    for level in range(3):
        for i in range(5):
            results.append({
                'similarity': 0.9 - (level * 0.1) - (i * 0.05),
                'level': level,
                'text': f"Result {i+1} at level {level}"
            })
    multilevel_path = visualize_multi_level_search(
        results,
        save_path="./visualizations/multilevel_test.png"
    )
    print(f"Multi-level search visualization saved to: {multilevel_path}")

    # Test cluster visualization
    embeddings = np.random.random((50, embedding_dim))
    labels = np.random.randint(0, 3, 50)
    clusters_path = visualize_embedding_clusters(
        embeddings,
        labels,
        save_path="./visualizations/clusters_test.png"
    )
    print(f"Embedding clusters visualization saved to: {clusters_path}")

    print("All visualizations completed.")

if __name__ == "__main__":
    # Run test visualizations when script is executed directly
    test_visualizations()