"""
Memory Embedding Visualization

This module creates visualizations of embedding spaces across different enhancement levels,
highlighting cross-level relationships and semantic clustering properties.

The visualization emphasizes how related concepts cluster across different enhancement levels,
showing the core idea of enhanced memory: semantically similar items should remain
relatively close in embedding space, despite transformations across levels.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib import cm
from matplotlib.patches import ConnectionPatch
import math
import time
import json
import pickle
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    # Try importing optional visualization libraries
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Install with: pip install plotly")

# Default embedding dimension to use if none is found
DEFAULT_DIMENSION = 64

def load_memory_data(memory_dir, max_items=100):
    """
    Load memory data directly from files without relying on the MemoryManager class.

    Args:
        memory_dir: Directory containing memory data files
        max_items: Maximum number of items to include

    Returns:
        Dictionary with embeddings and metadata or None if loading fails
    """
    print(f"Loading memory data from {memory_dir}...")

    # Check if items.json exists
    items_path = os.path.join(memory_dir, "items.json")
    if not os.path.exists(items_path):
        print(f"No items.json found in {memory_dir}")
        return None

    # Load items data
    try:
        with open(items_path, 'r', encoding='utf-8') as f:
            items_data = json.load(f)

        print(f"Found {len(items_data)} items")

        # Limit to max_items if needed
        if len(items_data) > max_items:
            print(f"Limiting to {max_items} items")
            items_data = items_data[:max_items]

        # Load embeddings if available
        embeddings_path = os.path.join(memory_dir, "embeddings.npy")
        if os.path.exists(embeddings_path):
            all_embeddings = np.load(embeddings_path)
            print(f"Loaded {len(all_embeddings)} embeddings")

            # Take the first max_items embeddings
            embeddings = all_embeddings[:len(items_data)]

            # Check embedding dimension
            if len(embeddings) > 0:
                embedding_dim = embeddings[0].shape[0]
                print(f"Embedding dimension: {embedding_dim}")
            else:
                embedding_dim = DEFAULT_DIMENSION
                print(f"No embeddings found, using default dimension: {embedding_dim}")
        else:
            # No embeddings file, create random embeddings
            embedding_dim = DEFAULT_DIMENSION
            embeddings = np.random.randn(len(items_data), embedding_dim)
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-10)
            print(f"No embeddings.npy found, created random embeddings with dimension {embedding_dim}")

        # Try to load enhanced embeddings if available
        enhanced_embeddings = {}
        enhanced_path = os.path.join(memory_dir, "enhanced_embeddings.pkl")
        enhancement_levels = []

        if os.path.exists(enhanced_path):
            try:
                with open(enhanced_path, 'rb') as f:
                    enhanced_data = pickle.load(f)
                print(f"Loaded enhanced embeddings for {len(enhanced_data)} items")

                # Find all enhancement levels
                for item_id, item_enhanced in enhanced_data.items():
                    for level in item_enhanced.keys():
                        if level not in enhancement_levels:
                            enhancement_levels.append(level)

                print(f"Found enhancement levels: {sorted(enhancement_levels)}")

                # For each level, create a list of embeddings
                for level in enhancement_levels:
                    # Initialize with zeros
                    level_embeddings = np.zeros((len(items_data), embedding_dim))

                    # Fill in available enhanced embeddings
                    for i, item in enumerate(items_data):
                        item_id = item.get('id')
                        if item_id in enhanced_data and level in enhanced_data[item_id]:
                            level_emb = enhanced_data[item_id][level]
                            # Check if dimensions match
                            if len(level_emb) != embedding_dim:
                                # Resize if needed
                                if len(level_emb) > embedding_dim:
                                    level_emb = level_emb[:embedding_dim]
                                else:
                                    level_emb = np.pad(level_emb, (0, embedding_dim - len(level_emb)))
                            level_embeddings[i] = level_emb

                            # Normalize
                            norm = np.linalg.norm(level_emb)
                            if norm > 1e-10:
                                level_embeddings[i] = level_embeddings[i] / norm

                    enhanced_embeddings[level] = level_embeddings
            except Exception as e:
                print(f"Error loading enhanced embeddings: {e}")
                # Create empty enhanced embeddings
                enhanced_embeddings = {}

        # Create metadata
        metadata = []
        for item in items_data:
            meta = {
                'id': item.get('id', 'unknown'),
                'content_preview': item.get('content', '')[:100],
                'content': item.get('content', ''),
                'retrieval_count': item.get('metadata', {}).get('retrieval_count', 0)
            }

            # Extract category information
            item_metadata = item.get('metadata', {})
            if 'main_category' in item_metadata:
                meta['main_category'] = item_metadata['main_category']

            metadata.append(meta)

        return {
            'base_embeddings': embeddings,
            'enhanced_embeddings': enhanced_embeddings,
            'metadata': metadata,
            'enhancement_levels': sorted(enhancement_levels),
            'embedding_dim': embedding_dim
        }

    except Exception as e:
        print(f"Error loading memory data: {e}")
        traceback.print_exc()
        return None

def create_semantic_cluster_visualization(data, output_dir):
    """
    Create a visualization showing semantic clusters across different levels,
    with a polar coordinate representation.

    Args:
        data: Dictionary with embedding data
        output_dir: Directory to save visualization

    Returns:
        Path to saved visualization
    """
    print("Creating semantic cluster visualization...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get embedding data
    base_embeddings = data['base_embeddings']
    enhanced_embeddings = data['enhanced_embeddings']
    metadata = data['metadata']
    enhancement_levels = data['enhancement_levels']

    # Set up figure with polar coordinates for a sunburst-like view
    fig = plt.figure(figsize=(20, 20))

    # Apply dimensionality reduction to all levels combined
    all_embeddings = [base_embeddings]
    for level in enhancement_levels:
        # Filter out zero embeddings
        level_emb = enhanced_embeddings[level]
        non_zero = np.any(level_emb != 0, axis=1)
        if np.any(non_zero):
            filtered_emb = level_emb[non_zero]
            all_embeddings.append(filtered_emb)

    # Skip if we don't have enough data
    if len(all_embeddings) <= 1 or len(base_embeddings) == 0:
        print("Not enough data for semantic cluster visualization")
        return None

    combined_embeddings = np.vstack(all_embeddings)

    # Use PCA to reduce to 2D for the base coordinates
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(combined_embeddings)

    # Get categories for coloring
    categories = [meta.get('main_category', 'unknown') for meta in metadata]
    # print("CATEGORIES", metadata)
    unique_categories = sorted(set(categories))
    cat_to_color = {cat: i for i, cat in enumerate(unique_categories)}

    # Set up colormaps
    base_cmap = plt.cm.tab10
    level_cmaps = [
        plt.cm.viridis,
        plt.cm.plasma,
        plt.cm.inferno,
        plt.cm.magma
    ]

    # Calculate starting indices for each level in the reduced embeddings
    level_start_idx = {}
    start_idx = len(base_embeddings)
    for level in enhancement_levels:
        level_start_idx[level] = start_idx
        level_emb = enhanced_embeddings[level]
        non_zero = np.any(level_emb != 0, axis=1)
        start_idx += np.sum(non_zero)

    # Create a sunburst-like visualization using polar coordinates
    ax_base = fig.add_subplot(111, polar=True)

    # Extract base level coordinates
    base_reduced = reduced_embeddings[:len(base_embeddings)]

    # Convert to polar coordinates (r, θ)
    # Normalize to range [0, 1] for radius and [0, 2π] for angle
    x_min, x_max = np.min(base_reduced[:, 0]), np.max(base_reduced[:, 0])
    y_min, y_max = np.min(base_reduced[:, 1]), np.max(base_reduced[:, 1])

    # Handle case where min equals max
    x_range = max(1e-6, x_max - x_min)
    y_range = max(1e-6, y_max - y_min)

    # Compute radius and angle for base embeddings
    base_radius = np.sqrt(((base_reduced[:, 0] - x_min) / x_range)**2 +
                         ((base_reduced[:, 1] - y_min) / y_range)**2)
    base_radius = base_radius / np.maximum(np.max(base_radius), 1e-6) * 0.4  # Scale to inner 40% of plot

    base_theta = np.arctan2(base_reduced[:, 1] - y_min, base_reduced[:, 0] - x_min)

    # Plot base level points
    for i, (r, theta) in enumerate(zip(base_radius, base_theta)):
        category = categories[i]
        color = base_cmap(cat_to_color[category] / max(1, len(unique_categories)))
        ax_base.scatter(theta, r, c=[color], s=100, alpha=0.8, edgecolors='white')

        # Annotate with small category indicator
        if i % 3 == 0:  # Only annotate every 3rd point to avoid clutter
            ax_base.annotate(category[0].upper(),
                           (theta, r),
                           fontsize=8,
                           ha='center', va='center',
                           bbox=dict(boxstyle="circle,pad=0.1", fc=color, alpha=0.7))

    # Now add the enhancement levels as outer rings
    for level_idx, level in enumerate(enhancement_levels):
        # Get level-specific colormap
        level_cmap = level_cmaps[level_idx % len(level_cmaps)]

        # Get embeddings for this level
        level_emb = enhanced_embeddings[level]
        non_zero = np.any(level_emb != 0, axis=1)
        filtered_indices = np.where(non_zero)[0]

        if len(filtered_indices) == 0:
            continue

        # Get reduced embeddings for this level
        start_idx = level_start_idx[level]
        end_idx = start_idx + np.sum(non_zero)
        level_reduced = reduced_embeddings[start_idx:end_idx]

        # Compute radius and angle for this level
        # Scale radius to create concentric rings
        base_scale = 0.4  # Inner 40% for base level
        ring_width = (1.0 - base_scale) / max(1, len(enhancement_levels))
        inner_radius = base_scale + level_idx * ring_width
        outer_radius = inner_radius + ring_width

        # Normalize coordinates within this level
        level_x_min, level_x_max = np.min(level_reduced[:, 0]), np.max(level_reduced[:, 0])
        level_y_min, level_y_max = np.min(level_reduced[:, 1]), np.max(level_reduced[:, 1])

        # Handle case where min equals max
        level_x_range = max(1e-6, level_x_max - level_x_min)
        level_y_range = max(1e-6, level_y_max - level_y_min)

        # Compute radius and angle
        level_radius = np.sqrt(((level_reduced[:, 0] - level_x_min) / level_x_range)**2 +
                              ((level_reduced[:, 1] - level_y_min) / level_y_range)**2)

        # Scale to the appropriate ring
        level_radius = inner_radius + (level_radius / np.maximum(np.max(level_radius), 1e-6) * ring_width)

        level_theta = np.arctan2(level_reduced[:, 1] - level_y_min, level_reduced[:, 0] - level_x_min)

        # Plot this level's points
        for i, idx in enumerate(filtered_indices):
            category = categories[idx]
            color_idx = cat_to_color[category] / max(1, len(unique_categories))
            # Blend base category color with level color
            base_color = np.array(base_cmap(color_idx))
            level_color = np.array(level_cmap(i / max(1, len(filtered_indices))))
            blended_color = base_color * 0.7 + level_color * 0.3

            r, theta = level_radius[i], level_theta[i]
            ax_base.scatter(theta, r, c=[blended_color], s=80, alpha=0.8, edgecolors='white')

            # Draw connection lines to base embedding for some points
            if idx % 5 == 0:  # Connect every 5th point to reduce visual clutter
                base_r, base_t = base_radius[idx], base_theta[idx]
                ax_base.plot([base_t, theta], [base_r, r], color=blended_color, alpha=0.3, linewidth=1)

                # Add level indicator
                if idx % 10 == 0:  # Only label every 10th point
                    ax_base.annotate(f"L{level}",
                                   (theta, r),
                                   fontsize=7,
                                   ha='center', va='center',
                                   bbox=dict(boxstyle="circle,pad=0.1", fc=blended_color, alpha=0.7))

    # Add concentric circles to delineate levels
    radii = [0.4]  # Start with base level boundary
    for i in range(len(enhancement_levels)):
        radii.append(0.4 + ((i+1) * (1.0 - 0.4) / max(1, len(enhancement_levels))))

    for radius in radii:
        circle = plt.Circle((0, 0), radius, transform=ax_base.transData._b,
                           fill=False, edgecolor='gray', linestyle='--', alpha=0.3)
        ax_base.add_artist(circle)

    # Add level labels
    ax_base.annotate("Base Level", xy=(0, 0.2), xycoords='data',
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))

    for i, level in enumerate(enhancement_levels):
        radius = 0.4 + (i + 0.5) * ((1.0 - 0.4) / max(1, len(enhancement_levels)))
        angle = 0  # Place at 0 radians (right side)
        ax_base.annotate(f"Level {level}", xy=(angle, radius), xycoords='data',
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))

    # Add category legend outside the plot
    handles = []
    for cat in unique_categories:
        color = base_cmap(cat_to_color[cat] / max(1, len(unique_categories)))
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                           markersize=10, label=cat)
        handles.append(handle)

    ax_base.legend(handles=handles, title="Categories", loc='upper right',
                  bbox_to_anchor=(1.2, 1.0))

    # Set title
    ax_base.set_title("Memory Embedding Visualization: Semantic Clusters Across Levels",
                     y=1.05, fontsize=16)

    # Remove radial ticks and labels
    ax_base.set_yticklabels([])
    ax_base.set_xticklabels([])
    ax_base.grid(False)

    # Save figure
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"semantic_clusters_{timestr}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved semantic cluster visualization to {output_path}")

    return output_path

def create_embedding_fingerprints(data, output_dir):
    """
    Create visualizations showing how embeddings transform across levels,
    like a fingerprint or signature for each item.

    Args:
        data: Dictionary with embedding data
        output_dir: Directory to save visualization

    Returns:
        Path to saved visualization
    """
    print("Creating embedding fingerprint visualization...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get embedding data
    base_embeddings = data['base_embeddings']
    enhanced_embeddings = data['enhanced_embeddings']
    metadata = data['metadata']
    enhancement_levels = data['enhancement_levels']

    # Skip if we don't have enhancement levels
    if not enhancement_levels:
        print("No enhancement levels found, skipping fingerprint visualization")
        return None

    # Select a subset of items to visualize (to avoid overcrowding)
    max_items = min(9, len(base_embeddings))

    # Prioritize items with high retrieval counts or interesting content
    item_scores = []
    for i, meta in enumerate(metadata):
        # Calculate an interestingness score
        score = 0

        # Prioritize items with higher retrieval counts
        score += meta.get('retrieval_count', 0) * 10

        # Ensure variety of categories
        category = meta.get('main_category', 'unknown')
        if category not in ['unknown']:
            score += 20

        # Check if item has enhanced embeddings
        has_enhanced = False
        for level in enhancement_levels:
            if np.any(enhanced_embeddings[level][i] != 0):
                has_enhanced = True
                score += 50
                break

        # Only include items with enhanced embeddings
        if has_enhanced:
            item_scores.append((i, score))

    # Sort by score and select top items
    if not item_scores:
        print("No items with enhanced embeddings found, skipping fingerprint visualization")
        return None

    selected_indices = [idx for idx, _ in sorted(item_scores, key=lambda x: -x[1])[:max_items]]

    # Create a grid of embedding fingerprints
    rows = int(np.ceil(np.sqrt(max_items)))
    cols = int(np.ceil(max_items / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

    # Ensure axes is a 2D array
    if max_items == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    # Helper function to get distinctive color for a level
    level_colors = {
        0: '#1f77b4',  # Base level - blue
        1: '#ff7f0e',  # Level 1 - orange
        2: '#2ca02c',  # Level 2 - green
        3: '#d62728',  # Level 3 - red
        4: '#9467bd',  # Level 4 - purple
        5: '#8c564b',  # Level 5 - brown
    }

    # For each selected item, create a fingerprint
    for plot_idx, item_idx in enumerate(selected_indices):
        row, col = plot_idx // cols, plot_idx % cols
        ax = axes[row, col]

        # Get item metadata
        meta = metadata[item_idx]
        category = meta.get('main_category', 'unknown')
        title = meta.get('content_preview', '').strip()[:50]

        # To create a fingerprint, we'll visualize how the embedding values change across levels
        embeddings_by_level = []

        # Add base embedding
        embeddings_by_level.append((0, base_embeddings[item_idx]))

        # Add enhanced embeddings if available
        for level in enhancement_levels:
            level_emb = enhanced_embeddings[level][item_idx]
            if not np.all(level_emb == 0):
                embeddings_by_level.append((level, level_emb))

        # If we have embeddings from at least 2 levels, create the fingerprint
        if len(embeddings_by_level) >= 2:
            # For visualization, use a subset of the embedding dimensions
            # We'll take the first 50 dimensions or fewer if the embeddings are smaller
            max_dims = min(50, embeddings_by_level[0][1].shape[0])

            # Create a 2D visualization with levels on Y-axis and dimensions on X-axis
            level_values = []
            level_labels = []

            for level, emb in embeddings_by_level:
                # Normalize the embedding
                if np.max(np.abs(emb[:max_dims])) > 0:
                    norm_emb = emb[:max_dims] / np.max(np.abs(emb[:max_dims]))
                else:
                    norm_emb = emb[:max_dims]
                level_values.append(norm_emb)
                level_labels.append(f"L{level}")

            # Create a heatmap of the embedding values
            sns.heatmap(level_values, ax=ax, cmap='coolwarm', center=0,
                      cbar=False, yticklabels=level_labels)

            # Add a small line plot overlaid on the heatmap for key dimensions
            # This gives a more fingerprint-like appearance
            dims = np.arange(max_dims)
            for i, (level, values) in enumerate(zip(level_labels, level_values)):
                ax_twin = ax.twinx()
                ax_twin.plot(dims, values, color=level_colors.get(i, 'black'),
                           alpha=0.7, linewidth=1.5)
                ax_twin.set_ylim(-1.1, 1.1)
                ax_twin.axis('off')

            # Add title
            ax.set_title(f"{category}: {title}", fontsize=10)

            # Remove x-axis ticks to reduce clutter
            ax.set_xticks([])

        else:
            # If not enough levels, show a message
            ax.text(0.5, 0.5, f"No enhanced embeddings for\n{category}: {title}",
                  ha='center', va='center', fontsize=10)
            ax.axis('off')

    # If we have empty subplots, hide them
    for i in range(plot_idx + 1, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')

    # Add overall title
    plt.suptitle("Embedding Fingerprints Across Enhancement Levels", fontsize=16, y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"embedding_fingerprints_{timestr}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved embedding fingerprints to {output_path}")

    return output_path

def create_content_similarity_matrix(data, output_dir):
    """
    Create a matrix visualization showing similarity between memory items.

    Args:
        data: Dictionary with embedding data
        output_dir: Directory to save visualization

    Returns:
        Path to saved visualization
    """
    print("Creating content similarity matrix...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get embedding data
    base_embeddings = data['base_embeddings']
    metadata = data['metadata']

    # Skip if we don't have enough items
    if len(base_embeddings) <= 1:
        print("Not enough items for similarity matrix")
        return None

    # Calculate similarity matrix (cosine similarity)
    # Normalize embeddings
    norms = np.linalg.norm(base_embeddings, axis=1, keepdims=True)
    normalized_embeddings = base_embeddings / np.maximum(norms, 1e-10)

    # Calculate similarity matrix
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    # Create figure with appropriate size (larger for more items)
    plt.figure(figsize=(min(20, max(12, len(base_embeddings) // 2)),
                       min(20, max(10, len(base_embeddings) // 2))))

    # Create heatmap
    sns.heatmap(
        similarity_matrix,
        cmap='viridis',
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Cosine Similarity'}
    )

    # Add category and ID labels
    categories = [f"{i}: {meta.get('main_category', 'unknown')}" for i, meta in enumerate(metadata)]
    plt.xticks(np.arange(len(categories)) + 0.5, categories, rotation=90, fontsize=8)
    plt.yticks(np.arange(len(categories)) + 0.5, categories, fontsize=8)

    # Add title
    plt.title("Memory Content Similarity Matrix", fontsize=16)

    # Adjust layout for better readability
    plt.tight_layout()

    # Save figure
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"similarity_matrix_{timestr}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved similarity matrix to {output_path}")

    return output_path

def create_retrieval_heatmap(data, output_dir):
    """
    Create a heatmap showing retrieval frequency of memory items.

    Args:
        data: Dictionary with embedding data
        output_dir: Directory to save visualization

    Returns:
        Path to saved visualization
    """
    print("Creating retrieval heatmap...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get embedding data
    base_embeddings = data['base_embeddings']
    metadata = data['metadata']

    # Skip if we don't have enough items
    if len(base_embeddings) <= 1:
        print("Not enough items for retrieval heatmap")
        return None

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(base_embeddings)

    # Get retrieval counts
    retrieval_counts = [meta.get('retrieval_count', 0) for meta in metadata]

    # Create figure
    plt.figure(figsize=(12, 10))

    # Create a scatter plot with retrieval count determining size and color
    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=retrieval_counts,
        s=[50 + (count * 20) for count in retrieval_counts],  # Size based on count
        cmap='viridis',
        alpha=0.7,
        edgecolors='white'
    )

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Retrieval Count')

    # Add labels for frequently retrieved items
    for i, (x, y) in enumerate(reduced):
        if retrieval_counts[i] > 0:
            content = metadata[i].get('content_preview', '')[:30]
            plt.annotate(f"{content}... (×{retrieval_counts[i]})",
                        (x, y),
                        fontsize=8,
                        xytext=(5, 5),
                        textcoords='offset points')

    # Add title and labels
    plt.title("Memory Retrieval Frequency Heatmap", fontsize=16)
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2f})")

    # Add grid for readability
    plt.grid(alpha=0.3)

    # Save figure
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"retrieval_heatmap_{timestr}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved retrieval heatmap to {output_path}")

    return output_path

def create_concept_flow_visualization(data, output_dir):
    """
    Create a visualization showing how concepts flow and transform across levels,
    like a river or stream diagram.

    Args:
        data: Dictionary with embedding data
        output_dir: Directory to save visualization

    Returns:
        Path to saved visualization
    """
    print("Creating concept flow visualization...")

    # Skip if we don't have enhancement levels
    if not data['enhancement_levels']:
        print("No enhancement levels found, skipping concept flow visualization")
        return None

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get embedding data
    base_embeddings = data['base_embeddings']
    enhanced_embeddings = data['enhanced_embeddings']
    metadata = data['metadata']
    enhancement_levels = data['enhancement_levels']

    # Apply dimensionality reduction to all levels
    embeddings_by_level = {0: base_embeddings}
    for level in enhancement_levels:
        # Filter out zero embeddings
        level_emb = enhanced_embeddings[level]
        non_zero = np.any(level_emb != 0, axis=1)
        if np.any(non_zero):
            filtered_indices = np.where(non_zero)[0]
            filtered_emb = level_emb[filtered_indices]
            embeddings_by_level[level] = filtered_emb

    # Available levels
    available_levels = sorted(embeddings_by_level.keys())

    # Skip if we only have base level
    if len(available_levels) <= 1:
        print("No enhanced levels available, skipping concept flow visualization")
        return None

    # Use PCA to reduce all embeddings to 2D
    try:
        all_embeddings = np.vstack([embeddings_by_level[level] for level in available_levels])
        pca = PCA(n_components=2)
        all_reduced = pca.fit_transform(all_embeddings)
    except Exception as e:
        print(f"Error in PCA reduction: {e}")
        return None

    # Split the reduced embeddings by level
    reduced_by_level = {}
    start_idx = 0
    for level in available_levels:
        size = len(embeddings_by_level[level])
        reduced_by_level[level] = all_reduced[start_idx:start_idx+size]
        start_idx += size

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))

    # Set up colormaps
    # Use a different colormap for each level
    level_cmaps = {
        0: plt.cm.Blues,
        1: plt.cm.Oranges,
        2: plt.cm.Greens,
        3: plt.cm.Reds,
        4: plt.cm.Purples,
        5: plt.cm.Greys
    }

    # Get categories for coloring and markers
    categories = [meta.get('main_category', 'unknown') for meta in metadata]
    unique_categories = sorted(set(categories))
    cat_to_marker = {
        'unknown': 'o',                 # Circle
        'declarative': 's',             # Square
        'procedural_knowledge': '^',    # Triangle
        'experiential': 'P',            # Plus
        'tacit': '*',                   # Star
        'explicit': 'X',                # X
        'conceptual_knowledge': 'D',    # Diamond
        'contextual': 'v'               # Triangle down
        # 'general': 'd'      # Thin diamond
    }


    # Assign markers to any missing categories
    available_markers = ['o', 's', '^', 'P', '*', 'X', 'D', 'v', 'd', 'p', 'h', '8']
    next_marker = 0
    for cat in unique_categories:
        if cat not in cat_to_marker:
            cat_to_marker[cat] = available_markers[next_marker % len(available_markers)]
            next_marker += 1

    # Define level positions along x-axis
    level_positions = {}
    total_levels = len(available_levels)
    for i, level in enumerate(available_levels):
        level_positions[level] = i / max(1, total_levels - 1)

    # Plot each level's points
    level_points = {}
    base_indices_by_level = {}

    for level in available_levels:
        reduced = reduced_by_level[level]

        # Normalize y-coordinates to [0, 1]
        y_min, y_max = np.min(reduced[:, 1]), np.max(reduced[:, 1])
        y_range = max(1e-6, y_max - y_min)

        norm_y = (reduced[:, 1] - y_min) / y_range

        # Normalize x-coordinates and center around level position
        x_min, x_max = np.min(reduced[:, 0]), np.max(reduced[:, 0])
        x_range = max(1e-6, x_max - x_min)

        # Use a narrower range for x to avoid overlap between levels
        x_scale = 0.2

        # Get level position (scalar value)
        level_pos = level_positions[level]

        # Calculate normalized x coordinates centered around level position
        norm_x = level_pos + (reduced[:, 0] - x_min) / x_range * x_scale - x_scale/2

        # Store points for connecting later
        if level == 0:
            # For base level, we want all points
            level_points[level] = [(norm_x[i], norm_y[i]) for i in range(len(norm_x))]
        else:
            # For other levels, we need to track which base indices correspond to these points
            level_emb = enhanced_embeddings[level]
            non_zero = np.any(level_emb != 0, axis=1)
            base_indices = np.where(non_zero)[0]
            base_indices_by_level[level] = base_indices

            # Store mapping from base index to position
            level_points[level] = {}
            for i, base_idx in enumerate(base_indices):
                if i < len(norm_x) and i < len(norm_y):
                    level_points[level][base_idx] = (norm_x[i], norm_y[i])

        # Plot points at this level
        cmap = level_cmaps.get(level, plt.cm.viridis)

        if level == 0:
            # For base level, use category markers
            for i, (x, y) in enumerate(zip(norm_x, norm_y)):
                if i < len(categories):
                    category = categories[i]
                    marker = cat_to_marker.get(category, 'o')
                    ax.scatter(x, y, c=[cmap(0.5)], marker=marker, s=100,
                              alpha=0.8, edgecolors='white')

                    # Add category label for some points to avoid clutter
                    if i % 5 == 0:
                        ax.annotate(category[:3], (x, y), fontsize=8, ha='center', va='center')
        else:
            # For other levels, use simpler markers
            ax.scatter(norm_x, norm_y, c=[cmap(0.5)], marker='o', s=80,
                      alpha=0.7, edgecolors='white')

    # Connect points across levels to show concept flow
    for base_idx in range(len(base_embeddings)):
        if base_idx >= len(level_points[0]):
            continue

        base_x, base_y = level_points[0][base_idx]

        if base_idx < len(categories):
            category = categories[base_idx]
        else:
            category = "unknown"

        # Connect to each available higher level
        for level in available_levels[1:]:
            if base_idx in level_points[level]:
                level_x, level_y = level_points[level][base_idx]

                # Draw connection line
                ax.plot([base_x, level_x], [base_y, level_y],
                       alpha=0.3, linewidth=1, color='gray')

                # Draw small indicator of the transformation
                mid_x, mid_y = (base_x + level_x) / 2, (base_y + level_y) / 2
                ax.annotate(f"L{level}", (mid_x, mid_y), fontsize=7,
                          alpha=0.7, ha='center', va='center',
                          bbox=dict(boxstyle="circle,pad=0.1", fc='white', alpha=0.5))

    # Add level labels
    for level in available_levels:
        # Use the scalar level position
        position = level_positions[level]
        ax.text(position, 1.05, f"Level {level}",
               ha='center', va='bottom', fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8))

    # Add category legend
    handles = []
    for cat in unique_categories:
        marker = cat_to_marker.get(cat, 'o')
        handle = plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray',
                           markersize=10, label=cat)
        handles.append(handle)

    ax.legend(handles=handles, title="Categories", loc='upper right',
             bbox_to_anchor=(1.15, 1.0))

    # Set title and labels
    ax.set_title("Concept Flow Across Enhancement Levels", fontsize=16)
    ax.set_xlabel("Enhancement Level Progression", fontsize=12)
    ax.set_ylabel("Semantic Position", fontsize=12)

    # Set axis limits
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Save figure
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"concept_flow_{timestr}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved concept flow visualization to {output_path}")

    return output_path

def create_interactive_viz(data, output_dir):
    """
    Create an interactive 3D visualization of enhanced embeddings.

    Args:
        data: Dictionary with embedding data
        output_dir: Directory to save visualization

    Returns:
        Path to saved visualization
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping interactive visualization.")
        return None

    # Skip if we don't have enhancement levels
    if not data['enhancement_levels']:
        print("No enhancement levels found, skipping interactive visualization")
        return None

    print("Creating interactive 3D visualization...")

    # Get embedding data
    base_embeddings = data['base_embeddings']
    enhanced_embeddings = data['enhanced_embeddings']
    metadata = data['metadata']
    enhancement_levels = data['enhancement_levels']

    # Apply PCA to reduce to 3D
    pca = PCA(n_components=3)
    base_3d = pca.fit_transform(base_embeddings)

    # Create a plotly figure
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'scatter3d'}]]
    )

    # Get categories for coloring
    categories = [meta.get('main_category', 'unknown') for meta in metadata]
    unique_categories = sorted(set(categories))

    # Add base embeddings
    fig.add_trace(
        go.Scatter3d(
            x=base_3d[:, 0],
            y=base_3d[:, 1],
            z=base_3d[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=[unique_categories.index(cat) for cat in categories],
                colorscale='Viridis',
                colorbar=dict(
                    title="Category",
                    tickvals=[i for i in range(len(unique_categories))],
                    ticktext=unique_categories
                ),
                showscale=True
            ),
            text=[f"Base - {meta.get('main_category', 'unknown')}: {meta.get('content_preview', '')[:30]}..."
                 for meta in metadata],
            name='Base Level'
        )
    )

    # Add enhancement levels
    for level in enhancement_levels:
        # Filter non-zero embeddings
        level_emb = enhanced_embeddings[level]
        non_zero = np.any(level_emb != 0, axis=1)

        if not np.any(non_zero):
            continue

        # Apply PCA
        level_filtered = level_emb[non_zero]
        level_3d = pca.transform(level_filtered)

        # Get metadata for filtered points
        level_categories = [categories[i] for i, is_nonzero in enumerate(non_zero) if is_nonzero]
        level_metadata = [metadata[i] for i, is_nonzero in enumerate(non_zero) if is_nonzero]

        # Add to figure
        fig.add_trace(
            go.Scatter3d(
                x=level_3d[:, 0],
                y=level_3d[:, 1],
                z=level_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=[unique_categories.index(cat) for cat in level_categories],
                    colorscale='Viridis',
                    showscale=False,
                    symbol='diamond'
                ),
                text=[f"L{level} - {meta.get('main_category', 'unknown')}: {meta.get('content_preview', '')[:30]}..."
                     for meta in level_metadata],
                name=f'Level {level}'
            )
        )

        # Add connections between base and level embeddings
        # (only for a subset to avoid clutter)
        base_indices = np.where(non_zero)[0]
        for i, base_idx in enumerate(base_indices):
            if i % 5 == 0:  # Only connect every 5th point
                fig.add_trace(
                    go.Scatter3d(
                        x=[base_3d[base_idx, 0], level_3d[i, 0]],
                        y=[base_3d[base_idx, 1], level_3d[i, 1]],
                        z=[base_3d[base_idx, 2], level_3d[i, 2]],
                        mode='lines',
                        line=dict(width=2, color='rgba(100, 100, 100, 0.2)'),
                        showlegend=False
                    )
                )

    # Update layout
    fig.update_layout(
        title='Interactive 3D Memory Embedding Visualization',
        scene=dict(
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.2f})",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.2f})",
            zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.2f})"
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # Save as HTML
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"interactive_3d_{timestr}.html")
    fig.write_html(output_path)
    print(f"Saved interactive 3D visualization to {output_path}")

    return output_path

def create_memory_visualizations(memory_dir, output_dir="./visualizations",
                               max_items=100, interactive=False, seed_query=None):
    """
    Create visualizations of memory data.

    Args:
        memory_dir: Directory containing memory data files
        output_dir: Directory to save visualizations
        max_items: Maximum number of items to include
        interactive: Whether to create interactive visualizations
        seed_query: Optional query to find semantically related items

    Returns:
        Dictionary with paths to generated visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load memory data directly from files
    data = load_memory_data(memory_dir, max_items)

    if data is None:
        print("Failed to load memory data.")
        return None

    results = {}

    # Create semantic cluster visualization
    cluster_path = create_semantic_cluster_visualization(data, output_dir)
    if cluster_path:
        results['semantic_clusters'] = cluster_path

    # Create embedding fingerprints
    fingerprint_path = create_embedding_fingerprints(data, output_dir)
    if fingerprint_path:
        results['embedding_fingerprints'] = fingerprint_path

    # Create concept flow visualization
    flow_path = create_concept_flow_visualization(data, output_dir)
    if flow_path:
        results['concept_flow'] = flow_path

    # Create similarity matrix
    matrix_path = create_content_similarity_matrix(data, output_dir)
    if matrix_path:
        results['similarity_matrix'] = matrix_path

    # Create retrieval heatmap
    heatmap_path = create_retrieval_heatmap(data, output_dir)
    if heatmap_path:
        results['retrieval_heatmap'] = heatmap_path

    # Create interactive 3D visualization if enabled
    if interactive and PLOTLY_AVAILABLE:
        interactive_path = create_interactive_viz(data, output_dir)
        if interactive_path:
            results['interactive_viz'] = interactive_path

    # Display summary
    if results:
        print("\nVisualization Summary:")
        for viz_type, path in results.items():
            print(f"- {viz_type}: {path}")
    else:
        print("No visualizations were created.")

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Memory Embedding Visualization")
    parser.add_argument("--memory-dir", type=str, default="./memory",
                      help="Directory containing memory data")
    parser.add_argument("--output-dir", type=str, default="./visualizations",
                      help="Directory to save visualizations")
    parser.add_argument("--max-items", type=int, default=10000,
                      help="Maximum number of items to include")
    parser.add_argument("--interactive", action="store_true",
                      help="Create interactive visualizations")
    parser.add_argument("--seed-query", type=str, default=None,
                      help="Seed query to find semantically related items")

    args = parser.parse_args()

    # Create visualizations
    create_memory_visualizations(
        memory_dir=args.memory_dir,
        output_dir=args.output_dir,
        max_items=args.max_items,
        interactive=args.interactive,
        seed_query=args.seed_query
    )