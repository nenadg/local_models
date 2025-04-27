"""
Fractal Embedding Visualization

This module creates a fractal-like visualization of embedding spaces across levels,
highlighting cross-level relationships and semantic clustering properties.

The visualization emphasizes how related concepts cluster across different fractal levels,
showing the core idea of fractal memory: semantically similar items should remain
relatively close in embedding space, despite transformations across levels.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib import cm
from matplotlib.patches import ConnectionPatch
import math
import time

import sys
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


def create_fractal_embedding_visualization(memory_manager, output_dir="./visualizations", 
                                          max_items=100, interactive=False, seed_query=None):
    """
    Create a fractal-like visualization of embedding spaces across different levels,
    highlighting cross-level relationships and concept clustering.
    
    Args:
        memory_manager: The UnifiedMemoryManager instance
        output_dir: Directory to save visualizations
        max_items: Maximum number of items to include
        interactive: Whether to create interactive visualizations
        seed_query: Optional query to find semantically related items
        
    Returns:
        Dictionary with paths to generated visualizations
    """
    print(f"Creating fractal embedding visualization with up to {max_items} items...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get embedding data
    data = extract_embeddings(memory_manager, max_items, seed_query)
    if not data:
        print("No embedding data available.")
        return None
    
    results = {}
    
    # Create visualization showing the fractal nature of the embeddings
    results['fractal_sunburst'] = create_semantic_cluster_visualization(data, output_dir)
    
    # Create a fingerprint visualization that shows how items transform across levels
    results['embedding_fingerprints'] = create_embedding_fingerprints(data, output_dir)
    
    # Create a concept flow visualization
    results['concept_flow'] = create_concept_flow_visualization(data, output_dir)
    
    # If interactive and plotly available, create 3D visualization
    if interactive and PLOTLY_AVAILABLE:
        results['interactive_fractal'] = create_interactive_fractal_viz(data, output_dir)
    
    # If matplotlib is available, show plots
    if interactive:
        plt.show()
    
    return results


def extract_embeddings(memory_manager, max_items=100, seed_query=None):
    """
    Extract embeddings from memory manager with option to find semantically related items.
    
    Args:
        memory_manager: The UnifiedMemoryManager instance
        max_items: Maximum number of items to include
        seed_query: Optional query to find semantically related items
        
    Returns:
        Dictionary with embedding data
    """
    print("Extracting embeddings for visualization...")
    
    # Check if memory contains items
    stats = memory_manager.get_stats()
    if stats['active_items'] == 0:
        print("Memory store is empty.")
        return None
    
    # Handle seed query to find related items if provided
    selected_indices = None
    if seed_query and hasattr(memory_manager, 'retrieve') and callable(memory_manager.retrieve):
        try:
            # Get items related to the seed query
            print(f"Finding items related to seed query: '{seed_query}'")
            results = memory_manager.retrieve(
                query=seed_query,
                top_k=max_items,
                min_similarity=0.1
            )
            
            # Get indices of related items
            if results:
                selected_indices = []
                for result in results:
                    item_id = result.get('id')
                    if item_id in memory_manager.id_to_index:
                        idx = memory_manager.id_to_index[item_id]
                        selected_indices.append(idx)
                
                print(f"Found {len(selected_indices)} related items.")
            else:
                print("No related items found, using random selection.")
        except Exception as e:
            print(f"Error using seed query: {e}")
            selected_indices = None
    
    # If no seed query or no results, select random items
    if selected_indices is None or len(selected_indices) == 0:
        # Limit to max_items random indices
        all_indices = [i for i in range(len(memory_manager.items)) 
                      if i not in memory_manager.deleted_ids]
        
        if len(all_indices) > max_items:
            np.random.seed(42)  # For reproducibility
            selected_indices = np.random.choice(all_indices, size=max_items, replace=False)
        else:
            selected_indices = all_indices
    
    # Extract base embeddings and metadata for selected indices
    base_embeddings = []
    metadata = []
    
    # Get available fractal levels
    fractal_levels = []
    for i in range(1, memory_manager.max_fractal_levels + 1):
        if any(i in item.fractal_embeddings for item in memory_manager.items if item.fractal_embeddings):
            fractal_levels.append(i)
    
    # Initialize fractal embeddings dictionary
    fractal_embeddings = {level: [] for level in fractal_levels}
    
    # Extract data for selected indices
    for idx in selected_indices:
        if idx in memory_manager.deleted_ids:
            continue
        
        item = memory_manager.items[idx]
        
        # Add base embedding
        base_embeddings.append(item.embedding)
        
        # Add metadata
        meta = {
            'id': item.id,
            'content_preview': item.content[:100],
            'retrieval_count': item.metadata.get('retrieval_count', 0)
        }
        
        # Extract category information from metadata or content
        if 'category' in item.metadata:
            meta['category'] = item.metadata['category']
        elif 'source_hint' in item.metadata:
            meta['category'] = item.metadata['source_hint']
        elif 'filename' in item.metadata:
            # If it's a file, use the extension as category
            filename = item.metadata['filename']
            ext = os.path.splitext(filename)[1][1:] if '.' in filename else 'unknown'
            meta['category'] = ext
            meta['is_mcp'] = True
        else:
            # Infer category from content
            content = item.content.lower()
            if any(kw in content for kw in ['def ', 'class ', 'function', 'var ']):
                meta['category'] = 'code'
            elif any(kw in content for kw in ['#', '##', 'markdown']):
                meta['category'] = 'markdown'
            elif any(kw in content for kw in ['<html', '<div', '<p>']):
                meta['category'] = 'html'
            else:
                meta['category'] = 'text'
        
        metadata.append(meta)
        
        # Add fractal embeddings
        for level in fractal_levels:
            if level in item.fractal_embeddings:
                fractal_embeddings[level].append(item.fractal_embeddings[level])
            else:
                # Use zeros as placeholder
                fractal_embeddings[level].append(np.zeros_like(item.embedding))
    
    # Convert to numpy arrays
    base_embeddings = np.array(base_embeddings)
    for level in fractal_embeddings:
        fractal_embeddings[level] = np.array(fractal_embeddings[level])
    
    return {
        'base_embeddings': base_embeddings,
        'fractal_embeddings': fractal_embeddings,
        'metadata': metadata,
        'fractal_levels': fractal_levels
    }


def create_semantic_cluster_visualization(data, output_dir):
    """
    Create a visualization showing semantic clusters across different levels,
    inspired by fractal patterns in nature.
    
    Args:
        data: Dictionary with embedding data
        output_dir: Directory to save visualization
        
    Returns:
        Path to saved visualization
    """
    print("Creating semantic cluster visualization...")
    
    # Get embedding data
    base_embeddings = data['base_embeddings']
    fractal_embeddings = data['fractal_embeddings']
    metadata = data['metadata']
    fractal_levels = data['fractal_levels']
    
    # Set up figure with polar coordinates for a sunburst-like view
    # This creates a more "fractal-like" appearance
    fig = plt.figure(figsize=(20, 20))
    
    # Apply dimensionality reduction to all levels combined
    all_embeddings = [base_embeddings]
    for level in fractal_levels:
        # Filter out zero embeddings
        level_emb = fractal_embeddings[level]
        non_zero = np.any(level_emb != 0, axis=1)
        if np.any(non_zero):
            filtered_emb = level_emb[non_zero]
            all_embeddings.append(filtered_emb)
    
    combined_embeddings = np.vstack(all_embeddings)
    
    # Use PCA to reduce to 2D for the base coordinates
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(combined_embeddings)
    
    # Get categories for coloring
    categories = [meta.get('category', 'unknown') for meta in metadata]
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
    for level in fractal_levels:
        level_start_idx[level] = start_idx
        level_emb = fractal_embeddings[level]
        non_zero = np.any(level_emb != 0, axis=1)
        start_idx += np.sum(non_zero)
    
    # Create a sunburst-like visualization using polar coordinates
    axes = []
    
    # First, create the base level in the center
    ax_base = fig.add_subplot(111, polar=True)
    
    # Extract base level coordinates
    base_reduced = reduced_embeddings[:len(base_embeddings)]
    
    # Convert to polar coordinates (r, θ)
    # Normalize to range [0, 1] for radius and [0, 2π] for angle
    x_min, x_max = np.min(base_reduced[:, 0]), np.max(base_reduced[:, 0])
    y_min, y_max = np.min(base_reduced[:, 1]), np.max(base_reduced[:, 1])
    
    # Compute radius and angle for base embeddings
    base_radius = np.sqrt(((base_reduced[:, 0] - x_min) / (x_max - x_min))**2 + 
                         ((base_reduced[:, 1] - y_min) / (y_max - y_min))**2)
    base_radius = base_radius / np.max(base_radius) * 0.4  # Scale to inner 40% of plot
    
    base_theta = np.arctan2(base_reduced[:, 1] - y_min, base_reduced[:, 0] - x_min)
    
    # Plot base level points
    for i, (r, theta) in enumerate(zip(base_radius, base_theta)):
        category = categories[i]
        color = base_cmap(cat_to_color[category] / len(unique_categories))
        ax_base.scatter(theta, r, c=[color], s=100, alpha=0.8, edgecolors='white')
        
        # Annotate with small category indicator
        if i % 3 == 0:  # Only annotate every 3rd point to avoid clutter
            ax_base.annotate(category[0].upper(), 
                           (theta, r), 
                           fontsize=8,
                           ha='center', va='center',
                           bbox=dict(boxstyle="circle,pad=0.1", fc=color, alpha=0.7))
    
    # Now add the fractal levels as outer rings
    for level_idx, level in enumerate(fractal_levels):
        # Get level-specific colormap
        level_cmap = level_cmaps[level_idx % len(level_cmaps)]
        
        # Get embeddings for this level
        level_emb = fractal_embeddings[level]
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
        ring_width = (1.0 - base_scale) / len(fractal_levels)
        inner_radius = base_scale + level_idx * ring_width
        outer_radius = inner_radius + ring_width
        
        # Normalize coordinates within this level
        level_x_min, level_x_max = np.min(level_reduced[:, 0]), np.max(level_reduced[:, 0])
        level_y_min, level_y_max = np.min(level_reduced[:, 1]), np.max(level_reduced[:, 1])
        
        # Compute radius and angle
        level_radius = np.sqrt(((level_reduced[:, 0] - level_x_min) / (level_x_max - level_x_min))**2 +
                              ((level_reduced[:, 1] - level_y_min) / (level_y_max - level_y_min))**2)
        
        # Scale to the appropriate ring
        level_radius = inner_radius + (level_radius / np.max(level_radius) * ring_width)
        
        level_theta = np.arctan2(level_reduced[:, 1] - level_y_min, level_reduced[:, 0] - level_x_min)
        
        # Plot this level's points
        for i, idx in enumerate(filtered_indices):
            category = categories[idx]
            color_idx = cat_to_color[category] / len(unique_categories)
            # Blend base category color with level color
            base_color = np.array(base_cmap(color_idx))
            level_color = np.array(level_cmap(i / len(filtered_indices)))
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
    for i in range(len(fractal_levels)):
        radii.append(0.4 + ((i+1) * (1.0 - 0.4) / len(fractal_levels)))
    
    for radius in radii:
        circle = plt.Circle((0, 0), radius, transform=ax_base.transData._b, 
                           fill=False, edgecolor='gray', linestyle='--', alpha=0.3)
        ax_base.add_artist(circle)
    
    # Add level labels
    ax_base.annotate("Base Level", xy=(0, 0.2), xycoords='data', 
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
    
    for i, level in enumerate(fractal_levels):
        radius = 0.4 + (i + 0.5) * ((1.0 - 0.4) / len(fractal_levels))
        angle = 0  # Place at 0 radians (right side)
        ax_base.annotate(f"Level {level}", xy=(angle, radius), xycoords='data',
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
    
    # Add category legend outside the plot
    handles = []
    for cat in unique_categories:
        color = base_cmap(cat_to_color[cat] / len(unique_categories))
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                           markersize=10, label=cat)
        handles.append(handle)
    
    ax_base.legend(handles=handles, title="Categories", loc='upper right',
                  bbox_to_anchor=(1.2, 1.0))
    
    # Set title
    ax_base.set_title("Fractal Embedding Visualization: Semantic Clusters Across Levels", 
                     y=1.05, fontsize=16)
    
    # Remove radial ticks and labels
    ax_base.set_yticklabels([])
    ax_base.set_xticklabels([])
    ax_base.grid(False)
    
    # Save figure
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"fractal_semantic_clusters_{timestr}.png")
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
    
    # Get embedding data
    base_embeddings = data['base_embeddings']
    fractal_embeddings = data['fractal_embeddings']
    metadata = data['metadata']
    fractal_levels = data['fractal_levels']
    
    # Select a subset of items to visualize (to avoid overcrowding)
    max_items = min(9, len(base_embeddings))
    
    # Prioritize items with MCP content or high retrieval counts
    item_scores = []
    for i, meta in enumerate(metadata):
        # Calculate an interestingness score
        score = 0
        
        # Prioritize MCP content
        if meta.get('is_mcp', False):
            score += 100
            
        # Prioritize items with higher retrieval counts
        score += meta.get('retrieval_count', 0) * 10
        
        # Ensure variety of categories
        category = meta.get('category', 'unknown')
        if category in ['code', 'markdown', 'html']:
            score += 20
            
        item_scores.append((i, score))
    
    # Sort by score and select top items
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
        category = meta.get('category', 'unknown')
        title = meta.get('content_preview', '').strip()[:50]
        
        # To create a fingerprint, we'll visualize how the embedding values change across levels
        embeddings_by_level = []
        
        # Add base embedding
        embeddings_by_level.append((0, base_embeddings[item_idx]))
        
        # Add fractal embeddings if available
        for level in fractal_levels:
            level_emb = fractal_embeddings[level][item_idx]
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
                norm_emb = emb[:max_dims] / np.max(np.abs(emb[:max_dims]))
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
            ax.text(0.5, 0.5, f"No fractal embeddings for\n{category}: {title}",
                  ha='center', va='center', fontsize=10)
            ax.axis('off')
    
    # If we have empty subplots, hide them
    for i in range(plot_idx + 1, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    # Add overall title
    plt.suptitle("Embedding Fingerprints Across Fractal Levels", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"embedding_fingerprints_{timestr}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved embedding fingerprints to {output_path}")
    
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
    
    # Get embedding data
    base_embeddings = data['base_embeddings']
    fractal_embeddings = data['fractal_embeddings']
    metadata = data['metadata']
    fractal_levels = data['fractal_levels']
    
    # Apply dimensionality reduction to all levels
    embeddings_by_level = {0: base_embeddings}
    for level in fractal_levels:
        # Filter out zero embeddings
        level_emb = fractal_embeddings[level]
        non_zero = np.any(level_emb != 0, axis=1)
        if np.any(non_zero):
            embeddings_by_level[level] = level_emb[non_zero]
            
    # Available levels
    available_levels = sorted(embeddings_by_level.keys())
    
    # Use PCA to reduce all embeddings to 2D
    all_embeddings = np.vstack([embeddings_by_level[level] for level in available_levels])
    pca = PCA(n_components=2)
    all_reduced = pca.fit_transform(all_embeddings)
    
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
    categories = [meta.get('category', 'unknown') for meta in metadata]
    unique_categories = sorted(set(categories))
    cat_to_marker = {
        'code': 'o',        # Circle
        'markdown': 's',    # Square
        'html': '^',        # Triangle
        'text': 'P',        # Plus
        'js': '*',          # Star
        'py': 'X',          # X
        'md': 'D',          # Diamond
        'htm': 'v',         # Triangle down
        'general': 'd'      # Thin diamond
    }
    
    # Assign markers to any missing categories
    available_markers = ['o', 's', '^', 'P', '*', 'X', 'D', 'v', 'd', 'p', 'h', '8']
    next_marker = 0
    for cat in unique_categories:
        if cat not in cat_to_marker:
            cat_to_marker[cat] = available_markers[next_marker % len(available_markers)]
            next_marker += 1
    
    # Define level positions along x-axis
    level_x = {}
    total_levels = len(available_levels)
    for i, level in enumerate(available_levels):
        level_x[level] = i / (total_levels - 1)
    
    # Plot each level's points
    level_points = {}
    for level in available_levels:
        reduced = reduced_by_level[level]
        
        # Normalize y-coordinates to [0, 1]
        y_min, y_max = np.min(reduced[:, 1]), np.max(reduced[:, 1])
        y_range = y_max - y_min
        if y_range < 1e-10:
            y_range = 1.0
        
        norm_y = (reduced[:, 1] - y_min) / y_range
        
        # Normalize x-coordinates and center around level position
        x_min, x_max = np.min(reduced[:, 0]), np.max(reduced[:, 0])
        x_range = x_max - x_min
        if x_range < 1e-10:
            x_range = 1.0
            
        # Use a narrower range for x to avoid overlap between levels
        x_scale = 0.2
        norm_x = level_x[level] + (reduced[:, 0] - x_min) / x_range * x_scale - x_scale/2
        
        # Store points for connecting later
        if level == 0:
            # For base level, we want all points
            level_points[level] = [(norm_x[i], norm_y[i]) for i in range(len(norm_x))]
        else:
            # For other levels, we may not have all points
            # Find which base indices have embeddings at this level
            level_emb = fractal_embeddings[level]
            non_zero = np.any(level_emb != 0, axis=1)
            base_indices = np.where(non_zero)[0]
            
            # Store mapping from base index to position
            level_points[level] = {}
            for i, base_idx in enumerate(base_indices):
                level_points[level][base_idx] = (norm_x[i], norm_y[i])
        
        # Plot points at this level
        cmap = level_cmaps.get(level, plt.cm.viridis)
        
        if level == 0:
            # For base level, use category markers
            for i, (x, y) in enumerate(zip(norm_x, norm_y)):
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
        base_x, base_y = level_points[0][base_idx]
        category = categories[base_idx]
        
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
        ax.text(level_x[level], 1.05, f"Level {level}", 
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
    ax.set_title("Concept Flow Across Fractal Levels", fontsize=16)
    ax.set_xlabel("Fractal Level Progression", fontsize=12)
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


def create_interactive_fractal_viz(data, output_dir):
    """
    Create an interactive 3D visualization of fractal embeddings.
    
    Args:
        data: Dictionary with embedding data
        output_dir: Directory to save visualization
        
    Returns:
        Path to saved visualization
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping interactive visualization.")
        return None
    
    print("Creating interactive 3D fractal visualization...")
    
    # Get embedding data
    base_embeddings = data['base_embeddings']
    fractal_embeddings = data['fractal_embeddings']
    metadata = data['metadata']
    fractal_levels = data['fractal_levels']
    
    # Apply PCA to reduce to 3D
    pca = PCA(n_components=3)
    base_3d = pca.fit_transform(base_embeddings)
    
    # Create a plotly figure
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'scatter3d'}]]
    )
    
    # Get categories for coloring
    categories = [meta.get('category', 'unknown') for meta in metadata]
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
            text=[f"Base - {meta.get('category', 'unknown')}: {meta.get('content_preview', '')[:30]}..." 
                 for meta in metadata],
            name='Base Level'
        )
    )
    
    # Add fractal levels
    for level in fractal_levels:
        # Filter non-zero embeddings
        level_emb = fractal_embeddings[level]
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
                text=[f"L{level} - {meta.get('category', 'unknown')}: {meta.get('content_preview', '')[:30]}..." 
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
        title='Interactive 3D Fractal Embedding Visualization',
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
    output_path = os.path.join(output_dir, f"interactive_fractal_3d_{timestr}.html")
    fig.write_html(output_path)
    print(f"Saved interactive 3D visualization to {output_path}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fractal Embedding Visualization")
    parser.add_argument("--memory-dir", type=str, default="./memory",
                      help="Directory containing the memory store")
    parser.add_argument("--output-dir", type=str, default="./visualizations",
                      help="Directory to save visualizations")
    parser.add_argument("--max-items", type=int, default=100,
                      help="Maximum number of items to include")
    parser.add_argument("--seed-query", type=str, default=None,
                      help="Seed query to find semantically related items")
    parser.add_argument("--interactive", action="store_true",
                      help="Show interactive visualizations")
    
    args = parser.parse_args()
    
    # Try to import the memory manager
    try:
        from unified_memory import UnifiedMemoryManager
        
        # Create memory manager
        memory_manager = UnifiedMemoryManager(storage_path=args.memory_dir)
        
        # Create visualizations
        create_fractal_embedding_visualization(
            memory_manager=memory_manager,
            output_dir=args.output_dir,
            max_items=args.max_items,
            interactive=args.interactive,
            seed_query=args.seed_query
        )
    except ImportError:
        print("UnifiedMemoryManager couldn't be imported.")
        print("Make sure you're running this from the project root directory.")