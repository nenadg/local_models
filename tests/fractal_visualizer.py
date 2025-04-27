"""
Fractal Memory Visualization Tool

This tool provides advanced visualization capabilities for the fractal memory system
in the TinyLlama Chat project. It offers interactive explorations of embedding spaces
across different fractal levels, similarity analysis, and embedding transformations.

Usage:
    python fractal_visualizer.py [--memory-dir DIR] [--interactive] [--max-items N]
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

# insert the project root (one level up) at the front of sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try to import interactive 3D visualization libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Install with: pip install plotly")

# Import memory system
try:
    from unified_memory import UnifiedMemoryManager
except ImportError:
    print("Warning: UnifiedMemoryManager couldn't be imported.")
    print("Make sure you're running this from the project root directory.")


class FractalVisualizer:
    """Visualization tool for fractal embeddings in the memory system."""
    
    def __init__(self, memory_dir="./memory", max_items=500):
        """
        Initialize the visualizer.
        
        Args:
            memory_dir: Directory containing the memory store
            max_items: Maximum number of items to include in visualizations
        """
        self.memory_dir = memory_dir
        self.max_items = max_items
        self.output_dir = "./visualizations"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load memory store
        print(f"Loading memory from {memory_dir}...")
        self.memory_manager = UnifiedMemoryManager(storage_path=memory_dir)
        
        # Check if the memory has items
        stats = self.memory_manager.get_stats()
        print(f"Memory stats: {stats['active_items']} active items")
        
        if stats['active_items'] == 0:
            print("Warning: Memory store is empty. Run the chat system or tests first.")
        
        # Get memory configuration
        self.use_fractal = stats.get('fractal_enabled', False)
        self.max_fractal_levels = stats.get('fractal_levels', 3)
        
        print(f"Fractal enabled: {self.use_fractal}")
        print(f"Max fractal levels: {self.max_fractal_levels}")
        
        # Color theme setup
        self.setup_color_theme()
    
    def setup_color_theme(self):
        """Setup consistent color themes for visualizations."""
        # Use seaborn color palettes
        self.category_palette = sns.color_palette("tab10")
        self.level_palette = sns.color_palette("viridis", self.max_fractal_levels + 1)
        self.cluster_palette = sns.color_palette("Set2", 10)
        
        # Set default style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("notebook", font_scale=1.2)
    
    def extract_embeddings(self, level=None):
        """
        Extract embeddings and metadata for visualization.
        
        Args:
            level: Specific fractal level to extract (None for all levels)
            
        Returns:
            Dictionary with embeddings and metadata
        """
        print(f"Extracting embeddings for visualization...")
        
        # Check memory store
        stats = self.memory_manager.get_stats()
        if stats['active_items'] == 0:
            print("Error: No items in memory.")
            return None
        
        # Limit to max_items
        item_count = min(stats['active_items'], self.max_items)
        print(f"Processing {item_count} items...")
        
        # Create data containers
        data = {
            'base_embeddings': [],
            'metadata': [],
            'fractal_embeddings': {level: [] for level in range(1, self.max_fractal_levels + 1)}
        }
        
        # Extract data from memory items
        for i, item in enumerate(self.memory_manager.items[:item_count]):
            if i in self.memory_manager.deleted_ids:
                continue
                
            # Store base embedding
            data['base_embeddings'].append(item.embedding)
            
            # Store metadata
            metadata = {}
            
            # Extract category if available
            if 'category' in item.metadata:
                metadata['category'] = item.metadata['category']
            elif 'source_hint' in item.metadata:
                metadata['category'] = item.metadata['source_hint']
            else:
                # Infer category from content
                content = item.content.lower()
                if any(term in content for term in ['code', 'function', 'class', 'variable']):
                    metadata['category'] = 'code'
                elif any(term in content for term in ['math', 'calculation', 'equation']):
                    metadata['category'] = 'math'
                elif any(term in content for term in ['history', 'ancient', 'century']):
                    metadata['category'] = 'history'
                else:
                    metadata['category'] = 'general'
            
            # Add other useful metadata
            metadata['item_id'] = item.id
            metadata['retrieval_count'] = item.metadata.get('retrieval_count', 0)
            metadata['content_preview'] = item.content[:50]
            metadata['timestamp'] = item.metadata.get('timestamp', 0)
            
            # Add to metadata list
            data['metadata'].append(metadata)
            
            # Extract fractal embeddings
            for level_num in range(1, self.max_fractal_levels + 1):
                if level_num in item.fractal_embeddings:
                    data['fractal_embeddings'][level_num].append(item.fractal_embeddings[level_num])
                else:
                    # Use zeros as placeholder
                    data['fractal_embeddings'][level_num].append(np.zeros_like(item.embedding))
        
        # Convert to numpy arrays
        data['base_embeddings'] = np.array(data['base_embeddings'])
        for level_num in data['fractal_embeddings']:
            data['fractal_embeddings'][level_num] = np.array(data['fractal_embeddings'][level_num])
        
        # Filter to specific level if requested
        if level is not None:
            if level == 0:
                data['filtered_embeddings'] = data['base_embeddings']
            else:
                data['filtered_embeddings'] = data['fractal_embeddings'][level]
        
        return data
    
    def create_embedding_stats(self, data):
        """
        Calculate statistics on embeddings.
        
        Args:
            data: Embedding data dictionary
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        # Calculate statistics for base embeddings
        base_embeddings = data['base_embeddings']
        stats['base'] = {
            'mean': np.mean(base_embeddings, axis=0),
            'std': np.std(base_embeddings, axis=0),
            'min': np.min(base_embeddings, axis=0),
            'max': np.max(base_embeddings, axis=0),
            'norm_mean': np.mean(np.linalg.norm(base_embeddings, axis=1))
        }
        
        # Calculate statistics for each fractal level
        stats['fractal'] = {}
        for level, embeddings in data['fractal_embeddings'].items():
            # Skip if no embeddings at this level
            if np.all(embeddings == 0):
                continue
                
            # Calculate statistics for non-zero embeddings
            non_zero_mask = np.any(embeddings != 0, axis=1)
            if not np.any(non_zero_mask):
                continue
                
            filtered_embeddings = embeddings[non_zero_mask]
            
            stats['fractal'][level] = {
                'mean': np.mean(filtered_embeddings, axis=0),
                'std': np.std(filtered_embeddings, axis=0),
                'min': np.min(filtered_embeddings, axis=0),
                'max': np.max(filtered_embeddings, axis=0),
                'norm_mean': np.mean(np.linalg.norm(filtered_embeddings, axis=1)),
                'count': np.sum(non_zero_mask)
            }
        
        # Calculate similarity matrices (for a subset if too many items)
        max_sim_items = min(100, len(base_embeddings))
        
        # Base similarity matrix
        base_sample = base_embeddings[:max_sim_items]
        base_sample_norm = base_sample / np.maximum(np.linalg.norm(base_sample, axis=1, keepdims=True), 1e-8)
        stats['base_similarity'] = np.dot(base_sample_norm, base_sample_norm.T)
        
        # Fractal similarity matrices
        stats['fractal_similarity'] = {}
        for level, embeddings in data['fractal_embeddings'].items():
            # Skip if no embeddings at this level
            if np.all(embeddings == 0):
                continue
                
            # Calculate similarity matrix
            level_sample = embeddings[:max_sim_items]
            level_sample_norm = level_sample / np.maximum(np.linalg.norm(level_sample, axis=1, keepdims=True), 1e-8)
            stats['fractal_similarity'][level] = np.dot(level_sample_norm, level_sample_norm.T)
        
        return stats
    
    def visualize_similarity_heatmaps(self, data, stats):
        """
        Create similarity heatmaps for different embedding levels.
        
        Args:
            data: Embedding data dictionary
            stats: Statistics dictionary
            
        Returns:
            The figure object
        """
        print("Creating similarity heatmaps...")
        
        # Get available similarity matrices
        available_levels = [0] + sorted(stats['fractal_similarity'].keys())
        
        # Create figure with subplots for each level
        fig, axes = plt.subplots(1, len(available_levels), 
                                figsize=(5*len(available_levels), 5),
                                sharex=True, sharey=True)
        
        # Create colormap
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        
        # If only one level, ensure axes is a list
        if len(available_levels) == 1:
            axes = [axes]
        
        # Plot similarity matrices
        for i, level in enumerate(available_levels):
            if level == 0:
                # Base embeddings
                sim_matrix = stats['base_similarity']
                title = "Base Embeddings"
            else:
                # Fractal embeddings
                sim_matrix = stats['fractal_similarity'][level]
                title = f"Level {level} Embeddings"
            
            # Plot the heatmap
            im = axes[i].imshow(sim_matrix, cmap=cmap, vmin=-1, vmax=1)
            axes[i].set_title(title)
            
            # Remove ticks
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes, shrink=0.8)
        cbar.set_label('Cosine Similarity')
        
        # Add title
        plt.suptitle("Similarity Matrices Across Embedding Levels", fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        timestr = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(self.output_dir, f"similarity_heatmaps_{timestr}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved similarity heatmaps to {output_path}")
        
        return fig
    
    def visualize_embedding_clusters(self, data, method='pca', n_clusters=5):
        """
        Visualize embedding clusters for each level.
        
        Args:
            data: Embedding data dictionary
            method: Dimensionality reduction method ('pca' or 'tsne')
            n_clusters: Number of clusters to use
            
        Returns:
            The figure object
        """
        print(f"Creating embedding cluster visualization using {method}...")
        
        # Get embeddings for each level
        base_embeddings = data['base_embeddings']
        fractal_levels = [level for level in sorted(data['fractal_embeddings'].keys())
                         if not np.all(data['fractal_embeddings'][level] == 0)]
        
        # Create figure
        n_levels = 1 + len(fractal_levels)
        fig = plt.figure(figsize=(15, 5 * n_levels))
        
        # Create GridSpec for flexible layout
        gs = gridspec.GridSpec(n_levels, 3, width_ratios=[2, 2, 1])
        
        # Process metadata for coloring
        categories = [m.get('category', 'unknown') for m in data['metadata']]
        unique_categories = sorted(set(categories))
        category_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
        category_colors = [category_to_idx.get(cat, 0) for cat in categories]
        
        # Create colormaps
        category_cmap = ListedColormap(sns.color_palette("tab10", len(unique_categories)))
        cluster_cmap = ListedColormap(sns.color_palette("Set2", n_clusters))
        
        # Process each level
        for i, level in enumerate([0] + fractal_levels):
            if level == 0:
                # Base embeddings
                embeddings = base_embeddings
                title = "Base Embeddings"
            else:
                # Filter non-zero embeddings
                embeddings = data['fractal_embeddings'][level]
                non_zero_mask = np.any(embeddings != 0, axis=1)
                if not np.any(non_zero_mask):
                    continue
                    
                embeddings = embeddings[non_zero_mask]
                level_categories = [categories[j] for j, mask in enumerate(non_zero_mask) if mask]
                level_colors = [category_colors[j] for j, mask in enumerate(non_zero_mask) if mask]
                
                title = f"Level {level} Embeddings"
            
            # Apply dimensionality reduction
            if method == 'tsne':
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
                reduced = tsne.fit_transform(embeddings)
            else:  # Default to PCA
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(embeddings)
                var_ratio = pca.explained_variance_ratio_
            
            # Apply clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            # Plot by category (left plot)
            ax1 = fig.add_subplot(gs[i, 0])
            scatter = ax1.scatter(reduced[:, 0], reduced[:, 1], 
                                 c=level_colors if level > 0 else category_colors,
                                 cmap=category_cmap, alpha=0.7, s=50)
            ax1.set_title(f"{title} - By Category")
            
            # Add legend for categories
            handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=category_cmap(category_to_idx[cat]/len(unique_categories)), 
                                 markersize=8, label=cat) 
                      for cat in unique_categories]
            ax1.legend(handles=handles, title="Categories", loc="best")
            
            # Plot by cluster (middle plot)
            ax2 = fig.add_subplot(gs[i, 1])
            scatter2 = ax2.scatter(reduced[:, 0], reduced[:, 1], 
                                  c=clusters, cmap=cluster_cmap, alpha=0.7, s=50)
            ax2.set_title(f"{title} - Clusters")
            
            # Add cluster centers
            if method == 'pca':
                # Transform cluster centers
                centers_reduced = pca.transform(kmeans.cluster_centers_)
                ax2.scatter(centers_reduced[:, 0], centers_reduced[:, 1], 
                           marker='X', s=200, c='black', alpha=0.8)
            
            # Plot cluster distribution (right plot)
            ax3 = fig.add_subplot(gs[i, 2])
            
            # Count categories in each cluster
            cluster_data = []
            for cluster_id in range(n_clusters):
                cluster_mask = clusters == cluster_id
                if not np.any(cluster_mask):
                    continue
                    
                if level > 0:
                    cluster_categories = [level_categories[j] for j, mask in enumerate(cluster_mask) if mask]
                else:
                    cluster_categories = [categories[j] for j, mask in enumerate(cluster_mask) if mask]
                
                cat_counts = {}
                for cat in cluster_categories:
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1
                
                for cat, count in cat_counts.items():
                    cluster_data.append({
                        'cluster': f'Cluster {cluster_id}',
                        'category': cat,
                        'count': count
                    })
            
            # Convert to DataFrame for seaborn
            import pandas as pd
            cluster_df = pd.DataFrame(cluster_data)
            
            if not cluster_df.empty:
                # Plot stacked bar chart
                sns.barplot(x='cluster', y='count', hue='category', data=cluster_df, ax=ax3)
                ax3.set_title("Category Distribution in Clusters")
                ax3.set_xlabel("")
                ax3.tick_params(axis='x', rotation=45)
                ax3.legend().set_visible(False)  # Hide legend (already shown in first plot)
        
        # Add overall title
        plt.suptitle(f"Embedding Clusters Across Levels ({method.upper()}, k={n_clusters})", 
                    fontsize=16, y=0.99)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        timestr = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(self.output_dir, 
                                  f"embedding_clusters_{method}_{n_clusters}_{timestr}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved cluster visualization to {output_path}")
        
        return fig
    
    def visualize_fractal_transformation(self, data, n_samples=10):
        """
        Visualize how embeddings transform across fractal levels.
        
        Args:
            data: Embedding data dictionary
            n_samples: Number of sample points to visualize
            
        Returns:
            The figure object
        """
        print(f"Visualizing fractal transformations for {n_samples} sample points...")
        
        # Get embeddings
        base_embeddings = data['base_embeddings']
        metadata = data['metadata']
        
        # Only proceed if we have fractal levels
        fractal_levels = [level for level in sorted(data['fractal_embeddings'].keys())
                         if not np.all(data['fractal_embeddings'][level] == 0)]
        
        if not fractal_levels:
            print("No fractal embeddings found. Skipping transformation visualization.")
            return None
        
        # Use PCA to reduce to 2D for visualization
        all_points = np.vstack([base_embeddings] + 
                              [data['fractal_embeddings'][level] for level in fractal_levels])
        
        # Filter out zero embeddings
        non_zero_mask = np.any(all_points != 0, axis=1)
        filtered_points = all_points[non_zero_mask]
        
        # Apply PCA
        pca = PCA(n_components=2)
        reduced_points = pca.fit_transform(filtered_points)
        
        # Reconstruct the results by level
        points_by_level = {}
        current_idx = 0
        
        # Base level
        n_base = len(base_embeddings)
        points_by_level[0] = reduced_points[current_idx:current_idx+n_base]
        current_idx += n_base
        
        # Fractal levels
        for level in fractal_levels:
            level_embeddings = data['fractal_embeddings'][level]
            n_level = len(level_embeddings)
            points_by_level[level] = reduced_points[current_idx:current_idx+n_level]
            current_idx += n_level
        
        # Random sample of points to visualize (for clarity)
        np.random.seed(42)
        sample_indices = np.random.choice(len(base_embeddings), size=min(n_samples, len(base_embeddings)), replace=False)
        
        # Create figure for static visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot connections between levels
        for idx in sample_indices:
            level_points = []
            
            # Collect points at each level
            for level in [0] + fractal_levels:
                if idx < len(points_by_level[level]):
                    level_points.append(points_by_level[level][idx])
            
            # Plot connections
            if len(level_points) > 1:
                level_points = np.array(level_points)
                ax.plot(level_points[:, 0], level_points[:, 1], 'o-', alpha=0.5,
                       color=plt.cm.tab10(sample_indices.tolist().index(idx) % 10))
                
                # Add level labels
                for i, level in enumerate([0] + fractal_levels):
                    if i < len(level_points):
                        ax.annotate(f"L{level}", level_points[i], fontsize=8,
                                   xytext=(5, 5), textcoords='offset points')
        
        # Add title and labels
        ax.set_title("Embedding Transformations Across Fractal Levels", fontsize=14)
        ax.set_xlabel(f"PCA-1 (var: {pca.explained_variance_ratio_[0]:.2f})")
        ax.set_ylabel(f"PCA-2 (var: {pca.explained_variance_ratio_[1]:.2f})")
        
        # Add legend for the sample points
        handles = []
        for i, idx in enumerate(sample_indices):
            if idx < len(metadata):
                label = f"{metadata[idx].get('category', 'unknown')}: {metadata[idx].get('content_preview', '')[:20]}..."
                handle = plt.Line2D([0], [0], marker='o', color=plt.cm.tab10(i % 10),
                                   label=label, markersize=8)
                handles.append(handle)
        
        ax.legend(handles=handles, title="Sample Points", loc="best", fontsize=8)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        timestr = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(self.output_dir, f"fractal_transformations_{timestr}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved transformation visualization to {output_path}")
        
        return fig
    
    def create_interactive_3d_visualization(self, data, level=0):
        """
        Create an interactive 3D visualization of embeddings using Plotly.
        
        Args:
            data: Embedding data dictionary
            level: Fractal level to visualize (0 for base embeddings)
            
        Returns:
            The plotly figure object or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with: pip install plotly")
            return None
        
        print(f"Creating interactive 3D visualization for level {level}...")
        
        # Get embeddings
        if level == 0:
            embeddings = data['base_embeddings']
            title = "Base Embeddings - 3D Visualization"
        else:
            # Check if level exists
            if level not in data['fractal_embeddings']:
                print(f"Level {level} not found in data")
                return None
                
            # Filter non-zero embeddings
            embeddings = data['fractal_embeddings'][level]
            non_zero_mask = np.any(embeddings != 0, axis=1)
            
            if not np.any(non_zero_mask):
                print(f"No embeddings found for level {level}")
                return None
                
            embeddings = embeddings[non_zero_mask]
            filtered_metadata = [data['metadata'][i] for i, mask in enumerate(non_zero_mask) if mask]
            
            title = f"Level {level} Embeddings - 3D Visualization"
        
        # Apply PCA for dimensionality reduction
        if len(embeddings) < 3:
            print("Not enough embeddings for 3D visualization")
            return None
            
        pca = PCA(n_components=3)
        points_3d = pca.fit_transform(embeddings)
        
        # Prepare data for plotting
        categories = []
        hover_texts = []
        retrieval_counts = []
        
        # Choose the right metadata based on level
        if level == 0:
            metadata_to_use = data['metadata']
        else:
            metadata_to_use = filtered_metadata
            
        # Extract metadata for visualization
        for meta in metadata_to_use:
            categories.append(meta.get('category', 'unknown'))
            hover_texts.append(f"ID: {meta.get('item_id', 'unknown')}<br>"
                             f"Category: {meta.get('category', 'unknown')}<br>"
                             f"Retrieved: {meta.get('retrieval_count', 0)} times<br>"
                             f"Content: {meta.get('content_preview', '')}")
            retrieval_counts.append(meta.get('retrieval_count', 0))
        
        # Create figure
        fig = px.scatter_3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            color=categories,
            size=[min(20, count + 5) for count in retrieval_counts],  # Size based on retrieval count
            hover_name=["Item " + str(i) for i in range(len(points_3d))],
            hover_data={
                "Category": categories,
                "Retrieval Count": retrieval_counts
            },
            text=["Item " + str(i) for i in range(len(points_3d))],
            title=title,
            labels={
                "x": f"PCA-1 ({pca.explained_variance_ratio_[0]:.2f})",
                "y": f"PCA-2 ({pca.explained_variance_ratio_[1]:.2f})",
                "z": f"PCA-3 ({pca.explained_variance_ratio_[2]:.2f})"
            }
        )
        
        # Customize layout
        fig.update_layout(
            scene=dict(
                xaxis_title=f"PCA-1 ({pca.explained_variance_ratio_[0]:.2f})",
                yaxis_title=f"PCA-2 ({pca.explained_variance_ratio_[1]:.2f})",
                zaxis_title=f"PCA-3 ({pca.explained_variance_ratio_[2]:.2f})"
            ),
            title=dict(
                text=title,
                x=0.5,
                y=0.95
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Save as HTML
        timestr = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(self.output_dir, f"interactive_3d_level_{level}_{timestr}.html")
        
        try:
            fig.write_html(output_path)
            print(f"Saved interactive 3D visualization to {output_path}")
        except Exception as e:
            print(f"Error saving interactive visualization: {e}")
        
        return fig
    
    def create_level_comparison_dashboard(self, data):
        """
        Create a dashboard comparing patterns across different fractal levels.
        
        Args:
            data: Embedding data dictionary
            
        Returns:
            The plotly figure object or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with: pip install plotly")
            return None
        
        print("Creating fractal level comparison dashboard...")
        
        # Get available levels
        base_embeddings = data['base_embeddings']
        fractal_levels = [level for level in sorted(data['fractal_embeddings'].keys())
                         if not np.all(data['fractal_embeddings'][level] == 0)]
        
        if not fractal_levels:
            print("No fractal embeddings found. Skipping level comparison.")
            return None
        
        # Apply PCA to base embeddings
        pca = PCA(n_components=2)
        base_2d = pca.fit_transform(base_embeddings)
        
        # Create subplot grid
        rows = len(fractal_levels) + 1
        fig = go.Figure()
        
        # Get categories for coloring
        categories = [m.get('category', 'unknown') for m in data['metadata']]
        unique_categories = sorted(set(categories))
        
        # Create plot for base embeddings
        base_trace = go.Scatter(
            x=base_2d[:, 0],
            y=base_2d[:, 1],
            mode='markers',
            marker=dict(
                size=10,
                color=[unique_categories.index(cat) for cat in categories],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Category",
                    tickvals=[i for i in range(len(unique_categories))],
                    ticktext=unique_categories
                )
            ),
            text=[f"Category: {cat}<br>Item: {i}" for i, cat in enumerate(categories)],
            name='Base Embeddings'
        )
        
        fig.add_trace(base_trace)
        
        # Process each fractal level
        for level in fractal_levels:
            # Get embeddings for this level
            level_embeddings = data['fractal_embeddings'][level]
            
            # Apply PCA to level embeddings
            try:
                level_2d = pca.transform(level_embeddings)
                
                # Create trace for this level
                level_trace = go.Scatter(
                    x=level_2d[:, 0],
                    y=level_2d[:, 1],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=[unique_categories.index(cat) for cat in categories],
                        colorscale='Viridis',
                        showscale=False
                    ),
                    text=[f"Category: {cat}<br>Item: {i}" for i, cat in enumerate(categories)],
                    name=f'Level {level}',
                    visible=False  # Initially hidden
                )
                
                fig.add_trace(level_trace)
            except Exception as e:
                print(f"Error processing level {level}: {e}")
        
        # Create buttons for switching between levels
        buttons = []
        
        # Button for base level
        buttons.append(
            dict(
                method="update",
                label="Base",
                args=[{"visible": [True] + [False] * len(fractal_levels)}],
                args2=[{"title": "Base Embeddings"}]
            )
        )
        
        # Buttons for each fractal level
        for i, level in enumerate(fractal_levels):
            visibility = [False] * (len(fractal_levels) + 1)
            visibility[i + 1] = True  # +1 because base embeddings are at index 0
            
            buttons.append(
                dict(
                    method="update",
                    label=f"Level {level}",
                    args=[{"visible": visibility}],
                    args2=[{"title": f"Level {level} Embeddings"}]
                )
            )
        
        # Add dropdown menu
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    x=0.25,
                    y=1.12,
                    buttons=buttons
                )
            ],
            title="Fractal Level Comparison (use buttons to switch levels)",
            xaxis_title=f"PCA-1 ({pca.explained_variance_ratio_[0]:.2f})",
            yaxis_title=f"PCA-2 ({pca.explained_variance_ratio_[1]:.2f})",
            legend_title="Dataset",
            height=600,
            width=800
        )
        
        # Save as HTML
        timestr = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(self.output_dir, f"level_comparison_dashboard_{timestr}.html")
        
        try:
            fig.write_html(output_path)
            print(f"Saved level comparison dashboard to {output_path}")
        except Exception as e:
            print(f"Error saving dashboard: {e}")
        
        return fig
    
    def create_mcp_integration_visualization(self, data):
        """
        Visualize how MCP-generated content is integrated into the fractal memory system.
        
        Args:
            data: Embedding data dictionary
            
        Returns:
            The figure object
        """
        print("Creating MCP integration visualization...")
        
        # Extract metadata
        metadata = data['metadata']
        
        # Identify MCP-generated content
        mcp_indices = []
        for i, meta in enumerate(metadata):
            # Check for MCP-related metadata
            if (meta.get('source', '') == 'mcp' or
                meta.get('source_hint', '') == 'mcp' or
                'filename' in meta):
                mcp_indices.append(i)
        
        if not mcp_indices:
            print("No MCP-generated content found in memory.")
            return None
        
        print(f"Found {len(mcp_indices)} items from MCP in memory.")
        
        # Get base embeddings
        base_embeddings = data['base_embeddings']
        
        # Apply PCA
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(base_embeddings)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot all points
        categories = [m.get('category', 'unknown') for m in metadata]
        unique_categories = sorted(set(categories))
        color_map = {cat: i for i, cat in enumerate(unique_categories)}
        
        # Get colors for each point
        colors = [plt.cm.tab10(color_map.get(cat, 0) % 10) for cat in categories]
        
        # Plot non-MCP points
        non_mcp_mask = np.ones(len(points_2d), dtype=bool)
        non_mcp_mask[mcp_indices] = False
        
        ax.scatter(points_2d[non_mcp_mask, 0], points_2d[non_mcp_mask, 1],
                  c=[colors[i] for i in range(len(colors)) if non_mcp_mask[i]],
                  alpha=0.5, s=30, label="Regular content")
        
        # Plot MCP points with different style
        ax.scatter(points_2d[mcp_indices, 0], points_2d[mcp_indices, 1],
                 c=[colors[i] for i in mcp_indices], marker='*', s=200,
                 edgecolors='black', linewidths=1, alpha=0.9, label="MCP content")
        
        # Add labels for MCP points
        for idx in mcp_indices:
            if 'filename' in metadata[idx]:
                label = metadata[idx]['filename']
            else:
                label = f"MCP item {idx}"
                
            ax.annotate(label, points_2d[idx], fontsize=8,
                      xytext=(5, 5), textcoords='offset points')
        
        # Add title and labels
        ax.set_title("MCP Content Integration in Embedding Space", fontsize=16)
        ax.set_xlabel(f"PCA-1 (var: {pca.explained_variance_ratio_[0]:.2f})")
        ax.set_ylabel(f"PCA-2 (var: {pca.explained_variance_ratio_[1]:.2f})")
        
        # Add legend for categories
        handles = [plt.Line2D([0], [0], marker='o', color=plt.cm.tab10(i % 10),
                             linestyle='', label=cat) 
                  for i, cat in enumerate(unique_categories)]
        
        # Add legend for MCP vs regular
        handles.append(plt.Line2D([0], [0], marker='o', color='gray',
                                linestyle='', label='Regular content'))
        handles.append(plt.Line2D([0], [0], marker='*', color='gray',
                                markersize=10, linestyle='', label='MCP content'))
        
        ax.legend(handles=handles, title="Content Types", loc="best", fontsize=8)
        
        # Save the figure
        timestr = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(self.output_dir, f"mcp_integration_{timestr}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved MCP integration visualization to {output_path}")
        
        return fig
    
    def run_all_visualizations(self, interactive=False):
        """
        Run all visualization methods.
        
        Args:
            interactive: Whether to display interactive visualizations
            
        Returns:
            Dictionary of results
        """
        print("\nRunning all fractal memory visualizations...")
        
        # Extract embeddings
        data = self.extract_embeddings()
        if not data:
            return {'error': 'No data available'}
        
        # Calculate statistics
        stats = self.create_embedding_stats(data)
        
        # Run visualizations
        results = {}
        
        # Create similarity heatmaps
        results['similarity'] = self.visualize_similarity_heatmaps(data, stats)
        
        # Create cluster visualizations
        results['clusters_pca'] = self.visualize_embedding_clusters(data, method='pca')
        results['clusters_tsne'] = self.visualize_embedding_clusters(data, method='tsne')
        
        # Create fractal transformation visualization
        results['transformation'] = self.visualize_fractal_transformation(data)
        
        # Create animated transformation
        results['animation'], animation_path = self.create_animated_transformation(data)
        
        # Create MCP integration visualization
        results['mcp'] = self.create_mcp_integration_visualization(data)
        
        # Create interactive visualizations if requested and available
        if interactive and PLOTLY_AVAILABLE:
            # Interactive 3D visualization for base embeddings
            results['interactive_3d_base'] = self.create_interactive_3d_visualization(data, level=0)
            
            # Interactive 3D for first fractal level (if available)
            if 1 in data['fractal_embeddings'] and not np.all(data['fractal_embeddings'][1] == 0):
                results['interactive_3d_level1'] = self.create_interactive_3d_visualization(data, level=1)
            
            # Level comparison dashboard
            results['level_dashboard'] = self.create_level_comparison_dashboard(data)
        
        # Display visualizations
        if interactive:
            plt.show()
        
        print("\nAll visualizations completed successfully!")
        return results
    
    def create_animated_transformation(self, data, n_samples=10):
        """
        Create an animated visualization of embeddings transforming across levels.
        
        Args:
            data: Embedding data dictionary
            n_samples: Number of sample points to visualize
            
        Returns:
            The animation object and save path
        """
        print(f"Creating animated fractal transformation visualization...")
        
        # Get embeddings
        base_embeddings = data['base_embeddings']
        metadata = data['metadata']
        
        # Only proceed if we have fractal levels
        fractal_levels = [level for level in sorted(data['fractal_embeddings'].keys())
                         if not np.all(data['fractal_embeddings'][level] == 0)]
        
        if not fractal_levels:
            print("No fractal embeddings found. Skipping animation.")
            return None, None
        
        # Use PCA to reduce to 2D for visualization
        all_embeddings = []
        all_embeddings.append(('base', base_embeddings))
        for level in fractal_levels:
            # Filter out zero embeddings
            level_embeddings = data['fractal_embeddings'][level]
            non_zero_mask = np.any(level_embeddings != 0, axis=1)
            if np.any(non_zero_mask):
                all_embeddings.append((f'level_{level}', level_embeddings))
        
        # Combine all embeddings
        combined_embeddings = np.vstack([emb for _, emb in all_embeddings])
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca.fit(combined_embeddings)
        
        # Transform each level's embeddings
        transformed_embeddings = []
        for level_name, embeddings in all_embeddings:
            transformed = pca.transform(embeddings)
            transformed_embeddings.append((level_name, transformed))
        
        # Random sample of points to visualize (for clarity)
        np.random.seed(42)
        sample_indices = np.random.choice(len(base_embeddings), size=min(n_samples, len(base_embeddings)), replace=False)
        
        # Extract categories for coloring
        categories = [metadata[idx].get('category', 'unknown') for idx in sample_indices]
        unique_categories = sorted(set(categories))
        category_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
        category_indices = [category_to_idx.get(cat, 0) for cat in categories]
        
        # Create figure for animation
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare frame data
        frame_data = []
        for i in range(len(transformed_embeddings) - 1):
            start_level_name, start_points = transformed_embeddings[i]
            end_level_name, end_points = transformed_embeddings[i + 1]
            
            # Interpolate between levels
            n_frames = 10
            for j in range(n_frames):
                alpha = j / (n_frames - 1)
                interp_points = (1 - alpha) * start_points + alpha * end_points
                frame_data.append((f"{start_level_name} → {end_level_name} ({j+1}/{n_frames})", interp_points))
        
        # Animation update function
        def update(frame):
            ax.clear()
            frame_title, points = frame_data[frame]
            
            # Plot each sample point
            for k, idx in enumerate(sample_indices):
                if idx < len(points):
                    color = plt.cm.tab10(category_indices[k] % 10)
                    ax.scatter(points[idx, 0], points[idx, 1], color=color, 
                              s=100, alpha=0.7, edgecolors='white')
                    
                    # Add label
                    if frame == 0:  # Only add labels in the first frame
                        label = metadata[idx].get('category', 'unknown')
                        ax.annotate(label, points[idx], fontsize=8,
                                   xytext=(5, 5), textcoords='offset points')
            
            # Set title and labels
            ax.set_title(f"Fractal Transformation: {frame_title}", fontsize=14)
            ax.set_xlabel(f"PCA-1 (var: {pca.explained_variance_ratio_[0]:.2f})")
            ax.set_ylabel(f"PCA-2 (var: {pca.explained_variance_ratio_[1]:.2f})")
            
            # Add legend
            handles = []
            for i, cat in enumerate(unique_categories):
                handle = plt.Line2D([0], [0], marker='o', color=plt.cm.tab10(i % 10),
                                   label=cat, markersize=8, linestyle='')
                handles.append(handle)
            
            ax.legend(handles=handles, title="Categories", loc="upper right", fontsize=8)
            
            # Set consistent axis limits
            ax.set_xlim(combined_min[0], combined_max[0])
            ax.set_ylim(combined_min[1], combined_max[1])
        
        # Calculate overall min/max for consistent axes
        all_transformed_points = np.vstack([points for _, points in transformed_embeddings])
        combined_min = np.min(all_transformed_points, axis=0)
        combined_max = np.max(all_transformed_points, axis=0)
        
        # Create the animation
        anim = FuncAnimation(fig, update, frames=len(frame_data), interval=300, blit=False)
        
        # Save the animation
        timestr = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(self.output_dir, f"fractal_animation_{timestr}.gif")
        
        try:
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=5)
            anim.save(output_path, writer=writer)
            print(f"Saved animation to {output_path}")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Install Pillow for GIF support: pip install Pillow")
            output_path = None
        
        return anim, output_path