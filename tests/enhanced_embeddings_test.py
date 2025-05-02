"""
Enhanced Embeddings Visualization Test

This script tests the enhanced (formerly "fractal") embedding system in the unified memory manager.
It creates visualizations showing how embeddings transform across different levels and analyzes
retrieval performance improvements from these transformations.

Requirements:
- matplotlib
- numpy
- sklearn (for PCA and TSNE visualizations)
- seaborn (optional, for better visualizations)

Usage:
    python enhanced_embeddings_test.py
"""

import os
import sys
import time
import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Try to import optional visualization packages
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Seaborn not available. Basic visualizations will be used.")

try:
    # Import project modules
    from unified_memory import MemoryManager
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)

# Use scikit-learn for dimensionality reduction if available
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not available. Some visualizations will be disabled.")


def create_test_memory(memory_dir="./test_memory", sample_size=50):
    """
    Create a test memory instance with categorized data for visualization.
    
    Args:
        memory_dir: Directory to store memory data
        sample_size: Number of samples to generate
        
    Returns:
        Initialized MemoryManager with test data
    """
    os.makedirs(memory_dir, exist_ok=True)
    
    # Initialize memory manager with enhanced embeddings enabled
    memory = MemoryManager(
        storage_path=memory_dir,
        embedding_dim=384,
        enable_enhanced_embeddings=True,
        max_enhancement_levels=3,
        similarity_enhancement_factor=0.4,
        auto_save=True
    )
    
    # Test data categories with example templates
    categories = {
        "programming": [
            "Python is {adj} for {task}.",
            "JavaScript can be used to {action} in web development.",
            "The {framework} framework helps developers {action}.",
            "Software engineers use {tool} to {action}.",
            "The {algorithm} algorithm is used to solve {problem}."
        ],
        "science": [
            "The theory of {concept} explains how {entity} {action}.",
            "{scientist} discovered {discovery} in the field of {field}.",
            "The {element} element has properties that allow it to {action}.",
            "Research in {field} has shown that {entity} can {action}.",
            "The {process} process occurs when {condition}."
        ],
        "history": [
            "In {year}, {event} changed the course of {entity}.",
            "The {civilization} civilization was known for their {achievement}.",
            "{leader} led {group} during the {period}.",
            "The {war} resulted in significant changes to {entity}.",
            "Historians believe that {entity} was responsible for {event}."
        ],
        "arts": [
            "The {movement} movement influenced {field} during the {period}.",
            "{artist} created {artwork} using {technique}.",
            "The {color} in the painting represents {concept}.",
            "The {style} style is characterized by {feature}.",
            "In {art_form}, the {element} is used to convey {emotion}."
        ],
        "technology": [
            "The latest advancements in {field} allow for {capability}.",
            "{company} developed {product} to {action}.",
            "The {device} uses {technology} to {action}.",
            "Engineers at {company} are working on {project}.",
            "The {protocol} protocol enables {system} to {action}."
        ]
    }
    
    # Fill-in values for templates
    fillers = {
        "adj": ["useful", "efficient", "powerful", "great", "effective", "ideal"],
        "task": ["data analysis", "web development", "automation", "machine learning", "game development"],
        "action": ["create interactive elements", "optimize performance", "process data", "develop applications", "implement features"],
        "framework": ["React", "Django", "TensorFlow", "Angular", "Flask", "Spring"],
        "tool": ["Git", "Docker", "Kubernetes", "Jenkins", "VSCode", "PyCharm"],
        "algorithm": ["sorting", "search", "machine learning", "graph traversal", "optimization"],
        "problem": ["performance bottlenecks", "data organization", "resource allocation", "pattern recognition"],
        
        "concept": ["relativity", "evolution", "quantum mechanics", "thermodynamics", "gravity"],
        "entity": ["particles", "organisms", "planets", "elements", "molecules", "ecosystems"],
        "scientist": ["Einstein", "Newton", "Darwin", "Curie", "Hawking", "Tesla"],
        "discovery": ["radioactivity", "penicillin", "DNA structure", "the Higgs boson", "gravitational waves"],
        "field": ["physics", "biology", "chemistry", "astronomy", "geology", "medicine"],
        "element": ["carbon", "hydrogen", "oxygen", "gold", "uranium", "silicon"],
        "process": ["photosynthesis", "nuclear fusion", "oxidation", "fermentation", "mitosis"],
        "condition": ["temperature increases", "pressure decreases", "catalysts are present", "energy is applied"],
        
        "year": ["1776", "1945", "1969", "1989", "2001", "1492"],
        "event": ["World War II", "the Industrial Revolution", "the moon landing", "the Renaissance", "the French Revolution"],
        "civilization": ["Roman", "Egyptian", "Chinese", "Incan", "Greek", "Mayan"],
        "achievement": ["architecture", "mathematics", "astronomy", "agriculture", "writing systems"],
        "leader": ["Napoleon", "Lincoln", "Churchill", "Cleopatra", "Alexander", "Caesar"],
        "group": ["France", "the Allied forces", "ancient Egypt", "the Roman Empire", "the revolution"],
        "period": ["18th century", "Renaissance", "Cold War", "Bronze Age", "Middle Ages"],
        "war": ["American Civil War", "World War I", "Peloponnesian War", "Hundred Years' War"],
        
        "movement": ["Impressionist", "Cubist", "Surrealist", "Renaissance", "Romantic"],
        "artist": ["Picasso", "Van Gogh", "Monet", "Leonardo da Vinci", "Frida Kahlo"],
        "artwork": ["Starry Night", "Guernica", "Mona Lisa", "The Persistence of Memory", "The Scream"],
        "technique": ["oil painting", "watercolor", "sculpture", "photography", "digital art"],
        "color": ["blue", "red", "yellow", "green", "black", "white"],
        "style": ["Baroque", "Gothic", "Minimalist", "Abstract", "Realist"],
        "feature": ["geometric shapes", "vibrant colors", "emotional expression", "detailed realism", "symbolism"],
        "art_form": ["music", "dance", "theater", "literature", "film", "painting"],
        "element": ["rhythm", "composition", "harmony", "contrast", "motion", "perspective"],
        "emotion": ["joy", "sorrow", "fear", "hope", "love", "despair"],
        
        "company": ["Google", "Apple", "Microsoft", "Amazon", "Tesla", "IBM"],
        "product": ["smartphones", "AI assistants", "cloud platforms", "electric vehicles", "quantum computers"],
        "device": ["smartphone", "laptop", "smart speaker", "wearable device", "virtual reality headset"],
        "technology": ["artificial intelligence", "blockchain", "5G", "quantum computing", "augmented reality"],
        "protocol": ["HTTP", "TCP/IP", "Bluetooth", "Wi-Fi", "NFC", "MQTT"],
        "system": ["computers", "networks", "smartphones", "databases", "autonomous vehicles"],
        "project": ["self-driving cars", "quantum computing", "smart cities", "space exploration", "renewable energy"]
    }
    
    # Helper function to fill a template
    def fill_template(template):
        # Find all placeholders in the template
        import re
        placeholders = re.findall(r'\{(\w+)\}', template)
        
        # Fill each placeholder
        filled_template = template
        for placeholder in placeholders:
            if placeholder in fillers:
                value = random.choice(fillers[placeholder])
                filled_template = filled_template.replace(f"{{{placeholder}}}", value)
            else:
                filled_template = filled_template.replace(f"{{{placeholder}}}", f"unknown-{placeholder}")
                
        return filled_template
    
    # Define a deterministic but unique embedding function
    def category_based_embedding(text, category=None):
        """Generate a contextualized embedding based on text content and category"""
        # Use a hash of the text for deterministic randomness
        text_hash = sum(ord(c) for c in text[:20])
        np.random.seed(text_hash)
        
        # Create a base random embedding
        embedding = np.random.randn(384)
        
        # If category is provided, bias the embedding in category-specific ways
        if category:
            # Create category biases
            category_biases = {
                "programming": np.array([1.0, 0.2, 0.1, 0.0, 0.1] + [0.0] * (384 - 5)),
                "science": np.array([0.1, 1.0, 0.2, 0.1, 0.0] + [0.0] * (384 - 5)),
                "history": np.array([0.1, 0.1, 1.0, 0.2, 0.1] + [0.0] * (384 - 5)),
                "arts": np.array([0.0, 0.1, 0.2, 1.0, 0.1] + [0.0] * (384 - 5)),
                "technology": np.array([0.8, 0.1, 0.0, 0.1, 1.0] + [0.0] * (384 - 5))
            }
            
            # Get bias for this category
            bias = category_biases.get(category, np.zeros(384))
            
            # Apply bias to first few dimensions
            embedding[:5] = embedding[:5] * 0.3 + bias[:5]
            
            # Add category signature to other dimensions
            cat_sig_start = (hash(category) % 30) + 10  # Start position between 10-40
            cat_signature = np.array([0.5, 0.8, 0.3, -0.2, 0.7, -0.5, 0.1, 0.9, -0.4, 0.6])
            embedding[cat_sig_start:cat_sig_start+10] = embedding[cat_sig_start:cat_sig_start+10] * 0.3 + cat_signature
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm
            
        return embedding
    
    # Create batch version for efficiency
    def batch_embedding_function(texts):
        """Process a batch of texts, ignoring category"""
        return [category_based_embedding(text) for text in texts]
    
    # Set embedding functions
    memory.set_embedding_function(category_based_embedding, batch_embedding_function)
    
    # Generate test data
    print(f"Generating {sample_size} test memory items across {len(categories)} categories...")
    
    for i in range(sample_size):
        # Cycle through categories
        category = list(categories.keys())[i % len(categories)]
        
        # Select random template and fill it
        template = random.choice(categories[category])
        content = fill_template(template)
        
        # Create embedding with category bias
        embedding = category_based_embedding(content, category)
        
        # Add to memory
        item_id = memory.add(
            content=content,
            metadata={"category": category, "index": i}
        )
        
        # Show progress
        if (i + 1) % 10 == 0 or i + 1 == sample_size:
            print(f"  Added {i+1}/{sample_size} items")
    
    # Save memory
    memory.cleanup()
    print(f"Memory created with {sample_size} items in {memory_dir}")
    
    return memory


def visualize_embeddings(memory, output_dir="./visualizations"):
    """
    Create visualizations of the enhanced embedding space.
    
    Args:
        memory: Initialized MemoryManager with test data
        output_dir: Directory to save visualizations
    """
    if not SKLEARN_AVAILABLE:
        print("Skipping embedding visualization - scikit-learn not available")
        return
        
    print("\nCreating embedding visualizations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract embeddings and metadata
    base_embeddings = []
    enhanced_embeddings = {1: [], 2: [], 3: []}
    categories = []
    
    # Get all items
    item_count = len(memory.items)
    print(f"Extracting embeddings from {item_count} memory items...")
    
    # Extract data for each item
    for idx, item in enumerate(memory.items):
        # Skip deleted items
        if idx in memory.deleted_ids:
            continue
            
        # Extract base embedding
        base_embeddings.append(item.embedding)
        
        # Extract enhanced embeddings for each level
        for level in range(1, 4):
            if hasattr(item, 'additional_embeddings') and level in item.additional_embeddings:
                enhanced_embeddings[level].append(item.additional_embeddings[level])
            else:
                # Use zeros as placeholder for missing embeddings
                enhanced_embeddings[level].append(np.zeros(item.embedding.shape))
        
        # Extract category
        category = item.metadata.get('category', 'unknown')
        categories.append(category)
        
        # Show progress
        if (idx + 1) % 20 == 0 or idx + 1 == item_count:
            print(f"  Processed {idx+1}/{item_count} items")
    
    # Convert to numpy arrays
    base_embeddings = np.array(base_embeddings)
    for level in enhanced_embeddings:
        enhanced_embeddings[level] = np.array(enhanced_embeddings[level])
    
    # Get unique categories for coloring
    unique_categories = sorted(set(categories))
    category_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    category_indices = [category_to_idx[cat] for cat in categories]
    
    # Set up a timestamp for unique filenames
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    # ============ 1. PCA Visualization for Each Level ============
    print("\nCreating PCA visualizations...")
    
    # Apply PCA to all embeddings together for consistent projection
    all_embeddings = [base_embeddings]
    for level in range(1, 4):
        # Filter out zero embeddings
        non_zero = np.any(enhanced_embeddings[level] != 0, axis=1)
        if np.any(non_zero):
            level_embeddings = enhanced_embeddings[level][non_zero]
            all_embeddings.append(level_embeddings)
    
    combined_embeddings = np.vstack(all_embeddings)
    pca = PCA(n_components=2)
    combined_reduced = pca.fit_transform(combined_embeddings)
    
    # Split back into levels
    start_idx = 0
    base_reduced = combined_reduced[:len(base_embeddings)]
    start_idx += len(base_embeddings)
    
    level_reduced = {}
    for level in range(1, 4):
        non_zero = np.any(enhanced_embeddings[level] != 0, axis=1)
        if np.any(non_zero):
            level_size = np.sum(non_zero)
            level_reduced[level] = combined_reduced[start_idx:start_idx+level_size]
            start_idx += level_size
    
    # Create separate plots for each level
    for level_name, reduced_data in [("Base", base_reduced)] + [(f"Level {level}", level_reduced.get(level)) 
                                                               for level in range(1, 4) if level in level_reduced]:
        if reduced_data is None or len(reduced_data) == 0:
            continue
            
        plt.figure(figsize=(10, 8))
        
        # For base level, we have all data points
        if level_name == "Base":
            # Use a good-looking colormap
            if SEABORN_AVAILABLE:
                sns.set_style("whitegrid")
                palette = sns.color_palette("hls", len(unique_categories))
                cmap = ListedColormap(palette)
            else:
                cmap = plt.cm.get_cmap('tab10', len(unique_categories))
            
            scatter = plt.scatter(base_reduced[:, 0], base_reduced[:, 1], 
                               c=category_indices, cmap=cmap, 
                               s=100, alpha=0.7, edgecolors='w')
            
            # Add a legend
            handles, labels = scatter.legend_elements()
            plt.legend(handles, unique_categories, title="Categories", loc="upper right")
            
            # Annotate some points for clarity
            for i, category in enumerate(categories):
                if i % 10 == 0:  # Label every 10th point to avoid clutter
                    x, y = base_reduced[i]
                    plt.annotate(category, (x, y), fontsize=8, 
                                ha='center', va='center',
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        
        # For enhanced levels, we only have embeddings for some points
        else:
            # Get the level number
            level = int(level_name.split()[-1])
            
            # Get indices of points that have this level embedding
            non_zero = np.any(enhanced_embeddings[level] != 0, axis=1)
            level_categories = [categories[i] for i, has_emb in enumerate(non_zero) if has_emb]
            level_cat_indices = [category_to_idx[cat] for cat in level_categories]
            
            # Create scatter plot
            if SEABORN_AVAILABLE:
                sns.set_style("whitegrid")
                palette = sns.color_palette("hls", len(unique_categories))
                cmap = ListedColormap(palette)
            else:
                cmap = plt.cm.get_cmap('tab10', len(unique_categories))
                
            scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                               c=level_cat_indices, cmap=cmap, 
                               s=100, alpha=0.7, edgecolors='w')
            
            # Add a legend
            handles, labels = scatter.legend_elements()
            legend_labels = [unique_categories[idx] for idx in sorted(set(level_cat_indices))]
            plt.legend(handles, legend_labels, title="Categories", loc="upper right")
            
            # Annotate some points
            for i, category in enumerate(level_categories):
                if i % 10 == 0:  # Label every 10th point
                    x, y = reduced_data[i]
                    plt.annotate(category, (x, y), fontsize=8, 
                                ha='center', va='center',
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        
        # Add title and labels
        plt.title(f"{level_name} Embeddings - PCA Visualization")
        explained_var = pca.explained_variance_ratio_
        plt.xlabel(f"Component 1 ({explained_var[0]:.2%} variance)")
        plt.ylabel(f"Component 2 ({explained_var[1]:.2%} variance)")
        
        # Save figure
        filename = f"pca_{level_name.lower().replace(' ', '_')}_{timestr}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved {filepath}")
    
    # ============ 2. Level Transformation Visualization ============
    print("\nCreating level transformation visualizations...")
    
    # Create a transformation visualization for a few sample points
    # Select one sample from each category
    samples = {}
    for i, category in enumerate(categories):
        if category not in samples:
            samples[category] = i
    
    # Create figure showing transformations across levels
    plt.figure(figsize=(15, 10))
    
    # Use a good-looking colormap
    if SEABORN_AVAILABLE:
        sns.set_style("whitegrid")
        palette = sns.color_palette("hls", len(unique_categories))
    else:
        palette = [plt.cm.tab10(i) for i in range(len(unique_categories))]
    
    # Plot each sample and its transformations
    for category, idx in samples.items():
        # Get base embedding's position
        cat_idx = category_to_idx[category]
        color = palette[cat_idx]
        
        x, y = base_reduced[idx]
        
        # Plot base point
        plt.scatter(x, y, color=color, s=150, label=f"{category} (Base)", 
                   marker='o', edgecolors='white', linewidth=1.5, alpha=0.8)
        
        # Plot enhanced embeddings at each level
        for level in range(1, 4):
            if not np.any(enhanced_embeddings[level][idx] != 0):
                continue  # Skip if no embedding at this level
                
            # Get the level embedding's position
            # Find the corresponding index in the level's reduced data
            level_non_zero = np.any(enhanced_embeddings[level] != 0, axis=1)
            if not level_non_zero[idx]:
                continue
                
            level_idx = np.where(level_non_zero[:idx+1])[0].size - 1
            level_x, level_y = level_reduced[level][level_idx]
            
            # Plot level point with different marker
            markers = ['s', '^', 'd']  # square, triangle, diamond
            plt.scatter(level_x, level_y, color=color, s=150,
                       marker=markers[level-1], edgecolors='white', linewidth=1.5,
                       alpha=0.8)
            
            # Draw connection line
            plt.plot([x, level_x], [y, level_y], color=color, linestyle='--', alpha=0.6)
            
            # Add label
            plt.annotate(f"L{level}", (level_x, level_y), fontsize=9,
                        ha='center', va='center', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    
    # Add legend for categories (manually to avoid duplicates)
    handles = []
    for i, category in enumerate(unique_categories):
        if category in samples:
            handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], 
                               markersize=10, label=category)
            handles.append(handle)
    
    # Add legend for level markers
    handles.append(plt.Line2D([0], [0], marker='o', color='gray', 
                            markersize=10, label='Base'))
    handles.append(plt.Line2D([0], [0], marker='s', color='gray', 
                            markersize=10, label='Level 1'))
    handles.append(plt.Line2D([0], [0], marker='^', color='gray', 
                            markersize=10, label='Level 2'))
    handles.append(plt.Line2D([0], [0], marker='d', color='gray', 
                            markersize=10, label='Level 3'))
    
    plt.legend(handles=handles, title="Legend", loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add title and labels
    plt.title("Enhanced Embedding Transformations Across Levels", fontsize=16)
    explained_var = pca.explained_variance_ratio_
    plt.xlabel(f"Component 1 ({explained_var[0]:.2%} variance)")
    plt.ylabel(f"Component 2 ({explained_var[1]:.2%} variance)")
    
    # Save figure
    filename = f"level_transformations_{timestr}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved {filepath}")
    
    # ============ 3. Create a 3D visualization if matplotlib 3D is available ============
    try:
        print("\nCreating 3D visualization...")
        from mpl_toolkits.mplot3d import Axes3D
        
        # Apply PCA to get 3 components
        pca3d = PCA(n_components=3)
        base_3d = pca3d.fit_transform(base_embeddings)
        
        # Create 3D figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points with category colors
        for i, category in enumerate(unique_categories):
            # Get indices for this category
            cat_indices = [j for j, cat in enumerate(categories) if cat == category]
            
            # Plot these points
            cat_points = base_3d[cat_indices]
            ax.scatter(cat_points[:, 0], cat_points[:, 1], cat_points[:, 2],
                      label=category, s=70, alpha=0.7)
        
        # Add labels and legend
        explained_var = pca3d.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({explained_var[0]:.2%})")
        ax.set_ylabel(f"PC2 ({explained_var[1]:.2%})")
        ax.set_zlabel(f"PC3 ({explained_var[2]:.2%})")
        
        plt.title("3D PCA Visualization of Base Embeddings", fontsize=14)
        plt.legend(title="Categories", loc="upper right")
        
        # Save figure
        filename = f"pca_3d_visualization_{timestr}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved {filepath}")
        
    except Exception as e:
        print(f"Skipping 3D visualization: {e}")
    
    # Close all figures
    plt.close('all')
    print(f"All visualizations saved to {os.path.abspath(output_dir)}")


def test_retrieval_performance(memory, output_dir="./visualizations"):
    """
    Test retrieval performance with and without enhanced embeddings.
    
    Args:
        memory: Initialized MemoryManager
        output_dir: Directory to save visualization
    """
    print("\nTesting retrieval performance...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test queries - one for each category
    test_queries = {
        "programming": [
            "How is Python used in software development?",
            "What are the best frameworks for web development?",
            "Tell me about algorithms used in computer science."
        ],
        "science": [
            "What discoveries have been made in physics recently?",
            "How do chemical reactions work?",
            "Tell me about important scientific theories."
        ],
        "history": [
            "What important events happened in the 20th century?",
            "Tell me about ancient civilizations and their achievements.",
            "Who were influential leaders throughout history?"
        ],
        "arts": [
            "Describe important art movements and styles.",
            "What techniques do artists use in their work?",
            "How does color theory apply to paintings?"
        ],
        "technology": [
            "What are the latest advancements in technology?",
            "How are companies developing new products?",
            "Tell me about devices that use artificial intelligence."
        ]
    }
    
    # Initialize results tracking
    results = {
        "standard": {cat: [] for cat in test_queries},
        "enhanced": {cat: [] for cat in test_queries}
    }
    
    # Test each query with and without enhanced embeddings
    for category, queries in test_queries.items():
        print(f"\nTesting retrieval for category: {category}")
        
        for query in queries:
            print(f"  Query: '{query}'")
            
            # Test with standard retrieval (no enhanced embeddings)
            standard_results = memory.retrieve(
                query=query,
                top_k=5,
                min_similarity=0.1,
                use_enhanced_embeddings=False
            )
            
            # Calculate average relevance (items matching the query category)
            correct_standard = sum(1 for r in standard_results if r['metadata'].get('category') == category)
            avg_standard_similarity = sum(r['similarity'] for r in standard_results) / len(standard_results) if standard_results else 0
            
            print(f"    Standard: {correct_standard}/5 correct, avg similarity: {avg_standard_similarity:.3f}")
            
            # Test with enhanced retrieval
            enhanced_results = memory.retrieve(
                query=query,
                top_k=5,
                min_similarity=0.1,
                use_enhanced_embeddings=True
            )
            
            # Calculate average relevance
            correct_enhanced = sum(1 for r in enhanced_results if r['metadata'].get('category') == category)
            avg_enhanced_similarity = sum(r['similarity'] for r in enhanced_results) / len(enhanced_results) if enhanced_results else 0
            
            print(f"    Enhanced: {correct_enhanced}/5 correct, avg similarity: {avg_enhanced_similarity:.3f}")
            
            # Record results
            results["standard"][category].append({
                "query": query,
                "correct": correct_standard,
                "avg_similarity": avg_standard_similarity,
                "top_result_similarity": standard_results[0]['similarity'] if standard_results else 0
            })
            
            results["enhanced"][category].append({
                "query": query,
                "correct": correct_enhanced,
                "avg_similarity": avg_enhanced_similarity,
                "top_result_similarity": enhanced_results[0]['similarity'] if enhanced_results else 0,
                "level_used": enhanced_results[0].get('level', 'Base') if enhanced_results else 'None'
            })
    
    # Calculate overall stats
    overall_standard_correct = sum(r["correct"] for cat in results["standard"] for r in results["standard"][cat])
    overall_enhanced_correct = sum(r["correct"] for cat in results["enhanced"] for r in results["enhanced"][cat])
    
    overall_standard_similarity = sum(r["avg_similarity"] for cat in results["standard"] for r in results["standard"][cat]) / sum(len(results["standard"][cat]) for cat in results["standard"])
    overall_enhanced_similarity = sum(r["avg_similarity"] for cat in results["enhanced"] for r in results["enhanced"][cat]) / sum(len(results["enhanced"][cat]) for cat in results["enhanced"])
    
    total_queries = sum(len(queries) for queries in test_queries.values())
    max_correct = total_queries * 5  # 5 results per query
    
    print(f"\nOverall results:")
    print(f"  Standard: {overall_standard_correct}/{max_correct} correct ({overall_standard_correct/max_correct:.1%}), avg similarity: {overall_standard_similarity:.3f}")
    print(f"  Enhanced: {overall_enhanced_correct}/{max_correct} correct ({overall_enhanced_correct/max_correct:.1%}), avg similarity: {overall_enhanced_similarity:.3f}")
    print(f"  Improvement: {overall_enhanced_correct - overall_standard_correct} more correct results ({(overall_enhanced_correct - overall_standard_correct)/max_correct:.1%})")
    
    # Create visualization
    if not plt:
        print("Skipping visualization - matplotlib not available")
        return
        
    # Create bar chart comparing correct results
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    categories = list(test_queries.keys())
    standard_correct = [sum(r["correct"] for r in results["standard"][cat]) for cat in categories]
    enhanced_correct = [sum(r["correct"] for r in results["enhanced"][cat]) for cat in categories]
    
    # Calculate max possible correct for each category
    max_possible = [len(test_queries[cat]) * 5 for cat in categories]
    
    # Calculate percentages
    standard_percent = [100 * c / m for c, m in zip(standard_correct, max_possible)]
    enhanced_percent = [100 * c / m for c, m in zip(enhanced_correct, max_possible)]
    
    # Set up bar chart
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, standard_percent, width, label='Standard', color='skyblue')
    rects2 = ax.bar(x + width/2, enhanced_percent, width, label='Enhanced', color='orange')
    
    # Add details
    ax.set_title('Retrieval Accuracy by Category: Standard vs Enhanced Embeddings', fontsize=16)
    ax.set_xlabel('Category', fontsize=14)
    ax.set_ylabel('Correct Results (%)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 105)  # Allow a little room above 100%
    ax.legend()
    
    # Add value labels on bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    
    # Add improvement percentages
    for i, (std, enh) in enumerate(zip(standard_percent, enhanced_percent)):
        improvement = enh - std
        color = 'green' if improvement > 0 else 'red'
        ax.annotate(f'+{improvement:.1f}%' if improvement > 0 else f'{improvement:.1f}%',
                   xy=(i, max(std, enh) + 2),
                   ha='center', va='bottom',
                   color=color, weight='bold')
    
    # Adjust layout and save
    fig.tight_layout()
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = f"retrieval_comparison_{timestr}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nSaved retrieval comparison visualization to {filepath}")
    
    # Also create a plot showing which levels were used
    plt.figure(figsize=(10, 6))
    
    # Collect level usage data
    level_counts = {'Base': 0, 'Level 1': 0, 'Level 2': 0, 'Level 3': 0, 'None': 0}
    
    for cat in results["enhanced"]:
        for query_result in results["enhanced"][cat]:
            level = query_result.get("level_used")
            if level is None or level == 'None':
                level_counts["None"] += 1
            elif level == "Base":
                level_counts["Base"] += 1
            else:
                level_counts[f"Level {level}"] += 1
    
    # Create pie chart
    labels = [f"{level}: {count}" for level, count in level_counts.items() if count > 0]
    sizes = [count for level, count in level_counts.items() if count > 0]
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
           colors=['lightblue', 'lightgreen', 'lightsalmon', 'lightpink', 'lightgray'])
    plt.axis('equal')
    plt.title('Enhanced Retrieval: Level Usage for Top Results')
    
    # Save figure
    filename = f"level_usage_{timestr}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved level usage visualization to {filepath}")
    
    # Close all figures
    plt.close('all')


def main():
    """Main function to run all tests"""
    print(f"Starting enhanced embeddings tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    memory_dir = "./test_memory"
    output_dir = "./visualizations"
    os.makedirs(memory_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create test memory with enhanced embeddings
        memory = create_test_memory(memory_dir, sample_size=100)
        
        # Create visualizations
        visualize_embeddings(memory, output_dir)
        
        # Test retrieval performance
        test_retrieval_performance(memory, output_dir)
        
    except Exception as e:
        print(f"Error during tests: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nTests completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output files available in {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()