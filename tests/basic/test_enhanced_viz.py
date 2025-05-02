import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_memory import MemoryManager

def test_enhanced_embeddings_visualization():
    """Visualize how enhanced embeddings transform the vector space"""

    # Initialize memory manager with enhanced embeddings
    memory_dir = "./test_memory"
    os.makedirs(memory_dir, exist_ok=True)
    memory = MemoryManager(
        storage_path=memory_dir,
        embedding_dim=384,
        enable_enhanced_embeddings=True,
        max_enhancement_levels=3
    )

    # Define category-based embedding function
    def category_embedding(text):
        categories = {
            "sports": [1.0, 0.2, 0.1],
            "technology": [0.1, 1.0, 0.2],
            "arts": [0.2, 0.1, 1.0],
            "science": [0.8, 0.8, 0.2],
            "history": [0.3, 0.3, 0.9]
        }

        # Determine which category the text belongs to
        for category, seed in categories.items():
            if category in text.lower():
                np.random.seed(sum(ord(c) for c in category))
                break
        else:
            # Default if no category matches
            np.random.seed(sum(ord(c) for c in text[:10]))
            seed = [0.5, 0.5, 0.5]

        # Create embedding with bias toward category
        embedding = np.random.randn(384)
        embedding[:3] = embedding[:3] * 0.1 + np.array(seed)  # Inject category signal
        return embedding / np.linalg.norm(embedding)

    # Set the embedding function
    memory.set_embedding_function(category_embedding)

    # Add test items in different categories
    print("Adding test items to memory...")
    categories = ["sports", "technology", "arts", "science", "history"]
    test_items = []

    for category in categories:
        for i in range(5):  # 5 items per category
            content = f"This is a {category} item about {category} topic {i+1}."
            item_id = memory.add(
                content=content,
                metadata={"category": category}
            )
            item = memory.get(item_id)
            if item:
                test_items.append(item)

    print(f"Added {len(test_items)} items across {len(categories)} categories")

    # Get base and enhanced embeddings
    base_embeddings = []
    enhanced_embeddings = {1: [], 2: [], 3: []}
    labels = []

    for i, item_dict in enumerate(test_items):
        idx = memory.id_to_index.get(item_dict["id"])
        if idx is not None:
            item = memory.items[idx]
            base_embeddings.append(item.embedding)

            for level in range(1, 4):
                if level in item.additional_embeddings:
                    enhanced_embeddings[level].append(item.additional_embeddings[level])
                else:
                    # Create placeholder embedding if level doesn't exist
                    enhanced_embeddings[level].append(np.zeros(384))

            labels.append(item_dict["metadata"]["category"])

    # Use PCA to visualize in 2D
    def plot_embeddings(embeddings, level_name):
        if not embeddings:
            return

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))

        # Map categories to colors
        category_colors = {
            "sports": "red",
            "technology": "blue",
            "arts": "green",
            "science": "purple",
            "history": "orange"
        }

        # Plot points
        for i, (x, y) in enumerate(reduced):
            category = labels[i]
            color = category_colors.get(category, "gray")
            plt.scatter(x, y, color=color)
            plt.text(x, y, category, fontsize=8)

        plt.title(f"{level_name} Embeddings - PCA Visualization")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")

        # Add legend
        for category, color in category_colors.items():
            plt.scatter([], [], color=color, label=category)
        plt.legend()

        plt.tight_layout()
        output_file = f"embedding_viz_{level_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_file)
        print(f"Saved visualization to {output_file}")
        plt.close()

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_embeddings(base_embeddings, "Base")

    for level, embs in enhanced_embeddings.items():
        if embs:
            plot_embeddings(embs, f"Level {level}")

    # Clean up
    memory.cleanup()
    print("\nTest complete")

if __name__ == "__main__":
    test_enhanced_embeddings_visualization()