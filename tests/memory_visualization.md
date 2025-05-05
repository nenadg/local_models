# Memory Visualization Utility Documentation

## Overview

The Memory Visualization utility creates visual representations of the embedding space in the unified memory system. It helps users understand how memory items are organized, how they relate to each other, and how enhancement levels affect semantic relationships.
The tool works by loading memory data directly from storage files, processing the embeddings through dimensionality reduction techniques, and generating various visualizations that offer different perspectives on the memory system's organization.

## Installation Requirements

- Python 3.6+
- NumPy
- Matplotlib
- scikit-learn
- seaborn
- Plotly (optional, for interactive 3D visualizations)

Install dependencies with:

```bash
pip install numpy matplotlib scikit-learn seaborn
pip install plotly  # Optional, for interactive visualizations
```

## Usage

Run the visualization utility from the command line:

```bash
python memory_visualization.py --memory-dir ./memory --output-dir ./visualizations
```

## Command Line Arguments

`--memory-dir`: Directory containing memory data files (default: "./memory")
`--output-dir`: Directory to save visualizations (default: "./visualizations")
`--max-items`: Maximum number of memory items to include (default: 100)
`--interactive`: Enable interactive 3D visualization (requires Plotly)
`--seed-query`: Optional query to find semantically related items

## Visualization Types

The utility generates up to five different visualizations:

## 1. Semantic Cluster Visualization

A polar plot showing how memory items cluster semantically across different enhancement levels. Items are arranged in concentric rings, with the base level in the center and enhancement levels in outer rings. Items are colored by category, and connections between enhancement levels show how concepts transform.

## 2. Embedding Fingerprints

A grid of heatmaps showing how individual memory items' embeddings transform across enhancement levels. Each "fingerprint" represents a memory item, with rows showing different enhancement levels and columns showing embedding dimensions. The overlay lines highlight patterns of transformation.

## 3. Concept Flow Visualization

A flow diagram showing how memory items move through the embedding space across enhancement levels. The x-axis represents enhancement level progression, and the y-axis represents semantic position. Lines connect the same item across different levels, illustrating transformation patterns.

## 4. Content Similarity Matrix

A heatmap showing the cosine similarity between memory items based on their base embeddings. Brighter colors indicate higher similarity. This helps identify clusters of related content in the memory store.

## 5. Retrieval Frequency Heatmap

A scatter plot showing memory items in a 2D embedding space, with color and size indicating how frequently each item has been retrieved. Larger, brighter dots represent more frequently accessed items.

## How It Works

1. Data Loading: The tool loads memory item data from JSON files and embeddings from NumPy arrays, along with any enhanced embeddings from pickle files.
2. Dimensionality Reduction: High-dimensional embeddings are reduced to 2D or 3D using Principal Component Analysis (PCA) for visualization.
3. Category Detection: Memory items are categorized based on their metadata or content patterns.
4. Visualization Generation: Each visualization type is created using the processed data, with careful handling of edge cases like missing enhancement levels.
5. Interactive Elements: When the --interactive flag is set and Plotly is available, the tool creates an interactive 3D visualization that can be explored in a web browser.

## Understanding the Visualizations

### Enhancement Levels

The visualization shows how embeddings transform across different "enhancement levels":

- Level 0 (Base): The original embeddings without transformation
- Levels 1-3: Progressively transformed versions of the embeddings, designed to emphasize different semantic aspects

### Color Coding

- Items are colored by their category (e.g., "qa_pair", "command_output", "code", "text")
- Different enhancement levels use different color palettes for clarity
- Lines connecting the same item across levels help track transformations

## Similarity and Distance

- Items positioned closer together are semantically similar
- The distance between items in the visualization correlates with their semantic distance
- Enhancement levels may change these relationships, revealing different semantic aspects

## Troubleshooting

- No visualizations generated: Check that your memory directory contains the required files (items.json, embeddings.npy).
- Missing enhancement levels: If no enhanced embeddings are found, some visualizations may be skipped.
- Dimension mismatch errors: The tool attempts to handle mismatches automatically by resizing embeddings.
- Interactive visualization not working: Ensure Plotly is installed and use the --interactive flag.

## Example Outputs

The visualizations provide insights into your memory system:

1. Semantic clusters reveal how items form conceptual groups
2. Fingerprints show how enhancement affects different memory items
3. Concept flow demonstrates how items move through semantic space
4. Similarity matrix helps identify related content
5. Retrieval heatmap shows which items are most frequently accessed

These insights can help optimize memory retrieval, understand the effect of enhancement levels, and identify patterns in how your AI system uses its memory.
