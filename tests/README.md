# Fractal Memory Testing and Visualization Tools

This package provides comprehensive testing and visualization tools for the fractal memory system in TinyLlama Chat. It includes facilities for testing the integration between Model Content Protocol (MCP) and fractal memory, as well as advanced visualization of embedding transformations across fractal levels.

## Contents

1. **Fractal Memory Tester** (`fractal_tests.py`): Core testing functions for the fractal memory system.
2. **Fractal Memory Visualizer** (`fractal_visualizer.py`): Advanced visualization tools for exploring fractal embeddings.
3. **MCP-Fractal Integration Tester** (`mcp_fractal_integration_tester.py`): Specific tests for MCP integration with fractal memory.

## Requirements

-  Python 3.8+
-  NumPy
-  Matplotlib
-  Scikit-learn
-  Seaborn
-  (Optional) Plotly for interactive 3D visualizations

You can install the required packages using:

```bash
pip install numpy matplotlib scikit-learn seaborn
# Optional for interactive visualizations
pip install plotly
```

## Testing Fractal Memory

The `fractal_tests.py` script provides comprehensive tests for the fractal memory system:

```bash
# Run all tests
python fractal_tests.py --test all

# Run specific tests
python fractal_tests.py --test add --items 100
python fractal_tests.py --test retrieve --queries 30
python fractal_tests.py --test visualize --interactive

# Create 3D visualization
python fractal_tests.py --test 3d --interactive
```

### Available Test Types

-  `add`: Tests adding items to fractal memory
-  `retrieve`: Tests retrieving items from fractal memory
-  `visualize`: Creates static visualizations of fractal embeddings
-  `mcp`: Tests MCP integration with fractal memory
-  `3d`: Creates 3D visualization of fractal embeddings

## Visualizing Fractal Memory

The `fractal_visualizer.py` script provides advanced visualization tools:

```bash
# Run all visualizations
python fractal_visualizer.py --interactive

# Create specific visualizations
python fractal_visualizer.py --vis-type similarity
python fractal_visualizer.py --vis-type clusters
python fractal_visualizer.py --vis-type transformation
python fractal_visualizer.py --vis-type animation
python fractal_visualizer.py --vis-type 3d --level 1
python fractal_visualizer.py --vis-type dashboard
```

### Available Visualization Types

-  `similarity`: Heatmaps of similarity matrices across levels
-  `clusters`: Cluster visualizations of embedding spaces
-  `transformation`: Visualization of how embeddings transform across levels
-  `animation`: Animated transformation between fractal levels
-  `mcp`: Visualization of MCP content in embedding space
-  `3d`: Interactive 3D visualization of embeddings
-  `dashboard`: Interactive dashboard for comparing fractal levels

## Testing MCP Integration

The `mcp_fractal_integration_tester.py` script specifically tests the integration between MCP and fractal memory:

```bash
# Run basic integration test
python mcp_fractal_integration_tester.py

# Run with visualization
python mcp_fractal_integration_tester.py --visualize

# Specify custom directories
python mcp_fractal_integration_tester.py --memory-dir ./custom_memory --output-dir ./custom_output
```

## Example Workflow

1. First, test adding and retrieving from fractal memory:

   ```bash
   python fractal_tests.py --test add --items 100
   python fractal_tests.py --test retrieve --queries 20
   ```

2. Then visualize the fractal embeddings:

   ```bash
   python fractal_visualizer.py --vis-type all --interactive
   ```

3. Finally, test MCP integration:
   ```bash
   python mcp_fractal_integration_tester.py --visualize
   ```

## Understanding the Visualizations

### Similarity Heatmaps

These show the cosine similarity between embeddings at different fractal levels. Stronger clustering indicates that the fractal system is effectively organizing semantically similar content.

### Embedding Clusters

Shows how items are grouped in different embedding spaces, with PCA or t-SNE projections. The clustering should reflect the semantic organization of the content.

### Fractal Transformation

Shows how embeddings transform across fractal levels. This helps understand how the system creates different perspectives on the same content.

### MCP Integration

Shows how MCP-generated content is integrated into the memory system, highlighting the relationship between content types and embedding organization.

## Output

All visualizations are saved to the `./visualizations` directory by default (configurable with `--output-dir`).

Test results, including metrics and performance data, are saved to the `./test_output` directory.

## Advanced Usage

### Custom Embedding Function

You can modify the `simple_embedding_function` in the tester classes to use different embedding algorithms or to test specific semantic patterns.

### Benchmarking

For benchmarking retrieval performance, use the retrieval test with increasing numbers of items:

```bash
python fractal_tests.py --test retrieve --items 100 --queries 50
python fractal_tests.py --test retrieve --items 500 --queries 50
python fractal_tests.py --test retrieve --items 1000 --queries 50
```

### Integration with TinyLlama Chat

To integrate these tests with the main TinyLlama Chat system, you can import the tester classes and run selected tests programmatically:

```python
from fractal_tests import FractalMemoryTester

tester = FractalMemoryTester(memory_dir="./memory")
results = tester.test_add_to_fractal_memory(num_items=50)
print(f"Added {results['items_added']} items with {results['success_rate']*100:.1f}% success rate")
```
