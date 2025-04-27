"""
MCP and Fractal Memory Integration Tester

This script specifically tests the integration between the Model Content Protocol (MCP)
and the fractal memory system in TinyLlama Chat.

It performs several key tests:
1. Adding content via MCP and retrieving it from fractal memory
2. Testing how MCP-generated content is organized in fractal levels
3. Visualizing the integration between MCP and fractal memory
4. Benchmarking retrieval performance for MCP content

Usage:
    python mcp_fractal_integration_tester.py [--memory-dir DIR] [--output-dir DIR] [--visualize]
"""

import os
import argparse
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple


import sys
# insert the project root (one level up) at the front of sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


try:
    from unified_memory import UnifiedMemoryManager
    from mcp_handler import MCPHandler
    from fractal_visualizer import FractalVisualizer
except ImportError:
    print("Warning: Some modules couldn't be imported.")
    print("Make sure you're running this from the project root directory.")


class MCPFractalIntegrationTester:
    """Tests the integration between MCP and the fractal memory system."""
    
    def __init__(self, memory_dir="./test_memory", output_dir="./test_output"):
        """Initialize the tester with specified parameters."""
        self.memory_dir = memory_dir
        self.output_dir = output_dir
        
        # Create test directories
        os.makedirs(memory_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize memory manager
        self.memory_manager = UnifiedMemoryManager(
            storage_path=memory_dir,
            embedding_function=self.simple_embedding_function,
            embedding_dim=384,
            use_fractal=True,
            max_fractal_levels=3,
            auto_save=True,
            enable_entity_separation=True
        )
        
        # Initialize MCP handler
        self.mcp_handler = MCPHandler(output_dir=output_dir, allow_shell_commands=False)
        
        print(f"[INIT] Test environment ready")
        print(f"[INIT] Memory directory: {memory_dir}")
        print(f"[INIT] MCP output directory: {output_dir}")
    
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
        embedding = np.random.randn(384).astype(np.float32)
        
        # Add some content-specific patterns
        # Different content types get different embedding patterns
        if "python" in text.lower() or "def " in text or "class " in text:
            # Python code
            embedding[:50] += 0.5
        elif "javascript" in text.lower() or "function" in text or "const " in text:
            # JavaScript code
            embedding[50:100] += 0.5
        elif "markdown" in text.lower() or "# " in text or "## " in text:
            # Markdown content
            embedding[100:150] += 0.5
        elif "html" in text.lower() or "<div" in text or "<p>" in text:
            # HTML content
            embedding[150:200] += 0.5
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def generate_test_mcp_content(self):
        """Generate test MCP content for different file types."""
        print(f"\n{self.get_time()} Generating test MCP content...")
        
        # Test MCP content for different file types
        test_blocks = {
            # Python code
            "fibonacci.py": """
def fibonacci(n):
    # Calculate the nth Fibonacci number.
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Calculate the 10th Fibonacci number
result = fibonacci(10)
print(f"The 10th Fibonacci number is {result}")
""",
            # JavaScript code
            "greeting.js": """
function greet(name) {
    return `Hello, ${name}!`;
}

// Test the greeting function
const message = greet('World');
console.log(message);
""",
            # Markdown document
            "project_notes.md": """
# Project Notes

## Features to Implement

- Fractal memory system
- MCP integration
- Visualization tools
- Performance benchmarks

## Known Issues

1. Memory leaks in long sessions
2. Slow retrieval for large datasets
3. Need to optimize embedding functions
""",
            # HTML content
            "sample.html": """
<!DOCTYPE html>
<html>
<head>
    <title>Sample Page</title>
    <style>
        body { font-family: Arial, sans-serif; }
        .container { max-width: 800px; margin: 0 auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sample HTML Document</h1>
        <p>This is a test HTML document for the MCP system.</p>
    </div>
</body>
</html>
"""
        }
        
        # Create MCP formatted content
        mcp_content = ""
        for filename, content in test_blocks.items():
            mcp_content += f">>> FILE: {filename}\n{content}\n<<<\n\n"
        
        return mcp_content, test_blocks
    
    def test_mcp_fractal_integration(self):
        """Test MCP integration with fractal memory."""
        print(f"\n{self.get_time()} Testing MCP and fractal memory integration...")
        
        # Generate test content
        mcp_content, test_blocks = self.generate_test_mcp_content()
        
        # Extract MCP blocks
        cleaned_content, mcp_blocks = self.mcp_handler.extract_mcp_blocks(mcp_content)
        
        # Save MCP blocks to files
        save_results = self.mcp_handler.save_mcp_blocks(mcp_blocks)
        
        # Report file saving results
        print(f"\n{self.get_time()} MCP file saving results:")
        for filename, success in save_results.items():
            print(f"  - {filename}: {'Success' if success else 'Failed'}")
        
        # Add MCP content to memory
        add_results = {}
        for filename, content in mcp_blocks.items():
            # Get file extension
            ext = os.path.splitext(filename)[1][1:]
            
            # Create metadata
            metadata = {
                "source": "mcp",
                "filename": filename,
                "filetype": ext,
                "timestamp": time.time()
            }
            
            # Add to memory
            item_id = self.memory_manager.add(
                content=content,
                metadata=metadata,
                use_fractal=True
            )
            
            add_results[filename] = {
                "id": item_id,
                "success": item_id is not None
            }
            
            print(f"  - Added to memory: {filename}: {item_id is not None}")
        
        # Test retrieving MCP content with different query types
        retrieve_results = {}
        
        # Test queries for each file type
        test_queries = {
            "fibonacci.py": [
                "Python function to calculate Fibonacci numbers",
                "Recursive implementation of Fibonacci sequence",
                "Calculate the 10th number in the Fibonacci series"
            ],
            "greeting.js": [
                "JavaScript greeting function",
                "Function that says hello in JavaScript",
                "Template string greeting in JS"
            ],
            "project_notes.md": [
                "Project features to implement",
                "Known issues in the project",
                "Markdown notes about the project"
            ],
            "sample.html": [
                "HTML page with container class",
                "Simple HTML document with CSS",
                "Sample web page structure"
            ]
        }
        
        print(f"\n{self.get_time()} Testing retrieval from fractal memory...")
        
        for filename, queries in test_queries.items():
            retrieve_results[filename] = []
            
            for query in queries:
                start_time = time.time()
                # Retrieve from memory
                results = self.memory_manager.retrieve(
                    query=query,
                    top_k=5,
                    min_similarity=0.1
                )
                retrieval_time = time.time() - start_time
                
                # Check if target file was found
                found = False
                result_position = -1
                
                for i, result in enumerate(results):
                    meta = result.get("metadata", {})
                    if meta.get("filename", "") == filename:
                        found = True
                        result_position = i
                        break
                
                # Store result
                retrieve_results[filename].append({
                    "query": query,
                    "found": found,
                    "position": result_position if found else -1,
                    "result_count": len(results),
                    "time": retrieval_time
                })
                
                # Report result
                status = "✓" if found else "✗"
                pos_str = f"at position {result_position}" if found else "not found"
                print(f"  - Query: '{query}' {status} {pos_str} in {retrieval_time:.3f}s")
        
        # Analyze performance across fractal levels
        print(f"\n{self.get_time()} Analyzing retrieval across fractal levels...")
        
        level_analysis = {}
        for level in range(4):  # 0 = base, 1-3 = fractal levels
            # Create a level-specific query
            if level == 0:
                query = "Find all MCP-generated content"
            else:
                query = f"Find content at fractal level {level}"
            
            # Try retrieval with this level
            use_fractal = level > 0
            results = self.memory_manager.retrieve(
                query=query,
                top_k=10,
                min_similarity=0.1,
                use_fractal=use_fractal
            )
            
            # Count MCP files found
            found_files = []
            for result in results:
                meta = result.get("metadata", {})
                filename = meta.get("filename", "")
                if filename in test_blocks.keys():
                    found_files.append(filename)
            
            # Store results
            level_analysis[level] = {
                "query": query,
                "use_fractal": use_fractal,
                "result_count": len(results),
                "mcp_files_found": found_files,
                "mcp_file_count": len(found_files)
            }
            
            # Report result
            print(f"  - Level {level}: Found {len(found_files)}/{len(test_blocks)} MCP files")
            if found_files:
                print(f"    > Files: {', '.join(found_files)}")
        
        # Check for cross-level results
        cross_level_matches = 0
        for result in results:
            if "found_in_levels" in result and len(result["found_in_levels"]) > 1:
                cross_level_matches += 1
                
                # Find which file this is
                meta = result.get("metadata", {})
                filename = meta.get("filename", "unknown")
                levels = result["found_in_levels"]
                
                print(f"  - {filename} found across levels: {levels}")
                cross_level_matches += 1
        
        print(f"  - Cross-level matches: {cross_level_matches}")
        
        # Return comprehensive results
        return {
            "mcp_blocks": len(mcp_blocks),
            "save_results": save_results,
            "add_results": add_results,
            "retrieve_results": retrieve_results,
            "level_analysis": level_analysis,
            "cross_level_matches": cross_level_matches
        }
    
    def visualize_mcp_fractal_integration(self):
        """Visualize the integration between MCP and fractal memory."""
        print(f"\n{self.get_time()} Creating visualizations of MCP and fractal integration...")
        
        try:
            # Import the visualizer
            from fractal_visualizer import FractalVisualizer
            visualizer = FractalVisualizer(memory_dir=self.memory_dir)
            
            # Extract data
            data = visualizer.extract_embeddings()
            if not data:
                print("  - Error: Could not extract embeddings. Skipping visualization.")
                return None
            
            # Create MCP integration visualization
            mcp_fig = visualizer.create_mcp_integration_visualization(data)
            
            # Create embedding cluster visualization
            cluster_fig = visualizer.visualize_embedding_clusters(data, method='pca')
            
            # Create transformation visualization if there are fractal levels
            has_fractal = False
            for level in data['fractal_embeddings']:
                if not np.all(data['fractal_embeddings'][level] == 0):
                    has_fractal = True
                    break
            
            if has_fractal:
                transform_fig = visualizer.visualize_fractal_transformation(data)
            else:
                transform_fig = None
                print("  - No fractal embeddings found. Skipping transformation visualization.")
            
            return {
                "mcp_fig": mcp_fig,
                "cluster_fig": cluster_fig,
                "transform_fig": transform_fig
            }
            
        except ImportError:
            print("  - Error: FractalVisualizer not available. Skipping visualization.")
            return None
    
    def run_full_integration_test(self, visualize=False):
        """Run a full integration test and optionally visualize results."""
        # First, test the integration
        results = self.test_mcp_fractal_integration()
        
        # Then visualize if requested
        if visualize:
            viz_results = self.visualize_mcp_fractal_integration()
            results["visualizations"] = viz_results
            
            # Show plots
            if viz_results:
                plt.show()
        
        # Save test results
        timestr = time.strftime("%Y%m%d-%H%M%S")
        results_path = os.path.join(self.output_dir, f"mcp_fractal_test_results_{timestr}.json")
        
        # Convert numpy values to Python primitives
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(i) for i in obj]
            else:
                return obj
        
        # Filter out non-serializable entries
        serializable_results = {k: v for k, v in results.items() 
                              if k not in ["visualizations", "mcp_fig", "cluster_fig", "transform_fig"]}
        serializable_results = convert_to_json_serializable(serializable_results)
        
        # Save results
        try:
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"\n{self.get_time()} Test results saved to {results_path}")
        except Exception as e:
            print(f"\n{self.get_time()} Error saving test results: {e}")
        
        return results

def main():
    """Main function to run the MCP fractal integration test."""
    parser = argparse.ArgumentParser(description="Test MCP and Fractal Memory Integration")
    parser.add_argument("--memory-dir", type=str, default="./test_memory",
                       help="Directory to store test memory")
    parser.add_argument("--output-dir", type=str, default="./test_output",
                       help="Directory to store test output files")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualizations of the integration")
    
    args = parser.parse_args()
    
    # Create and run tester
    tester = MCPFractalIntegrationTester(
        memory_dir=args.memory_dir,
        output_dir=args.output_dir
    )
    
    # Run the full test
    results = tester.run_full_integration_test(visualize=args.visualize)
    
    # Print summary
    print(f"\n{tester.get_time()} Test completed!")
    print(f"  - MCP blocks processed: {results['mcp_blocks']}")
    
    # Calculate retrieval success rate
    total_queries = 0
    successful_queries = 0
    for filename, queries in results['retrieve_results'].items():
        for query in queries:
            total_queries += 1
            if query['found']:
                successful_queries += 1
    
    print(f"  - Retrieval success rate: {successful_queries}/{total_queries} ({successful_queries/total_queries*100:.1f}%)")
    print(f"  - Cross-level matches: {results['cross_level_matches']}")

if __name__ == "__main__":
    main()