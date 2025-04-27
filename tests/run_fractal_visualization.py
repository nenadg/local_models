#!/usr/bin/env python3
"""
Fractal Memory Testing and Visualization Script

This script combines testing and visualization for the fractal memory system.
It runs a series of tests to populate the memory system with diverse content,
then creates intuitive visualizations showing how concepts cluster and transform
across different fractal levels.

Usage:
    python run_fractal_visualization.py [--memory-dir DIR] [--output-dir DIR] [--clean] [--interactive]
"""

import os
import sys
import argparse
import time
import shutil
import numpy as np
import random
from datetime import datetime

# insert the project root (one level up) at the front of sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Try importing the necessary components
try:
    from unified_memory import UnifiedMemoryManager
    from mcp_handler import MCPHandler
    from fractal_embedding_visualizer import create_fractal_embedding_visualization
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


def get_time():
    """Get a formatted timestamp for logging."""
    return datetime.now().strftime("[%d/%m/%y %H:%M:%S]")


def simple_embedding_function(text):
    """
    Create a semantic-aware embedding vector based on text content.
    This function creates more realistic embeddings than random ones.
    """
    # Use hash of text for deterministic randomness
    import hashlib
    text_hash = hashlib.md5(text.encode()).digest()
    seed = int.from_bytes(text_hash[:4], byteorder='little')
    np.random.seed(seed)

    # Base embedding with controlled randomness
    embedding = np.random.randn(384).astype(np.float32)

    # Add semantic patterns based on content
    text_lower = text.lower()

    # Domain-specific patterns
    patterns = [
        ("math", ["math", "calculation", "formula", "equation", "number", "fibonacci"]),
        ("code", ["code", "function", "class", "variable", "programming", "algorithm"]),
        ("science", ["science", "physics", "chemistry", "biology", "experiment"]),
        ("history", ["history", "ancient", "century", "war", "civilization"]),
        ("art", ["art", "painting", "music", "creative", "design"]),
        ("geography", ["geography", "map", "country", "city", "location"]),
        ("literature", ["literature", "book", "novel", "story", "author"]),
        ("philosophy", ["philosophy", "concept", "ethics", "meaning", "thinking"])
    ]

    # Programming language patterns
    languages = [
        ("python", ["def ", "class ", "import ", "python", ".py"]),
        ("javascript", ["function", "const ", "let ", "var ", "javascript", ".js"]),
        ("html", ["<html", "<div", "<p>", "html", ".html"]),
        ("markdown", ["# ", "## ", "**", "markdown", ".md"])
    ]

    # Check for domain matches
    domain_matches = []
    for domain, keywords in patterns:
        score = sum(3 for kw in keywords if kw in text_lower)
        if score > 0:
            domain_matches.append((domain, score))

    # Check for language matches
    language_matches = []
    for language, keywords in languages:
        score = sum(5 for kw in keywords if kw in text_lower)
        if score > 0:
            language_matches.append((language, score))

    # Apply domain influence
    step = 60  # chunk size for embedding dimensions
    for i, (domain, score) in enumerate(domain_matches):
        if i < 6:  # limit to 6 domains to avoid overflow
            start_idx = i * step
            end_idx = start_idx + step
            # Influence strength proportional to match score
            embedding[start_idx:end_idx] += 0.1 * score

    # Apply language influence (stronger than domain)
    for i, (language, score) in enumerate(language_matches):
        if i < 4:  # limit to 4 languages
            start_idx = 300 + i * 20  # Use last part of embedding
            end_idx = start_idx + 20
            # Stronger influence for language
            embedding[start_idx:end_idx] += 0.2 * score

    # Normalize to unit length
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def initialize_memory_system(memory_dir, clean=False):
    """
    Initialize or load the memory system.

    Args:
        memory_dir: Directory for storing memory
        clean: Whether to start with a clean memory

    Returns:
        Initialized UnifiedMemoryManager
    """
    print(f"{get_time()} Initializing memory system in {memory_dir}...")

    # If clean is True, remove the existing memory directory
    if clean and os.path.exists(memory_dir):
        print(f"{get_time()} Cleaning existing memory...")
        shutil.rmtree(memory_dir)

    # Create directory if it doesn't exist
    os.makedirs(memory_dir, exist_ok=True)

    # Initialize memory manager with fractal embeddings enabled
    memory_manager = UnifiedMemoryManager(
        storage_path=memory_dir,
        embedding_function=simple_embedding_function,
        embedding_dim=384,
        use_fractal=True,
        max_fractal_levels=3,
        auto_save=True,
        enable_entity_separation=True
    )

    # Get memory stats
    stats = memory_manager.get_stats()
    print(f"{get_time()} Memory initialized with {stats['active_items']} existing items")

    return memory_manager


def generate_diverse_content():
    """
    Generate diverse content to populate the memory system.

    Returns:
        List of content items with metadata
    """
    print(f"{get_time()} Generating diverse test content...")

    content_items = []

    # Math content
    math_items = [
        {
            "content": "The Fibonacci sequence is defined by F(n) = F(n-1) + F(n-2) with F(0) = 0 and F(1) = 1. The first 10 Fibonacci numbers are 0, 1, 1, 2, 3, 5, 8, 13, 21, 34.",
            "metadata": {"category": "math"}
        },
        {
            "content": "The quadratic formula for solving ax² + bx + c = 0 is x = (-b ± √(b² - 4ac)) / 2a.",
            "metadata": {"category": "math"}
        },
        {
            "content": "Pi (π) is the ratio of a circle's circumference to its diameter, approximately equal to 3.14159.",
            "metadata": {"category": "math"}
        }
    ]
    content_items.extend(math_items)

    # Code content
    code_items = [
        {
            "content": """def factorial(n):
    \"\"\"Calculate the factorial of n.\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n-1)

# Calculate factorial of 5
result = factorial(5)
print(f"The factorial of 5 is {result}")""",
            "metadata": {"category": "code", "language": "python"}
        },
        {
            "content": """function calculateAverage(numbers) {
    if (numbers.length === 0) return 0;
    const sum = numbers.reduce((a, b) => a + b, 0);
    return sum / numbers.length;
}

// Calculate average of an array
const avg = calculateAverage([10, 20, 30, 40, 50]);
console.log(`The average is ${avg}`);""",
            "metadata": {"category": "code", "language": "javascript"}
        },
        {
            "content": """<!DOCTYPE html>
<html>
<head>
    <title>Simple Page</title>
    <style>
        body { font-family: Arial, sans-serif; }
        .container { max-width: 800px; margin: 0 auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Hello World</h1>
        <p>This is a simple HTML page.</p>
    </div>
</body>
</html>""",
            "metadata": {"category": "code", "language": "html"}
        }
    ]
    content_items.extend(code_items)

    # Science content
    science_items = [
        {
            "content": "Photosynthesis is the process by which plants convert light energy into chemical energy. The basic equation is: 6 CO₂ + 6 H₂O + light energy → C₆H₁₂O₆ + 6 O₂",
            "metadata": {"category": "science"}
        },
        {
            "content": "Newton's Second Law of Motion states that force equals mass times acceleration (F = ma).",
            "metadata": {"category": "science"}
        },
        {
            "content": "DNA (deoxyribonucleic acid) is a molecule composed of two polynucleotide chains that coil around each other to form a double helix carrying genetic instructions.",
            "metadata": {"category": "science"}
        }
    ]
    content_items.extend(science_items)

    # History content
    history_items = [
        {
            "content": "The Roman Empire was one of the largest empires in history, spanning from Britain to Egypt at its height around 117 CE under Emperor Trajan.",
            "metadata": {"category": "history"}
        },
        {
            "content": "The Industrial Revolution began in Great Britain in the late 18th century and spread to other parts of Europe and the United States.",
            "metadata": {"category": "history"}
        },
        {
            "content": "World War II lasted from 1939 to 1945 and involved most of the world's nations forming two opposing military alliances: the Allies and the Axis.",
            "metadata": {"category": "history"}
        }
    ]
    content_items.extend(history_items)

    # Geography content
    geography_items = [
        {
            "content": "The Amazon River is the largest river by discharge volume of water in the world, and the second in length after the Nile.",
            "metadata": {"category": "geography"}
        },
        {
            "content": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas.",
            "metadata": {"category": "geography"}
        },
        {
            "content": "The Sahara is the largest hot desert in the world, covering most of North Africa.",
            "metadata": {"category": "geography"}
        }
    ]
    content_items.extend(geography_items)

    # Literature content
    literature_items = [
        {
            "content": "William Shakespeare was an English playwright, poet, and actor, widely regarded as the greatest writer in the English language.",
            "metadata": {"category": "literature"}
        },
        {
            "content": "To Kill a Mockingbird is a novel by Harper Lee published in 1960 that explores themes of racial injustice and moral growth.",
            "metadata": {"category": "literature"}
        },
        {
            "content": "The Odyssey is one of two major ancient Greek epic poems attributed to Homer. It mainly focuses on the Greek hero Odysseus and his journey home after the fall of Troy.",
            "metadata": {"category": "literature"}
        }
    ]
    content_items.extend(literature_items)

    # Art content
    art_items = [
        {
            "content": "The Mona Lisa is a half-length portrait painting by Italian Renaissance artist Leonardo da Vinci.",
            "metadata": {"category": "art"}
        },
        {
            "content": "Impressionism is a 19th-century art movement characterized by small, thin brush strokes and an emphasis on accurate depiction of light.",
            "metadata": {"category": "art"}
        },
        {
            "content": "Van Gogh's Starry Night is one of the most recognized paintings in Western culture, painted in June 1889.",
            "metadata": {"category": "art"}
        }
    ]
    content_items.extend(art_items)

    # Music content
    music_items = [
        {
            "content": "Ludwig van Beethoven was a German composer and pianist whose music spans the transition from the classical period to the romantic era.",
            "metadata": {"category": "music"}
        },
        {
            "content": "Jazz is a music genre that originated in the African-American communities of New Orleans in the late 19th and early 20th centuries.",
            "metadata": {"category": "music"}
        },
        {
            "content": "The Beatles were an English rock band formed in Liverpool in 1960, widely regarded as the most influential band of all time.",
            "metadata": {"category": "music"}
        }
    ]
    content_items.extend(music_items)

    # Create some MCP content
    mcp_handler = MCPHandler()
    mcp_content = """
>>> FILE: fibonacci.py
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
<<<

>>> FILE: physics_formulas.md
# Common Physics Formulas

## Kinematics
- Velocity: v = dx/dt
- Acceleration: a = dv/dt
- Distance: x = x₀ + v₀t + ½at²

## Dynamics
- Newton's Second Law: F = ma
- Weight: W = mg
- Momentum: p = mv

## Energy
- Kinetic Energy: KE = ½mv²
- Potential Energy: PE = mgh
- Work: W = Fd cos(θ)
<<<

>>> FILE: periodic_table.html
<!DOCTYPE html>
<html>
<head>
    <title>Periodic Table</title>
    <style>
        .element {
            display: inline-block;
            width: 100px;
            border: 1px solid #ccc;
            padding: 5px;
            margin: 2px;
            text-align: center;
        }
        .metal { background-color: #ffd700; }
        .nonmetal { background-color: #90ee90; }
        .noble-gas { background-color: #add8e6; }
    </style>
</head>
<body>
    <h1>Periodic Table of Elements (Sample)</h1>
    <div class="element metal">
        <h2>Fe</h2>
        <p>Iron</p>
        <p>26</p>
    </div>
    <div class="element nonmetal">
        <h2>O</h2>
        <p>Oxygen</p>
        <p>8</p>
    </div>
    <div class="element noble-gas">
        <h2>He</h2>
        <p>Helium</p>
        <p>2</p>
    </div>
</body>
</html>
<<<
"""

    # Extract and process MCP blocks
    cleaned_content, mcp_blocks = mcp_handler.extract_mcp_blocks(mcp_content)

    # Add MCP content to items
    for filename, content in mcp_blocks.items():
        # Get file extension
        ext = os.path.splitext(filename)[1][1:] if '.' in filename else 'unknown'

        # Determine category based on filename or extension
        category = 'code'
        if 'fibonacci' in filename:
            category = 'math'
        elif 'physics' in filename:
            category = 'science'
        elif 'periodic' in filename:
            category = 'science'

        content_items.append({
            "content": content,
            "metadata": {
                "category": category,
                "source": "mcp",
                "filename": filename,
                "filetype": ext
            }
        })

    print(f"{get_time()} Generated {len(content_items)} content items")
    return content_items


def populate_memory(memory_manager, content_items, min_items=30):
    """
    Populate memory with diverse content.

    Args:
        memory_manager: The UnifiedMemoryManager instance
        content_items: List of content items to add
        min_items: Minimum number of items to have in memory

    Returns:
        Number of items added
    """
    print(f"{get_time()} Populating memory with diverse content...")

    # Check current item count
    stats = memory_manager.get_stats()
    current_count = stats['active_items']

    # If we already have enough items, we can skip adding more
    if current_count >= min_items:
        print(f"{get_time()} Memory already has {current_count} items, skipping population")
        return 0

    # Determine how many items to add
    items_to_add = max(0, min_items - current_count)

    # Add items in a random order to ensure diversity
    random.shuffle(content_items)

    # Add items to memory
    added_count = 0
    for i, item in enumerate(content_items):
        if added_count >= items_to_add:
            break

        try:
            item_id = memory_manager.add(
                content=item["content"],
                metadata=item["metadata"],
                use_fractal=True
            )

            if item_id:
                added_count += 1
                if added_count % 5 == 0:
                    print(f"{get_time()} Added {added_count}/{items_to_add} items...")
        except Exception as e:
            print(f"{get_time()} Error adding item: {e}")

    print(f"{get_time()} Added {added_count} new items to memory")

    # Get updated stats
    stats = memory_manager.get_stats()
    print(f"{get_time()} Memory now has {stats['active_items']} items")

    return added_count


def run_test_queries(memory_manager, queries=None):
    """
    Run test queries to verify memory functionality.

    Args:
        memory_manager: The UnifiedMemoryManager instance
        queries: Optional list of test queries

    Returns:
        Dictionary with query results
    """
    print(f"{get_time()} Testing memory with sample queries...")

    # Default queries if none provided
    if queries is None:
        queries = [
            "Fibonacci sequence",
            "Python function",
            "Newton's laws of motion",
            "HTML page structure",
            "Famous paintings",
            "World history",
            "Mathematical formulas",
            "Programming languages"
        ]

    results = {}

    # Run each query
    for query in queries:
        try:
            print(f"{get_time()} Query: '{query}'")
            start_time = time.time()

            # Retrieve results
            query_results = memory_manager.retrieve(
                query=query,
                top_k=5,
                min_similarity=0.1
            )

            retrieval_time = time.time() - start_time

            # Store results
            results[query] = {
                "count": len(query_results),
                "time": retrieval_time,
                "items": []
            }

            # Process results
            for i, result in enumerate(query_results):
                # Extract metadata
                item_id = result.get('id', 'unknown')
                category = result.get('metadata', {}).get('category', 'unknown')
                similarity = result.get('similarity', 0.0)

                # Get content snippet
                content = result.get('content', '')
                snippet = content[:100] + '...' if len(content) > 100 else content

                # Check if result is from MCP
                is_mcp = 'filename' in result.get('metadata', {})

                # Check for cross-level matches
                cross_level = False
                if 'found_in_levels' in result:
                    cross_level = len(result['found_in_levels']) > 1

                # Store item info
                results[query]['items'].append({
                    'id': item_id,
                    'rank': i,
                    'category': category,
                    'similarity': similarity,
                    'snippet': snippet,
                    'is_mcp': is_mcp,
                    'cross_level': cross_level
                })

                # Print result summary
                mcp_str = " (MCP)" if is_mcp else ""
                cross_level_str = " (Cross-level)" if cross_level else ""
                print(f"  - [{i+1}] {category}{mcp_str}{cross_level_str}: {similarity:.3f} - {snippet[:50]}...")

            print(f"  - Found {len(query_results)} results in {retrieval_time:.3f}s")
        except:
        	pass

    # Calculate overall statistics
    total_queries = len(queries)
    total_results = sum(r['count'] for r in results.values())
    avg_results = total_results / max(1, total_queries)

    # Count cross-level matches
    cross_level_count = 0
    for query_result in results.values():
        for item in query_result.get('items', []):
            if item.get('cross_level', False):
                cross_level_count += 1

    print(f"\n{get_time()} Query testing complete!")
    print(f"  - Total queries: {total_queries}")
    print(f"  - Total results: {total_results}")
    print(f"  - Average results per query: {avg_results:.1f}")
    print(f"  - Cross-level matches: {cross_level_count}")

    return results


def run_fractal_tests(memory_dir, output_dir, clean=False, interactive=False):
    """
    Run a complete fractal memory test suite with visualizations.

    Args:
        memory_dir: Directory for memory storage
        output_dir: Directory for output files
        clean: Whether to start with a clean memory
        interactive: Whether to show interactive visualizations

    Returns:
        Dictionary with test results
    """
    # Step 1: Initialize memory system
    memory_manager = initialize_memory_system(memory_dir, clean)

    # Step 2: Generate diverse test content
    content_items = generate_diverse_content()

    # Step 3: Populate memory if needed
    populate_memory(memory_manager, content_items, min_items=30)

    # Step 4: Run test queries
    query_results = run_test_queries(memory_manager)

    # Step 5: Create fractal visualizations
    print(f"\n{get_time()} Creating fractal visualizations...")

    # Generate seed queries for focused visualizations
    seed_queries = [
        "Fibonacci sequence and mathematical formulas",
        "Computer programming and HTML",
        "Physics and scientific concepts"
    ]

    visualization_results = {}

    # Create general fractal visualization
    general_viz = create_fractal_embedding_visualization(
        memory_manager=memory_manager,
        output_dir=output_dir,
        max_items=50,
        interactive=interactive
    )
    visualization_results['general'] = general_viz

    # Create focused visualizations for each seed query
    for i, query in enumerate(seed_queries):
        print(f"\n{get_time()} Creating focused visualization for: {query}")
        focused_viz = create_fractal_embedding_visualization(
            memory_manager=memory_manager,
            output_dir=output_dir,
            max_items=30,
            interactive=False,
            seed_query=query
        )
        visualization_results[f'focused_{i}'] = focused_viz

    print(f"\n{get_time()} All tests and visualizations complete!")

    # Return combined results
    return {
        'query_results': query_results,
        'visualization_results': visualization_results
    }


def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Fractal Memory Testing and Visualization")
    parser.add_argument("--memory-dir", type=str, default="./test_memory",
                      help="Directory for memory storage")
    parser.add_argument("--output-dir", type=str, default="./visualizations",
                      help="Directory for output files")
    parser.add_argument("--clean", action="store_true",
                      help="Start with a clean memory")
    parser.add_argument("--interactive", action="store_true",
                      help="Show interactive visualizations")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run tests
    run_fractal_tests(
        memory_dir=args.memory_dir,
        output_dir=args.output_dir,
        clean=args.clean,
        interactive=args.interactive
    )


if __name__ == "__main__":
    main()