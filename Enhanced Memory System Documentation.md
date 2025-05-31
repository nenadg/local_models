# Enhanced Memory System Documentation

## Overview

The Enhanced Memory System is an advanced memory architecture designed for embedding-based retrieval with multi-level representations and sophisticated search algorithms. It enables enhanced similarity search across different abstraction levels, improving recall performance when exact wording differs but conceptual meaning is similar.

## Core Components

### Memory Items

Each memory item contains:

-   Original content (text)
-   Base embedding vector
-   Enhanced embeddings at multiple levels
-   Comprehensive metadata (source, timestamp, retrieval count, classification, quality scores, etc.)

## Multi-level Embeddings

The system maintains multiple representations of each memory at different abstraction levels:

| Level | Purpose                  | Transformation                |
| ----- | ------------------------ | ----------------------------- |
| 0     | Base representation      | Original embedding            |
| 1     | First-order abstraction  | Tanh transformation           |
| 2     | Second-order abstraction | Power transformation          |
| 3     | Third-order abstraction  | Logarithmic transformation    |
| 4     | Fourth-order abstraction | Enhanced power transformation |

## Embedding Transformation

Each level applies progressively more abstract transformations:

```python
# Level-specific transformations (unified_memory.py)
if level % 3 == 0:
    # Enhance magnitude differences
    transformed = np.sign(shifted) * np.power(np.abs(shifted), 0.8)
elif level % 3 == 1:
    # Apply sigmoid-like function
    transformed = np.tanh(shifted * (1.0 + 0.1 * level))
else:
    # Log transformation
    transformed = np.sign(shifted) * np.log(1 + np.abs(shifted) * (1.0 + 0.05 * level))
```

## Rotation Matrices

Each level has its own rotation matrix with increasing randomness:

```python
# Create rotation matrix with increasing randomness by level
rotation_factor = 0.15 + (i * 0.05)  # Increased from original
rotation = np.random.normal(0, rotation_factor, (embedding_dim, embedding_dim))

# For earlier levels, preserve more of the original structure
if i <= 2:  # More aggressive than original (was 3)
    # Add identity matrix component to preserve original information
    preservation_factor = 0.8 - (i * 0.2)  # More aggressive reduction
    rotation = rotation + np.eye(embedding_dim) * preservation_factor
```

## Search Architecture

## Enhanced Parallax Search Process

The system uses a sophisticated **parallax search** algorithm that provides different perspectives on the same query:

1. **Base Index Search**: Initial search using the original query embedding to find top candidates
2. **Parallax Vector Calculation**: Compute the centroid of top base results and calculate a parallax shift vector
3. **Progressive Perspective Shifts**: Apply increasing parallax shifts for each enhancement level
4. **Weighted Level Combination**: Combine results with decreasing weights for higher levels (to maintain base relevance)
5. **Cross-level Verification**: Boost confidence for memories found across multiple levels

```python
# Parallax search implementation (unified_memory.py)
def _enhanced_search(self, query_embedding, top_k, min_similarity):
    # Step 1: Search base index first
    base_results = self.index.search(query_array, search_k)

    # Step 2: Calculate parallax shift vector from top results
    centroid = np.mean(base_embeddings, axis=0)
    parallax_vector = centroid - query_embedding

    # Step 3: Search enhanced levels with progressive parallax shifts
    for level in sorted(self.enhanced_indices.keys()):
        parallax_strength = level * 0.2  # 0.2, 0.4, 0.6, 0.8 for levels 1-4
        shifted_query = query_embedding + (parallax_vector * parallax_strength)

        # Generate level embedding for shifted query
        level_query = self._generate_level_embedding(shifted_query, level)

        # Apply decreasing level weights to maintain base relevance
        level_weight = 1.0 / (1.0 + level * 0.3)  # 0.77, 0.63, 0.53, 0.45
```

## Cross-level Verification

The system gives higher confidence to memories found across multiple levels:

```python
# Apply boost for items found in multiple levels
if len(level_info) > 1:
    # Calculate weighted average based on level
    weighted_sum = 0.0
    total_weight = 0.0

    for level, sim in level_info:
        # Higher weight for higher levels
        level_weight = 1.0 + (level * 0.2)
        weighted_sum += sim * level_weight
        total_weight += level_weight

    # Apply cross-level bonus
    cross_level_bonus = min(0.2, 0.05 * len(level_info))
```

## Content Classification System

The system includes a sophisticated content classification system with 7 main knowledge categories:

### Knowledge Categories

| Category               | Index | Description                 | Use Case                           |
| ---------------------- | ----- | --------------------------- | ---------------------------------- |
| `declarative`          | 0     | Facts and information       | "What is the capital of France?"   |
| `procedural_knowledge` | 1     | How to do something         | "How to bake a cake"               |
| `experiential`         | 2     | Personal experiences        | "What's it like to climb Everest?" |
| `tacit`                | 3     | Intuition, insights         | "What do you think about..."       |
| `explicit`             | 4     | Articulated knowledge       | "According to the manual..."       |
| `conceptual_knowledge` | 5     | Principles and theories     | "Explain quantum mechanics"        |
| `contextual`           | 6     | Environmental understanding | "In the context of..."             |

### Subcategories

Each main category has detailed subcategories for precise classification:

```python
# Example subcategories (question_classifier.py)
'declarative': [
    'historical', 'scientific', 'geographic', 'mathematical',
    'linguistic', 'cultural', 'biographical', 'legal',
    'technological', 'literary'
],
'procedural_knowledge': [
    'cooking', 'programming', 'mechanical', 'artistic',
    'musical', 'sports', 'medical', 'crafting',
    'language_usage', 'problem_solving'
]
```

## Quality Assessment & Reliability

### Hallucination Detection

The system includes advanced quality assessment to detect potentially unreliable content:

```python
# Quality assessment integration (local_ai.py)
if is_hallucination:
    response_classification['hallucination_detected'] = True
    response_classification['hallucination_score'] = details.get('uncertainty_score', 0.5)
    response_classification['reliability_penalty'] = 0.7  # Apply 70% penalty
```

### Content Quality Metrics

-   **Confidence scores** from token-level analysis
-   **Semantic entropy** calculations
-   **Pattern detection** for repetitive or corrupted content
-   **Cross-reference validation** across multiple sources

## Dimension Management

## Automatic Dimension Detection

The system automatically detects the embedding dimension from the model:

```python
# Enhanced dimension detection (unified_memory.py)
if hasattr(self.model, 'config'):
    if hasattr(self.model.config, 'hidden_size'):
        embedding_dim = self.model.config.hidden_size
    elif hasattr(self.model.config, 'hidden_dim'):
        embedding_dim = self.model.config.hidden_dim
    elif hasattr(self.model.config, 'd_model'):
        embedding_dim = self.model.config.d_model
    elif hasattr(self.model.config, 'n_embd'):
        embedding_dim = self.model.config.n_embd
```

## Dimension Migration

When a model with a different embedding size is used, the system migrates all memories:

1. **Resize Base Embeddings**: Truncate or pad embeddings to match the new dimension
2. **Migrate Enhanced Embeddings**: Update all enhanced embeddings to the new dimension
3. **Rebuild Indices**: Rebuild FAISS indices for all levels
4. **Reinitialize Matrices**: Create new rotation matrices for the new dimension

## Web Search Integration

The system includes comprehensive web search integration for real-time information:

### Search Triggers

Automatic web search is triggered for queries containing:

-   Current information indicators: `latest`, `current`, `today`, `recent`, `2024`, `2025`
-   Specific entity queries: `who is`, `what is`, `where is`
-   Explicit search requests: `search`, `look up`, `find`

### Content Extraction Pipeline

1. **Google Custom Search API**: Retrieve relevant URLs
2. **Content Extraction**: Use `python-readability` for clean text extraction
3. **Fact Parsing**: Extract key facts and statements
4. **Classification**: Categorize extracted content
5. **Memory Storage**: Save structured facts to memory system

```python
# Web search implementation (web_search_integration.py)
class WebSearchIntegration:
    def search_and_extract(self, query: str, max_pages: int = 3)
    def save_to_memory(self, content_data: Dict[str, str], query: str)
    def process_query(self, query: str) -> Dict[str, any]
```

## Multi-Model Support

The system supports multiple model architectures with specialized formatting:

### Supported Models

-   **TinyLlama**: Standard chat template formatting
-   **Gemma**: Specialized `<start_of_turn>` formatting with knowledge blocks
-   **Qwen**: Thinking mode integration with memory context

### Model-Specific Memory Integration

```python
# Gemma formatting (local_ai.py)
if is_gemma:
    if memory_content:
        prompt += f"<knowledge>\n{memory_content}\n</knowledge>\n\n"

# Qwen formatting
elif is_qwen:
    # Wrap memory in Qwen-specific format
    memory_text = "- " + content + "\n" for each memory
```

## Performance & Optimization

### Batch Processing

Efficient processing of large datasets using adaptive batch sizing:

```python
# Batch processing integration (batch_utils.py)
def batch_embed_texts(texts, tokenizer, model, device,
                     batch_size=32, adaptive=True, handle_oom=True)

# Memory-aware batch sizing
def estimate_optimal_batch_size(tensor_shape, dtype, target_memory_usage=0.7)
```

### CUDA Optimization

-   **Memory management**: Automatic cache clearing and optimization
-   **Device consistency**: Ensures all tensors are on the correct device
-   **Efficient inference**: Uses optimized attention mechanisms when available

### Resource Management

```python
# Resource management (resource_manager.py)
class ResourceManager:
    def optimize_for_inference(self)
    def clear_cache(self)
    def batch_process_embeddings(self, texts, model, tokenizer)
```

## Memory Integration

The memory system integrates seamlessly with the chat system:

### Automatic Integration Points

1. **User Queries**: Automatic memory retrieval based on query classification
2. **Response Generation**: Context-aware memory insertion
3. **Command Outputs**: Automatic preservation of command results
4. **Web Content**: Integration of search results into memory
5. **Quality Control**: Filtering and reliability assessment

### Context Integration

```python
# Memory integration in conversation (local_ai.py)
def _integrate_memory(self, messages, query):
    # Classify query
    query_classification = classify_content(query, self.question_classifier)

    # Retrieve relevant memories
    memories = self.memory_manager.retrieve(
        query=query,
        top_k=5,
        min_similarity=0.3,
        metadata_filter={"main_category": main_category}
    )

    # Format for model-specific templates
    memory_text = format_memories_by_category(memories, main_category, subcategory)
```

## Configuration Options

### Default Configuration

```python
# Current default settings (unified_memory.py)
MemoryManager(
    storage_path="./memory",
    embedding_dim=None,  # Auto-detected from model
    enable_enhanced_embeddings=True,
    max_enhancement_levels=4,  # Increased from 3
    auto_save=True,
    similarity_enhancement_factor=0.3
)
```

### Advanced Settings

-   `embedding_dim`: Dimension of embedding vectors (auto-detected)
-   `enable_enhanced_embeddings`: Toggle for multi-level embeddings
-   `max_enhancement_levels`: Number of enhancement levels (default: 4)
-   `similarity_enhancement_factor`: Controls non-linear similarity transformation (0.0-1.0)
-   `auto_save`: Whether to automatically save changes
-   `batch_size`: Adaptive batch processing size
-   `target_memory_usage`: GPU memory utilization target (0.7 = 70%)

### Quality Control Settings

```python
# Response filtering configuration (response_filter.py)
ResponseFilter(
    confidence_threshold=0.45,
    entropy_threshold=3.5,
    perplexity_threshold=25.0,
    sharpening_factor=0.3,
    use_relative_filtering=True,
    pattern_detection_weight=0.6
)
```

## API Examples

### Basic Usage

```python
# Initialize memory manager
memory_manager = MemoryManager(
    storage_path="./memory",
    enable_enhanced_embeddings=True,
    max_enhancement_levels=4
)

# Set embedding function
memory_manager.set_embedding_function(
    model=model,
    tokenizer=tokenizer,
    device="cuda"
)

# Add content
memory_id = memory_manager.add(
    content="Paris is the capital of France.",
    metadata={"category": "declarative", "source": "factual"}
)

# Retrieve relevant content
results = memory_manager.retrieve(
    query="What is the capital of France?",
    top_k=5,
    min_similarity=0.3
)
```

### Advanced Integration

```python
# Full integration with chat system
chat = MemoryEnhancedChat(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    enable_memory=True,
    enable_enhanced_embeddings=True,
    similarity_enhancement_factor=0.3
)

# Chat with automatic memory integration
response = chat.chat(
    messages=[{"role": "user", "content": "Tell me about Paris"}],
    max_new_tokens=128,
    show_confidence=True
)
```

## Performance Characteristics

### Memory Usage

-   **Base embeddings**: ~4 bytes per dimension per memory
-   **Enhanced embeddings**: ~16 bytes per dimension per memory (4 levels)
-   **FAISS indices**: Efficient sparse storage with sub-linear scaling
-   **Metadata**: JSON storage with compression

### Search Performance

-   **Base search**: O(log n) with FAISS indexing
-   **Enhanced search**: O(k \* log n) where k is number of levels
-   **Parallax computation**: O(1) overhead per level
-   **Cross-level verification**: O(m) where m is number of matching items

### Scalability

-   **Memory items**: Tested with 100K+ items
-   **Batch processing**: Adaptive sizing for available memory
-   **Incremental updates**: Efficient index rebuilding
-   **Dimension migration**: Handles model changes gracefully

## Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce `batch_size` or `target_memory_usage`
2. **Dimension mismatch**: System auto-migrates, but may take time
3. **Low retrieval quality**: Adjust `similarity_enhancement_factor`
4. **Search latency**: Consider reducing `max_enhancement_levels`

### Debug Options

```python
# Enable debug modes
chat.memory_debug = True  # Memory integration debugging
response_filter.debug_mode = True  # Quality assessment debugging
```

### Memory Statistics

```python
# Get system statistics
stats = memory_manager.get_stats()
print(f"Active items: {stats['active_items']}")
print(f"Enhancement levels: {stats['enhancement_levels']}")
print(f"Index size: {stats['index_size']}")
```
