# Fractal Memory System Documentation
## Overview

The Fractal Memory System is an advanced memory architecture designed for embedding-based retrieval with multi-level representations. It enables enhanced similarity search across different abstraction levels, improving recall performance when exact wording differs but conceptual meaning is similar.

## Core Components
### Memory Items

Each memory item contains:

- Original content (text)
- Base embedding vector
- Enhanced embeddings at multiple levels
- Metadata (source, timestamp, retrieval count, etc.)

## Multi-level Embeddings
The system maintains multiple representations of each memory at different abstraction levels:

Level 	Purpose 									Transformation
0 			Base representation 			Original embedding
1				First-order abstraction		Tanh transformation
2				Second-order abstraction	Logarithmic transformation
3				Third-order abstraction		Power transformation

## Embedding Transformation
Each level applies progressively more abstract transformations:

```python
# Level-specific transformations
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
rotation_factor = 0.1 + (i * 0.03)
rotation = np.random.normal(0, rotation_factor, (embedding_dim, embedding_dim))

# For earlier levels, preserve more of the original structure
if i <= 3:
    # Add identity matrix component to preserve original information
    preservation_factor = 0.9 - (i * 0.15)
    rotation = rotation + np.eye(embedding_dim) * preservation_factor
```

## Search Architecture
## Enhanced Search Process

1. *Multi-level Query Generation*: The system transforms the query to match each enhancement level
2. *Parallel Search*: Searches are conducted across all levels simultaneously
3. *Result Combination*: Results from all levels are combined with priority to higher-confidence matches
4. *Non-linear Similarity Enhancement*: Similarity scores undergo non-linear enhancement

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

## Dimension Management
## Automatic Dimension Detection
The system automatically detects the embedding dimension from the model:

```python
if hasattr(self.model, 'config'):
    if hasattr(self.model.config, 'hidden_size'):
        embedding_dim = self.model.config.hidden_size
    elif hasattr(self.model.config, 'hidden_dim'):
        embedding_dim = self.model.config.hidden_dim
    elif hasattr(self.model.config, 'd_model'):
        embedding_dim = self.model.config.d_model
```

## Dimension Migration
When a model with a different embedding size is used, the system migrates all memories:

1. *Resize Base Embeddings*: Truncate or pad embeddings to match the new dimension
2. *Migrate Enhanced Embeddings*: Update all enhanced embeddings to the new dimension
3. *Rebuild Indices*: Rebuild FAISS indices for all levels
4. *Reinitialize Matrices*: Create new rotation matrices for the new dimension

## Cache System
The system includes a quickstart cache for faster loading:

1. *Cache Creation*: On shutdown, a lightweight cache is created with metadata
2. *Rapid Loading*: On startup, the cache enables quick initialization
3. *Background Processing*: Full embedding loading happens in the background

## Memory Integration
The memory system integrates with:

1. *Command Outputs*: Automatically saves command results with metadata
2. *User Interactions*: Automatically saves important exchanges
3. *External Content*: Can store content from files and other sources

## Configuration Options

- `embedding_dim`: Dimension of embedding vectors
- `enable_enhanced_embeddings`: Toggle for multi-level embeddings
- `max_enhancement_levels`: Number of enhancement levels
- `similarity_enhancement_factor`: Controls non-linear similarity transformation
- `auto_save`: Whether to automatically save changes
