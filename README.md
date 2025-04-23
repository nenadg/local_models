# local_models AI

A lightweight, local LLM chat application with memory capabilities, confidence visualization, and file output support. The aim of this project is to try running AI stuff on local consumer grade hardware.

## Features

- **Local LLM Inference**: Run small language models like TinyLlama locally on consumer hardware
- **Speculative Decoding**: Accelerated generation using optimized draft models
- **Long-term Memory**: Remembers facts from previous conversations with vector storage
- **Confidence Visualization**: See model certainty with color-coded confidence heatmaps
- **Response Filtering**: Detect and filter low-confidence/hallucinated responses
- **Model Content Protocol (MCP)**: Easily save outputs to files and execute shell commands
- **Enhanced Vector Store**: Efficient memory consolidation with FAISS indexing

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (optional, but recommended for better performance)
- At least 4GB of RAM (8GB+ recommended)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/nenadg/local_models.git
cd local_models
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv local_models_venv
# On Windows
local_models_venv\Scripts\activate
# On Linux/Mac
source local_models_venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch transformers faiss-cpu numpy scikit-learn accelerate peft datasets spacy textpipeliner matplotlib nltk rake-nltk prompt_toolkit
```

If you have CUDA, install the GPU version of FAISS for better performance:

```bash
pip install faiss-gpu
```

For language (url search query building):

```bash
python -m spacy download en_core_web_sm bs4
```

(for keyword exctraction tests `python -m spacy download en_core_sci_lg`)

## Usage

### Basic Usage

Run the chat application with default settings:

```bash
python local_ai.py
```

### Command-line Arguments

```bash
python local_ai.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --temperature 0.7 --max-tokens 128 --heatmap
```

(\*some of the features and options can vary from version to version)

Available options:

- `--model`: Model identifier to use (default: "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
- `--device`: Device to use (cuda, cpu). Auto-detects if not specified
- `--system-prompt`: Custom system prompt
- `--temperature`: Temperature for response generation (default: 0.7)
- `--max-tokens`: Maximum tokens to generate per response (default: 128)
- `--no-stream`: Disable streaming output (word-by-word generation)
- `--turbo`: Enable turbo mode with speculative decoding (default: true)
- `--output-dir`: Directory to store output files (default: "./output")
- `--confidence_threshold`: Threshold for confidence filtering (default: 0.7)
- `--heatmap`: Show confidence heatmap visualization (default: false)
- `--no-memory`: Disable automatic memory features
- `--all-metrics`: Show all detailed metrics instead of just truthiness

### In-Chat Commands

During chat, you can use these special commands:

- `!teach: [fact]` - Add knowledge to the model
- `!correct: [correction]` - Correct the model's understanding
- `!save` - Save the current conversation
- `!system: [message]` - Change the system message
- `!toggle-stream` - Toggle streaming output on/off
- `!toggle-turbo` - Toggle turbo mode on/off
- `!mcp-help` - Show MCP commands for directing output to files
- `!toggle-filter` - Toggle uncertainty filtering on/off
- `!confidence: [0.0-1.0]` - Set confidence threshold
- `!toggle-heatmap` - Toggle confidence heatmap visualization
- `!toggle-all-metrics` - Toggle between showing all metrics or just truthiness
- `!memorize` - Force save the entire conversation to memory
- `!toggle-memory` - Toggle automatic memorization on/off
- `!memory-stats` - Display info about memories

To exit the chat, type `exit`.

## Model Content Protocol (MCP)

Local models Chat supports special commands for directing model outputs to files:

### User-Directed Output

Include a file directive in your query to save the entire response:

```
Explain quantum computing @{quantum.md}
```

### Model-Directed Output

The model can save specific content to files using:

```
>>> FILE: filename.ext
Content to save goes here...
<<<
```

### Shell Commands

You can execute shell commands (if enabled) using:

```
!{ls -la}
```

## Memory System

Local models Chat features an advanced memory system:

- **Automatic Memorization**: The model automatically saves key information from conversations
- **Vector Embeddings**: Uses FAISS for efficient similarity search
- **Memory Consolidation**: Deduplicates and optimizes memories periodically
- **Prioritized Corrections**: Corrections are given higher priority in future responses

Memories are stored in the `./memory` directory and persist between sessions.

## Code Structure

- `local_ai.py`: Main chat application
- `enhanced_memory_store.py`: Vector-based memory management
- `mcp_handler.py`: Model Content Protocol for file operations
- `terminal_heatmap.py`: Confidence visualization with color coding
- `response_filter.py`: Low-confidence response filtering
- `confidence_metrics.py`: Token confidence calculations
- `logit_capture_processor.py`: Captures token probabilities during generation

## How It Works

### Speculative Decoding

Local models Chat uses speculative decoding to accelerate generation by:

1. Creating a reduced-layer "draft" model from the full model
2. Generating candidate tokens with the faster draft model
3. Verifying those tokens with the full model in parallel
4. Accepting correct predictions and regenerating from the first mismatch

This approach can achieve significant speedups without sacrificing quality.

### Memory Management

The memory system works by:

1. Extracting key information from conversations
2. Converting text to vector embeddings (using sentence-transformers)
3. Storing information in a FAISS index for efficient retrieval
4. Retrieving relevant memories based on query similarity
5. Incorporating memories into the system prompt for context

### Confidence Metrics

The confidence system:

1. Captures token-by-token probabilities during generation
2. Calculates entropy, perplexity, and confidence scores
3. Visualizes confidence with color-coded text output
4. Filters responses that fall below confidence thresholds

## Limitations

- Works best with smaller models (1-3B parameters)
- CUDA acceleration is recommended for good performance
- Memory usage increases with conversation history
- Some features are simplified for demonstrations

## Troubleshooting

- **Out of memory errors**: Reduce --max-tokens or use a smaller model
- **Slow generation**: Enable --turbo mode, use GPU if available
- **Model not found**: Check internet connection, model name spelling
- **Unexpected hallucinations**: Increase confidence threshold, add corrections
- **Corrupt vector story erros**: `rm -rf ./memory/*_store*` or --no-memory flag to temporarily run without loading the existing memory

# GPU

If you're experiencing problems with GPU, try these:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
export GPU_SINGLE_ALLOC_PERCENT=100
export HSA_ENABLE_SDMA=0
```

or try running with:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python local_ai.py --temperature 0.1 --max-tokens 512 --heatmap
```

# ROCm

For ROCm you have to install

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

In case bitsandbytes complain get this:

```bash
pip install --no-deps --force-reinstall 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl'
```

(source: https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1202)

# Gemma

Branch `unstable_18` contains changes related to Gemma models

run:

```bash
python local_ai.py --model google/gemma-3-1b-it --temperature 0.7 --repetition-penalty 1.3 --load-in-8bit --heatmap
```

or

```bash
python local_ai.py --model google/gemma-3-4b-it --temperature 0.7 --repetition-penalty 1.3 --load-in-4bit --disable-speculative --max-tokens 256
```
