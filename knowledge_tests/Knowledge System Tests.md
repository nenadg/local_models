# Knowledge System Tests

This set of test scripts helps verify the correct integration and functionality of the knowledge system components in TinyLlamaChat. The test suite checks the initialization, domain management, knowledge extraction, storage, search, and chat integration aspects of the system.

## Overview

The knowledge system consists of several components:

1. **Knowledge Domains** - Containers for domain-specific information
2. **Knowledge Registry** - Central registry for managing domains
3. **Knowledge Extractor** - Extracts structured knowledge from text
4. **Knowledge Validator** - Validates knowledge for consistency/quality
5. **Integration with TinyLlamaChat** - Uses knowledge to enhance responses

## Test Scripts

The test suite includes the following scripts:

1. `test_knowledge_initialization.py` - Verifies basic initialization of all components
2. `test_knowledge_domain_management.py` - Tests domain creation, loading, and management
3. `test_knowledge_extraction.py` - Tests extraction of knowledge from different types of text
4. `test_knowledge_search_and_storage.py` - Tests adding knowledge to domains and searching for it
5. `test_chat_integration.py` - Tests integration with TinyLlamaChat's response generation

Plus a main script to run all tests:

- `run_all_tests.py` - Runs all tests in sequence and reports results

## Requirements

- Python 3.8 or newer
- All dependencies for TinyLlamaChat
- The following files must be in your Python path or current directory:
  - `knowledge_domain.py`
  - `knowledge_extractor.py`
  - `knowledge_registry.py`
  - `knowledge_validator.py`
  - `memory_manager.py`
  - `tiny_llama_6_memory.py`

## Running the Tests

You can run individual tests directly:

```bash
python test_knowledge_initialization.py
```

Or run all tests using the main script:

```bash
python run_all_tests.py
```

### Options for Chat Integration Test

The chat integration test accepts additional parameters:

- `--model MODEL_NAME` - Specify a different model (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- `--no-generate` - Skip response generation (useful on systems without GPU)

Example:

```bash
python test_chat_integration.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --no-generate
```

Or with the main script:

```bash
python run_all_tests.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --no-generate
```

## Test Data

The tests use predefined sample texts for:

- Music information (about artist "Duett")
- Technical topics (about Python programming)
- Procedural content (how to make coffee)
- Mapping data (English to French translations)

These provide a diverse set of knowledge types to test the system.

## Troubleshooting

If you encounter errors:

1. Ensure all required files are in your Python path or current directory
2. Check that the required dependencies are installed
3. Verify that the memory directory (`./memory`) is writable
4. If you get CUDA/GPU errors, try running with `--no-generate` flag

For model loading issues, ensure:

- You have internet connectivity to download the model
- You have enough disk space for model files
- Your GPU has enough VRAM (or use CPU with `--no-generate`)

## Expected Output

When successful, each test will end with:

```
[SUCCESS] Knowledge [test-name] test passed
```

And the main runner will report:

```
===== TEST SUMMARY =====
Passed: 5/5

✅ PASSED - test_knowledge_initialization
✅ PASSED - test_knowledge_domain_management
✅ PASSED - test_knowledge_extraction
✅ PASSED - test_knowledge_search_and_storage
✅ PASSED - test_chat_integration

✅ ALL TESTS PASSED
```
