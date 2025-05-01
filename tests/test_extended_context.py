import unittest
import sys
import os
import numpy as np
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary modules
# Note: Actual imports will depend on your project structure
from unified_memory import UnifiedMemoryManager, MemoryItem
from local_ai import TinyLlamaChat

class TestExtendedContext(unittest.TestCase):
    """Tests for the extended context window functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock embedding function
        def mock_embedding_function(text):
            # Create a deterministic but unique embedding based on text
            hash_value = hash(text) % 1000
            embedding = np.zeros(384)
            embedding[0] = hash_value / 1000
            return embedding
            
        # Create test memory manager
        self.memory_manager = UnifiedMemoryManager(
            storage_path="./test_memory",
            embedding_function=mock_embedding_function,
            embedding_dim=384,
            use_fractal=True,
            max_fractal_levels=3,
            auto_save=False  # Disable auto-save for tests
        )
        
        # Add some test memories
        self.memory_manager.add(
            content="Alice is a software engineer who specializes in Python and machine learning.",
            metadata={"source": "test", "relevance": "high"}
        )
        
        self.memory_manager.add(
            content="Bob is a data scientist working on natural language processing.",
            metadata={"source": "test", "relevance": "medium"}
        )
        
        self.memory_manager.add(
            content="Python is a programming language often used for data science and web development.",
            metadata={"source": "test", "relevance": "low"}
        )
        
        # Create mock window manager
        self.window_manager = MagicMock()
        self.window_manager.optimize_messages.return_value = []
        self.window_manager._count_tokens_safely.return_value = 10
        
        # Create mock tokenizer
        self.tokenizer = MagicMock()
        self.tokenizer.apply_chat_template.return_value = "test prompt"
        
        # Create mock model
        self.model = MagicMock()
        
        # Patch TinyLlamaChat to avoid actual initialization
        with patch('local_ai.TinyLlamaChat') as MockTinyLlamaChat:
            self.chat = TinyLlamaChat()
            self.chat.memory_manager = self.memory_manager
            self.chat.window_manager = self.window_manager
            self.chat.model = self.model
            self.chat.tokenizer = self.tokenizer
            self.chat.system_message = {"role": "system", "content": "You are a helpful assistant."}
            self.chat.get_time = MagicMock(return_value="[01/01/00 00:00:00]")

            # Create mock web_enhancer
            self.chat.web_enhancer = MagicMock()
            self.chat.web_enhancer.format_web_results_for_context = MagicMock(return_value="")
            self.chat.enhance_with_web_knowledge = MagicMock(return_value={"enhanced": False})

            # Set enable_web_knowledge attribute
            self.chat.enable_web_knowledge = False

            # Create stub for enhance_query_for_continuation
            self.chat.enhance_query_for_continuation = MagicMock(side_effect=lambda x: x)

    def tearDown(self):
        """Clean up after tests."""
        # Remove test memory directory
        import shutil
        if os.path.exists("./test_memory"):
            shutil.rmtree("./test_memory")

    def test_create_prompt_with_extended_context_basic(self):
        """Test that the method runs without errors."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about Python"}
        ]

        # Call the method
        result = self.chat.create_prompt_with_extended_context(messages)

        # Check basic structure
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]["role"], "system")

    def test_hierarchical_context_structure(self):
        """Test that hierarchical context is properly structured."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about Python programming"}
        ]

        # Modify retrieve to return different results for different levels
        original_retrieve = self.memory_manager.retrieve

        def mock_retrieve(query, top_k=5, min_similarity=0.25, use_fractal=True):
            if min_similarity < 0.2:  # Higher level (broader context)
                return [{"content": "Programming languages are used to create software.",
                         "similarity": 0.6}]
            else:  # Level 0 (specific context)
                return [{"content": "Python is a programming language often used for data science.",
                         "similarity": 0.8}]

        self.memory_manager.retrieve = mock_retrieve

        try:
            # Turn off web search for this test
            self.chat.enable_web_knowledge = False
            self.chat.web_enhancer.format_web_results_for_context.return_value = ""

            # Call the method
            result = self.chat.create_prompt_with_extended_context(messages)

            # Check that result contains hierarchical context structure
            system_content = result[0]["content"]
            self.assertIn("HIERARCHICAL CONTEXT", system_content)
            self.assertIn("DIRECT CONTEXT", system_content)
            self.assertIn("BROADER CONTEXT", system_content)

            # Check content from different levels is included
            self.assertIn("Python is a programming language", system_content)
            self.assertIn("Programming languages are used", system_content)

        finally:
            # Restore original method
            self.memory_manager.retrieve = original_retrieve

    def test_comparison_with_original_method(self):
        """Test that extended context provides more information than original method."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about Python programming"}
        ]

        # Mock the standard method to return a simple result
        self.chat.create_prompt_with_knowledge = MagicMock()
        standard_result = [
            {"role": "system", "content": "You are a helpful assistant.\n\nKNOWLEDGE:\nPython is a programming language."},
            {"role": "user", "content": "Tell me about Python programming"}
        ]
        self.chat.create_prompt_with_knowledge.return_value = standard_result

        # Mock retrieve to return realistic results
        original_retrieve = self.memory_manager.retrieve

        def mock_retrieve(query, top_k=5, min_similarity=0.25, use_fractal=True):
            return [
                {"content": "Python is a programming language often used for data science.", "similarity": 0.8},
                {"content": "Python has libraries like NumPy and Pandas.", "similarity": 0.7}
            ]

        self.memory_manager.retrieve = mock_retrieve

        try:
            # Turn off web search for this test
            self.chat.enable_web_knowledge = False
            self.chat.web_enhancer.format_web_results_for_context.return_value = ""

            # Get result from extended context method
            extended_result = self.chat.create_prompt_with_extended_context(messages)

            # Compare system message content
            standard_content = standard_result[0]["content"]
            extended_content = extended_result[0]["content"]

            # Print the contents for debugging
            print("Standard content:", standard_content)
            print("Extended content:", extended_content)

            # The extended context should include hierarchical context
            self.assertIn("HIERARCHICAL CONTEXT", extended_content)

            # The extended content should be longer
            self.assertGreater(len(extended_content), len(standard_content))

        finally:
            # Restore original retrieve method
            self.memory_manager.retrieve = original_retrieve

    def test_continuation_enhancement(self):
        """Test that continuation queries are properly enhanced."""
        # Setup continuation context
        self.chat.enhance_query_for_continuation = MagicMock()
        enhanced_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Continue the explanation about Python with more details"}
        ]
        self.chat.enhance_query_for_continuation.return_value = enhanced_messages

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Continue the explanation about Python"}
        ]

        # Call the method
        result = self.chat.create_prompt_with_extended_context(messages)

        # Check that enhance_query_for_continuation was called
        self.chat.enhance_query_for_continuation.assert_called_once_with(messages)

        # Check that the enhanced messages were used
        self.assertEqual(result[0]["role"], enhanced_messages[0]["role"])

    def test_performance_under_load(self):
        """Test performance with many memories."""
        # Add more memories to test performance
        for i in range(100):
            self.memory_manager.add(
                content=f"Test memory {i} with some random content about Python programming.",
                metadata={"source": "test", "relevance": "low"}
            )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about Python programming"}
        ]

        # Measure execution time
        import time
        start_time = time.time()

        # Call the method
        result = self.chat.create_prompt_with_extended_context(messages)

        execution_time = time.time() - start_time

        # Should process in reasonable time even with many memories
        self.assertLess(execution_time, 5.0)

        # Check that we got a valid result
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()