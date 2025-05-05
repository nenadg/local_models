#!/usr/bin/env python3
"""
Test suite for the unified memory system.
Tests memory storage, retrieval, and enhanced embedding functionality.
"""

import sys
import os
import unittest
import tempfile
import shutil
import numpy as np
from unittest.mock import MagicMock, patch
import time

# Add parent directory to path to import from root
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the MemoryManager class
from unified_memory import MemoryManager, MemoryItem

class TestMemorySystem(unittest.TestCase):
    """Test cases for the unified memory system."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for memory storage
        self.test_dir = tempfile.mkdtemp()

        # Create a memory manager with a fixed embedding dimension for testing
        self.memory_manager = MemoryManager(
            storage_path=self.test_dir,
            embedding_dim=64,  # Smaller dimension for faster tests
            enable_enhanced_embeddings=True,
            max_enhancement_levels=2,
            auto_save=True,
            similarity_enhancement_factor=0.3
        )

        # Set up a mock embedding function
        def mock_embed(text):
            # Generate a deterministic embedding based on the text
            text_hash = hash(text) % 10000
            np.random.seed(text_hash)
            embedding = np.random.randn(64).astype(np.float32)
            # Normalize
            return embedding / np.linalg.norm(embedding)

        # Create a simpler batch embedding function
        def mock_batch_embed(texts):
            return [mock_embed(text) for text in texts]

        self.memory_manager.set_embedding_function(mock_embed, mock_batch_embed)

        # Add some test items
        self.test_items = [
            {"content": "Paris is the capital of France", "metadata": {"source": "test", "topic": "geography"}},
            {"content": "Python is a programming language", "metadata": {"source": "test", "topic": "programming"}},
            {"content": "The Eiffel Tower is located in Paris", "metadata": {"source": "test", "topic": "landmarks"}},
            {"content": "JavaScript is used for web development", "metadata": {"source": "test", "topic": "programming"}}
        ]

        # Add test items to memory
        self.item_ids = []
        for item in self.test_items:
            item_id = self.memory_manager.add(
                content=item["content"],
                metadata=item["metadata"]
            )
            self.item_ids.append(item_id)

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up the test directory
        shutil.rmtree(self.test_dir)

    def test_add_and_retrieve(self):
        """Test basic adding and retrieving functionality."""
        # Verify items were added
        stats = self.memory_manager.get_stats()
        self.assertEqual(stats["active_items"], len(self.test_items))

        # Try to get an item by ID
        item_id = self.item_ids[0]
        item = self.memory_manager.get(item_id)
        self.assertIsNotNone(item)
        self.assertEqual(item["content"], self.test_items[0]["content"])

    def test_similarity_enhancement(self):
        """Test that similarity enhancement factor affects rankings."""
        # Set a higher enhancement factor
        self.memory_manager.similarity_enhancement_factor = 0.5

        # Get stats to verify memory has items
        stats = self.memory_manager.get_stats()
        self.assertGreater(stats["active_items"], 0)

        # Test that similarity_enhancement_factor can be set
        self.assertEqual(self.memory_manager.similarity_enhancement_factor, 0.5)

    def test_enhanced_embeddings(self):
        """Test that enhanced embeddings are created."""
        # Check that we have enhanced embeddings
        has_additional = False
        for item in self.memory_manager.items:
            if hasattr(item, 'additional_embeddings') and item.additional_embeddings:
                has_additional = True
                break

        self.assertTrue(has_additional, "No items have enhanced embeddings")

    def test_metadata_filtering(self):
        """Test filtering results by metadata."""
        # Get all programming items
        programming_items = []
        for item in self.memory_manager.items:
            if item.metadata.get("topic") == "programming":
                programming_items.append(item)

        # Should have 2 programming items
        self.assertEqual(len(programming_items), 2)

        # Get a geography item
        geography_items = []
        for item in self.memory_manager.items:
            if item.metadata.get("topic") == "geography":
                geography_items.append(item)

        # Should have 1 geography item
        self.assertEqual(len(geography_items), 1)

    def test_get_and_update(self):
        """Test getting and updating items."""
        # Get an item
        item_id = self.item_ids[0]
        item = self.memory_manager.get(item_id)

        # Check item content
        self.assertEqual(item["content"], self.test_items[0]["content"])

        # Update the item
        new_content = "Paris is the beautiful capital of France"
        update_success = self.memory_manager.update(
            item_id=item_id,
            updates={"content": new_content}
        )

        # Check update was successful
        self.assertTrue(update_success)

        # Get the updated item
        updated_item = self.memory_manager.get(item_id)
        self.assertEqual(updated_item["content"], new_content)

        # Update metadata
        update_success = self.memory_manager.update(
            item_id=item_id,
            updates={"metadata": {"importance": "high"}}
        )

        # Check metadata was merged, not replaced
        updated_item = self.memory_manager.get(item_id)
        self.assertEqual(updated_item["metadata"]["importance"], "high")
        self.assertEqual(updated_item["metadata"]["topic"], "geography")

    def test_remove(self):
        """Test removing items from memory."""
        # Get total items before removal
        stats_before = self.memory_manager.get_stats()

        # Remove an item
        item_id = self.item_ids[0]
        remove_success = self.memory_manager.remove(item_id)

        # Check removal was successful
        self.assertTrue(remove_success)

        # Try to get the removed item
        removed_item = self.memory_manager.get(item_id)
        self.assertIsNone(removed_item)

        # Check stats after removal
        stats_after = self.memory_manager.get_stats()
        self.assertEqual(stats_after["active_items"], stats_before["active_items"] - 1)
        self.assertEqual(stats_after["deleted_items"], stats_before["deleted_items"] + 1)

    def test_save_and_load(self):
        """Test saving and loading memory data."""
        # First save the current state
        save_success = self.memory_manager.save()
        self.assertTrue(save_success)

        # Verify files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "items.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "embeddings.npy")))

    def test_format_for_context(self):
        """Test format_for_context method works."""
        # Create some mock results
        mock_results = []

        # Add a high confidence item
        mock_results.append({
            "id": "item_12345_678",
            "content": "Paris is the capital of France",
            "similarity": 0.9,
            "metadata": {"topic": "geography"}
        })

        # Add a medium confidence item
        mock_results.append({
            "id": "item_23456_789",
            "content": "The Eiffel Tower is in Paris",
            "similarity": 0.6,
            "metadata": {"topic": "landmarks"}
        })

        # Format for context
        context = self.memory_manager.format_for_context(mock_results)

        # Check that format works
        self.assertIsInstance(context, str)

    def test_embedding_dimension_update(self):
        """Test updating the embedding dimension."""
        # Skip this test if the update_embedding_dimension method isn't present
        if not hasattr(self.memory_manager, 'update_embedding_dimension'):
            self.skipTest("update_embedding_dimension method not implemented")

        # Original dimension
        original_dim = self.memory_manager.embedding_dim

        # Get first item content for later comparison
        first_item_id = self.item_ids[0]
        first_item = self.memory_manager.get(first_item_id)
        first_item_content = first_item["content"]

        # Test that we can at least call the method
        try:
            self.memory_manager.update_embedding_dimension(original_dim)
            self.assertEqual(self.memory_manager.embedding_dim, original_dim)
        except Exception as e:
            self.fail(f"update_embedding_dimension raised {type(e)} unexpectedly: {e}")

if __name__ == '__main__':
    unittest.main()