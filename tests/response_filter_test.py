#!/usr/bin/env python3
"""
Test suite for the ResponseFilter class.
Tests filter thresholds, confidence metrics, and filtering behavior.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Add parent directory to path to import from root
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the ResponseFilter class
from response_filter import ResponseFilter

class TestResponseFilter(unittest.TestCase):
    """Test cases for the ResponseFilter class."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.filter = ResponseFilter(
            confidence_threshold=0.7,
            entropy_threshold=2.0,
            perplexity_threshold=10.0,
            question_classifier=None
        )
        
        # Override continuation phrases to make testing more predictable
        self.filter.continuation_phrases = [
            "please continue",
            "continue anyway",
            "speculate anyway",
            "give it your best guess"
        ]

        # Mock a question classifier
        self.mock_classifier = MagicMock()
        self.mock_classifier.get_domain_settings.return_value = {
            'domain': 'factual',
            'memory_weight': 0.8,
            'sharpening_factor': 0.5,
            'confidence_threshold': 0.7,
            'retrieval_count': 12
        }

    def test_should_filter_high_confidence(self):
        """Test that high confidence responses aren't filtered."""
        # High confidence metrics
        metrics = {
            'confidence': 0.85,
            'entropy': 1.5,
            'perplexity': 5.0
        }

        should_filter, reason = self.filter.should_filter(metrics)

        self.assertFalse(should_filter)
        self.assertEqual(reason, "acceptable")

    def test_should_filter_low_confidence(self):
        """Test that low confidence responses are filtered."""
        # Low confidence metrics
        metrics = {
            'confidence': 0.5,
            'entropy': 3.0,
            'perplexity': 12.0
        }

        should_filter, reason = self.filter.should_filter(metrics)

        self.assertTrue(should_filter)
        self.assertEqual(reason, "low_confidence")

    def test_should_filter_high_entropy(self):
        """Test that high entropy responses are filtered."""
        # High entropy metrics
        metrics = {
            'confidence': 0.75,
            'entropy': 2.8,
            'perplexity': 8.0
        }

        should_filter, reason = self.filter.should_filter(metrics)

        self.assertTrue(should_filter)
        self.assertEqual(reason, "high_entropy")

    def test_should_filter_high_perplexity(self):
        """Test that high perplexity responses are filtered."""
        # High perplexity metrics
        metrics = {
            'confidence': 0.75,
            'entropy': 1.8,
            'perplexity': 15.0
        }

        should_filter, reason = self.filter.should_filter(metrics)

        self.assertTrue(should_filter)
        self.assertEqual(reason, "high_perplexity")

    def test_domain_specific_thresholds(self):
        """Test that domain-specific thresholds are applied."""
        # Create filter with mock classifier
        filter_with_classifier = ResponseFilter(
            confidence_threshold=0.6,  # Lower than domain setting
            entropy_threshold=2.5,     # Higher than domain setting
            perplexity_threshold=12.0, # Higher than domain setting
            question_classifier=self.mock_classifier
        )

        # Borderline metrics - would pass with default thresholds
        # but should fail with stricter domain thresholds
        metrics = {
            'confidence': 0.65,  # Below factual domain threshold (0.7)
            'entropy': 2.2,      # Below default (2.5) but above domain threshold (1.8)
            'perplexity': 9.0    # Below default (12.0) but above domain threshold (8.0)
        }

        # Query to trigger 'factual' domain
        query = "What is the capital of France?"

        should_filter, reason = filter_with_classifier.should_filter(metrics, query)

        # Should be filtered with domain-specific thresholds
        self.assertTrue(should_filter)

        # Check that the classifier was called with the query
        self.mock_classifier.get_domain_settings.assert_called_once_with(query)

    def test_sharpening_confidence_scores(self):
        """Test that confidence scores are properly sharpened."""
        # Test values
        token_probs = [0.3, 0.4, 0.5, 0.6, 0.7]

        # Set sharpening factor
        self.filter.sharpening_factor = 0.5

        # Sharpen scores
        sharpened = self.filter.sharpen_confidence_scores(token_probs)

        # Just verify that sharpening changes the values
        self.assertNotEqual(sharpened, token_probs)

        # Check that we get the right number of values back
        self.assertEqual(len(sharpened), len(token_probs))

    def test_check_override_instruction(self):
        """Test that override instructions are properly detected."""
        # Test with override phrase
        query_with_override = "I'm not sure about this. Please continue anyway."
        self.assertTrue(self.filter.check_override_instruction(query_with_override))

        # Test without override phrase
        query_without_override = "I'm not sure about this fact."
        self.assertFalse(self.filter.check_override_instruction(query_without_override))

        # Test adding a custom override phrase
        self.filter.add_continuation_phrase("keep going")
        query_with_custom = "I'm not sure about this. Keep going."
        self.assertTrue(self.filter.check_override_instruction(query_with_custom))

    def test_filter_response(self):
        """Test that filter_response returns a disclaimer with override."""
        # Original response
        response = "The capital of France is Paris. It is located on the Seine River."

        # Low confidence metrics
        low_metrics = {
            'confidence': 0.5,
            'entropy': 3.0,
            'perplexity': 12.0
        }

        # Filter without override
        filtered = self.filter.filter_response(
            response=response,
            metrics=low_metrics,
            query="What is the capital of France?",
            allow_override=True
        )

        # Check that response was filtered
        self.assertNotEqual(filtered, response)

        # Now test with override
        filtered_with_override = self.filter.filter_response(
            response=response,
            metrics=low_metrics,
            query="What is the capital of France? Please continue anyway.",
            allow_override=True
        )

        # Check that original response was kept with a disclaimer
        self.assertIn(response, filtered_with_override)
        self.assertIn("not entirely confident", filtered_with_override.lower())

    def test_repetition_detection(self):
        """Test detection of repetitive content."""
        # Create a response with repetition
        repeated_content = "This is a test.\nThis is a test.\nThis is a test.\nThis is a test."

        # Check repetition
        result = self.filter._check_for_repetition(repeated_content)

        # Should detect excessive repetition
        self.assertTrue(result['excessive_repetition'])
        self.assertEqual(result['max_repeats'], 4)

        # Create a response without repetition
        normal_content = "This is a test.\nThis is another test.\nHere is more content.\nFinal line."

        # Check repetition
        result = self.filter._check_for_repetition(normal_content)

        # Should not detect excessive repetition
        self.assertFalse(result['excessive_repetition'])

    def test_mcp_preservation(self):
        """Test that MCP commands are preserved during filtering with override."""
        # Response with MCP commands
        response_with_mcp = """The answer is 42.

>>> FILE: test.py
print("Hello, world!")
<<<"""

        # Low confidence metrics
        low_metrics = {
            'confidence': 0.5,
            'entropy': 3.0,
            'perplexity': 12.0
        }

        # Filter with MCP preservation and override
        filtered = self.filter.filter_response(
            response=response_with_mcp,
            metrics=low_metrics,
            query="What is the answer? Please continue anyway.",
            preserve_mcp=True,
            allow_override=True
        )

        # Check that MCP commands were preserved
        self.assertIn(">>> FILE: test.py", filtered)
        self.assertIn("print(\"Hello, world!\")", filtered)
        self.assertIn("<<<", filtered)

        # Response is preserved due to override
        self.assertIn("The answer is 42", filtered)

if __name__ == '__main__':
    unittest.main()