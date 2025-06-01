import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from transformers import LogitsProcessor, LogitsProcessorList

class TokenProbabilityCaptureProcessor(LogitsProcessor):
    """
    A LogitsProcessor that captures token probabilities during generation.
    This processor captures the probabilities and entropy for each token
    to provide accurate confidence metrics for the generated text.
    """
    def __init__(self, confidence_metrics):
        """
        Initialize the processor with a reference to the confidence metrics object.
        Args:
            confidence_metrics: EnhancedConfidenceMetrics instance to store token scores
        """
        self.confidence_metrics = confidence_metrics

    def __call__(self, input_ids, scores):
        """
        Process logits during generation and capture metrics.
        Args:
            input_ids: Current sequence of input IDs
            scores: Logits for the next token prediction
        Returns:
            The original scores (unchanged)
        """
        # For each item in the batch
        for i in range(scores.shape[0]):
            # Get most likely token id
            token_id = torch.argmax(scores[i]).item()
            # Record metrics for this token
            self.confidence_metrics.add_token_score(scores[i], token_id)
        # Return scores unchanged (we're just observing)
        return scores

class EnhancedConfidenceMetrics:
    """
    Enhanced confidence metrics for model generation.
    Tracks token probabilities and entropies without sharpening.
    """
    def __init__(self):
        """
        Initialize the confidence metrics.
        """
        self.token_probabilities = []
        self.token_entropies = []
        self.accumulated = False

    def reset(self):
        """Reset metrics for a new generation."""
        self.token_probabilities = []
        self.token_entropies = []
        self.accumulated = False

    def add_token_score(self, logits, token_id):
        """
        Record probability and entropy for a single token.
        Args:
            logits: Model logits for current token
            token_id: The selected token ID
        """
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Get probability of selected token
        token_prob = probs[token_id].item()

        # Calculate entropy for this distribution
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()

        # Store raw values
        self.token_probabilities.append(token_prob)
        self.token_entropies.append(entropy)

    def get_metrics(self, apply_sharpening: bool = False) -> Dict[str, float]:
        """
        Get the confidence metrics.

        Args:
            apply_sharpening: Ignored, kept for compatibility

        Returns:
            Dictionary of confidence metrics
        """
        # Calculate basic metrics from token probabilities
        if not self.token_probabilities:
            # Default placeholder metrics if no tokens processed
            return {
                'confidence': 0.8,    # Default high confidence
                'perplexity': 2.0,    # Default good perplexity
                'entropy': 1.0,       # Default good entropy
                'tokens': 0
            }

        # Calculate average confidence
        avg_confidence = sum(self.token_probabilities) / max(1, len(self.token_probabilities))

        # Calculate perplexity
        # Ensure no zeros to avoid log(0)
        safe_probs = [max(p, 1e-10) for p in self.token_probabilities]
        avg_log_prob = sum(-np.log(p) for p in safe_probs) / len(safe_probs)
        perplexity = float(np.exp(avg_log_prob))

        # Calculate average entropy
        if self.token_entropies:
            avg_entropy = sum(self.token_entropies) / len(self.token_entropies)
        else:
            # Fallback calculation if entropies weren't tracked
            entropy_vals = [-p * np.log2(p) if p > 0 else 0 for p in safe_probs]
            avg_entropy = float(sum(entropy_vals) / max(1, len(entropy_vals)))

        metrics = {
            'confidence': float(avg_confidence),
            'perplexity': float(perplexity),
            'entropy': float(avg_entropy),
            'tokens': len(self.token_probabilities)
        }

        return metrics

    def get_token_stats(self):
        """Get detailed statistics about token probabilities for analysis."""
        if not self.token_probabilities:
            return {
                "count": 0,
                "mean": 0,
                "median": 0,
                "min": 0,
                "max": 0,
                "std_dev": 0
            }

        return {
            "count": len(self.token_probabilities),
            "mean": float(np.mean(self.token_probabilities)),
            "median": float(np.median(self.token_probabilities)),
            "min": float(np.min(self.token_probabilities)),
            "max": float(np.max(self.token_probabilities)),
            "std_dev": float(np.std(self.token_probabilities))
        }

    def get_token_probabilities(self) -> List[float]:
        """Get raw token probabilities for pattern detection."""
        return self.token_probabilities.copy() if hasattr(self, 'token_probabilities') else []