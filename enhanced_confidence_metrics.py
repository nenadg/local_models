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
    Enhanced confidence metrics for model generation with sharpening capabilities.
    Combines functionality from original ConfidenceMetrics with enhanced features.
    """
    def __init__(self, sharpening_factor: float = 0.3):
        """
        Initialize the confidence metrics.
        Args:
            sharpening_factor: How strongly to sharpen metrics (typically 1.0-2.0)
        """
        self.token_probabilities = []
        self.token_entropies = []
        self.accumulated = False
        self.sharpening_factor = sharpening_factor

        # Store original values for comparison
        self.original_token_probabilities = []
        self.original_token_entropies = []

    def reset(self):
        """Reset metrics for a new generation."""
        self.token_probabilities = []
        self.token_entropies = []
        self.original_token_probabilities = []
        self.original_token_entropies = []
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

        # Always store original values first
        self.original_token_probabilities.append(token_prob)
        self.original_token_entropies.append(entropy)

        # Apply sharpening
        sharpened_prob, sharpened_entropy = self._sharpen_token_metrics(token_prob, entropy)

        # Store sharpened values
        self.token_probabilities.append(sharpened_prob)
        self.token_entropies.append(sharpened_entropy)

        # print(f"DEBUG - Added token: prob={token_prob:.3f}, entropy={entropy:.3f}")
        # print(f"DEBUG - Original tokens recorded: {len(self.original_token_probabilities)}")


    def _sharpen_token_metrics(self, probability: float, entropy: float, sharpening_factor: float = 0.3) -> Tuple[float, float]:
        """
        Apply sharpening to token probability and entropy.

        Args:
            probability: Original token probability
            entropy: Original entropy value
            sharpening_factor: Strength of sharpening effect (higher = stronger)

        Returns:
            Tuple of (sharpened_probability, sharpened_entropy)
        """

        if self.sharpening_factor:
            sharpening_factor = self.sharpening_factor

        # Skip if no sharpening requested
        if sharpening_factor <= 0:
            return probability, entropy

        # Apply non-linear sharpening to probability
        # Higher probabilities get boosted, lower ones reduced
        if probability > 0.5:
            # For high probabilities, boost them higher (more confidence)
            boost = (probability - 0.5) * sharpening_factor
            sharpened_probability = min(1.0, probability + boost)
        else:
            # For low probabilities, reduce them further (less confidence)
            reduction = (0.5 - probability) * sharpening_factor
            sharpened_probability = max(0.0, probability - reduction)

        # For entropy, lower is better (less uncertainty), so invert the effect
        if entropy < 1.0:
            # For low entropy (good), make it even lower
            reduction = entropy * sharpening_factor * 0.5
            sharpened_entropy = max(0.0, entropy - reduction)
        else:
            # For high entropy (bad), make it higher to emphasize uncertainty
            boost = (entropy - 1.0) * sharpening_factor * 0.5
            sharpened_entropy = entropy + boost

        return sharpened_probability, sharpened_entropy

    def sharpen_values(self, values: List[float], threshold: float, higher_is_better: bool = True) -> List[float]:
        """
        Apply sharpening effect to a list of values.
        Args:
            values: List of metric values
            threshold: Reference threshold for good/bad values
            higher_is_better: Whether higher values are better
        Returns:
            List of sharpened values
        """
        if not values:
            return []

        sharpened = []

        for val in values:
            if higher_is_better:
                # For metrics where higher is better (like probability)
                if val > threshold:
                    # Above threshold - make it higher
                    sharp_val = min(1.0, val + (val - threshold) * (self.sharpening_factor - 1.0))
                else:
                    # Below threshold - make it lower
                    ratio = val / threshold
                    sharp_val = val * (0.8 + (ratio * 0.2))
            else:
                # For metrics where lower is better (like entropy/perplexity)
                if val < threshold:
                    # Below threshold (good) - make it even lower
                    ratio = val / threshold
                    sharp_val = val * (ratio ** (self.sharpening_factor - 1.0))
                else:
                    # Above threshold (bad) - make it higher to create contrast
                    excess = val - threshold
                    sharp_val = threshold + (excess * self.sharpening_factor)

            sharpened.append(sharp_val)

        return sharpened

    def get_metrics(self, apply_sharpening: bool = False) -> Dict[str, float]:
        """
        Get the confidence metrics with optional sharpening.

        Args:
            apply_sharpening: Whether to apply sharpening to metrics

        Returns:
            Dictionary of confidence metrics
        """
        # Calculate basic metrics from token probabilities
        if not self.token_probabilities and not self.original_token_probabilities:
            # Default placeholder metrics if no tokens processed
            return {
                'confidence': 0.8,    # Default high confidence
                'perplexity': 2.0,    # Default good perplexity
                'entropy': 1.0,       # Default good entropy
                'tokens': 0
            }

        # Decide which probabilities to use
        probs = self.token_probabilities if self.token_probabilities else self.original_token_probabilities

        # Calculate average confidence
        avg_confidence = sum(probs) / max(1, len(probs))

        # Calculate perplexity (using original unsharpened probabilities for accuracy)
        orig_probs = self.original_token_probabilities if self.original_token_probabilities else probs
        # Ensure no zeros to avoid log(0)
        safe_probs = [max(p, 1e-10) for p in orig_probs]
        avg_log_prob = sum(-np.log(safe_probs)) / len(safe_probs)
        perplexity = float(np.exp(avg_log_prob))

        # Calculate entropy (uncertainty) - higher entropy = more uncertainty
        try:
            # Need to convert to scalar values explicitly to avoid array truth value error
            entropy_vals = [-p * np.log2(p) if isinstance(p, (int, float)) and p > 0 else 0 for p in safe_probs]
            entropy = float(sum(entropy_vals) / max(1, len(entropy_vals)))
        except Exception as e:
            print(f"Error calculating entropy: {e}")
            entropy = 1.0  # Default to moderate entropy

        metrics = {
            'confidence': float(avg_confidence),
            'perplexity': float(perplexity),
            'entropy': float(entropy),
            'tokens': len(probs)
        }

        # Apply sharpening if requested
        if apply_sharpening and self.sharpening_factor > 0:
            # Store original metrics
            original_metrics = metrics.copy()

            # Apply sharpening (calculate again with sharpened probs)
            if self.token_probabilities != self.original_token_probabilities:
                # Calculate with already sharpened probabilities
                sharpened_confidence = avg_confidence

                # Calculate enhancement percentage
                if self.original_token_probabilities:
                    orig_confidence = sum(self.original_token_probabilities) / max(1, len(self.original_token_probabilities))
                    enhancement = ((sharpened_confidence - orig_confidence) / max(0.01, orig_confidence)) * 100
                else:
                    enhancement = 0.0

                # Add sharpening info to metrics
                metrics['original'] = original_metrics
                metrics['enhancement'] = float(enhancement)

            else:
                # Apply fresh sharpening
                sharpened_probs = self.sharpen_confidence_scores(probs)

                # Recalculate with sharpened values
                sharpened_confidence = sum(sharpened_probs) / max(1, len(sharpened_probs))

                # Calculate enhancement
                enhancement = ((sharpened_confidence - avg_confidence) / max(0.01, avg_confidence)) * 100

                # Update metrics with sharpened values
                metrics['confidence'] = float(sharpened_confidence)
                metrics['original'] = original_metrics
                metrics['enhancement'] = float(enhancement)

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

    def set_sharpening_factor(self, factor: float) -> None:
        """
        Set the sharpening factor and recalculate metrics.
        Args:
            factor: New sharpening factor (typically 1.0-2.0)
        """
        self.sharpening_factor = factor

        # Reset sharpened values
        self.token_probabilities = []
        self.token_entropies = []

        # Recalculate all sharpened values
        for i in range(len(self.original_token_probabilities)):
            prob = self.original_token_probabilities[i]
            entropy = self.original_token_entropies[i]
            sharp_prob, sharp_entropy = self._sharpen_token_metrics(prob, entropy)
            self.token_probabilities.append(sharp_prob)
            self.token_entropies.append(sharp_entropy)

        """Set the sharpening factor for memory retrieval and confidence metrics"""
        # Update confidence metrics sharpening
        # if hasattr(self.confidence_metrics, 'set_sharpening_factor'):
        #     old_factor = self.confidence_metrics.sharpening_factor
        #     self.confidence_metrics.set_sharpening_factor(factor)
        #     print(f"DEBUG - Updated confidence sharpening factor: {old_factor} -> {factor}")

        #     # Force recalculation with sharpening applied
        #     if hasattr(self.confidence_metrics, 'original_token_probabilities'):
        #         print(f"DEBUG - Original tokens: {len(self.confidence_metrics.original_token_probabilities)}")
        #         if self.confidence_metrics.original_token_probabilities:
        #             metrics = self.confidence_metrics.get_metrics(apply_sharpening=True)
        #             print(f"DEBUG - Recalculated metrics: {metrics.keys()}")

    def get_token_probabilities(self) -> List[float]:
        """Get raw token probabilities for pattern detection."""
        return self.token_probabilities.copy() if hasattr(self, 'token_probabilities') else []