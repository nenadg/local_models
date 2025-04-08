import torch
import numpy as np
from typing import List, Dict, Any
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

    def _sharpen_token_metrics(self, probability: float, entropy: float) -> tuple:
        """
        Apply sharpening to token metrics, with improved algorithm for better results.

        Args:
            probability: Original token probability
            entropy: Original entropy value
        Returns:
            Tuple of (sharpened_probability, sharpened_entropy)
        """
        # IMPROVED: More precise probability sharpening based on ranges
        if probability > 0.75:
            # For very high confidence, small boosting with diminishing returns
            boost = min(0.15, (probability - 0.75) * self.sharpening_factor * 0.5)
            sharpened_prob = min(0.99, probability + boost)
        elif probability > 0.5:
            # For medium-high confidence, moderate boost
            boost = (probability - 0.5) * self.sharpening_factor * 0.4
            sharpened_prob = probability + boost
        elif probability > 0.3:
            # For medium confidence, slight reduction
            reduction = (0.5 - probability) * self.sharpening_factor * 0.2
            sharpened_prob = probability - reduction
        else:
            # For low confidence, stronger reduction
            reduction = (0.3 - probability) * self.sharpening_factor * 0.8
            sharpened_prob = max(0.01, probability - reduction)

        # IMPROVED: More nuanced entropy sharpening with reference points
        reference_low = 1.0   # Low entropy (good)
        reference_high = 3.0  # High entropy (bad)

        if entropy < reference_low:
            # Good (low) entropy - make it even better (lower) with diminishing returns
            factor = 1.0 - (self.sharpening_factor * 0.3 * (reference_low - entropy) / reference_low)
            sharpened_entropy = entropy * factor
        elif entropy < reference_high:
            # Medium entropy - interpolate between references
            position = (entropy - reference_low) / (reference_high - reference_low)
            # Less sharpening in the middle range
            adjustment = self.sharpening_factor * 0.2 * (0.5 - abs(position - 0.5))
            sharpened_entropy = entropy * (1.0 + adjustment)
        else:
            # Bad (high) entropy - make it worse (higher) with more aggressive factor for very poor entropy
            excess = entropy - reference_high
            scaling = 1.0 + (self.sharpening_factor * 0.4 * min(1.0, excess / 2.0))
            sharpened_entropy = reference_high + (excess * scaling)

        return sharpened_prob, sharpened_entropy


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

    def get_metrics(self, apply_sharpening: bool = True):
        """
        Calculate and return confidence metrics with sharpening.
        Args:
            apply_sharpening: Whether to apply sharpening effects
        Returns:
            Dict of confidence metrics
        """
        if not self.token_probabilities and not self.original_token_probabilities:
            return {
                "confidence": 0.0,
                "perplexity": 0.0,
                "entropy": 0.0,
                "raw_prob": 0.0
            }

        # Make sure we have original values to compare against
        if not self.original_token_probabilities and self.token_probabilities:
            self.original_token_probabilities = self.token_probabilities.copy()
            self.original_token_entropies = self.token_entropies.copy()

        # Choose which values to use
        probs = self.token_probabilities if apply_sharpening else self.original_token_probabilities
        entropies = self.token_entropies if apply_sharpening else self.original_token_entropies

        # Mean probability (direct confidence measure)
        mean_prob = np.mean(probs)

        # Calculate perplexity (transformed into 0-1 range where 1 is best)
        mean_entropy = np.mean(entropies)
        perplexity = 2 ** mean_entropy

        # More aggressive scaling for perplexity
        perplexity_score = max(0.0, min(1.0, 1.0 - (perplexity / 20.0)))

        # More aggressive entropy-based confidence
        max_possible_entropy = 6.0  # Lower threshold makes this more sensitive
        entropy_confidence = max(0.0, min(1.0, 1.0 - (mean_entropy / max_possible_entropy)))

        # More weight on entropy/perplexity, less on raw probability
        confidence = 0.3 * mean_prob + 0.35 * perplexity_score + 0.35 * entropy_confidence

        # Apply power function to create greater separation between values
        confidence = confidence ** 1.5  # Exponentiate to create more separation

        # Analyze the distribution of token probabilities for signs of inconsistency
        if len(probs) > 10:
            # Calculate the standard deviation of token probabilities
            std_dev = np.std(probs)
            # Higher standard deviation often indicates more uncertainty
            if std_dev > 0.2:  # High variation in token confidence
                confidence = confidence * 0.8  # Reduce confidence

            # Check for confidence decay (early tokens confident, later tokens uncertain)
            # This pattern often appears in hallucination
            early_tokens = probs[:10]
            later_tokens = probs[-10:]
            early_mean = np.mean(early_tokens)
            later_mean = np.mean(later_tokens)

            if early_mean - later_mean > 0.15:  # Significant decay
                confidence = confidence * 0.9  # Penalize confidence

        result = {
            "confidence": round(confidence, 2),
            "perplexity": round(perplexity, 2),
            "entropy": round(mean_entropy, 2),
            "raw_prob": round(mean_prob, 2)  # Include raw probability for debugging
        }

        # Include original (unsharpened) metrics for comparison if sharpening was applied
        if apply_sharpening and self.original_token_probabilities:
            # Calculate original metrics
            orig_mean_prob = np.mean(self.original_token_probabilities)
            orig_mean_entropy = np.mean(self.original_token_entropies)
            orig_perplexity = 2 ** orig_mean_entropy

            # Calculate original confidence score with the same formula
            orig_perplexity_score = max(0.0, min(1.0, 1.0 - (orig_perplexity / 20.0)))
            orig_entropy_confidence = max(0.0, min(1.0, 1.0 - (orig_mean_entropy / max_possible_entropy)))
            orig_confidence = 0.3 * orig_mean_prob + 0.35 * orig_perplexity_score + 0.35 * orig_entropy_confidence
            orig_confidence = orig_confidence ** 1.5

            result["original"] = {
                "confidence": round(orig_confidence, 2),
                "perplexity": round(orig_perplexity, 2),
                "entropy": round(orig_mean_entropy, 2),
                "raw_prob": round(orig_mean_prob, 2)
            }

            # Calculate enhancement percentage
            conf_change = ((result["confidence"] - result["original"]["confidence"]) /
                         max(0.01, result["original"]["confidence"])) * 100
            result["enhancement"] = round(conf_change, 1)

        return result

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
        if hasattr(self.confidence_metrics, 'set_sharpening_factor'):
            old_factor = self.confidence_metrics.sharpening_factor
            self.confidence_metrics.set_sharpening_factor(factor)
            print(f"DEBUG - Updated confidence sharpening factor: {old_factor} -> {factor}")

            # Force recalculation with sharpening applied
            if hasattr(self.confidence_metrics, 'original_token_probabilities'):
                print(f"DEBUG - Original tokens: {len(self.confidence_metrics.original_token_probabilities)}")
                if self.confidence_metrics.original_token_probabilities:
                    metrics = self.confidence_metrics.get_metrics(apply_sharpening=True)
                    print(f"DEBUG - Recalculated metrics: {metrics.keys()}")