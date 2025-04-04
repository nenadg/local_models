import torch
import numpy as np

class ConfidenceMetrics:
    """
    Enhanced confidence metrics for model generation with more aggressive
    scaling to better detect uncertainty.
    """
    def __init__(self):
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
        self.token_probabilities.append(token_prob)

        # Calculate entropy for this token distribution
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
        self.token_entropies.append(entropy)

    def get_metrics(self):
        """
        Calculate and return confidence metrics with more aggressive scaling.

        Returns:
            Dict of confidence metrics
        """
        if not self.token_probabilities:
            return {
                "confidence": 0.0,
                "perplexity": 0.0,
                "entropy": 0.0
            }

        # Mean probability (direct confidence measure)
        mean_prob = np.mean(self.token_probabilities)

        # Calculate perplexity (transformed into 0-1 range where 1 is best)
        mean_entropy = np.mean(self.token_entropies)
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
        if len(self.token_probabilities) > 10:
            # Calculate the standard deviation of token probabilities
            std_dev = np.std(self.token_probabilities)
            # Higher standard deviation often indicates more uncertainty
            if std_dev > 0.2:  # High variation in token confidence
                confidence = confidence * 0.8  # Reduce confidence

            # Check for confidence decay (early tokens confident, later tokens uncertain)
            # This pattern often appears in hallucination
            early_tokens = self.token_probabilities[:10]
            later_tokens = self.token_probabilities[-10:]
            early_mean = np.mean(early_tokens)
            later_mean = np.mean(later_tokens)

            if early_mean - later_mean > 0.15:  # Significant decay
                confidence = confidence * 0.9  # Penalize confidence

        return {
            "confidence": round(confidence, 2),
            "perplexity": round(perplexity, 2),
            "entropy": round(mean_entropy, 2),
            "raw_prob": round(mean_prob, 2)  # Include raw probability for debugging
        }

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