"""
Unified sharpening utilities for the local_models project.
Provides consistent sharpening functions for use across different components.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple

class SharpeningUtils:
    """
    Centralized utilities for sharpening various metrics and values.
    
    This class provides consistent sharpening behavior across different components:
    - Token probabilities and entropy (for EnhancedConfidenceMetrics)
    - Vector similarities (for VectorStore)
    - Confidence metrics (for ResponseFilter)
    """
    
    @staticmethod
    def sharpen_token_metrics(
        probability: float, 
        entropy: float, 
        sharpening_factor: float = 0.3
    ) -> Tuple[float, float]:
        """
        Apply sharpening to token metrics, with improved algorithm for better results.

        Args:
            probability: Original token probability
            entropy: Original entropy value
            sharpening_factor: Factor to control sharpening strength (0.0-1.0)
            
        Returns:
            Tuple of (sharpened_probability, sharpened_entropy)
        """
        # Probability sharpening based on ranges
        if probability > 0.75:
            # For very high confidence, small boosting with diminishing returns
            boost = min(0.15, (probability - 0.75) * sharpening_factor * 0.5)
            sharpened_prob = min(0.99, probability + boost)
        elif probability > 0.5:
            # For medium-high confidence, moderate boost
            boost = (probability - 0.5) * sharpening_factor * 0.4
            sharpened_prob = probability + boost
        elif probability > 0.3:
            # For medium confidence, slight reduction
            reduction = (0.5 - probability) * sharpening_factor * 0.2
            sharpened_prob = probability - reduction
        else:
            # For low confidence, stronger reduction
            reduction = (0.3 - probability) * sharpening_factor * 0.8
            sharpened_prob = max(0.01, probability - reduction)

        # More nuanced entropy sharpening with reference points
        reference_low = 1.0   # Low entropy (good)
        reference_high = 3.0  # High entropy (bad)

        if entropy < reference_low:
            # Good (low) entropy - make it even better (lower) with diminishing returns
            factor = 1.0 - (sharpening_factor * 0.3 * (reference_low - entropy) / reference_low)
            sharpened_entropy = entropy * factor
        elif entropy < reference_high:
            # Medium entropy - interpolate between references
            position = (entropy - reference_low) / (reference_high - reference_low)
            # Less sharpening in the middle range
            adjustment = sharpening_factor * 0.2 * (0.5 - abs(position - 0.5))
            sharpened_entropy = entropy * (1.0 + adjustment)
        else:
            # Bad (high) entropy - make it worse (higher) with more aggressive factor
            excess = entropy - reference_high
            scaling = 1.0 + (sharpening_factor * 0.4 * min(1.0, excess / 2.0))
            sharpened_entropy = reference_high + (excess * scaling)

        return sharpened_prob, sharpened_entropy

    @staticmethod
    def sharpen_similarity(
        similarity: float, 
        sharpening_factor: float = 0.3, 
        is_correction: bool = False,
        is_high_priority: bool = False
    ) -> float:
        """
        Apply sharpening to similarity scores for vector search.
        
        Args:
            similarity: Original similarity score (0.0-1.0)
            sharpening_factor: Factor to control sharpening strength (0.0-1.0)
            is_correction: Whether this is for a correction entry
            is_high_priority: Whether this is a high priority item
            
        Returns:
            Sharpened similarity score
        """
        # Apply priority boost if applicable
        if is_correction:
            # Always boost corrections significantly
            boost_factor = 0.3 + (sharpening_factor * 0.2)
            return min(1.0, similarity + boost_factor)
        elif is_high_priority:
            # Boost high priority items
            boost_factor = 0.2 + (sharpening_factor * 0.1)
            return min(1.0, similarity + boost_factor)
            
        # Standard similarity sharpening
        if similarity > 0.7:  # Very high similarity
            # Logarithmic boost (diminishing returns for very high similarities)
            boost = np.log1p(similarity) * sharpening_factor * 0.3
            return min(1.0, similarity + boost)
        elif similarity > 0.5:  # Medium-high similarity
            # Linear boost
            boost = (similarity - 0.5) * sharpening_factor * 0.5
            return min(1.0, similarity + boost)
        elif similarity > 0.35:  # Medium similarity
            # Neutral zone - minimal change
            return similarity
        else:  # Low similarity
            # Decrease low similarities more aggressively with higher sharpening factors
            reduction = (0.35 - similarity) * sharpening_factor * 0.6
            return max(0.0, similarity - reduction)

    @staticmethod
    def sharpen_metrics(
        metrics: Dict[str, float], 
        sharpening_factor: float = 0.3
    ) -> Dict[str, float]:
        """
        Apply sharpening to confidence metrics for response filtering.
        
        Args:
            metrics: Dictionary of confidence metrics (confidence, perplexity, entropy)
            sharpening_factor: Factor to control sharpening strength (0.0-1.0)
            
        Returns:
            Dictionary with sharpened metrics
        """
        # Create a copy to avoid modifying the original
        sharpened = metrics.copy()

        # Sharpen confidence score
        if 'confidence' in metrics:
            # Apply non-linear sharpening to confidence
            conf = metrics['confidence']
            # Favor high confidence, penalize low confidence
            if conf > 0.6:
                # Boost high confidence
                sharpened['confidence'] = min(1.0, conf + (conf - 0.6) * (sharpening_factor - 1.0))
            else:
                # Reduce low confidence
                sharpened['confidence'] = conf * (0.7 + (conf * 0.3))

        # For entropy and perplexity (where lower is better), apply inverse sharpening
        if 'entropy' in metrics:
            # Get normalized entropy (0-1 range where 0 is best)
            norm_entropy = min(1.0, metrics['entropy'] / 3.0)  # 3.0 as reasonable max entropy
            # Apply sharpening
            sharpened_norm = norm_entropy ** sharpening_factor
            # Convert back to original scale
            sharpened['entropy'] = sharpened_norm * 3.0

        if 'perplexity' in metrics:
            # Similar approach for perplexity
            norm_perp = min(1.0, metrics['perplexity'] / 15.0)  # 15.0 as reasonable max perplexity
            sharpened_norm = norm_perp ** sharpening_factor
            sharpened['perplexity'] = sharpened_norm * 15.0

        return sharpened

    @staticmethod
    def sharpen_list(
        values: List[float], 
        threshold: float, 
        higher_is_better: bool = True,
        sharpening_factor: float = 0.3
    ) -> List[float]:
        """
        Apply sharpening to a list of values.
        
        Args:
            values: List of metric values
            threshold: Reference threshold for good/bad values
            higher_is_better: Whether higher values are better
            sharpening_factor: Factor to control sharpening strength (0.0-1.0)
            
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
                    sharp_val = min(1.0, val + (val - threshold) * (sharpening_factor - 1.0))
                else:
                    # Below threshold - make it lower
                    ratio = val / threshold
                    sharp_val = val * (0.8 + (ratio * 0.2))
            else:
                # For metrics where lower is better (like entropy/perplexity)
                if val < threshold:
                    # Below threshold (good) - make it even lower
                    ratio = val / threshold
                    sharp_val = val * (ratio ** (sharpening_factor - 1.0))
                else:
                    # Above threshold (bad) - make it higher to create contrast
                    excess = val - threshold
                    sharp_val = threshold + (excess * sharpening_factor)

            sharpened.append(sharp_val)

        return sharpened