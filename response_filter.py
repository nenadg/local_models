import torch
import numpy as np
import re
from typing import Dict, Optional, Tuple, List, Union, Any

class ResponseFilter:
    """
    Enhanced filter for LLM responses based on confidence metrics.
    Prevents hallucinations by replacing low-confidence responses
    with uncertainty messages while allowing user override.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.65,  # Lower this from 0.65
        entropy_threshold: float = 2.5,      # Lower this from 2.5
        perplexity_threshold: float = 15.0,  # Lower this from 15.0
        fallback_messages: Optional[List[str]] = None,
        continuation_phrases: Optional[List[str]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        sharpening_factor: float = 0.3,
        question_classifier=None
    ):
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.perplexity_threshold = perplexity_threshold
        self.user_context = user_context or {}
        self.sharpening_factor = sharpening_factor
        self.question_classifier = question_classifier

        # Default fallback messages when low confidence is detected
        self.fallback_messages = fallback_messages or [
            "... I don't know :/.",
            "... I don't know :(.",
            "... I don't know :|.",
            "... I don't know :\\.",
            "... I don't know :[.",
            "... I don't know !|."
       ]

        # Phrases that indicate user wants to continue despite uncertainty
        self.continuation_phrases = continuation_phrases or [
            "please continue",
            "continue anyway",
            "speculate anyway",
            "give it your best guess",
            "go ahead anyway",
            "try anyway",
            "speculate",
            "just guess",
            "make something up",
            "proceed anyway",
            "...",
            ".",
            ""
        ]

    def sharpen_confidence_scores(self, token_probs: List[float]) -> List[float]:
        """
        Sharpen confidence scores by amplifying the difference between high and low values.

        Args:
            token_probs: List of token probabilities

        Returns:
            List of sharpened probabilities
        """
        if not token_probs:
            return []

        # Calculate mean confidence
        mean_prob = sum(token_probs) / len(token_probs)

        # Apply sharpening (similar to contrast enhancement in images)
        sharpened_probs = []
        for prob in token_probs:
            # Compute difference from mean
            diff = prob - mean_prob
            # Apply sharpening factor
            sharpened = mean_prob + (diff * self.sharpening_factor)
            # Clamp to valid range [0, 1]
            sharpened = max(0.0, min(1.0, sharpened))
            sharpened_probs.append(sharpened)

        return sharpened_probs

    def sharpen_metrics(self, metrics: Dict[str, float], sharpening_factor: float = 0.3) -> Dict[str, float]:
        """
        Apply sharpening to a dictionary of confidence metrics.

        Args:
            metrics: Dictionary of metrics (confidence, perplexity, entropy)
            sharpening_factor: Strength of sharpening effect

        Returns:
            Dictionary with sharpened metrics
        """
        # Skip if no sharpening requested
        if sharpening_factor <= 0:
            return metrics

        # Get key metrics
        confidence = metrics.get('confidence', 0.5)
        perplexity = metrics.get('perplexity', 1.0)
        entropy = metrics.get('entropy', 0.0)

        # Apply sharpening to each metric
        sharpened_metrics = metrics.copy()

        # Sharpen confidence (higher is better)
        if confidence > 0.5:
            # Boost high confidence
            boost = (confidence - 0.5) * sharpening_factor
            sharpened_metrics['confidence'] = min(1.0, confidence + boost)
        else:
            # Reduce low confidence
            reduction = (0.5 - confidence) * sharpening_factor
            sharpened_metrics['confidence'] = max(0.1, confidence - reduction)

        # Sharpen perplexity (lower is better)
        if perplexity < 5.0:
            # Decrease low perplexity (good)
            reduction = perplexity * sharpening_factor * 0.2
            sharpened_metrics['perplexity'] = max(1.0, perplexity - reduction)
        else:
            # Increase high perplexity (bad)
            boost = (perplexity - 5.0) * sharpening_factor * 0.2
            sharpened_metrics['perplexity'] = perplexity + boost

        # Sharpen entropy (lower is better)
        if entropy < 1.0:
            # Decrease low entropy (good)
            reduction = entropy * sharpening_factor * 0.3
            sharpened_metrics['entropy'] = max(0.0, entropy - reduction)
        else:
            # Increase high entropy (bad)
            boost = (entropy - 1.0) * sharpening_factor * 0.3
            sharpened_metrics['entropy'] = entropy + boost

        return sharpened_metrics

    def should_filter(self, metrics: Dict[str, float], query: Optional[str] = None) -> Tuple[bool, str]:
        """
        Determine if a response should be filtered based on metrics and domain.

        Args:
            metrics: Dictionary containing confidence, perplexity, and entropy values
            query: The user's query (for domain detection)

        Returns:
            Tuple of (should_filter, reason)
        """
        # Ensure metrics contains basic values
        confidence = float(metrics.get("confidence", 0.5))
        entropy = float(metrics.get("entropy", 2.0))
        perplexity = float(metrics.get("perplexity", 10.0))

        # Set default thresholds
        domain_confidence_threshold = self.confidence_threshold
        domain_entropy_threshold = self.entropy_threshold
        domain_perplexity_threshold = self.perplexity_threshold

        # Apply domain-specific thresholds if possible
        domain = None
        if query and hasattr(self, 'question_classifier') and self.question_classifier:
            try:
                # Get domain and adjust thresholds
                domain_settings = self.question_classifier.get_domain_settings(query)
                domain = domain_settings.get('domain', 'unknown')

                # Apply domain-specific thresholds
                if domain == 'arithmetic' or domain == 'factual':
                    # Stricter thresholds for factual/math queries
                    domain_confidence_threshold = 0.75   # Higher threshold = 0.75
                    domain_entropy_threshold = 1.8      # Lower threshold = 1.8
                    domain_perplexity_threshold = 8.0   # Lower threshold = 8.0
                elif domain == 'translation':
                    # Stricter thresholds for translations
                    domain_confidence_threshold = 0.4
                    domain_entropy_threshold = 3.9
                    domain_perplexity_threshold = 12.0
            except Exception as e:
                print(f"Error detecting domain: {e}")
                # Continue with default thresholds if domain detection fails

        # Apply sharpening if available
        if hasattr(self, 'sharpen_metrics'):
            sharpened_metrics = self.sharpen_metrics(metrics)
        else:
            # Fallback to original metrics
            sharpened_metrics = metrics.copy()

        # Convert any numpy values to Python primitives to avoid array truth value errors
        for key in sharpened_metrics:
            if hasattr(sharpened_metrics[key], 'item'):
                sharpened_metrics[key] = sharpened_metrics[key].item()

        # Calculate a combined uncertainty score
        uncertainty_score = (
            (1 - float(sharpened_metrics.get("confidence", 0.5))) * 0.5 +     # Weight confidence more
            (float(sharpened_metrics.get("entropy", 2.0)) / float(domain_entropy_threshold)) * 0.3 +
            (float(sharpened_metrics.get("perplexity", 10.0)) / float(domain_perplexity_threshold)) * 0.2
        )

        # Check the confidence value explicitly
        sharpened_confidence = float(sharpened_metrics.get("confidence", 0.5))
        if sharpened_confidence < domain_confidence_threshold:
            return True, "low_confidence"

        # Check entropy value explicitly
        sharpened_entropy = float(sharpened_metrics.get("entropy", 2.0))
        if sharpened_entropy > domain_entropy_threshold:
            return True, "high_entropy"

        # Check perplexity value explicitly
        sharpened_perplexity = float(sharpened_metrics.get("perplexity", 10.0))
        if sharpened_perplexity > domain_perplexity_threshold:
            return True, "high_perplexity"

        # High uncertainty score indicates problem
        if uncertainty_score > 0.65:
            return True, "high_uncertainty"

        # If we get here, the response passes all checks
        return False, "acceptable"

    def check_override_instruction(self, query: str) -> bool:
        """
        Check if the user's query contains an instruction to continue
        despite uncertainty.

        Args:
            query: The user's input query

        Returns:
            Boolean indicating whether to override the filter
        """
        if not query:
            return False

        query_lower = query.lower()

        # Check for continuation phrases
        for phrase in self.continuation_phrases:
            if phrase in query_lower:
                return True

        return False

    def update_user_context(self, query: str, response: str, metrics: Dict[str, float]):
        """
        Update user context with information about the query and response.

        Args:
            query: The user's input query
            response: The model's response
            metrics: Confidence metrics for the response
        """
        # Track the last uncertain query
        should_filter, reason = self.should_filter(metrics, query)
        if should_filter:
            self.user_context["last_uncertain_query"] = query
            self.user_context["last_uncertain_reason"] = reason
        else:
            # Clear if we have a confident response
            self.user_context.pop("last_uncertain_query", None)
            self.user_context.pop("last_uncertain_reason", None)

    def filter_response(
        self,
        response: str,
        metrics: Dict[str, float],
        query: Optional[str] = None,
        preserve_mcp: bool = True,
        allow_override: bool = True
    ) -> str:
        """
        Filter a response based on confidence metrics, with user override option.

        Args:
            response: Original model response
            metrics: Dictionary containing confidence, perplexity, and entropy values
            query: Original query (optional, for contextual fallbacks)
            preserve_mcp: Whether to preserve MCP commands from the original response
            allow_override: Whether to allow the user to override filtering

        Returns:
            Filtered response or original response if confidence is high enough
        """
        should_filter, reason = self.should_filter(metrics, query)

        # Skip filtering if metrics are good
        if not should_filter:
            return response

        # Check for override instruction
        if allow_override and query and self.check_override_instruction(query):
            # User has explicitly requested to continue despite uncertainty
            # Add a brief uncertainty disclaimer
            disclaimer = "Note: I'm not entirely confident about this information, but as requested, I'll provide my best attempt:\n\n"
            return disclaimer + response

        # Check if this is a follow-up to a previously uncertain query
        if "last_uncertain_query" in self.user_context and allow_override and query:
            if self.check_override_instruction(query):
                # This is an override for the previous uncertain query
                disclaimer = "As requested, I'll try to answer despite my uncertainty:\n\n"
                return disclaimer + response

        # If we get here, we need to filter the response

        # Check for unusual repetition patterns
        repetition_check = self._check_for_repetition(response)
        if repetition_check.get('excessive_repetition', False):
            # Use a stronger filtering threshold for repetitive content
            if metrics['confidence'] < self.confidence_threshold * 1.2:
                should_filter = True
                reason = "repetitive_content"

        # Extract any MCP commands if needed
        mcp_commands = []
        if preserve_mcp:
            # Simple regex-free extraction of MCP commands for preservation
            lines = response.split('\n')
            filtered_lines = []
            in_mcp_block = False

            for line in lines:
                if ">>>" in line and "FILE:" in line:
                    in_mcp_block = True
                    mcp_commands.append(line)
                elif "<<<" in line and in_mcp_block:
                    in_mcp_block = False
                    mcp_commands.append(line)
                elif in_mcp_block:
                    mcp_commands.append(line)
                else:
                    filtered_lines.append(line)

        # Choose a fallback message
        import random
        fallback = random.choice(self.fallback_messages)

        # Add the query topic to the fallback message if available
        if query:
            topic_match = re.search(r'(?:about|on|regarding)\s+(["\']?[\w\s]+["\']?)', query)
            if topic_match:
                topic = topic_match.group(1)
                fallback = fallback.replace("this topic", f"'{topic}'")
                fallback = fallback.replace("this subject", f"'{topic}'")

        # Add metrics explanation in debug mode
        debug_info = f"\n\nResponse filtered due to {reason}: confidence={metrics['confidence']:.2f}, " \
                    f"entropy={metrics['entropy']:.2f}, perplexity={metrics['perplexity']:.2f}"

        # Combine fallback with preserved MCP commands
        if preserve_mcp and mcp_commands:
            return fallback + "\n\n" + "\n".join(mcp_commands)

        return fallback


    def _check_for_repetition(self, text: str) -> Dict[str, Any]:
        """Check for unusual repetition patterns in text."""
        # Split into lines or items
        lines = text.split('\n')

        # Count repeats of the same content
        content_counts = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove numbering like "1. ", "2. "
            clean_line = re.sub(r'^\d+\.\s+', '', line)
            if clean_line in content_counts:
                content_counts[clean_line] += 1
            else:
                content_counts[clean_line] = 1

        # Check for excessive repetition
        max_repeats = max(content_counts.values()) if content_counts else 0
        excessive_repetition = max_repeats > 3

        return {
            'excessive_repetition': excessive_repetition,
            'max_repeats': max_repeats,
            'repeat_items': [item for item, count in content_counts.items() if count > 1]
        }

    def set_thresholds(
        self,
        confidence: Optional[float] = None,
        entropy: Optional[float] = None,
        perplexity: Optional[float] = None
    ) -> None:
        """Update the thresholds used for filtering."""
        if confidence is not None:
            self.confidence_threshold = confidence
        if entropy is not None:
            self.entropy_threshold = entropy
        if perplexity is not None:
            self.perplexity_threshold = perplexity

    def add_continuation_phrase(self, phrase: str) -> None:
        """Add a new continuation phrase to the list."""
        if phrase.lower() not in [p.lower() for p in self.continuation_phrases]:
            self.continuation_phrases.append(phrase.lower())

    def should_stream_fallback(self, metrics: Dict[str, float], query: str) -> bool:
        """
        Determine if we should stream a fallback message based on metrics.

        Args:
            metrics: Dictionary containing confidence metrics
            query: User query for context

        Returns:
            Boolean indicating whether to stream a fallback
        """
        # Check if filtering should occur
        should_filter, reason = self.should_filter(metrics, query)
        if not should_filter:
            return False

        # Check for override instruction
        if query and self.check_override_instruction(query):
            return False

        # Check if this is a follow-up with override
        if "last_uncertain_query" in self.user_context and query:
            if self.check_override_instruction(query):
                return False

        # If we reach here, we should stream a fallback
        return True

    def get_streamable_fallback(self, query: Optional[str] = None) -> str:
        """
        Get a fallback message that can be streamed.

        Args:
            query: Original query for context

        Returns:
            Fallback message as a string
        """
        # Choose a fallback message
        import random
        fallback = random.choice(self.fallback_messages)

        # Add the query topic to the fallback message if available
        if query:
            import re
            topic_match = re.search(r'(?:about|on|regarding)\s+(["\']?[\w\s]+["\']?)', query)
            if topic_match:
                topic = topic_match.group(1)
                fallback = fallback.replace("this topic", f"'{topic}'")
                fallback = fallback.replace("this subject", f"'{topic}'")

        return fallback