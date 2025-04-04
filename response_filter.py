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
        confidence_threshold: float = 0.55,  # Lower this from 0.65
        entropy_threshold: float = 2.0,      # Lower this from 2.5
        perplexity_threshold: float = 10.0,  # Lower this from 15.0
        fallback_messages: Optional[List[str]] = None,
        continuation_phrases: Optional[List[str]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        sharpening_factor: float = 0.3
    ):
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.perplexity_threshold = perplexity_threshold
        self.user_context = user_context or {}
        self.sharpening_factor = sharpening_factor

        # Default fallback messages when low confidence is detected
        self.fallback_messages = fallback_messages or [
            "... I don't know man, but I've heard some theories about that, if you want me to continue just say 'please continue' or 'speculate anyway'"
            # "... I don't know enough about this topic to provide a reliable answer. If you'd like me to speculate anyway, please let me know by saying 'please continue' or 'speculate anyway'.",
            # "... I'm not confident in my knowledge about this. I might make mistakes if I answer. If you still want me to try, please say 'continue anyway' or 'please speculate'.",
            # "... I don't have enough information to answer this accurately. I could try to speculate if you reply with 'please continue' or 'give it your best guess'.",
            # "... I'm not familiar enough with this topic to give a reliable response. If you'd like me to attempt an answer despite my uncertainty, please say 'go ahead anyway'.",
            # "... I should acknowledge that I don't have sufficient knowledge on this subject. If you'd still like me to provide my best attempt, please reply with 'please continue'."
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
            "proceed anyway"
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

    def sharpen_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Apply sharpening to confidence metrics.

        Args:
            metrics: Dictionary of confidence metrics

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
                sharpened['confidence'] = min(1.0, conf + (conf - 0.6) * (self.sharpening_factor - 1.0))
            else:
                # Reduce low confidence
                sharpened['confidence'] = conf * (0.7 + (conf * 0.3))

        # For entropy and perplexity (where lower is better), apply inverse sharpening
        if 'entropy' in metrics:
            # Get normalized entropy (0-1 range where 0 is best)
            norm_entropy = min(1.0, metrics['entropy'] / self.entropy_threshold)
            # Apply sharpening
            sharpened_norm = norm_entropy ** self.sharpening_factor
            # Convert back to original scale
            sharpened['entropy'] = sharpened_norm * self.entropy_threshold

        if 'perplexity' in metrics:
            # Similar approach for perplexity
            norm_perp = min(1.0, metrics['perplexity'] / self.perplexity_threshold)
            sharpened_norm = norm_perp ** self.sharpening_factor
            sharpened['perplexity'] = sharpened_norm * self.perplexity_threshold

        return sharpened

    def should_filter(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        Determine if a response should be filtered based on sharpened confidence metrics.

        Args:
            metrics: Dictionary containing confidence, perplexity, and entropy values

        Returns:
            Tuple of (should_filter, reason)
        """
        # if metrics["confidence"] < self.confidence_threshold:
        #     return True, "low_confidence"

        # if metrics["entropy"] > self.entropy_threshold:
        #     return True, "high_entropy"

        # if metrics["perplexity"] > self.perplexity_threshold:
        #     return True, "high_perplexity"

        sharpened_metrics = self.sharpen_metrics(metrics)

        # Use sharpened metrics for filtering decision
        if (sharpened_metrics["confidence"] < self.confidence_threshold and
            sharpened_metrics["entropy"] > self.entropy_threshold and
            sharpened_metrics["perplexity"] > self.perplexity_threshold):
            return True, "low_confidence"

        if (metrics["confidence"] < self.confidence_threshold and metrics["entropy"] > self.entropy_threshold and metrics["perplexity"] > self.perplexity_threshold):
            return True, "low_confidence"

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
        should_filter, reason = self.should_filter(metrics)
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
        should_filter, reason = self.should_filter(metrics)

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
        should_filter, reason = self.should_filter(metrics)
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