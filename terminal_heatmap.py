import torch
import numpy as np
from typing import List, Dict, Tuple, Optional

class TerminalHeatmap:
    """
    Provides colorized terminal output for displaying confidence levels
    in generated text. Each token's color reflects its confidence score.
    """
    
    def __init__(self, 
                tokenizer=None, 
                use_background=False, 
                color_scheme="sepia-red"):
        """
        Initialize the terminal heatmap.
        
        Args:
            tokenizer: The tokenizer used to tokenize the text
            use_background: Whether to color the background (True) or foreground (False)
            color_scheme: Color scheme to use ("sepia-red", "green-to-red", "blue-to-red")
        """
        self.tokenizer = tokenizer
        self.use_background = use_background
        self.color_scheme = color_scheme

        # ANSI color codes
        self.RESET = "\033[0m"

        # Foreground colors (text)
        self.FG_BLACK = "\033[30m"
        self.FG_RED = "\033[31m"
        self.FG_GREEN = "\033[32m"
        self.FG_YELLOW = "\033[33m"
        self.FG_BLUE = "\033[34m"
        self.FG_MAGENTA = "\033[35m"
        self.FG_CYAN = "\033[36m"
        self.FG_WHITE = "\033[37m"

        # Background colors
        self.BG_BLACK = "\033[40m"
        self.BG_RED = "\033[41m"
        self.BG_GREEN = "\033[42m"
        self.BG_YELLOW = "\033[43m"
        self.BG_BLUE = "\033[44m"
        self.BG_MAGENTA = "\033[45m"
        self.BG_CYAN = "\033[46m"
        self.BG_WHITE = "\033[47m"

        # Initialize color palettes for different schemes
        self._initialize_color_palette()

        # Window for smoothing confidence values (only used by EnhancedHeatmap)
        self.confidence_window = []

    def _initialize_color_palette(self):
        """Initialize color palette based on selected scheme"""
        if self.color_scheme == "sepia-red":
            # Sepia-to-red color scheme (warm tones)
            self.fg_colors = [
                "\033[38;5;223m",  # Light sepia (high confidence)
                "\033[38;5;216m",  # Light tan
                "\033[38;5;173m",  # Medium tan
                "\033[38;5;166m",  # Dark orange
                "\033[38;5;160m",  # Light red
                "\033[38;5;124m"   # Dark red (low confidence)
            ]
            self.bg_colors = [
                "\033[48;5;223m",  # Light sepia
                "\033[48;5;216m",  # Light tan
                "\033[48;5;173m",  # Medium tan
                "\033[48;5;166m",  # Dark orange
                "\033[48;5;160m",  # Light red
                "\033[48;5;124m"   # Dark red
            ]
        elif self.color_scheme == "green-to-red":
            self.fg_colors = [
                self.FG_GREEN,      # High confidence (1.0-0.8)
                self.FG_CYAN,       # Good confidence (0.8-0.7)
                self.FG_BLUE,       # Moderate confidence (0.7-0.6)
                self.FG_YELLOW,     # Low confidence (0.6-0.5)
                self.FG_MAGENTA,    # Very low confidence (0.5-0.4)
                self.FG_RED         # Extremely low confidence (<0.4)
            ]
            self.bg_colors = [
                self.BG_GREEN,
                self.BG_CYAN,
                self.BG_BLUE,
                self.BG_YELLOW,
                self.BG_MAGENTA,
                self.BG_RED
            ]
        elif self.color_scheme == "blue-to-red":
            # Define alternative color scheme
            self.fg_colors = [
                self.FG_BLUE,
                self.FG_CYAN,
                self.FG_GREEN,
                self.FG_YELLOW,
                self.FG_MAGENTA,
                self.FG_RED
            ]
            self.bg_colors = [
                self.BG_BLUE,
                self.BG_CYAN,
                self.BG_GREEN,
                self.BG_YELLOW,
                self.BG_MAGENTA,
                self.BG_RED
            ]

    def _get_color_for_confidence(self, confidence: float) -> str:
        """
        Get the appropriate color code for a confidence value.

        Args:
            confidence: Confidence score (0-1)

        Returns:
            ANSI color code
        """
        # Define thresholds for different colors
        thresholds = [0.8, 0.7, 0.6, 0.5, 0.4, 0.0]

        # Select the appropriate color based on confidence
        for i, threshold in enumerate(thresholds):
            if confidence >= threshold:
                if self.use_background:
                    return self.bg_colors[i] + self.FG_BLACK  # Black text on colored background
                else:
                    return self.fg_colors[i]  # Colored text on default background

        # Fallback
        return self.RESET

    def colorize_tokens(self, text: str, token_confidences: List[float]) -> str:
        """
        Colorize text based on token confidence values.

        Args:
            text: The text to colorize
            token_confidences: List of confidence values for each token

        Returns:
            Colorized text with ANSI escape codes
        """
        if not self.tokenizer:
            # If no tokenizer, treat each character as a token (simplified)
            result = ""
            for i, char in enumerate(text):
                # Get confidence for this position (or use last if out of range)
                conf_idx = min(i, len(token_confidences) - 1)
                if conf_idx < 0:
                    conf = 1.0  # Default high confidence if no values
                else:
                    conf = token_confidences[conf_idx]

                # Apply color based on confidence
                color = self._get_color_for_confidence(conf)
                result += color + char + self.RESET
            return result
        else:
            # Tokenize the text
            tokens = self.tokenizer.encode(text)
            token_texts = [self.tokenizer.decode([token]) for token in tokens]

            # Colorize each token
            result = ""
            for i, token_text in enumerate(token_texts):
                # Get confidence for this token (or use last if out of range)
                conf_idx = min(i, len(token_confidences) - 1)
                if conf_idx < 0:
                    conf = 1.0  # Default high confidence if no values
                else:
                    conf = token_confidences[conf_idx]

                # Apply color based on confidence
                color = self._get_color_for_confidence(conf)
                result += color + token_text + self.RESET
            return result

    def colorize_streaming_token(self, token: str, confidence: float) -> str:
        """
        Colorize a single token during streaming generation.

        Args:
            token: The token to colorize
            confidence: Confidence value for this token

        Returns:
            Colorized token with ANSI escape codes
        """
        color = self._get_color_for_confidence(confidence)
        return color + token + self.RESET

    def print_legend(self) -> None:
        """Print a legend showing what each color represents."""
        print("\nConfidence Heatmap Legend:")

        thresholds = [
            ("Very High", 0.8, 1.0),
            ("High", 0.7, 0.8),
            ("Moderate", 0.6, 0.7),
            ("Low", 0.5, 0.6),
            ("Very Low", 0.4, 0.5),
            ("Extremely Low", 0.0, 0.4)
        ]

        for i, (label, lower, upper) in enumerate(thresholds):
            if self.use_background:
                color = self.bg_colors[i] + self.FG_BLACK
            else:
                color = self.fg_colors[i]

            print(f"{color}█████{self.RESET} {label} ({lower:.1f}-{upper:.1f})")


class EnhancedHeatmap(TerminalHeatmap):
    """
    Enhanced Terminal Heatmap that uses geometric mean normalization for
    smoothing confidence transitions between words, creating a more natural
    gradient effect across the text.
    """

    def __init__(self,
                tokenizer=None,
                use_background=False,
                color_scheme="sepia-red",
                window_size=3):  # Window size for rolling geometric mean
        """
        Initialize the enhanced heatmap.

        Args:
            tokenizer: The tokenizer used to tokenize the text
            use_background: Whether to color the background (True) or foreground (False)
            color_scheme: Color scheme to use
            window_size: Size of the sliding window for geometric mean calculation
        """
        # Initialize the parent class
        super().__init__(tokenizer, use_background, color_scheme)

        # Store additional parameters
        self.window_size = window_size

        # Expand the color palette for more gradual transitions
        self._expand_color_palette()

    def _expand_color_palette(self):
        """Expand the color palette for more gradual transitions"""
        if self.color_scheme == "sepia-red":
            # More gradual sepia-to-red color scheme
            self.fg_colors = [
                "\033[38;5;223m",  # Lightest sepia (highest confidence)
                "\033[38;5;222m",
                "\033[38;5;221m",
                "\033[38;5;220m",
                "\033[38;5;215m",
                "\033[38;5;214m",
                "\033[38;5;208m",
                "\033[38;5;202m",
                "\033[38;5;196m",
                "\033[38;5;160m"   # Darkest red (lowest confidence)
            ]
            self.bg_colors = [
                "\033[48;5;223m",  # Lightest sepia
                "\033[48;5;222m",
                "\033[48;5;221m",
                "\033[48;5;220m",
                "\033[48;5;215m",
                "\033[48;5;214m",
                "\033[48;5;208m",
                "\033[48;5;202m",
                "\033[48;5;196m",
                "\033[48;5;160m"   # Darkest red
            ]
        elif self.color_scheme == "blue-to-red":
            # More gradual blue-to-red scheme
            self.fg_colors = [
                "\033[38;5;39m",   # Light blue
                "\033[38;5;38m",
                "\033[38;5;37m",
                "\033[38;5;36m",
                "\033[38;5;35m",
                "\033[38;5;41m",
                "\033[38;5;184m",  # Yellow/green transitional
                "\033[38;5;220m",  # Yellow
                "\033[38;5;208m",  # Orange
                "\033[38;5;196m"   # Red
            ]
            self.bg_colors = [
                "\033[48;5;39m",   # Light blue
                "\033[48;5;38m",
                "\033[48;5;37m",
                "\033[48;5;36m",
                "\033[48;5;35m",
                "\033[48;5;41m",
                "\033[48;5;184m",  # Yellow/green transitional
                "\033[48;5;220m",  # Yellow
                "\033[48;5;208m",  # Orange
                "\033[48;5;196m"   # Red
            ]

    def _geometric_mean(self, values):
        """
        Calculate the geometric mean of a list of values.

        Args:
            values: List of confidence values

        Returns:
            Geometric mean
        """
        if not values:
            return 0.0

        # Avoid values too close to zero
        safe_values = [max(0.01, v) for v in values]

        # Calculate geometric mean using numpy
        return np.power(np.prod(safe_values), 1.0 / len(safe_values))

    def _get_color_for_normalized_confidence(self, confidence):
        """
        Get the appropriate color code from expanded palette based on confidence.

        Args:
            confidence: Confidence score (0-1)

        Returns:
            ANSI color code
        """
        # Define evenly spaced thresholds for expanded colors
        thresholds = np.linspace(0, 1, len(self.fg_colors) + 1)[1:][::-1]

        # Select the appropriate color based on confidence
        for i, threshold in enumerate(thresholds):
            if confidence >= threshold:
                if self.use_background:
                    return self.bg_colors[i] + self.FG_BLACK
                else:
                    return self.fg_colors[i]

        # Fallback to lowest confidence color
        if self.use_background:
            return self.bg_colors[-1] + self.FG_BLACK
        else:
            return self.fg_colors[-1]

    def add_confidence(self, confidence):
        """
        Add a confidence value to the sliding window.

        Args:
            confidence: New confidence value
        """
        self.confidence_window.append(confidence)

        # Keep window at specified size
        if len(self.confidence_window) > self.window_size:
            self.confidence_window.pop(0)

    def get_normalized_confidence(self):
        """
        Calculate normalized confidence using geometric mean of window.

        Returns:
            Normalized confidence value
        """
        if not self.confidence_window:
            return 0.8  # Default high confidence if no values

        return self._geometric_mean(self.confidence_window)

    def colorize_streaming_token(self, token, confidence):
        """
        Colorize a single token during streaming generation using normalized confidence.

        Args:
            token: The token to colorize
            confidence: Raw confidence value for this token

        Returns:
            Colorized token with ANSI escape codes
        """
        # Add confidence to window
        self.add_confidence(confidence)

        # Get normalized confidence
        normalized_confidence = self.get_normalized_confidence()

        # Get color based on normalized confidence
        color = self._get_color_for_normalized_confidence(normalized_confidence)

        # Apply color to token
        return color + token + self.RESET

    def colorize_tokens(self, text, token_confidences):
        """
        Colorize text based on token confidence values with smoothing.

        Args:
            text: The text to colorize
            token_confidences: List of confidence values for each token

        Returns:
            Colorized text with ANSI escape codes
        """
        # Reset window
        self.confidence_window = []

        if not self.tokenizer:
            # If no tokenizer, treat each word or character as a token
            words = text.split()
            result = ""

            for i, word in enumerate(words):
                # Get confidence for this word
                conf_idx = min(i, len(token_confidences) - 1)
                if conf_idx < 0:
                    conf = 0.8  # Default if no confidence available
                else:
                    conf = token_confidences[conf_idx]

                # Add to window and get normalized confidence
                self.add_confidence(conf)
                normalized_confidence = self.get_normalized_confidence()

                # Apply color based on normalized confidence
                color = self._get_color_for_normalized_confidence(normalized_confidence)
                result += color + word + self.RESET + " "

            return result.rstrip()
        else:
            # With tokenizer, we can be more precise
            tokens = self.tokenizer.encode(text)
            token_texts = [self.tokenizer.decode([token]) for token in tokens]

            result = ""
            for i, token_text in enumerate(token_texts):
                # Get confidence for this token
                conf_idx = min(i, len(token_confidences) - 1)
                if conf_idx < 0:
                    conf = 0.8  # Default if no confidence available
                else:
                    conf = token_confidences[conf_idx]

                # Add to window and get normalized confidence
                self.add_confidence(conf)
                normalized_confidence = self.get_normalized_confidence()

                # Apply color based on normalized confidence
                color = self._get_color_for_normalized_confidence(normalized_confidence)
                result += color + token_text + self.RESET

            return result

    def print_legend(self):
        """Print a legend showing the expanded color gradient."""
        print("\nConfidence Heatmap Legend (Normalized using Geometric Mean):")

        # Create a gradient legend
        gradient = ""
        for color in self.fg_colors:
            gradient += color + "█" + self.RESET

        print(f"{gradient} High confidence → Low confidence")

        # Show thresholds for reference
        thresholds = np.linspace(0, 1, len(self.fg_colors) + 1)[1:][::-1]
        lowest = f"{thresholds[-1]:.2f}"
        highest = f"{thresholds[0]:.2f}"

        print(f"Window size for geometric mean: {self.window_size} tokens")
        print(f"Confidence range: {lowest} → {highest}")