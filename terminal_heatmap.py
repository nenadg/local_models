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
            color_scheme: Color scheme to use ("green-to-red", "blue-to-red", etc.)
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
        
        # Define color maps for different schemes
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
    
    def colorize_tokens(self, 
                        text: str, 
                        token_confidences: List[float]) -> str:
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
    
    def colorize_streaming_token(self, 
                               token: str, 
                               confidence: float) -> str:
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


# Example of integration with TinyLlamaChat
def integrate_with_tiny_llama():
    """
    Example of how to integrate the heatmap with TinyLlamaChat.
    
    For the generate_response method:
    1. Record token-by-token confidences during generation
    2. Apply colorization to tokens in streaming mode
    """
    # Pseudo-code for integration
    code = """
    # In the generate_response method:
    
    # Initialize token confidences list
    token_confidences = []
    
    # Initialize heatmap
    heatmap = TerminalHeatmap(self.tokenizer, use_background=False)
    
    # During streaming:
    for token in streamer:
        # Get confidence for this token from confidence_metrics
        token_confidence = ... # Extract from confidence_metrics
        token_confidences.append(token_confidence)
        
        # Colorize and display the token
        colored_token = heatmap.colorize_streaming_token(token, token_confidence)
        print(colored_token, end="", flush=True)
    
    # After generation, store token confidences with the response
    complete_response.token_confidences = token_confidences
    """
    return code

# Example usage
if __name__ == "__main__":
    # Sample text with confidence values
    sample_text = "The capital of France is Paris, but the capital of Atlantis is underwater."
    
    # Sample token probabilities (high for known facts, low for fiction)
    token_confidences = [
        0.95, 0.92, 0.90, 0.94, 0.93,  # "The capital of France is"
        0.96, 0.97, 0.94,              # "Paris,"
        0.85, 0.83, 0.80,              # "but the"
        0.75, 0.72, 0.68, 0.65,        # "capital of Atlantis"
        0.45, 0.35, 0.30, 0.25         # "is underwater."
    ]
    
    # Create heatmap
    heatmap = TerminalHeatmap(use_background=False)
    
    # Print colorized text
    colorized = heatmap.colorize_tokens(sample_text, token_confidences)
    print("\nSample with confidence heatmap:")
    print(colorized)
    
    # Print legend
    heatmap.print_legend()