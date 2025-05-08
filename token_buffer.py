import re
import collections
import itertools

class TokenBuffer:
    """
    Efficient token buffer for streaming generation.
    Avoids expensive string concatenation by using a token list.
    """

    def __init__(self, tokenizer=None):
        """
        Initialize the token buffer.

        Args:
            tokenizer: Optional tokenizer for more advanced operations
        """
        self.tokens = []
        self.tokenizer = tokenizer
        self.last_read_position = 0

        # FIXED: Don't perform internal pattern detection
        # Just track tokens efficiently

    def add_token(self, token: str) -> None:
        """
        Add a token to the buffer.

        Args:
            token: Token to add
        """
        self.tokens.append(token)

    def get_text(self) -> str:
        """
        Get the full text.

        Returns:
            Concatenated text
        """
        return "".join(self.tokens)

    def get_new_text(self) -> str:
        """
        Get only text added since last read.

        Returns:
            New text segment
        """
        if self.last_read_position >= len(self.tokens):
            return ""

        new_segment = "".join(self.tokens[self.last_read_position:])
        self.last_read_position = len(self.tokens)
        return new_segment

    def contains(self, text: str) -> bool:
        """
        Check if buffer contains specific text.

        Args:
            text: Text to check for

        Returns:
            True if text is found in buffer
        """
        return text in self.get_text()

    def clear(self) -> None:
        """Clear all tokens from the buffer."""
        self.tokens = []
        self.last_read_position = 0
    
    def __len__(self) -> int:
        """Get the number of tokens in the buffer."""
        return len(self.tokens)