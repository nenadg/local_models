"""
Enhanced RollingWindowManager with continuation tracking capabilities.
Handles context window management and enables precise continuations.
"""

import re
from typing import List, Dict, Any, Tuple, Optional, Set

class ContinuationTrackingWindowManager:
    """
    Enhanced window manager that tracks generated content for continuations.
    Builds on the RollingWindowManager to add continuation capabilities.
    """

    def __init__(self, tokenizer, max_window_size: int = 2048, memory_manager=None,
                 safety_margin: int = 50, continuation_buffer_size: int = 200):
        """
        Initialize the continuation-aware window manager.

        Args:
            tokenizer: The model tokenizer for token counting
            max_window_size: Maximum allowed tokens in context window
            memory_manager: Optional memory manager for improved context selection
            safety_margin: Token buffer to prevent exceeding limits
            continuation_buffer_size: Number of tokens to store for continuation
        """
        self.tokenizer = tokenizer
        self.max_window_size = max_window_size
        self.memory_manager = memory_manager
        self.safety_margin = safety_margin
        self.continuation_buffer_size = continuation_buffer_size

        # Tracking for continuations
        self.last_response = ""
        self.last_response_tokens = []
        self.continuation_context = {}  # Maps user_id to continuation context

    def optimize_messages(self, messages: List[Dict[str, str]], max_new_tokens: int = 128) -> List[Dict[str, str]]:
        """
        Optimize message history to fit within token limits.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_new_tokens: Expected generation length

        Returns:
            Optimized list of messages
        """
        # Calculate the available token budget
        available_budget = self.max_window_size - max_new_tokens - self.safety_margin

        # Always keep system message and last user message
        if len(messages) < 2:
            return messages

        system_message = messages[0] if messages[0]['role'] == 'system' else None
        user_message = messages[-1] if messages[-1]['role'] == 'user' else None

        # If no system or user message, return as is
        if not system_message or not user_message:
            return messages

        # Calculate tokens for essential messages
        system_tokens = len(self.tokenizer.encode(system_message['content']))
        user_tokens = len(self.tokenizer.encode(user_message['content']))

        essential_tokens = system_tokens + user_tokens

        # If essential messages alone exceed budget, truncate the user message
        if essential_tokens > available_budget:
            max_user_tokens = available_budget - system_tokens
            truncated_user_content = self._truncate_text(user_message['content'], max_user_tokens)

            # Return minimal context
            return [
                system_message,
                {"role": "user", "content": truncated_user_content}
            ]

        # We have room for conversation history - prioritize recent messages
        history_budget = available_budget - essential_tokens
        optimized_history = self._select_history_messages(messages[1:-1], history_budget)

        # Construct final message list
        result = [system_message] + optimized_history + [user_message]

        return result

    def _select_history_messages(self, history: List[Dict[str, str]], token_budget: int) -> List[Dict[str, str]]:
        """
        Select the most relevant history messages within the token budget.

        Args:
            history: List of history messages (excluding system and current user message)
            token_budget: Maximum tokens available for history

        Returns:
            Optimized history message list
        """
        # If history is empty or budget is zero, return empty list
        if not history or token_budget <= 0:
            return []

        # Keep track of tokens as we add messages
        used_tokens = 0
        selected_messages = []

        # Process messages from newest to oldest
        for msg in reversed(history):
            # Count tokens in this message
            msg_tokens = len(self.tokenizer.encode(msg['content']))

            # If adding this message would exceed budget, skip it
            if used_tokens + msg_tokens > token_budget:
                break

            # Add message and update token count
            selected_messages.insert(0, msg)  # Insert at beginning to maintain order
            used_tokens += msg_tokens

        return selected_messages

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit, attempting to break at sentence boundaries.

        Args:
            text: Text to truncate
            max_tokens: Maximum allowed tokens

        Returns:
            Truncated text
        """
        if not text:
            return ""

        # Try encoding the full text to check token count
        encoded = self.tokenizer.encode(text)

        # If already under limit, return unchanged
        if len(encoded) <= max_tokens:
            return text

        # Need to truncate - first try to find a good breakpoint
        sentences = re.split(r'(?<=[.!?])\s+', text)

        truncated = ""
        for sentence in sentences:
            # Check if adding this sentence would exceed limit
            test_text = truncated + sentence
            test_tokens = len(self.tokenizer.encode(test_text))

            if test_tokens <= max_tokens:
                truncated = test_text
            else:
                break

        # If we couldn't fit even one sentence, do hard truncation
        if not truncated:
            # Decode the first max_tokens tokens
            truncated = self.tokenizer.decode(encoded[:max_tokens])

        return truncated

    def calculate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Calculate the total number of tokens in a message list.

        Args:
            messages: List of message dictionaries

        Returns:
            Total token count
        """
        total = 0
        for msg in messages:
            total += len(self.tokenizer.encode(msg['content']))

        return total

    def track_generated_response(self, response: str, user_id: str = "default_user"):
        """
        Track a generated response for potential continuation.

        Args:
            response: The generated response text
            user_id: User identifier for multi-user systems
        """
        # Store the full response
        self.last_response = response

        # Tokenize the response
        self.last_response_tokens = self.tokenizer.encode(response)

        # Store the last portion for continuation context
        continuation_tokens = self.last_response_tokens[-self.continuation_buffer_size:]
        continuation_text = self.tokenizer.decode(continuation_tokens)

        # Store continuation context for this user
        self.continuation_context[user_id] = {
            "tokens": continuation_tokens,
            "text": continuation_text,
            "timestamp": self._get_timestamp(),
            "full_response": response
        }

    def detect_continuation_request(self, query: str) -> bool:
        """
        Detect if a query is asking for continuation.

        Args:
            query: User query text

        Returns:
            Boolean indicating if this is a continuation request
        """
        # Common phrases that indicate continuation requests
        continuation_phrases = [
            "continue", "go on", "keep going", "proceed", "what happens next",
            "continue from", "pick up where", "write more", "and then?",
            "next part", "next section", "rest of", "finish", "complete"
        ]

        query_lower = query.lower()

        # Check for continuation indicators
        for phrase in continuation_phrases:
            if phrase in query_lower:
                return True

        return False

    def prepare_continuation_prompt(self, query: str, user_id: str = "default_user") -> str:
        """
        Prepare an enhanced query for continuation requests.

        Args:
            query: Original user query
            user_id: User identifier

        Returns:
            Enhanced query with continuation context
        """
        # Check if we have continuation context for this user
        if user_id not in self.continuation_context:
            return query

        continuation_ctx = self.continuation_context[user_id]

        # Extract the last few lines of context for more precise continuation
        last_context = self._extract_last_context(continuation_ctx["text"])

        if not last_context:
            return query

        # Determine continuation type (code or prose)
        is_code = self._is_code_context(last_context)

        # Build a continuation prompt that works for both code and prose
        if is_code:
            # For code, include exact code block and line
            return f"""
{query}

Please continue the code EXACTLY from this point:

```
{last_context}
```

Continue writing from the EXACT point where the code was cut off, completing the current statement or function without restarting.
"""
        else:
            # For prose, provide context but don't use code blocks
            return f"""
{query}

Please continue from this exact point in the previous response:

"{last_context}"

Continue the text naturally from this point, maintaining the style, tone, and flow of the previous response.
"""

    def _extract_last_context(self, text: str, max_lines: int = 10) -> str:
        """
        Extract the last few lines from text for continuation context.

        Args:
            text: The text to extract context from
            max_lines: Maximum number of lines to extract

        Returns:
            Last lines of context
        """
        # Split into lines
        lines = text.split('\n')

        # Get last lines (up to max_lines)
        last_lines = lines[-max_lines:] if len(lines) > max_lines else lines

        # Rejoin into text
        return '\n'.join(last_lines)

    def _is_code_context(self, text: str) -> bool:
        """
        Determine if the context is code rather than prose.

        Args:
            text: Context text to analyze

        Returns:
            Boolean indicating if context is code
        """
        # Code indicators
        code_indicators = [
            # Symbols and patterns common in code
            "{", "}", "(", ")", ";", "==", "=>", "function", "var ", "let ", "const ",
            "if ", "for ", "while ", "class ", "import ", "export ", "return ",
            "public ", "private ", "static ", # Common keywords

            # Common language patterns
            "def ", "async ", "await ", "try ", "catch ", "else ", "elif ",

            # Indentation patterns
            re.compile(r'^\s{2,}[a-zA-Z0-9_]'),  # Indented line starts
            re.compile(r'[{([]$')  # Line ends with opening bracket
        ]

        # Check for code indicators
        for indicator in code_indicators:
            if isinstance(indicator, str) and indicator in text:
                return True
            elif hasattr(indicator, 'search') and indicator.search(text):
                return True

        # Check for code-like line patterns (important for languages like Python)
        lines = text.strip().split('\n')
        indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
        if indented_lines > len(lines) / 3:  # If more than 1/3 of lines are indented
            return True

        return False

    def _get_timestamp(self) -> str:
        """Get current timestamp for tracking."""
        from datetime import datetime
        return datetime.now().isoformat()