"""
Enhanced Continuation Manager with improved content type detection.
Provides seamless continuations for code, prose, and poetry.
"""

import re
from typing import List, Dict, Any, Tuple, Optional, Set

class ContinuationTrackingWindowManager:
    """
    Enhanced window manager that tracks generated content for continuations.
    Handles code, prose, and poetry with specialized continuations.
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

    def track_repetition_patterns(self, text: str, user_id: str = "default_user") -> bool:
        """Track and detect repetitive patterns with code-aware exceptions"""
        repetition_detected = False
        pattern_length = 20
        min_repetitions = 3

        # Skip detection for specific patterns common in code
        if self._detect_content_type(text) == "code":
            # Skip detection inside array literals
            if re.search(r'\[\s*\[\s*\d+\s*,', text[-50:]):
                return False

            # Skip detection in JSON objects with repeating structures
            if re.search(r'[{,]\s*[\'"]\w+[\'"]\s*:', text[-50:]):
                return False

        # Only check after accumulating enough tokens
        if len(text) > pattern_length * min_repetitions:
            # Standard repetition detection logic
            check_chunk = text[-pattern_length * min_repetitions:]

            for i in range(5, pattern_length + 1):
                pattern = check_chunk[-i:]
                if (check_chunk[-i*2:-i] == pattern and
                    check_chunk[-i*3:-i*2] == pattern):
                    repetition_detected = True
                    break

        # Store repetition status
        if user_id not in self.continuation_context:
            self.continuation_context[user_id] = {}

        self.continuation_context[user_id]["repetition_detected"] = repetition_detected

        return repetition_detected

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

    def is_generation_complete(self, user_id: str, current_response: str) -> bool:
        """
        Check if generation has reached completion based on target length or repetition.

        Args:
            user_id: User identifier
            current_response: The current response text

        Returns:
            Boolean indicating if generation should be terminated
        """
        # Get target length if set
        target_length = self.continuation_context.get(user_id, {}).get("target_length")

        # Count words in current response
        word_count = len(current_response.split())

        # Check if we've reached or exceeded target
        if target_length and word_count >= target_length:
            return True

        # Check for empty line padding (5+ consecutive newlines)
        newline_count = 0
        for char in current_response[-20:]:  # Check last 20 chars
            if char == '\n':
                newline_count += 1
            elif char.strip():  # Reset on non-whitespace
                newline_count = 0

        if newline_count >= 5:
            return True

        # Check for repetition patterns in code
        # This helps with the repeating const declarations seen in the log
        lines = current_response.split('\n')
        if len(lines) > 10:
            # Get last 10 lines
            last_lines = lines[-10:]
            # Create a set of stripped lines
            unique_lines = set(line.strip() for line in last_lines if line.strip())
            # If we have fewer than 3 unique lines in the last 10, we're likely in a loop
            if 0 < len(unique_lines) < 3 and len(last_lines) > 5:
                return True

        return False

    def _find_safe_termination_point(self, text: str) -> int:
        """Find a good position to terminate text (after punctuation)"""
        # Look for sentence-ending punctuation in the last 100 characters
        for punct in ['.', '!', '?', '。', ';']:
            last_punct = text.rfind(punct, -100)
            if last_punct > 0:
                return last_punct + 1
        return len(text)  # If no good breakpoint, use full text

    def extract_target_length (self, query: str) -> Optional[int]:
        """Extract target word/token length from user query."""
        # Use regex to find patterns like "500 word" or "300 words"
        word_count_match = re.search(r'(\d+)\s*(?:word|words)', query.lower())
        if word_count_match:
            return int(word_count_match.group(1))

        # Could add more patterns here (e.g., "2000 character", "5 pages", etc.)

        return None  # Return None if no target length found

    def set_target_length(self, user_id: str, target_length: int):
        """Set target completion length for a user"""
        if user_id not in self.continuation_context:
            self.continuation_context[user_id] = {}
        self.continuation_context[user_id]["target_length"] = target_length

    def _count_words(self, text: str) -> int:
        """
        Count the number of words in a text.

        Args:
            text: The text to count words in

        Returns:
            The word count
        """
        # Use a simple word counting approach
        # Split by whitespace and filter out empty strings
        words = [word for word in text.split() if word.strip()]
        return len(words)
        
    def track_generated_response(self, response: str, user_id: str = "default_user", target_length: int = None):
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

        # Add completion detection
        if target_length and self._count_words(response) >= target_length:
            # Mark as complete in the continuation context
            self.continuation_context[user_id]["is_complete"] = True
            self.continuation_context[user_id]["completion_reason"] = "target_length_reached"

    def prepare_continuation_prompt(self, query: str, user_id: str = "default_user") -> str:
        """Enhanced prompt preparation with fractal context awareness"""

        # Get continuation context first (existing code)
        if user_id not in self.continuation_context:
            return query

        continuation_ctx = self.continuation_context[user_id]

        # NEW: Get content type to apply domain-specific handling
        content_type = self._detect_content_type(continuation_ctx["text"])

        # Get vector store from memory manager (if available)
        vector_store = None
        if self.memory_manager:
            store = self.memory_manage
            if store and hasattr(store, 'enhanced_fractal_search'):
                vector_store = store

        # Extract appropriate context based on content type
        if content_type == "code":
            last_context = self._extract_code_context(continuation_ctx["text"])
            # Use code-specific continuation prompt - completely clean for simple continuations
            is_simple = any(word in query.lower() for word in ["go on", "continue", "next", "more"])
            if is_simple:
                return f"[CODE BLOCK START]\n{last_context}\n[CODE BLOCK END]"
            else:
                return self._create_code_continuation_prompt(query, last_context)
        else:
            # Use existing prose handling
            last_context = self._extract_last_context(continuation_ctx["text"])
            return self._create_prose_continuation_prompt(query, last_context)

    def _detect_content_type(self, text: str) -> str:
        """Detect if content is code, table, or prose"""
        # Code detection (existing _is_code_context logic)
        if self._is_code_context(text):
            return "code"

        # Add other content type detection as needed
        return "prose"

    def _extract_code_context(self, text: str, vector_store=None) -> str:
        """Extract code context with better structure awareness"""
        # Basic extraction (similar to _extract_last_context)
        lines = text.split('\n')
        last_lines = lines[-20:] if len(lines) > 20 else lines

        # If we have vector store, enhance context with relevant code patterns
        if vector_store:
            try:
                # Create embedding for code context
                if self.tokenizer and hasattr(self.memory_manager, 'embedding_function'):
                    code_embedding = self.memory_manager.embedding_function("\n".join(last_lines))

                    # Search for similar code patterns
                    results = vector_store.enhanced_fractal_search(
                        code_embedding,
                        top_k=3,
                        min_similarity=0.7,
                        apply_sharpening=True
                    )

                    # Extract relevant code patterns from results
                    for result in results:
                        if result.get('metadata', {}).get('memory_type') == 'code':
                            # Add comment about incorporating pattern
                            last_lines.append(f"// Pattern based on: {result.get('text', '')[:50]}...")
            except Exception as e:
                print(f"Error enhancing code context: {e}")

        return "\n".join(last_lines)

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

        # Very short queries are often continuation requests
        if len(query.strip().split()) <= 3 and any(word in query_lower for word in ["more", "next", "then"]):
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

        # Determine content type (code, poetry, or prose)
        is_code = self._is_code_context(last_context)
        is_poetry = self._is_poetry_context(last_context)

        # Build a continuation prompt based on content type
        if is_code:
            return self._create_code_continuation_prompt(query, last_context)
        elif is_poetry:
            return self._create_poetry_continuation_prompt(query, last_context)
        else:
            return self._create_prose_continuation_prompt(query, last_context)

    def _create_code_continuation_prompt(self, query: str, code_context: str) -> str:
        """Create a cleaner code-specific continuation prompt"""
        # Simple continuations should have zero instructions
        if any(phrase in query.lower() for phrase in ["go on", "continue", "keep going", "next", "more"]):
            return f"""
    [CODE BLOCK START]
    {code_context}
    [CODE BLOCK END]
    """
        # Only for specific questions add minimal guidance
        return f"""
    [CODE BLOCK START]
    {code_context}
    [CODE BLOCK END]
    """

    def _create_poetry_continuation_prompt(self, query: str, last_context: str) -> str:
        """
        Create a continuation prompt for poetry.

        Args:
            query: Original user query
            last_context: Last context to continue from

        Returns:
            Enhanced poetry continuation prompt
        """
        # Extract the final partial line
        lines = last_context.strip().split('\n')
        last_line = lines[-1] if lines else ""

        # Extract the last word or partial word
        last_word = last_line.strip().split()[-1] if last_line.strip() else ""

        # Create poetry-specific prompt
        return f"""
{query}

[POEM CONTINUATION]
The poem was cut off at this exact line:
"{last_line}"

The last word or partial word was: "{last_word}"

Continue the poem from this exact point. Complete the unfinished line and continue with the poem's structure, style, and rhyme scheme.
"""

    def _create_prose_continuation_prompt(self, query: str, last_context: str) -> str:
        """
        Create a continuation prompt for prose.

        Args:
            query: Original user query
            last_context: Last context to continue from

        Returns:
            Enhanced prose continuation prompt
        """
        # Extract the last few words
        words = last_context.strip().split()
        last_words = " ".join(words[-10:]) if len(words) > 10 else last_context.strip()

        # Create prose-specific prompt
        return f"""
{query}

[CONTINUATION POINT]
The previous text ended with these words:
"{last_words}"

Continue writing from this exact point without repeating any of the content shown. Pick up precisely where the text was cut off and maintain the same style, tone, and flow.
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

    def _is_poetry_context(self, text: str) -> bool:
        """
        Determine if the context is poetry.

        Args:
            text: Context text to analyze

        Returns:
            Boolean indicating if context is poetry
        """
        # Split into lines
        lines = text.strip().split('\n')

        # Too few lines can't be analyzed properly
        if len(lines) < 3:
            return False

        # Count lines with consistent patterns
        short_lines = 0
        capitalized_lines = 0
        lines_with_punctuation_at_end = 0

        for line in lines:
            clean_line = line.strip()

            # Skip empty lines
            if not clean_line:
                continue

            # Check line length (poetry tends to have shorter lines)
            if len(clean_line.split()) < 12:
                short_lines += 1

            # Check capitalization at beginning of line (common in poetry)
            if clean_line and clean_line[0].isupper():
                capitalized_lines += 1

            # Check for punctuation at end of line
            if clean_line and clean_line[-1] in '.,:;!?':
                lines_with_punctuation_at_end += 1

        # Calculate percentages
        non_empty_lines = len([l for l in lines if l.strip()])
        if non_empty_lines == 0:
            return False

        pct_short = short_lines / non_empty_lines
        pct_capitalized = capitalized_lines / non_empty_lines
        pct_punctuated = lines_with_punctuation_at_end / non_empty_lines

        # Poetry indicators:
        # 1. High percentage of short lines
        # 2. High percentage of lines starting with capital letter
        # 3. Variable punctuation at line ends
        if pct_short > 0.7 and pct_capitalized > 0.7:
            return True

        # Look for rhyming patterns (simplistic approach)
        line_endings = []
        for line in lines:
            words = line.strip().split()
            if words:
                # Get last 2-3 characters of last word as crude rhyme check
                last_word = words[-1].lower()
                ending = last_word[-3:] if len(last_word) > 3 else last_word
                line_endings.append(ending)

        # Check for repeating patterns in endings
        if len(line_endings) >= 4:
            potential_rhymes = 0
            for i in range(len(line_endings) - 2):
                for j in range(i + 2, len(line_endings)):
                    if line_endings[i] == line_endings[j]:
                        potential_rhymes += 1

            if potential_rhymes >= len(line_endings) / 3:
                return True

        return False

    def _clean_code_context(self, code: str) -> str:
        """
        Clean code context for better continuation.

        Args:
            code: Code context to clean

        Returns:
            Cleaned code context
        """
        # Remove any code block markers
        code = re.sub(r'^```\w*\s*', '', code)
        code = re.sub(r'\s*```$', '', code)

        # Ensure no trailing whitespace
        lines = code.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]

        return '\n'.join(cleaned_lines)

    def _get_timestamp(self) -> str:
        """Get current timestamp for tracking."""
        from datetime import datetime
        return datetime.now().isoformat()