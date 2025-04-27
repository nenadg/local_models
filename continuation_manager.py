"""
Enhanced Continuation Manager integrated with UnifiedMemoryManager.
Provides seamless continuations with knowledge integration and proper tensor handling.
"""

import re
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from datetime import datetime

class ContinuationTrackingWindowManager:
    """
    Enhanced window manager that tracks generated content for continuations
    and integrates with UnifiedMemoryManager for knowledge-aware context management.
    """

    def __init__(self,
                tokenizer,
                max_window_size: int = 2048,
                memory_manager=None,
                safety_margin: int = 50,
                continuation_buffer_size: int = 200):
        """
        Initialize the enhanced continuation-aware window manager.

        Args:
            tokenizer: The model tokenizer for token counting
            max_window_size: Maximum allowed tokens in context window
            memory_manager: UnifiedMemoryManager instance for knowledge integration
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
        self.continuation_context = {}  # Maps user_id to continuation context

        # Debug information
        self.debug_mode = False

    def log(self, message: str) -> None:
        """Log a message if debug mode is enabled"""
        if self.debug_mode:
            timestamp = datetime.now().strftime("[%d/%m/%y %H:%M:%S]")
            print(f"{timestamp} [ContinuationManager] {message}")

    def track_repetition_patterns(self, text: str, user_id: str = "default_user") -> bool:
        """
        Track and detect repetitive patterns with content type awareness.

        Args:
            text: Generated text to analyze
            user_id: User identifier

        Returns:
            Boolean indicating if repetition was detected
        """
        repetition_detected = False
        pattern_length = 20
        min_repetitions = 3

        # Skip detection for specific patterns common in code
        content_type = self._detect_content_type(text)

        if content_type == "code":
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

        # Store repetition status for this user
        if user_id not in self.continuation_context:
            self.continuation_context[user_id] = {}

        self.continuation_context[user_id]["repetition_detected"] = repetition_detected

        return repetition_detected

    def optimize_messages(self, messages: List[Dict[str, str]], max_new_tokens: int = 128) -> List[Dict[str, str]]:
        """
        Optimize message history to fit within token limits, with proper tensor handling.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_new_tokens: Expected generation length

        Returns:
            Optimized list of messages
        """
        self.log(f"Optimizing messages for {len(messages)} messages, max_new_tokens={max_new_tokens}")

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

        # Calculate tokens for essential messages with proper tensor handling
        with torch.no_grad():  # Prevent "Already borrowed" errors
            # Use safer method for token counting
            system_tokens = self._count_tokens_safely(system_message['content'])
            user_tokens = self._count_tokens_safely(user_message['content'])

        essential_tokens = system_tokens + user_tokens
        self.log(f"Essential tokens: {essential_tokens} (system={system_tokens}, user={user_tokens})")

        # If essential messages alone exceed budget, truncate the user message
        if essential_tokens > available_budget:
            self.log(f"Essential messages exceed budget, truncating user message")
            max_user_tokens = available_budget - system_tokens
            truncated_user_content = self._truncate_text(user_message['content'], max_user_tokens)

            # Return minimal context
            return [
                system_message,
                {"role": "user", "content": truncated_user_content}
            ]

        # We have room for conversation history - prioritize recent messages
        history_budget = available_budget - essential_tokens
        self.log(f"History budget: {history_budget} tokens")
        optimized_history = self._select_history_messages(messages[1:-1], history_budget)

        # Construct final message list
        result = [system_message] + optimized_history + [user_message]

        # Log token usage
        final_tokens = self.calculate_tokens(result)
        self.log(f"Final optimized message count: {len(result)}, tokens: {final_tokens}/{self.max_window_size}")

        return result

    def _count_tokens_safely(self, text: str) -> int:
        """
        Count tokens in text with proper error handling to prevent "Already borrowed" errors.

        Args:
            text: Text to count tokens in

        Returns:
            Number of tokens
        """
        try:
            # Create a new tensor each time to avoid "Already borrowed" errors
            with torch.no_grad():
                # Don't add special tokens to get more accurate counts
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                return len(token_ids)
        except RuntimeError as e:
            self.log(f"Error in token counting: {e}")
            # Fallback to character-based estimation (not accurate but safe)
            return len(text) // 4  # Rough estimate: ~4 chars per token

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
            # Count tokens safely
            msg_tokens = self._count_tokens_safely(msg['content'])

            # If adding this message would exceed budget, skip it
            if used_tokens + msg_tokens > token_budget:
                break

            # Add message and update token count
            selected_messages.insert(0, msg)  # Insert at beginning to maintain order
            used_tokens += msg_tokens

        self.log(f"Selected {len(selected_messages)}/{len(history)} history messages using {used_tokens}/{token_budget} tokens")
        return selected_messages

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit, breaking at sentence boundaries.

        Args:
            text: Text to truncate
            max_tokens: Maximum allowed tokens

        Returns:
            Truncated text
        """
        if not text:
            return ""

        # Early exit if the text is already under the limit
        if self._count_tokens_safely(text) <= max_tokens:
            return text

        # Try to find a good breakpoint using sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        truncated = ""
        for sentence in sentences:
            # Check if adding this sentence would exceed limit
            test_text = truncated + sentence
            test_tokens = self._count_tokens_safely(test_text)

            if test_tokens <= max_tokens:
                truncated = test_text
            else:
                break

        # If we couldn't fit even one sentence, do hard truncation
        if not truncated:
            self.log("Couldn't truncate at sentence boundary, performing hard truncation")

            # Tokenize and decode only max_tokens
            with torch.no_grad():
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                truncated = self.tokenizer.decode(token_ids[:max_tokens])

        return truncated

    def calculate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Calculate the total number of tokens in a message list safely.

        Args:
            messages: List of message dictionaries

        Returns:
            Total token count
        """
        total = 0
        for msg in messages:
            total += self._count_tokens_safely(msg['content'])

        return total

    def is_generation_complete(self, user_id: str, current_response: str) -> bool:
        """
        Check if generation has reached completion based on content analysis.

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
            self.log(f"Generation complete: reached target length of {target_length} words")
            return True

        # Content type specific checks
        content_type = self._detect_content_type(current_response)

        # Check for empty line padding (5+ consecutive newlines)
        newline_count = 0
        for char in current_response[-20:]:  # Check last 20 chars
            if char == '\n':
                newline_count += 1
            elif char.strip():  # Reset on non-whitespace
                newline_count = 0

        if newline_count >= 5:
            self.log("Generation complete: detected multiple consecutive empty lines")
            return True

        # Code-specific checks
        if content_type == "code":
            lines = current_response.split('\n')
            if len(lines) > 10:
                # Get last 10 lines
                last_lines = lines[-10:]
                # Create a set of stripped lines
                unique_lines = set(line.strip() for line in last_lines if line.strip())
                # If we have fewer than 3 unique lines in the last 10, we're likely in a loop
                if 0 < len(unique_lines) < 3 and len(last_lines) > 5:
                    self.log("Generation complete: detected code repetition pattern")
                    return True

            # Check for code completeness indicators
            if current_response.count('```') >= 2:  # Opening and closing code blocks
                # Look for function/class completeness
                # Count braces/indentation to detect completion
                open_braces = current_response.count('{')
                close_braces = current_response.count('}')
                if open_braces > 0 and open_braces == close_braces:
                    self.log("Generation complete: matched code block structure")
                    return True

        # Text-specific checks
        if content_type == "prose":
            # Check for conclusion indicators
            conclusion_markers = ["In conclusion", "To summarize", "In summary", "Thus", "Therefore"]

            # Get the last few sentences
            last_part = " ".join(current_response.split()[-30:])

            if any(marker in last_part for marker in conclusion_markers):
                # Check if we have a reasonable amount of content
                if word_count > 100:
                    self.log("Generation complete: detected conclusion marker with sufficient content")
                    return True

        return False

    def _find_safe_termination_point(self, text: str) -> int:
        """
        Find a good position to terminate text (after punctuation).

        Args:
            text: Text to analyze

        Returns:
            Index where text can be safely terminated
        """
        # Look for sentence-ending punctuation in the last 100 characters
        for punct in ['.', '!', '?', '。', ';']:
            last_punct = text.rfind(punct, -100)
            if last_punct > 0:
                return last_punct + 1

        return len(text)  # If no good breakpoint, use full text

    def extract_target_length(self, query: str) -> Optional[int]:
        """
        Extract target word/token length from user query with enhanced pattern matching.

        Args:
            query: User query

        Returns:
            Target length or None if not found
        """
        # Look for common length specifications
        word_count_match = re.search(r'(\d+)\s*(?:word|words)', query.lower())
        if word_count_match:
            return int(word_count_match.group(1))

        # Look for character count requirements
        char_count_match = re.search(r'(\d+)\s*(?:character|characters|chars)', query.lower())
        if char_count_match:
            # Convert approximate character count to words (roughly 5 chars per word)
            chars = int(char_count_match.group(1))
            return chars // 5

        # Look for page requirements
        page_count_match = re.search(r'(\d+)\s*(?:page|pages)', query.lower())
        if page_count_match:
            # Approximate 250 words per page
            pages = int(page_count_match.group(1))
            return pages * 250

        # Look for paragraph counts
        para_count_match = re.search(r'(\d+)\s*(?:paragraph|paragraphs)', query.lower())
        if para_count_match:
            # Approximate 75 words per paragraph
            paras = int(para_count_match.group(1))
            return paras * 75

        return None  # Return None if no target length found

    def set_target_length(self, user_id: str, target_length: int):
        """
        Set target completion length for a user.

        Args:
            user_id: User identifier
            target_length: Target length in words
        """
        if user_id not in self.continuation_context:
            self.continuation_context[user_id] = {}

        self.continuation_context[user_id]["target_length"] = target_length
        self.log(f"Set target length for user {user_id} to {target_length} words")

    def track_generated_response(self, response: str, user_id: str = "default_user", target_length: int = None):
        """
        Track a generated response for potential continuation.

        Args:
            response: The generated response text
            user_id: User identifier
            target_length: Optional target length in words
        """
        # Store the full response
        self.last_response = response

        # Tokenize the response (safely)
        with torch.no_grad():
            last_response_tokens = self.tokenizer.encode(response, add_special_tokens=False)

        # Store the last portion for continuation context
        continuation_tokens = last_response_tokens[-min(self.continuation_buffer_size, len(last_response_tokens)):]
        continuation_text = self.tokenizer.decode(continuation_tokens)

        # Store continuation context for this user
        self.continuation_context[user_id] = {
            "tokens": continuation_tokens,
            "text": continuation_text,
            "timestamp": self._get_timestamp(),
            "full_response": response,
            "content_type": self._detect_content_type(response)
        }

        # Add completion detection
        if target_length and self._count_words(response) >= target_length:
            # Mark as complete in the continuation context
            self.continuation_context[user_id]["is_complete"] = True
            self.continuation_context[user_id]["completion_reason"] = "target_length_reached"
            self.log(f"Response for user {user_id} marked as complete: target length reached")

        # Log continuation context preparation
        self.log(f"Stored continuation context for user {user_id} with {len(continuation_tokens)} tokens")

    def detect_continuation_request(self, query: str) -> bool:
        """
        Detect if a query is asking for continuation with enhanced patterns.

        Args:
            query: User query text

        Returns:
            Boolean indicating if this is a continuation request
        """
        # Common phrases that indicate continuation requests
        continuation_phrases = [
            "continue", "go on", "keep going", "proceed", "what happens next",
            "continue from", "pick up where", "write more", "and then?",
            "next part", "next section", "rest of", "finish", "complete",
            "what's next", "keep writing", "don't stop", "go ahead", "proceed"
        ]

        query_lower = query.lower()

        # Check for continuation indicators
        for phrase in continuation_phrases:
            if phrase in query_lower:
                self.log(f"Detected continuation request: '{phrase}' in query")
                return True

        # Very short queries are often continuation requests
        if len(query.strip().split()) <= 3 and any(word in query_lower for word in ["more", "next", "then"]):
            self.log("Detected simple continuation request (short query)")
            return True

        return False

    def prepare_continuation_prompt(self, query: str, user_id: str = "default_user") -> str:
        """
        Prepare an enhanced query for continuation requests with knowledge integration.

        Args:
            query: Original user query
            user_id: User identifier

        Returns:
            Enhanced query with continuation context
        """
        # Check if we have continuation context for this user
        if user_id not in self.continuation_context:
            self.log(f"No continuation context for user {user_id}")
            return query

        continuation_ctx = self.continuation_context[user_id]

        # Extract the last few lines of context for more precise continuation
        last_context = self._extract_last_context(continuation_ctx["text"])

        if not last_context:
            return query

        # Determine content type (default to "prose" if not set)
        content_type = continuation_ctx.get("content_type", self._detect_content_type(last_context))
        self.log(f"Detected content type: {content_type}")

        # Integrate with memory manager if available
        if content_type == "code" and self.memory_manager and hasattr(self.memory_manager, 'retrieve'):
            # Try to augment code continuation with relevant knowledge
            try:
                code_query = f"code {last_context.split()[-20:]}"
                knowledge_results = self.memory_manager.retrieve(
                    query=code_query,
                    top_k=2,
                    min_similarity=0.5
                )

                if knowledge_results:
                    # Extract relevant code patterns
                    self.log(f"Found {len(knowledge_results)} relevant code patterns")
                    code_knowledge = "\n".join([f"// Relevant pattern: {result['content'][:100]}..."
                                             for result in knowledge_results])
                    # Enhance the context with the patterns
                    last_context = f"{last_context}\n{code_knowledge}"
            except Exception as e:
                self.log(f"Error retrieving knowledge for code continuation: {e}")

        # Build a continuation prompt based on content type
        if content_type == "code":
            return self._create_code_continuation_prompt(query, last_context)
        else:
            return self._create_prose_continuation_prompt(query, last_context)

    def _create_code_continuation_prompt(self, query: str, code_context: str) -> str:
        """
        Create a code-specific continuation prompt with clean formatting.

        Args:
            query: Original user query
            code_context: Code context to continue from

        Returns:
            Formatted continuation prompt
        """
        # Simple continuations should have zero instructions
        if any(phrase in query.lower() for phrase in ["go on", "continue", "keep going", "next", "more"]):
            return f"""
[CODE BLOCK START]
{code_context}
[CODE BLOCK END]

Continue the code exactly from where it left off, maintaining the same style and structure.
"""
        # For specific requests, add minimal guidance
        return f"""
[CODE BLOCK START]
{code_context}
[CODE BLOCK END]

{query}

Continue the code implementation from exactly where it left off, addressing the request while maintaining the existing code structure and style.
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
        last_words = " ".join(words[-15:]) if len(words) > 15 else last_context.strip()

        # Create prose-specific prompt
        if any(phrase in query.lower() for phrase in ["go on", "continue", "keep going", "next", "more"]):
            # Simple continuation
            return f"""
[CONTINUATION POINT]
The previous text ended with:
"{last_words}"

Continue writing from this exact point without repeating any of the content shown. Pick up precisely where the text was cut off and maintain the same style, tone, and flow.
"""
        else:
            # Continuation with specific request
            return f"""
{query}

[CONTINUATION CONTEXT]
The previous text ended with:
"{last_words}"

Continue writing from this exact point, addressing the request while maintaining the same style and tone.
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

    def _detect_content_type(self, text: str) -> str:
        """
        Detect the content type (code, prose, etc.) from text.

        Args:
            text: Text to analyze

        Returns:
            Content type string
        """
        # Code indicators
        code_indicators = [
            # Symbols and patterns common in code
            "{", "}", "(", ")", ";", "==", "=>", "function", "var ", "let ", "const ",
            "if ", "for ", "while ", "class ", "import ", "export ", "return ",
            "public ", "private ", "static ",  # Common keywords

            # Common language patterns
            "def ", "async ", "await ", "try ", "catch ", "else ", "elif ",
        ]

        # Look for code block markers
        if "```" in text:
            return "code"

        # Check for code indicators
        for indicator in code_indicators:
            if indicator in text:
                return "code"

        # Check for code-like line patterns (important for languages like Python)
        lines = text.strip().split('\n')
        indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
        if indented_lines > len(lines) / 3:  # If more than 1/3 of lines are indented
            return "code"

        # If none of the above, assume prose
        return "prose"

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

    def _get_timestamp(self) -> str:
        """Get current timestamp for tracking."""
        return datetime.now().isoformat()

    def set_debug_mode(self, enabled: bool = True) -> None:
        """Enable or disable debug logging"""
        self.debug_mode = enabled
        self.log(f"Debug mode {'enabled' if enabled else 'disabled'}")

    def enhance_with_knowledge(self, context: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Enhance continuation context with knowledge retrieved from memory manager.

        Args:
            context: Current continuation context
            query: The original user query

        Returns:
            Enhanced continuation context
        """
        if not self.memory_manager or not hasattr(self.memory_manager, 'retrieve'):
            return context

        enhanced_context = context.copy()
        content_type = context.get("content_type", "prose")

        try:
            # Create a query appropriate for the content type
            if content_type == "code":
                knowledge_query = f"code implementation {' '.join(query.split()[:10])}"
            else:
                knowledge_query = f"continue writing {' '.join(query.split()[:10])}"

            # Retrieve relevant knowledge
            results = self.memory_manager.retrieve(
                query=knowledge_query,
                top_k=3,
                min_similarity=0.45
            )

            if results:
                self.log(f"Retrieved {len(results)} knowledge items for continuation")
                # Store knowledge in context
                enhanced_context["knowledge_items"] = [
                    {"content": result["content"], "similarity": result.get("similarity", 0.0)}
                    for result in results
                ]
        except Exception as e:
            self.log(f"Error enhancing with knowledge: {e}")

        return enhanced_context