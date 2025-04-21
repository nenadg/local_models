"""
Rolling Window Manager for TinyLlama Chat.
Handles context window management to prevent exceeding token limits.
"""

import re
from typing import List, Dict, Any, Tuple, Optional

class RollingWindowManager:
    """
    Manages the context window for text generation to prevent exceeding token limits.
    Integrates with existing memory systems to preserve critical context.
    """
    
    def __init__(self, tokenizer, max_window_size: int = 2048, memory_manager=None, safety_margin: int = 50):
        """
        Initialize the rolling window manager.
        
        Args:
            tokenizer: The model tokenizer for token counting
            max_window_size: Maximum allowed tokens in context window
            memory_manager: Optional memory manager for improved context selection
            safety_margin: Token buffer to prevent exceeding limits
        """
        self.tokenizer = tokenizer
        self.max_window_size = max_window_size
        self.memory_manager = memory_manager
        self.safety_margin = safety_margin
        
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
        
        # If no system or user message, return as is (shouldn't happen in normal operation)
        if not system_message or not user_message:
            return messages
            
        # Calculate tokens for essential messages
        system_tokens = len(self.tokenizer.encode(system_message['content']))
        user_tokens = len(self.tokenizer.encode(user_message['content']))
        
        essential_tokens = system_tokens + user_tokens
        
        # If essential messages alone exceed budget, truncate the user message
        if essential_tokens > available_budget:
            # Find a reasonable truncation point for the user message
            # This is a fallback to prevent complete failure
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