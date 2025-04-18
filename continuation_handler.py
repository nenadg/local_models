class ContinuationHandler:
    """
    Handles the continuation of text generation by maintaining context between responses.
    Allows users to request more text from a previous response that may have been truncated.
    """
    
    def __init__(self, chat_instance):
        """
        Initialize the continuation handler.
        
        Args:
            chat_instance: The chat instance to use for generation
        """
        self.chat = chat_instance
        self.last_response = ""
        self.last_message_idx = -1
        self.continuation_phrases = [
            "please continue", 
            "continue", 
            "please continue from where you left off",
            "please continue from the latest line", 
            "go on",
            "tell me more",
            "what happens next",
            "and then?",
            "continue from the last line"
        ]
        self.continuation_count = 0
        self.max_continuations = 5  # Prevent infinite continuations
        
    def is_continuation_request(self, user_input):
        """
        Check if the user input is a continuation request.
        
        Args:
            user_input: User input text
            
        Returns:
            Boolean indicating if this is a continuation request
        """
        user_input_lower = user_input.lower().strip()
        
        # Direct match with continuation phrases
        if any(phrase == user_input_lower for phrase in self.continuation_phrases):
            return True
            
        # More flexible matching
        if any(phrase in user_input_lower for phrase in self.continuation_phrases):
            return True
            
        # Check for reference to continuing from last line
        if "last line" in user_input_lower and ("continue" in user_input_lower or "more" in user_input_lower):
            return True
            
        return False
        
    def reset(self):
        """Reset the continuation state."""
        self.last_response = ""
        self.last_message_idx = -1
        self.continuation_count = 0
        
    def set_last_response(self, response, message_idx):
        """
        Set the last response for potential continuation.
        
        Args:
            response: The response text
            message_idx: Index of the message in the conversation
        """
        self.last_response = response
        self.last_message_idx = message_idx
        self.continuation_count = 0
        
    def continue_response(self, conversation, user_input, max_new_tokens=256, temperature=0.7):
        """
        Generate a continuation of the last response.
        
        Args:
            conversation: Current conversation history
            user_input: User's continuation request
            max_new_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Tuple of (continuation_text, confidence_data)
        """
        # Increment continuation count
        self.continuation_count += 1
        
        # Check if we've reached the maximum number of continuations
        if self.continuation_count > self.max_continuations:
            return ("I've provided several continuations. Let's move on to a new topic to ensure I'm being helpful across a range of questions.", None)
        
        # Create a custom prompt that includes the last response and an instruction to continue
        last_few_paragraphs = self.extract_last_paragraphs(self.last_response, 3)
        
        # Create system message with continuation instruction
        continuation_system = {
            "role": "system", 
            "content": self.chat.system_message["content"] + "\n\nIMPORTANT: Continue from exactly where you left off previously. Do not repeat information, summarize, or add new introductions. Simply continue the narrative, explanation, or text as if you were writing one continuous document."
        }
        
        # Create messages for continuation
        continuation_messages = [
            continuation_system,
            {"role": "user", "content": f"Here is the end of your previous response: {last_few_paragraphs}\n\nPlease continue exactly from where you left off, without repeating anything."}
        ]
        
        # Generate continuation
        print(f"Continuing from previous response (continuation #{self.continuation_count})...")
        continuation = self.chat.generate_response(
            continuation_messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        # Get confidence metrics
        confidence_data = self.chat.confidence_metrics.get_metrics()
        
        # Remove any overlap with the previous response
        cleaned_continuation = self.remove_overlap(self.last_response, continuation)
        
        # Update the last response to include this continuation
        self.last_response = self.last_response + "\n\n" + cleaned_continuation
        
        return (cleaned_continuation, confidence_data)
    
    def extract_last_paragraphs(self, text, num_paragraphs=3):
        """
        Extract the last few paragraphs from text.
        
        Args:
            text: Text to extract from
            num_paragraphs: Number of paragraphs to extract
            
        Returns:
            String with the last paragraphs
        """
        paragraphs = text.split('\n\n')
        
        # Get the last few non-empty paragraphs
        last_paragraphs = [p for p in paragraphs if p.strip()][-num_paragraphs:]
        
        return '\n\n'.join(last_paragraphs)
    
    def remove_overlap(self, original, continuation):
        """
        Remove any overlapping text from the continuation.
        
        Args:
            original: Original text
            continuation: Continuation text that might repeat the end of original
            
        Returns:
            Cleaned continuation without overlap
        """
        # Simplistic approach - check if continuation starts with the end of original
        original_words = original.split()
        continuation_words = continuation.split()
        
        # Try different overlap lengths
        for overlap_len in range(min(20, len(original_words), len(continuation_words)), 0, -1):
            if original_words[-overlap_len:] == continuation_words[:overlap_len]:
                # Found overlap, return continuation without the overlapping part
                return ' '.join(continuation_words[overlap_len:])
        
        # More aggressive check for partial overlap or repetition
        # Check if continuation repeats sentence fragments from the end of original
        last_sentences = '. '.join(original.split('.')[-3:])
        
        # Look for significant fragments
        for fragment_len in range(5, min(15, len(last_sentences.split()))):
            fragments = [' '.join(last_sentences.split()[i:i+fragment_len]) 
                        for i in range(max(0, len(last_sentences.split())-fragment_len))]
            
            # Check if any substantial fragment appears at start of continuation
            for fragment in fragments:
                if fragment in ' '.join(continuation_words[:min(25, len(continuation_words))]):
                    # Try to find where to start after the overlap
                    fragment_words = fragment.split()
                    for i in range(min(25, len(continuation_words))):
                        if i + len(fragment_words) <= len(continuation_words):
                            if ' '.join(continuation_words[i:i+len(fragment_words)]) == fragment:
                                return ' '.join(continuation_words[i+len(fragment_words):])
        
        # No significant overlap found
        return continuation