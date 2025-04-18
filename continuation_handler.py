import re
import numpy as np

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
        
    def continue_response(self, conversation, user_input, max_new_tokens=256, temperature=0.7, show_confidence=False):
	    """Generate a continuation with improved context management."""
	    # Reset interrupt handler
	    if hasattr(self.chat, 'stop_event'):
	        self.chat.stop_event.clear()

	    if hasattr(self.chat, 'interrupt_handler'):
	        self.chat.interrupt_handler.reset()
	        self.chat.interrupt_handler.install()

	    # Increment continuation count
	    self.continuation_count += 1

	    if self.continuation_count > self.max_continuations:
	        return ("I've provided several continuations. Let's move on to a new topic.", None)

	    # Get last response and detect if it contains code
	    contains_code = '```' in self.last_response
	    last_few_paragraphs = self.extract_last_paragraphs(self.last_response, 5 if contains_code else 3)

	    # If it's code, extract the last code block and the language
	    code_context = ""
	    if contains_code:
	        code_blocks = re.findall(r'```(\w*)\n([\s\S]*?)```', self.last_response)
	        if code_blocks:
	            lang, code = code_blocks[-1]  # Get the last code block
	            # Find the last line that's not empty
	            code_lines = code.split('\n')
	            last_code_lines = [line for line in code_lines[-10:] if line.strip()]
	            if last_code_lines:
	                code_context = f"\nThe last code line was: {last_code_lines[-1]}"

	    # Create continuation system message with explicit boundary markers
	    continuation_system = {
	        "role": "system",
	        "content": self.chat.system_message["content"] +
	                  "\n\nIMPORTANT: Continue exactly from where you left off. " +
	                  "Maintain the same topic, style, and approach." +
	                  (f"\nThis is a code continuation in {lang}. " if contains_code else "") +
	                  "Do not repeat information or summarize what's already been said."
	    }

	    # Create user message with explicit continuation markers
	    user_message = (f"Here is the end of your previous response:\n\n{last_few_paragraphs}\n\n" +
	                   f"// --- CONTINUE FROM THIS EXACT POINT --- {code_context}\n\n" +
	                   "Please continue exactly from where you left off, without repeating anything.")

	    # Create messages for continuation
	    continuation_messages = [
	        continuation_system,
	        {"role": "user", "content": user_message}
	    ]

	    # Generate continuation
	    print(f"Continuing from previous response (continuation #{self.continuation_count})...")
	    continuation = self.chat.generate_response(
	        continuation_messages,
	        max_new_tokens=max_new_tokens,
	        temperature=temperature,
	        show_confidence=show_confidence
	    )

	    # Use fractal embeddings to detect overlap if available
	    cleaned_continuation = ""
	    if hasattr(self.chat, 'memory_manager') and hasattr(self.chat.memory_manager, '_get_user_store'):
	        # Get user store for fractal embeddings
	        user_store = self.chat.memory_manager._get_user_store(self.chat.current_user_id)
	        if hasattr(user_store, 'fractal_enabled') and user_store.fractal_enabled:
	            # Generate embeddings for end of original and start of continuation
	            end_original = self.last_response[-500:]  # Last 500 chars
	            start_continuation = continuation[:500]   # First 500 chars

	            if hasattr(self.chat.memory_manager, 'generate_embedding'):
	                end_embedding = self.chat.memory_manager.generate_embedding(end_original)
	                start_embedding = self.chat.memory_manager.generate_embedding(start_continuation)

	                # Calculate similarity
	                end_norm = end_embedding / np.linalg.norm(end_embedding)
	                start_norm = start_embedding / np.linalg.norm(start_embedding)
	                similarity = np.dot(end_norm, start_norm)

	                # If high similarity, find a cutoff point
	                if similarity > 0.85:
	                    # More sophisticated overlap detection
	                    # Find sentence boundaries
	                    sentences = re.split(r'[.!?]+', start_continuation)
	                    accumulated = ""
	                    cutoff_idx = len(start_continuation)

	                    for i, sentence in enumerate(sentences):
	                        accumulated += sentence
	                        if accumulated in end_original:
	                            # Found overlap, continue after this sentence
	                            cutoff_idx = len(accumulated)
	                        else:
	                            break

	                    # Extract non-overlapping content
	                    cleaned_continuation = continuation[cutoff_idx:].strip()
	                else:
	                    cleaned_continuation = continuation
	            else:
	                cleaned_continuation = self.remove_overlap(self.last_response, continuation)
	        else:
	            cleaned_continuation = self.remove_overlap(self.last_response, continuation)
	    else:
	        cleaned_continuation = self.remove_overlap(self.last_response, continuation)

	    # Update last response
	    self.last_response = self.last_response + "\n\n" + cleaned_continuation

	    # Clean up interrupt handler
	    if hasattr(self.chat, 'interrupt_handler'):
	        self.chat.interrupt_handler.uninstall()

	    return (cleaned_continuation, self.chat.confidence_metrics.get_metrics())

    
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
        Enhanced method to remove any overlapping text from the continuation
        with improved duplicate detection.

        Args:
            original: Original text
            continuation: Continuation text that might repeat the end of original

        Returns:
            Cleaned continuation without overlap
        """
        # If either string is empty, return the continuation as is
        if not original or not continuation:
            return continuation

        # First approach: Check for exact repeats of the continuation prompt
        prompt_patterns = [
            "Please continue exactly from where you left off, without repeating anything.",
            "Please continue from the latest line",
            "Please continue",
            "Continue from where you left off"
        ]

        cleaned_continuation = continuation
        for pattern in prompt_patterns:
            cleaned_continuation = cleaned_continuation.replace(pattern, "")

        # Second approach: Check if the continuation begins with a repeat of the end of the original
        original_words = original.split()
        continuation_words = cleaned_continuation.split()

        # Try different overlap lengths to find the longest matching sequence
        max_check_length = min(50, len(original_words), len(continuation_words))
        overlap_found = False

        for overlap_len in range(max_check_length, 4, -1):  # Try from longest to shortest, minimum 5 words
            if len(original_words) >= overlap_len and len(continuation_words) >= overlap_len:
                original_end = original_words[-overlap_len:]
                continuation_start = continuation_words[:overlap_len]

                # Check for exact match
                if original_end == continuation_start:
                    cleaned_continuation = ' '.join(continuation_words[overlap_len:])
                    overlap_found = True
                    break

                # Check for fuzzy match (allow minor differences)
                matches = sum(1 for a, b in zip(original_end, continuation_start) if a.lower() == b.lower())
                if matches / overlap_len > 0.8:  # 80% match threshold
                    cleaned_continuation = ' '.join(continuation_words[overlap_len:])
                    overlap_found = True
                    break

        # Third approach: Check for repeated sentences
        if not overlap_found:
            original_sentences = [s.strip() for s in original.split('.') if s.strip()]
            continuation_sentences = [s.strip() for s in cleaned_continuation.split('.') if s.strip()]

            if original_sentences and continuation_sentences:
                # Check if any of the last 3 sentences of the original are repeated in the first 3 sentences of continuation
                last_sentences = original_sentences[-3:] if len(original_sentences) >= 3 else original_sentences
                first_sentences = continuation_sentences[:3] if len(continuation_sentences) >= 3 else continuation_sentences

                for orig_sent in last_sentences:
                    if len(orig_sent.split()) < 4:  # Skip very short sentences
                        continue

                    for i, cont_sent in enumerate(first_sentences):
                        # If there's significant overlap between sentences
                        orig_words = set(orig_sent.lower().split())
                        cont_words = set(cont_sent.lower().split())
                        common_words = orig_words.intersection(cont_words)

                        # If more than 70% words match
                        if len(common_words) > 0 and len(common_words) / max(len(orig_words), len(cont_words)) > 0.7:
                            # Skip this sentence and all before it
                            remaining_sentences = continuation_sentences[i+1:]
                            cleaned_continuation = '. '.join(remaining_sentences)
                            if cleaned_continuation and not cleaned_continuation.endswith('.'):
                                cleaned_continuation += '.'
                            overlap_found = True
                            break

                    if overlap_found:
                        break

        # Final pass: Check for duplicate paragraphs
        if not overlap_found and '\n\n' in cleaned_continuation:
            paragraphs = cleaned_continuation.split('\n\n')
            unique_paragraphs = []
            seen_paragraphs = set()

            for paragraph in paragraphs:
                # Create a simplified version for comparison
                simplified = ' '.join(paragraph.lower().split())
                if simplified and simplified not in seen_paragraphs:
                    seen_paragraphs.add(simplified)
                    unique_paragraphs.append(paragraph)

            cleaned_continuation = '\n\n'.join(unique_paragraphs)

        # Final cleanup - remove any leading/trailing whitespace
        cleaned_continuation = cleaned_continuation.strip()

        # If nothing's left after cleaning, return a message
        if not cleaned_continuation:
            cleaned_continuation = "I've completed my response. Is there something else you'd like to know about?"

        return cleaned_continuation