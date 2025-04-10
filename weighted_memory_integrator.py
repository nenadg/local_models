import re
import numpy as np
from typing import List, Dict, Any, Optional
from pattern_matching_utils import extract_arithmetic_expression, extract_example_pairs, extract_mapping_category, clean_duplicate_memories

class WeightedMemoryIntegrator:
    """
    Enhances memory retrieval and integration with domain-aware weighting
    to properly balance memory vs. model knowledge.

    Adapted to work with the refactored MemoryManager class.
    """

    def __init__(self, memory_manager, question_classifier=None):
        """
        Initialize the weighted memory integrator.

        Args:
            memory_manager: The memory manager instance
            question_classifier: Optional classifier for domain detection
        """
        self.memory_manager = memory_manager
        self.question_classifier = question_classifier

        # Domain-specific post-processors
        self.post_processors = {
            'arithmetic': self._post_process_arithmetic,
            'translation': self._post_process_translation,
            'transliteration': self._post_process_translation
        }

    def retrieve_and_integrate(self, user_id: str, query: str) -> Dict[str, Any]:
        """
        Retrieve memories and integrate them with domain-specific weighting.

        Args:
            user_id: The user ID
            query: The query string

        Returns:
            Dict containing the integrated memory and settings used
        """
        # Determine domain-specific settings
        settings = self._get_settings(query)

        # Get memories with domain-specific settings
        memory_text = self._retrieve_memories(user_id, query, settings)

        # Apply domain-specific post-processing if needed
        if settings['post_process'] and settings['domain'] in self.post_processors:
            memory_text = self.post_processors[settings['domain']](query, memory_text)

        # Return both the memory text and the settings used
        return {
            'memory_text': memory_text,
            'settings': settings
        }

    def _get_settings(self, query: str) -> Dict[str, Any]:
        """Get domain-specific settings based on query classification"""
        if self.question_classifier:
            return self.question_classifier.get_domain_settings(query)

        # Default settings if no classifier is available
        return {
            'memory_weight': 0.5,
            'sharpening_factor': 0.3,
            'confidence_threshold': 0.6,
            'domain': 'unknown',
            'domain_confidence': 0.0,
            'post_process': False,
            'retrieval_count': 8
        }

    def _format_memory_results(self, results: List[Dict[str, Any]], top_k: int) -> str:
        """
        Format search results into categorized memory text.

        Args:
            results: Search results from vector store
            top_k: Maximum number of results to include

        Returns:
            Formatted memory text
        """
        # If no results, return empty string
        if not results:
            return ""

        # Group results into clear categories
        corrections = []
        factual_info = []
        general_info = []

        for result in results[:top_k]:
            if result.get('metadata', {}).get('is_correction', False):
                corrections.append(result)
            elif any(term in result['text'].lower() for term in
                    ['definition', 'fact', 'rule', 'alphabet', 'order']):
                factual_info.append(result)
            else:
                general_info.append(result)

        # Compose the memory text with clear sections
        memory_text = ""

        if corrections:
            memory_text += "IMPORTANT CORRECTIONS (You MUST apply these):\n"
            for result in corrections:
                memory_text += f"- {result['text']}\n"
            memory_text += "\n"

        if factual_info:
            memory_text += "FACTUAL INFORMATION:\n"
            for result in factual_info:
                memory_text += f"- {result['text']}\n"
            memory_text += "\n"

        if general_info and (not corrections or not factual_info):
            memory_text += "OTHER RELEVANT INFORMATION:\n"
            for result in general_info:
                memory_text += f"- {result['text']}\n"

        return memory_text

    def _retrieve_memories(self, user_id: str, query: str, settings: Dict[str, Any]) -> str:
        """
        Retrieve memories with domain-specific parameters.

        Args:
            user_id: The user ID
            query: The query string
            settings: Domain-specific settings

        Returns:
            Formatted memory text
        """
        # Instead of modifying the memory manager's sharpening factor,
        # pass the appropriate sharpening factor directly to the retrieve method

        # Get memories with domain-specific count and sharpening
        retrieve_count = settings.get('retrieval_count', 8)
        sharpening_factor = settings.get('sharpening_factor', 0.3)
        apply_sharpening = self.memory_manager.sharpening_enabled if hasattr(self.memory_manager, 'sharpening_enabled') else True

        # Get the store directly
        store = self.memory_manager._get_user_store(user_id)

        # Generate embedding for the query
        query_embedding = self.memory_manager.generate_embedding(query)

        # Search with appropriate parameters
        results = store.search(
            query_embedding,
            top_k=retrieve_count*2,  # Get more results for better filtering
            min_similarity=0.25,
            apply_sharpening=apply_sharpening,
            sharpening_factor=sharpening_factor
        )

        # Format the results into memory text
        memory_text = self._format_memory_results(results, retrieve_count)

        # Check for domain-specific memory injection
        memory_text = self._inject_domain_knowledge(query, memory_text, settings)

        return memory_text

    def _inject_domain_knowledge(self, query: str, memory_text: str, settings: Dict[str, Any]) -> str:
        """
        Inject domain-specific knowledge for certain question types.

        Args:
            query: The query string
            memory_text: Retrieved memory text
            settings: Domain settings

        Returns:
            Enhanced memory text
        """
        domain = settings['domain']

        if domain == 'arithmetic':
            # Try to parse the arithmetic expression
            expression = self._extract_arithmetic_expression(query)
            if expression:
                try:
                    # Safely evaluate the expression
                    result = eval(expression)
                    # Add the result as a high-priority memory
                    arithmetic_memory = f"IMPORTANT FACTUAL INFORMATION:\n- The correct result of {expression} is {result}.\n\n"

                    # Clean up any existing memory text
                    cleaned_memory = self._clean_arithmetic_memories(memory_text, expression, result)

                    # Combine with existing memories but put arithmetic first
                    if cleaned_memory:
                        return arithmetic_memory + cleaned_memory
                    else:
                        return arithmetic_memory
                except:
                    pass  # If evaluation fails, just use existing memories

        elif domain == 'translation' and settings['domain_confidence'] > 0.8:
            # Extract word and language information
            word_match = re.search(r'["\']([^"\']+)["\']', query)
            if not word_match:
                word_match = re.search(r'translate\s+(\w+)', query)

            lang_match = re.search(r'(?:in|to|into)\s+(\w+)', query)

            if word_match and lang_match:
                word = word_match.group(1)
                language = lang_match.group(1).lower()

                # Check if we have a known translation
                translation = self._get_known_translation(word, language)

                if translation:
                    # Clean any incorrect memories
                    self.clean_memory_from_incorrect_translations(
                        user_id, word, language)

                    # Add the known translation to the top of the memory text
                    known_translation = f"IMPORTANT FACTUAL INFORMATION:\n- The word \"{word}\" in {language} is \"{translation}\".\n\n"

                    # Clean up memory text to remove duplicates and incorrect translations
                    cleaned_memory = self._clean_translation_memories(memory_text, word, language, translation)

                    if cleaned_memory:
                        return known_translation + cleaned_memory
                    else:
                        return known_translation

            # For high-confidence translation questions, add reminder about accuracy
            translation_reminder = "IMPORTANT NOTE:\n- Be careful with translations, especially for uncommon languages. If unsure, indicate your uncertainty.\n\n"

            # Clean up memory text
            cleaned_memory = self._clean_duplicate_memories(memory_text)

            if cleaned_memory:
                return translation_reminder + cleaned_memory
            else:
                return translation_reminder

        # For other domains, just clean up duplicates
        return self._clean_duplicate_memories(memory_text)

    def _extract_arithmetic_expression(self, query: str) -> Optional[str]:
        """Extract an arithmetic expression from a query."""
        return extract_arithmetic_expression(query)

    def _post_process_arithmetic(self, query: str, memory_text: str) -> str:
        """
        Apply post-processing for arithmetic queries.

        Args:
            query: The query string
            memory_text: Retrieved memory text

        Returns:
            Post-processed memory text
        """
        # This is similar to _inject_domain_knowledge but focused on verification
        expression = self._extract_arithmetic_expression(query)
        if not expression:
            return memory_text

        try:
            correct_result = eval(expression)

            # Look for incorrect arithmetic in the memories
            memory_lines = memory_text.split('\n')
            corrected_lines = []

            for line in memory_lines:
                # Check if this line has the arithmetic expression
                if expression in line.replace(' ', ''):
                    # Check if it has an incorrect result
                    result_match = re.search(rf'{re.escape(expression)}.*?(\d+)', line.replace(' ', ''))
                    if result_match:
                        stated_result = result_match.group(1)
                        if int(stated_result) != correct_result:
                            # Replace with correction
                            corrected_lines.append(f"- CORRECTION: The result of {expression} is {correct_result}, not {stated_result}.")
                            continue

                corrected_lines.append(line)

            return '\n'.join(corrected_lines)

        except:
            return memory_text

    def _post_process_translation(self, query: str, memory_text: str) -> str:
        """
        Apply post-processing for translation queries.

        Args:
            query: The query string
            memory_text: Retrieved memory text

        Returns:
            Post-processed memory text
        """
        # Extract language information
        word_match = re.search(r'["\']([^"\']+)["\']', query)
        if not word_match:
            word_match = re.search(r'translate\s+(\w+)', query)

        if not word_match:
            return memory_text

        word = word_match.group(1)

        # Look for target language
        lang_match = re.search(r'(?:in|to|into)\s+(\w+)', query)
        if not lang_match:
            return memory_text

        language = lang_match.group(1).lower()

        # We could add specific known translations here
        common_translations = {
            'french': {
                'glass': 'verre',
                'paper': 'papier',
                'water': 'eau',
                'book': 'livre',
                'house': 'maison',
                'car': 'voiture',
                'dog': 'chien',
                'cat': 'chat',
                'computer': 'ordinateur',
                'phone': 'téléphone',
                'device': 'appareil'
            },
            'spanish': {
                'glass': 'vaso',
                'paper': 'papel',
                'water': 'agua',
                'book': 'libro',
                'house': 'casa',
                'car': 'coche',
                'dog': 'perro',
                'cat': 'gato',
                'computer': 'computadora',
                'phone': 'teléfono',
                'device': 'dispositivo'
            },
            # Additional language dictionaries omitted for brevity
        }

        # Check if we have a known translation
        if language in common_translations and word.lower() in common_translations[language]:
            translation = common_translations[language][word.lower()]

            # Add the known translation to the top of the memory text
            known_translation = f"IMPORTANT FACTUAL INFORMATION:\n- The word \"{word}\" in {language} is \"{translation}\".\n\n"

            if memory_text:
                return known_translation + memory_text
            else:
                return known_translation

        return memory_text

    def _post_process_transliteration(self, query, memory_text):
        """Apply post-processing for transliteration queries"""
        # Extract word to transliterate
        word_match = re.search(r'"([^"]+)"', query)
        if not word_match:
            word_match = re.search(r'transliterate\s+(\w+)', query)

        if not word_match:
            return memory_text

        word = word_match.group(1)

        # Look for target script (cyrillic/latin)
        script_match = re.search(r'(latin|cyrillic)', query.lower())
        if not script_match:
            return memory_text

        script = script_match.group(1).lower()

        # Return only the transliteration without additional context
        # to avoid feedback loops
        return "IMPORTANT FACTUAL INFORMATION:\n- Transliteration information."

    def clean_memory_from_incorrect_translations(self, user_id: str, word: str, language: str) -> None:
        """
        Remove incorrect translation memories for a specific word and language.
        Adapted to work with the refactored VectorStore class.

        Args:
            user_id: The user ID
            word: The word that was incorrectly translated
            language: The target language
        """
        # Get the correct translation if available
        correct_translation = self._get_known_translation(word, language)

        if not correct_translation:
            # If we don't know the correct translation, we can't reliably clean up
            return

        # Get user's vector store
        store = self.memory_manager._get_user_store(user_id)

        # Since we can't directly access all documents anymore, we need a different approach
        # We'll search for relevant translation memories and filter them

        # Create a query specifically for this translation
        translation_query = f"translate {word} to {language}"
        query_embedding = self.memory_manager.generate_embedding(translation_query)

        # Search with a low threshold to find all potential matches
        results = store.search(
            query_embedding,
            top_k=20,  # Get a good number of potential matches
            min_similarity=0.2,  # Low threshold to catch more matches
            apply_sharpening=False  # Don't apply sharpening for this administrative task
        )

        # Find indices of memories to remove
        indices_to_remove = []
        for result in results:
            # Check if this is a translation memory for our word and language
            doc_text = result['text'].lower()
            idx = result['index']

            if ((f'translate "{word}"' in doc_text or
                f'say "{word}"' in doc_text or
                f'how to say "{word}"' in doc_text) and
                language.lower() in doc_text):

                # Check if it has an incorrect translation
                # Skip if it has the correct translation
                if correct_translation.lower() not in doc_text:
                    indices_to_remove.append(idx)

        # Remove the memories
        for idx in sorted(indices_to_remove, reverse=True):  # Remove from highest index to lowest
            store.remove(idx)

        if indices_to_remove:
            print(f"[Memory] Removed {len(indices_to_remove)} incorrect translation memories for '{word}' in {language}")

    def _get_known_translation(self, word: str, language: str) -> Optional[str]:
        """Get a known translation from the dictionary if available"""
        # This is to avoid duplicating the common_translations dictionary
        if not hasattr(self, '_common_translations'):
            self._common_translations = {
                'french': {
                    'glass': 'verre',
                    'paper': 'papier',
                    'water': 'eau',
                    'book': 'livre',
                    'house': 'maison',
                    'car': 'voiture',
                    'dog': 'chien',
                    'cat': 'chat',
                    'computer': 'ordinateur',
                    'phone': 'téléphone',
                    'device': 'appareil'
                },
                'spanish': {
                    'glass': 'vaso',
                    'paper': 'papel',
                    'water': 'agua',
                    'book': 'libro',
                    'house': 'casa',
                    'car': 'coche',
                    'dog': 'perro',
                    'cat': 'gato',
                    'computer': 'computadora',
                    'phone': 'teléfono',
                    'device': 'dispositivo'
                },
                # Additional language dictionaries omitted for brevity
            }

        language = language.lower()
        word = word.lower()

        if language in self._common_translations and word in self._common_translations[language]:
            return self._common_translations[language][word]

        return None

    def _clean_arithmetic_memories(self, memory_text: str, expression: str, correct_result: float) -> str:
        """Clean up arithmetic memories, correcting or removing incorrect calculations"""
        if not memory_text:
            return ""

        lines = memory_text.split('\n')
        cleaned_lines = []

        # Remove expression from memory to allow injection to take precedence
        for line in lines:
            # Skip lines that contain both the expression and a result
            if expression in line.replace(' ', '') and str(correct_result) not in line:
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _clean_translation_memories(self, memory_text, word, language, correct_translation):
        """Clean up translation memories, removing incorrect or duplicate translations"""
        if not memory_text:
            return ""

        lines = memory_text.split('\n')
        cleaned_lines = []

        # Keep track of seen transliterations to avoid duplication
        seen_transliterations = set()

        # Add the known correct transliteration only once
        correct_pattern = f"transliteration of {word.lower()}"
        seen_transliterations.add(correct_pattern)

        # Check each line
        for line in lines:
            # Skip if it's about transliteration and we've seen a similar pattern
            if "transliteration" in line.lower():
                # Skip all transliteration mentions after we've seen the correct one
                continue

            # For other lines, check if they're duplicates of what we've seen
            simplified = re.sub(r'[^\w\s]', '', line.lower())
            if simplified and simplified not in seen_transliterations:
                seen_transliterations.add(simplified)
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _clean_duplicate_memories(self, memory_text: str) -> str:
        """Remove duplicate memories from text."""
        return clean_duplicate_memories(memory_text)