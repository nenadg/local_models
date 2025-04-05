import re
import numpy as np
from typing import List, Dict, Any, Optional

class WeightedMemoryIntegrator:
    """
    Enhances memory retrieval and integration with domain-aware weighting
    to properly balance memory vs. model knowledge.
    """
    
    def __init__(self, memory_manager, question_classifier=None):
        """
        Initialize the weighted memory integrator.
        
        Args:
            memory_manager: The existing memory manager instance
            question_classifier: Optional classifier for domain detection
        """
        self.memory_manager = memory_manager
        self.question_classifier = question_classifier
        
        # Domain-specific post-processors
        self.post_processors = {
            'arithmetic': self._post_process_arithmetic,
            'translation': self._post_process_translation
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
        # Apply the domain-specific sharpening factor
        original_factor = self.memory_manager.sharpening_factor
        try:
            # Temporarily set the sharpening factor for this retrieval
            self.memory_manager.set_sharpening_factor(settings['sharpening_factor'])
            
            # Get memories with domain-specific count
            retrieve_count = settings.get('retrieval_count', 8)
            memory_text = self.memory_manager.retrieve_relevant_memories(
                user_id,
                query,
                top_k=retrieve_count,
                apply_sharpening=self.memory_manager.sharpening_enabled
            )
            
            # Check for domain-specific memory injection
            memory_text = self._inject_domain_knowledge(query, memory_text, settings)
            
            return memory_text
            
        finally:
            # Restore the original sharpening factor
            self.memory_manager.set_sharpening_factor(original_factor)
            
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
                        "default_user", word, language)

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
        """
        Extract an arithmetic expression from a query.
        
        Args:
            query: The query string
            
        Returns:
            Extracted expression or None
        """
        # Look for patterns like "what is 5 + 3" or "calculate 10 - 7"
        # or just "5 + 3"
        patterns = [
            r'(\d+\s*[\+\-\*\/]\s*\d+)',
            r'(?:what is|calculate|compute|evaluate|result of).*?(\d+\s*[\+\-\*\/]\s*\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Get the expression and clean it up
                expr = match.group(1).strip()
                # Replace any unicode math symbols with Python operators
                expr = expr.replace('×', '*').replace('÷', '/')
                # Remove any spaces
                expr = expr.replace(' ', '')
                return expr
                
        return None
        
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
            'german': {
                'glass': 'Glas',
                'paper': 'Papier',
                'water': 'Wasser',
                'book': 'Buch',
                'house': 'Haus',
                'car': 'Auto',
                'dog': 'Hund',
                'cat': 'Katze',
                'computer': 'Computer',
                'phone': 'Telefon',
                'device': 'Gerät'
            },
            'italian': {
                'glass': 'bicchiere',
                'paper': 'carta',
                'water': 'acqua',
                'book': 'libro',
                'house': 'casa',
                'car': 'macchina',
                'dog': 'cane',
                'cat': 'gatto',
                'computer': 'computer',
                'phone': 'telefono',
                'device': 'dispositivo'
            },
            'dutch': {
                'glass': 'glas',
                'paper': 'papier',
                'water': 'water',
                'book': 'boek',
                'house': 'huis',
                'car': 'auto',
                'dog': 'hond',
                'cat': 'kat',
                'computer': 'computer',
                'phone': 'telefoon',
                'device': 'apparaat'
            },
            'polish': {
                'glass': 'szklanka',
                'paper': 'papier',
                'water': 'woda',
                'book': 'książka',
                'house': 'dom',
                'car': 'samochód',
                'dog': 'pies',
                'cat': 'kot',
                'computer': 'komputer',
                'phone': 'telefon',
                'device': 'urządzenie'
            },
            'serbian': {
                'glass': 'чаша',
                'paper': 'папир',
                'water': 'вода',
                'book': 'књига',
                'house': 'кућа',
                'car': 'ауто',
                'dog': 'пас',
                'cat': 'мачка',
                'computer': 'рачунар',
                'phone': 'телефон', 
                'device': 'уређај'
            }
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

    def clean_memory_from_incorrect_translations(self, user_id: str, word: str, language: str) -> None:
        """
        Remove incorrect translation memories for a specific word and language.

        Args:
            user_id: The user ID
            word: The word that was incorrectly translated
            language: The target language
        """
        # Get the correct translation if available
        correct_translation = None
        language_lower = language.lower()

        # Check our dictionary of known translations
        if hasattr(self, 'post_processors') and 'translation' in self.post_processors:
            common_translations = getattr(self, '_common_translations', None)
            if not common_translations:
                # Extract common_translations from _post_process_translation method
                import inspect
                source = inspect.getsource(self._post_process_translation)
                # Parse the source to find common_translations
                if 'common_translations' in source:
                    try:
                        # This is a bit hacky but avoids duplicating the dictionary
                        exec(source.split('common_translations = {')[1].split('}')[0] + '}',
                             globals(), locals())
                        common_translations = locals().get('common_translations')
                    except:
                        common_translations = {}

            if common_translations and language_lower in common_translations:
                word_lower = word.lower()
                if word_lower in common_translations[language_lower]:
                    correct_translation = common_translations[language_lower][word_lower]

        if not correct_translation:
            # If we don't know the correct translation, we can't reliably clean up
            return

        # Get all memories
        store = self.memory_manager._get_user_store(user_id)
        if not store or not hasattr(store, 'documents'):
            return

        # Find indices of memories to remove
        indices_to_remove = []
        for i, doc in enumerate(store.documents):
            # Check if this is a translation memory for our word and language
            if (f'translate "{word}"' in doc.lower() or
                f'say "{word}"' in doc.lower() or
                f'how to say "{word}"' in doc.lower()):

                if language_lower in doc.lower():
                    # Check if it has an incorrect translation
                    # Skip if it has the correct translation
                    if correct_translation.lower() not in doc.lower():
                        indices_to_remove.append(i)

        # Remove the memories from newest to oldest to maintain correct indices
        indices_to_remove.sort(reverse=True)

        # Delete the memories
        for idx in indices_to_remove:
            if idx < len(store.documents):
                # Remove the document
                store.documents.pop(idx)
                # Remove the metadata
                if idx < len(store.metadata):
                    store.metadata.pop(idx)

        # Save the changes
        if indices_to_remove:
            store.save()
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
                'german': {
                    'glass': 'Glas',
                    'paper': 'Papier',
                    'water': 'Wasser',
                    'book': 'Buch',
                    'house': 'Haus',
                    'car': 'Auto',
                    'dog': 'Hund',
                    'cat': 'Katze',
                    'computer': 'Computer',
                    'phone': 'Telefon',
                    'device': 'Gerät'
                },
                'italian': {
                    'glass': 'bicchiere',
                    'paper': 'carta',
                    'water': 'acqua',
                    'book': 'libro',
                    'house': 'casa',
                    'car': 'macchina',
                    'dog': 'cane',
                    'cat': 'gatto',
                    'computer': 'computer',
                    'phone': 'telefono',
                    'device': 'dispositivo'
                },
                'dutch': {
                    'glass': 'glas',
                    'paper': 'papier',
                    'water': 'water',
                    'book': 'boek',
                    'house': 'huis',
                    'car': 'auto',
                    'dog': 'hond',
                    'cat': 'kat',
                    'computer': 'computer',
                    'phone': 'telefoon',
                    'device': 'apparaat'
                },
                'polish': {
                    'glass': 'szklanka',
                    'paper': 'papier',
                    'water': 'woda',
                    'book': 'książka',
                    'house': 'dom',
                    'car': 'samochód',
                    'dog': 'pies',
                    'cat': 'kot',
                    'computer': 'komputer',
                    'phone': 'telefon',
                    'device': 'urządzenie'
                },
                'serbian': {
                    'glass': 'чаша',
                    'paper': 'папир',
                    'water': 'вода',
                    'book': 'књига',
                    'house': 'кућа',
                    'car': 'ауто',
                    'dog': 'пас',
                    'cat': 'мачка',
                    'computer': 'рачунар',
                    'phone': 'телефон',
                    'device': 'уређај'
                }
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

    def _clean_translation_memories(self, memory_text: str, word: str, language: str, correct_translation: str) -> str:
        """Clean up translation memories, removing incorrect or duplicate translations"""
        if not memory_text:
            return ""

        lines = memory_text.split('\n')
        cleaned_lines = []

        # Check each line
        for line in lines:
            # Skip lines with incorrect translations for this word and language
            if (word.lower() in line.lower() and language.lower() in line.lower() and
                correct_translation.lower() not in line.lower()):
                continue
            cleaned_lines.append(line)

        return self._clean_duplicate_memories('\n'.join(cleaned_lines))

    def _clean_duplicate_memories(self, memory_text: str) -> str:
        """Remove duplicate or highly similar memories from the text"""
        if not memory_text:
            return ""

        # Split into sections
        sections = re.split(r'(IMPORTANT CORRECTIONS|FACTUAL INFORMATION|OTHER RELEVANT INFORMATION):', memory_text)

        if len(sections) <= 1:
            return memory_text

        cleaned_sections = []
        for i in range(0, len(sections), 2):
            if i+1 < len(sections):
                header = sections[i]
                content = sections[i+1]

                # Split content into bullet points
                bullets = content.split('\n- ')

                # Remove duplicates while preserving order
                seen = set()
                unique_bullets = []

                for bullet in bullets:
                    # Create a simplified key for comparison (lowercase, punctuation removed)
                    simplified = re.sub(r'[^\w\s]', '', bullet.lower())
                    simplified = ' '.join(simplified.split())  # Normalize whitespace

                    if simplified and simplified not in seen:
                        seen.add(simplified)
                        unique_bullets.append(bullet)

                # Rebuild content
                cleaned_content = '\n- '.join(unique_bullets)
                if i == 0:  # First section doesn't need the header
                    cleaned_sections.append(header + ":" + cleaned_content)
                else:
                    cleaned_sections.append(header + ":" + cleaned_content)
            else:
                cleaned_sections.append(sections[i])

        return ''.join(cleaned_sections)