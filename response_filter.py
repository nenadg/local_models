import numpy as np
import re
from typing import Dict, Optional, Tuple, List, Union, Any
import time

class ResponseFilter:
    """
    Comprehensive filter for LLM responses that integrates multiple approaches:
    1. Confidence metrics normalization using techniques from EnhancedHeatmap
    2. Context window overflow detection with specific patterns
    3. Content quality assessment based on coherence and structure
    4. Adaptive thresholds that adjust based on the query domain and response length
    """

    def __init__(
        self,
        confidence_threshold: float = 0.55,
        entropy_threshold: float = 2.5,
        perplexity_threshold: float = 15.0,
        fallback_messages: Optional[List[str]] = None,
        continuation_phrases: Optional[List[str]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        sharpening_factor: float = 0.3,
        window_size: int = 5,
        question_classifier=None,
        debug_mode: bool = False
    ):
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.perplexity_threshold = perplexity_threshold
        self.user_context = user_context or {}
        self.sharpening_factor = sharpening_factor
        self.window_size = window_size
        self.question_classifier = question_classifier
        self.debug_mode = debug_mode

        # Initialize tracking variables
        self.context_history = []
        self.confidence_window = []
        self.last_content_assessment = {}
        self.enable_aggressive_filtering = True
        self.token_count_threshold = 30  # Minimum tokens before aggressive filtering

        # Default fallback messages when low confidence is detected
        self.fallback_messages = fallback_messages or [
            "I don't have sufficient information to answer this question reliably.",
            "I'm uncertain about this topic and don't want to provide potentially incorrect information.",
            "I'm not confident in my ability to provide a good answer to this question.",
            "I'm not able to provide a satisfactory answer about this topic.",
        ]

        # Phrases that indicate user wants to continue despite uncertainty
        self.continuation_phrases = continuation_phrases or [
            "please continue",
            "continue anyway",
            "speculate anyway",
            "give it your best guess",
            "go ahead anyway",
            "try anyway",
            "speculate",
            "just guess",
            "make something up",
            "proceed anyway",
            "please try",
            "best estimate"
        ]

        # Pattern detection thresholds and configurations
        self.pattern_config = {
            # Content quality patterns
            'repeated_phrases': {
                'threshold': 2,        # Max identical phrases allowed
                'min_length': 5,       # Minimum length of phrase to check
                'max_distance': 100,   # Max tokens between repeats
            },
            'repeated_characters': {
                'threshold': 6,        # Max repeated characters
                'ignore_in_code': True # Ignore in code blocks
            },
            'inconsistency': {
                'contradictions': True, # Look for self-contradictions
            },

            # Context overflow indicators
            'overflow_patterns': [
                # File listing patterns
                r'(-rw-r--r--.*?root.*?\d+[KMG]?)',
                r'(drwx.*?root.*?\d+[KMG]?)',

                # Truncated/corrupted patterns
                r'(```\s*[a-z]*\s*```\s*){2,}',    # Empty code blocks
                r'(\$\s*\$\s*\$\s*\$+)',           # Repeated $ signs
                r'(\$\s*n\s*\$\s*\$\s*n+)',        # $n$$ patterns
                r'(r-[-rwxs]{8,})',                # File permissions fragments

                # Command output corruptions
                r'(total\s+\d+K\s*$)',
                r'(drwx[-rwxs]{8,}\s+\d+\s+\w+\s+\w+)',

                # Repetitive garbage
                r'(txt,txt,txt)',
                r'(,,+)',                         # Multiple commas
                r'(\.\.+\w+\.\.+\w+\.\.+)',       # Word.Word patterns
                r'(\w+,){5,}',                    # Comma-separated words
                r'(and,and|or,or|the,the)',       # Repeated connecting words

                # Special token leakage
                r'(\<\|[a-z]+\|\>)',              # Special tokens like <|user|>
            ],

            # Coherence assessment parameters
            'coherence': {
                'min_sentence_length': 5,         # Minimum words for a valid sentence
                'max_malformed_ratio': 0.3,       # Max ratio of malformed sentences allowed
                'subject_verb_check': True,       # Check for subject-verb structure
            }
        }

    def log(self, message: str) -> None:
        """Log messages if debug mode is enabled."""
        if self.debug_mode:
            print(f"[ResponseFilter] {message}")

    def detect_repetitive_patterns(self, response: str) -> dict:
        """
        Detect problematic repetitive patterns in the response.

        Args:
            response: Generated text response

        Returns:
            Dictionary with pattern detection results
        """
        results = {
            'has_repetitive_patterns': False,
            'detected_patterns': []
        }

        # Check for duplicate lines
        lines = response.split('\n')
        line_counts = {}
        for line in lines:
            clean_line = line.strip()
            if clean_line and len(clean_line) > 5:  # Skip short or empty lines
                if clean_line in line_counts:
                    line_counts[clean_line] += 1
                else:
                    line_counts[clean_line] = 1

        # Find lines that repeat too much
        duplicate_lines = [line for line, count in line_counts.items()
                          if count >= 3]  # Lines repeated 3+ times

        if duplicate_lines:
            results['has_repetitive_patterns'] = True
            results['detected_patterns'].append('duplicate_lines')
            results['duplicate_lines'] = duplicate_lines

        # Check for repetitive characters
        char_repeat_patterns = re.findall(r'(.)\1{8,}', response)  # 8+ repeated chars
        if char_repeat_patterns:
            results['has_repetitive_patterns'] = True
            results['detected_patterns'].append('repeated_characters')
            results['repeated_characters'] = char_repeat_patterns

        # Check for garbage patterns that frequently appear in broken responses
        garbage_patterns = [
            r'(```\s*[a-z]*\s*```\s*){2,}',    # Multiple empty code blocks
            r'(\$\s*\$\s*\$\s*\$+)',           # Repeated $ signs
            r'(\$\s*n\s*\$\s*\$\s*n+)',        # $n$$ patterns
            r'(r-[-rwxs]{8,})',                # File permissions fragments
            r'(txt,txt,txt)',                  # Repeated "txt" fragments
            r'(\w+,){5,}',                     # Comma-separated words
            r'(and,and|or,or|the,the)',        # Repeated connecting words
        ]

        for i, pattern in enumerate(garbage_patterns):
            matches = re.findall(pattern, response)
            if matches:
                results['has_repetitive_patterns'] = True
                results['detected_patterns'].append(f'garbage_pattern_{i}')
                results['garbage_matches'] = matches
                break

        return results

    def _geometric_mean(self, values: List[float]) -> float:
        """
        Calculate the geometric mean of a list of values.
        This provides better smoothing for confidence values.

        Args:
            values: List of confidence values

        Returns:
            Geometric mean
        """
        if not values:
            return 0.0

        # Avoid values too close to zero
        safe_values = [max(0.01, v) for v in values]

        # Calculate geometric mean
        return float(np.power(np.prod(safe_values), 1.0 / len(safe_values)))

    def add_confidence(self, confidence: float) -> None:
        """
        Add a confidence value to the sliding window.

        Args:
            confidence: New confidence value
        """
        self.confidence_window.append(confidence)

        # Keep window at specified size
        if len(self.confidence_window) > self.window_size:
            self.confidence_window.pop(0)

    def get_normalized_confidence(self) -> float:
        """
        Calculate normalized confidence using geometric mean of window.

        Returns:
            Normalized confidence value
        """
        if not self.confidence_window:
            return 0.8  # Default high confidence if no values

        return self._geometric_mean(self.confidence_window)

    def normalize_confidence_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Apply advanced normalization to confidence metrics using geometric mean.

        Args:
            metrics: Dictionary of metrics (confidence, perplexity, entropy)

        Returns:
            Dictionary with normalized metrics
        """
        # Add confidence to window
        if "confidence" in metrics:
            self.add_confidence(metrics["confidence"])

        # Get normalized confidence using window
        normalized_confidence = self.get_normalized_confidence()

        # Create normalized metrics
        normalized_metrics = metrics.copy()
        normalized_metrics["normalized_confidence"] = normalized_confidence

        # Apply sharpening if requested
        if self.sharpening_factor > 0:
            normalized_metrics = self.apply_sharpening(normalized_metrics)

        return normalized_metrics

    def apply_sharpening(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Apply non-linear sharpening to metrics.

        Args:
            metrics: Dictionary of metrics to sharpen

        Returns:
            Dictionary with sharpened metrics
        """
        # Skip if no sharpening requested
        if self.sharpening_factor <= 0:
            return metrics

        # Get key metrics
        confidence = metrics.get('normalized_confidence', metrics.get('confidence', 0.5))
        perplexity = metrics.get('perplexity', 1.0)
        entropy = metrics.get('entropy', 0.0)

        # Apply sharpening to each metric
        sharpened = metrics.copy()

        # Sharpen confidence (higher is better)
        if confidence > 0.5:
            # Boost high confidence
            boost = (confidence - 0.5) * self.sharpening_factor * 1.5
            sharpened['sharpened_confidence'] = min(1.0, confidence + boost)
        else:
            # Reduce low confidence
            reduction = (0.5 - confidence) * self.sharpening_factor * 1.5
            sharpened['sharpened_confidence'] = max(0.1, confidence - reduction)

        # Sharpen perplexity (lower is better)
        if perplexity < 5.0:
            # Decrease low perplexity (good)
            reduction = perplexity * self.sharpening_factor * 0.3
            sharpened['sharpened_perplexity'] = max(1.0, perplexity - reduction)
        else:
            # Increase high perplexity (bad)
            boost = (perplexity - 5.0) * self.sharpening_factor * 0.3
            sharpened['sharpened_perplexity'] = perplexity + boost

        # Sharpen entropy (lower is better)
        if entropy < 1.0:
            # Decrease low entropy (good)
            reduction = entropy * self.sharpening_factor * 0.4
            sharpened['sharpened_entropy'] = max(0.0, entropy - reduction)
        else:
            # Increase high entropy (bad)
            boost = (entropy - 1.0) * self.sharpening_factor * 0.4
            sharpened['sharpened_entropy'] = entropy + boost

        return sharpened

    def detect_context_overflow(self, response: str) -> Dict[str, Any]:
        """
        Detect signs of context window overflow.

        Args:
            response: Generated text response

        Returns:
            Dictionary with overflow detection results
        """
        results = {
            'is_overflow': False,
            'overflow_patterns': [],
            'pattern_matches': [],
        }

        # Check for overflow patterns
        for i, pattern in enumerate(self.pattern_config['overflow_patterns']):
            matches = re.findall(pattern, response)
            if matches:
                results['is_overflow'] = True
                results['overflow_patterns'].append(f"pattern_{i}")
                results['pattern_matches'].extend(matches[:3])  # Just store the first few matches

                self.log(f"Detected overflow pattern: {pattern[:30]}... with {len(matches)} matches")

        # Check for repetitive phrase patterns
        phrases = self._extract_repeated_phrases(response)
        if phrases and len(phrases) >= self.pattern_config['repeated_phrases']['threshold']:
            results['is_overflow'] = True
            results['overflow_patterns'].append('repetitive_phrases')
            results['repetitive_phrases'] = phrases

            self.log(f"Detected {len(phrases)} repetitive phrases: {phrases[:2]}")

        return results

    def _extract_repeated_phrases(self, text: str) -> List[str]:
        """
        Extract repeated meaningful phrases in the text.

        Args:
            text: Text to analyze

        Returns:
            List of repeated phrases
        """
        repeated = []

        # Get minimum length for phrases to check
        min_len = self.pattern_config['repeated_phrases']['min_length']

        # Split into words
        words = re.findall(r'\b\w+\b', text.lower())

        # Check for repeated phrases (n-grams)
        for n in range(min_len, min(15, len(words) // 2)):
            ngrams = {}

            # Build all n-grams
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                if len(ngram.strip()) < min_len * 2:  # Skip very short phrases
                    continue

                if ngram in ngrams:
                    if i - ngrams[ngram][-1] > self.pattern_config['repeated_phrases']['max_distance']:
                        ngrams[ngram].append(i)
                else:
                    ngrams[ngram] = [i]

            # Find repeated n-grams
            for ngram, positions in ngrams.items():
                if len(positions) >= self.pattern_config['repeated_phrases']['threshold']:
                    # Valid repeated phrase
                    repeated.append(ngram)

        return repeated

    def analyze_content_quality(self, response: str) -> Dict[str, Any]:
        """
        Analyze the quality of the generated content.

        Args:
            response: Generated text response

        Returns:
            Dictionary with content quality assessment
        """
        results = {
            'is_low_quality': False,
            'quality_issues': [],
            'quality_score': 1.0,  # Start with perfect score
        }

        # Skip if response is too short
        if len(response.strip()) < 50:
            return results

        # Split text into sentences
        sentences = re.split(r'[.!?]\s+', response)

        # Check sentence quality
        malformed_count = 0

        for sentence in sentences:
            sentence = sentence.strip()

            # Skip very short sentences
            if len(sentence.split()) < self.pattern_config['coherence']['min_sentence_length']:
                continue

            # Check for subject-verb structure if enabled
            if self.pattern_config['coherence']['subject_verb_check']:
                # Very simple check - look for a noun followed by a verb
                # This is a heuristic and not a proper parse
                has_subject_verb = re.search(r'\b[A-Z][a-z]+\b.*?\b[a-z]+s\b|\b[A-Z][a-z]+\b.*?\b[a-z]+ed\b', sentence)

                if not has_subject_verb:
                    malformed_count += 1

        # Calculate malformed ratio
        if sentences:
            malformed_ratio = malformed_count / len(sentences)

            if malformed_ratio > self.pattern_config['coherence']['max_malformed_ratio']:
                results['is_low_quality'] = True
                results['quality_issues'].append('high_malformed_sentence_ratio')
                results['malformed_ratio'] = malformed_ratio

                # Reduce quality score
                results['quality_score'] -= malformed_ratio

        # Check for abrupt topic shifts
        topics = self._extract_topics(response)

        if len(topics) > 3 and len(sentences) < 10:
            # Too many topics for a short response
            results['is_low_quality'] = True
            results['quality_issues'].append('topic_incoherence')
            results['topics'] = topics

            # Reduce quality score
            results['quality_score'] -= 0.2

        # Check for sentence fragments
        fragments = re.findall(r'\b[A-Z][^.!?]*(?<![.!?])(?:\n|\s{2,})', response)
        if fragments and len(fragments) > len(sentences) * 0.3:
            results['is_low_quality'] = True
            results['quality_issues'].append('sentence_fragments')

            # Reduce quality score
            results['quality_score'] -= 0.2

        # Analyze text structure for deterioration
        structure_result = self._analyze_text_structure(response)
        if structure_result['structure_deterioration']:
            results['is_low_quality'] = True
            results['quality_issues'].append('structure_deterioration')
            results['structure_analysis'] = structure_result

            # Reduce quality score
            results['quality_score'] -= 0.3

        # Ensure quality score stays in range [0, 1]
        results['quality_score'] = max(0.0, min(1.0, results['quality_score']))

        return results

    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract main topics from text.

        Args:
            text: Text to analyze

        Returns:
            List of main topics
        """
        # Simple implementation - just extract noun phrases
        topics = []

        # Use a simplified noun phrase regex pattern
        noun_phrases = re.findall(r'\b(?:[A-Z][a-z]+ )+(?:is|are|was|were|has|have)\b', text)

        # Add the first word of each sentence as potential topic
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in sentences:
            words = sentence.strip().split()
            if words and len(words[0]) > 3 and words[0][0].isupper():
                topics.append(words[0])

        return list(set(topics + noun_phrases))[:5]  # Return at most 5 topics

    def _analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze the structure of the text for signs of deterioration.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with structure analysis
        """
        result = {
            'structure_deterioration': False,
            'section_quality': []
        }

        # Split text into sections (e.g., paragraphs)
        sections = text.split('\n\n')

        # If no clear sections, split by sentences
        if len(sections) <= 1:
            sections = re.split(r'(?<=[.!?])\s+', text)

        # Analyze quality of each section
        prev_quality = 1.0
        degradation_count = 0

        for i, section in enumerate(sections):
            # Skip very short sections
            if len(section.strip()) < 20:
                continue

            # Calculate simple quality heuristic
            words = re.findall(r'\b\w+\b', section)
            unique_ratio = len(set(words)) / max(1, len(words))

            # Check for structure indicators
            has_punctuation = '.' in section or ',' in section
            has_capitalization = bool(re.search(r'\b[A-Z][a-z]+\b', section))

            # Combined quality score
            quality = (unique_ratio * 0.7) + (0.15 if has_punctuation else 0) + (0.15 if has_capitalization else 0)

            result['section_quality'].append(quality)

            # Check for degradation between sections
            if i > 0 and quality < prev_quality * 0.7:  # More than 30% drop
                degradation_count += 1

            prev_quality = quality

        # Detect deterioration if multiple sections show degradation
        if degradation_count >= 2:
            result['structure_deterioration'] = True

        # Check if quality scores show a clear downward trend
        if len(result['section_quality']) >= 3:
            first_third = np.mean(result['section_quality'][:len(result['section_quality'])//3])
            last_third = np.mean(result['section_quality'][-len(result['section_quality'])//3:])

            if last_third < first_third * 0.6:  # More than 40% drop
                result['structure_deterioration'] = True
                result['quality_decline'] = first_third - last_third

        return result

    def get_domain_thresholds(self, query: str) -> Dict[str, float]:
        """
        Get domain-specific thresholds based on the query.

        Args:
            query: User query

        Returns:
            Dictionary with adjusted thresholds
        """
        # Default thresholds
        thresholds = {
            'confidence': self.confidence_threshold,
            'entropy': self.entropy_threshold,
            'perplexity': self.perplexity_threshold
        }

        # Get domain settings if question classifier available
        if self.question_classifier and query:
            try:
                domain_settings = self.question_classifier.get_domain_settings(query)
                domain = domain_settings.get('domain', 'unknown')

                # Apply domain-specific thresholds
                if domain == 'arithmetic':
                    # Stricter for math
                    thresholds['confidence'] = 0.75
                    thresholds['entropy'] = 1.8
                    thresholds['perplexity'] = 8.0
                elif domain == 'factual':
                    # Stricter for factual knowledge
                    thresholds['confidence'] = 0.70
                    thresholds['entropy'] = 2.0
                    thresholds['perplexity'] = 10.0
                elif domain == 'translation':
                    # Slightly less strict for translations
                    thresholds['confidence'] = 0.60
                    thresholds['entropy'] = 2.8
                    thresholds['perplexity'] = 12.0
                elif domain == 'conceptual':
                    # Balanced for conceptual explanations
                    thresholds['confidence'] = 0.65
                    thresholds['entropy'] = 2.3
                    thresholds['perplexity'] = 12.0
                elif domain == 'procedural':
                    # Balanced for procedures
                    thresholds['confidence'] = 0.65
                    thresholds['entropy'] = 2.3
                    thresholds['perplexity'] = 12.0
                else:
                    # Unknown domain - use defaults with slight adjustment
                    thresholds['confidence'] = 0.68

                self.log(f"Using domain-specific thresholds for domain: {domain}")
            except Exception as e:
                self.log(f"Error getting domain settings: {e}")

        # Apply pattern-based adjustments based on query
        if query:
            # Technical queries may have more specialized vocabulary
            if re.search(r'\b(?:API|code|programming|function|algorithm|technical)\b', query, re.IGNORECASE):
                thresholds['entropy'] += 0.3  # Allow higher entropy

            # Factual queries need higher confidence
            if re.search(r'\bwhat is\b|\bwho is\b|\bwhen did\b|\bwhere is\b', query, re.IGNORECASE):
                thresholds['confidence'] += 0.05

            # Mathematical queries need higher confidence
            if re.search(r'\bcalculate\b|\bcompute\b|\bsolve\b|\bhow many\b', query, re.IGNORECASE):
                thresholds['confidence'] += 0.05

        return thresholds

    def should_filter(
        self,
        metrics: Dict[str, float],
        response: str,
        query: Optional[str] = None,
        tokens_generated: int = 0
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Determine if a response should be filtered based on comprehensive assessment.

        Args:
            metrics: Dictionary containing confidence metrics
            response: The generated response text
            query: Original query for context
            tokens_generated: Number of tokens generated so far

        Returns:
            Tuple of (should_filter, reason, details)
        """
        # Skip aggressive filtering for very short responses
        if tokens_generated < self.token_count_threshold and self.enable_aggressive_filtering:
            # Skip comprehensive filtering for short responses to allow model to build confidence
            normalized_metrics = self.normalize_confidence_metrics(metrics)
            confidence = normalized_metrics.get('normalized_confidence',
                                              normalized_metrics.get('confidence', 0.5))

            # Only filter extremely low confidence early output
            if confidence < 0.3:
                return True, "extremely_low_confidence", {'confidence': confidence}

            # Otherwise let it continue
            return False, "acceptable", {'confidence': confidence}

        # Apply normalization to confidence metrics
        normalized_metrics = self.normalize_confidence_metrics(metrics)

        # Get normalized confidence values
        confidence = normalized_metrics.get('sharpened_confidence',
                                           normalized_metrics.get('normalized_confidence',
                                                                 normalized_metrics.get('confidence', 0.5)))

        entropy = float(normalized_metrics.get('sharpened_entropy',
                                             normalized_metrics.get('entropy', 2.0)))

        perplexity = float(normalized_metrics.get('sharpened_perplexity',
                                                normalized_metrics.get('perplexity', 10.0)))

        # Get domain-specific thresholds
        thresholds = self.get_domain_thresholds(query)

        # Detect context overflow first (highest priority)
        overflow_results = self.detect_context_overflow(response)
        if overflow_results['is_overflow']:
            return True, f"context_overflow_{overflow_results['overflow_patterns']}", overflow_results

        # Analyze content quality
        quality_results = self.analyze_content_quality(response)
        if quality_results['is_low_quality'] and quality_results['quality_score'] < 0.5:
            return True, f"low_quality_{','.join(quality_results['quality_issues'])}", quality_results

        # Save content assessment for future reference
        self.last_content_assessment = quality_results

        # Check confidence against threshold
        if confidence < thresholds['confidence']:
            return True, "low_confidence", {'confidence': confidence, 'threshold': thresholds['confidence']}

        # Check entropy against threshold
        if entropy > thresholds['entropy']:
            return True, "high_entropy", {'entropy': entropy, 'threshold': thresholds['entropy']}

        # Check perplexity against threshold
        if perplexity > thresholds['perplexity']:
            return True, "high_perplexity", {'perplexity': perplexity, 'threshold': thresholds['perplexity']}

        # Calculate a combined uncertainty score with weighted components
        # - Higher weight on confidence
        # - Medium weight on content quality
        # - Lower weights on entropy and perplexity
        uncertainty_score = (
            (1 - confidence) * 0.5 +
            (1 - quality_results['quality_score']) * 0.3 +
            (entropy / thresholds['entropy']) * 0.1 +
            (perplexity / thresholds['perplexity']) * 0.1
        )

        if uncertainty_score > 0.55:
            return True, "high_uncertainty", {
                'uncertainty_score': uncertainty_score,
                'confidence': confidence,
                'quality_score': quality_results['quality_score'],
                'entropy_ratio': entropy / thresholds['entropy'],
                'perplexity_ratio': perplexity / thresholds['perplexity']
            }

        # If we get here, the response passes all checks
        return False, "acceptable", {
            'confidence': confidence,
            'quality_score': quality_results['quality_score'],
            'uncertainty_score': uncertainty_score
        }

    def check_override_instruction(self, query: str) -> bool:
        """
        Check if the user's query contains an instruction to continue
        despite uncertainty.

        Args:
            query: The user's input query

        Returns:
            Boolean indicating whether to override the filter
        """
        if not query:
            return False

        query_lower = query.lower()

        # Check for continuation phrases
        for phrase in self.continuation_phrases:
            if phrase in query_lower:
                return True

        return False

    def update_user_context(self, query: str, response: str, metrics: Dict[str, float]):
        """
        Update user context with information about the query and response.

        Args:
            query: The user's input query
            response: The model's response
            metrics: Confidence metrics for the response
        """
        # Apply normalization
        normalized_metrics = self.normalize_confidence_metrics(metrics)

        # Get current time
        current_time = time.time()

        # Check if filtering should occur
        should_filter, reason, details = self.should_filter(normalized_metrics, response, query)

        # Update conversation history
        self.context_history.append({
            'timestamp': current_time,
            'query': query,
            'response_length': len(response),
            'confidence': normalized_metrics.get('normalized_confidence', 0.5),
            'should_filter': should_filter,
            'reason': reason,
            'details': details
        })

        # Keep history limited to last 5 exchanges
        if len(self.context_history) > 5:
            self.context_history.pop(0)

        # Track uncertain queries
        if should_filter:
            self.user_context["last_uncertain_query"] = query
            self.user_context["last_uncertain_reason"] = reason
            self.user_context["last_uncertain_details"] = details
        else:
            # Clear if we have a confident response
            self.user_context.pop("last_uncertain_query", None)
            self.user_context.pop("last_uncertain_reason", None)
            self.user_context.pop("last_uncertain_details", None)

        # Track degradation pattern
        if len(self.context_history) >= 3:
            # Check for declining confidence trend
            confidences = [entry['confidence'] for entry in self.context_history[-3:]]
            if confidences[0] > confidences[1] > confidences[2]:
                self.user_context["declining_confidence"] = True

                # If confidence is declining, we should be more aggressive with filtering
                self.enable_aggressive_filtering = True
            else:
                self.user_context.pop("declining_confidence", None)

                # If confidence is stable or improving, we can be less aggressive
                self.enable_aggressive_filtering = False

    def extract_mcp_commands(self, response: str) -> Tuple[str, List[str]]:
        """
        Extract MCP commands from response.

        Args:
            response: Original model response

        Returns:
            Tuple of (cleaned_response, mcp_commands)
        """
        mcp_commands = []
        cleaned_response = response

        # Simple regex-free extraction of MCP commands
        lines = response.split('\n')
        filtered_lines = []
        in_mcp_block = False

        for line in lines:
            if ">>>" in line and "FILE:" in line:
                in_mcp_block = True
                mcp_commands.append(line)
            elif "<<<" in line and in_mcp_block:
                in_mcp_block = False
                mcp_commands.append(line)
            elif in_mcp_block:
                mcp_commands.append(line)
            else:
                filtered_lines.append(line)

        # Rebuild cleaned response
        if mcp_commands:
            cleaned_response = '\n'.join(filtered_lines)

        return cleaned_response, mcp_commands

    def filter_response(
        self,
        response: str,
        metrics: Dict[str, float],
        query: Optional[str] = None,
        preserve_mcp: bool = True,
        allow_override: bool = True,
        tokens_generated: int = 0
    ) -> str:
        """
        Filter a response based on comprehensive assessment.

        Args:
            response: Original model response
            metrics: Dictionary containing confidence metrics
            query: Original query for contextual fallbacks
            preserve_mcp: Whether to preserve MCP commands from the original response
            allow_override: Whether to allow the user to override filtering
            tokens_generated: Number of tokens generated

        Returns:
            Filtered response or original response if confidence is high enough
        """
        # Extract any MCP commands if needed
        cleaned_response, mcp_commands = self.extract_mcp_commands(response) if preserve_mcp else (response, [])

        # Apply comprehensive assessment
        should_filter, reason, details = self.should_filter(
            metrics, cleaned_response, query, tokens_generated
        )

        # Skip filtering if checks pass
        if not should_filter:
            return response

        # Log filtering reason if debug enabled
        self.log(f"Filtering response due to {reason}: {details}")

        # Check for override instruction
        if allow_override and query and self.check_override_instruction(query):
            # User has explicitly requested to continue despite uncertainty
            # Add a brief uncertainty disclaimer
            disclaimer = "Note: I'm not entirely confident about this information, but as requested, I'll provide my best attempt:\n\n"
            return disclaimer + response

        # Check if this is a follow-up to a previously uncertain query
        if "last_uncertain_query" in self.user_context and allow_override and query:
            if self.check_override_instruction(query):
                # This is an override for the previous uncertain query
                disclaimer = "As requested, I'll try to answer despite my uncertainty:\n\n"
                return disclaimer + response

        # If we get here, we need to filter the response

        # Choose a fallback message based on reason
        fallback = self._get_fallback_for_reason(reason, query, details)

        # Combine fallback with preserved MCP commands
        if preserve_mcp and mcp_commands:
            return fallback + "\n\n" + "\n".join(mcp_commands)

        return fallback

    def _get_fallback_for_reason(self, reason: str, query: Optional[str], details: Dict[str, Any]) -> str:
        """
        Get a fallback message tailored to the filtering reason.

        Args:
            reason: Reason for filtering
            query: Original query
            details: Details about the filtering decision

        Returns:
            Tailored fallback message
        """
        import random

        # Choose base fallback message
        fallback = random.choice(self.fallback_messages)

        # Tailor message based on reason
        if reason.startswith("context_overflow"):
            fallback = "I started generating an answer but encountered issues maintaining coherence. " + fallback
        elif reason.startswith("low_quality"):
            fallback = "I'm having trouble formulating a clear response to this question. " + fallback
        elif reason == "low_confidence":
            # Use base fallback
            pass
        elif reason == "high_entropy" or reason == "high_perplexity":
            fallback = "This topic is challenging for me to explain coherently. " + fallback
        elif reason == "high_uncertainty":
            # Use base fallback
            pass

        # Add topic reference if available
        if query:
            # Extract topic from query
            topic_match = re.search(r'(?:about|on|regarding)\s+(["\']?[\w\s]+["\']?)', query)
            if topic_match:
                topic = topic_match.group(1)
                # Add topic if not already in fallback
                if topic not in fallback:
                    fallback = fallback.replace("this topic", f"'{topic}'")
                    fallback = fallback.replace("this subject", f"'{topic}'")
                    fallback = fallback.replace("this question", f"this question about '{topic}'")

            # Look for specific capabilities being asked about
            if "file" in query.lower() or "table" in query.lower() or "csv" in query.lower():
                fallback += " I should be able to help with files and tables, so you may want to retry your question."

        return fallback

    def should_stream_fallback(
        self,
        metrics: Dict[str, float],
        response_so_far: str,
        query: str,
        tokens_generated: int
    ) -> bool:
        """
        Determine if we should stream a fallback message based on current generation.

        Args:
            metrics: Dictionary containing confidence metrics
            response_so_far: Partial response generated so far
            query: User query for context
            tokens_generated: Number of tokens generated so far

        Returns:
            Boolean indicating whether to stream a fallback
        """
        # Apply comprehensive assessment
        should_filter, reason, details = self.should_filter(
            metrics, response_so_far, query, tokens_generated
        )

        # Early check for overflow - if we detect this, stop right away
        if tokens_generated > 30 and reason.startswith("context_overflow"):
            self.log(f"Stopping generation due to context overflow after {tokens_generated} tokens")
            return True

        # For other issues, wait for a minimum number of tokens
        if tokens_generated < self.token_count_threshold:
            return False

        # Check if we should filter
        if not should_filter:
            return False

        # Check for override instruction
        if query and self.check_override_instruction(query):
            return False

        # Check if this is a follow-up with override
        if "last_uncertain_query" in self.user_context and query:
            if self.check_override_instruction(query):
                return False

        # If we reach here, we should stream a fallback
        return True

    def get_streamable_fallback(
        self,
        query: Optional[str] = None,
        reason: str = "unknown",
        details: Optional[Dict] = None
    ) -> str:
        """
        Get a fallback message that can be streamed.

        Args:
            query: Original query for context
            reason: Reason for the fallback
            details: Additional details about the issue

        Returns:
            Fallback message as a string
        """
        # Use the tailored fallback message function
        return self._get_fallback_for_reason(reason, query, details or {})

    def set_thresholds(
        self,
        confidence: Optional[float] = None,
        entropy: Optional[float] = None,
        perplexity: Optional[float] = None
    ) -> None:
        """Update the thresholds used for filtering."""
        if confidence is not None:
            self.confidence_threshold = confidence
        if entropy is not None:
            self.entropy_threshold = entropy
        if perplexity is not None:
            self.perplexity_threshold = perplexity

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information about the filter's state.

        Returns:
            Dictionary with debug information
        """
        return {
            'confidence_window': self.confidence_window.copy(),
            'context_history': self.context_history.copy(),
            'last_content_assessment': self.last_content_assessment.copy(),
            'enable_aggressive_filtering': self.enable_aggressive_filtering,
            'thresholds': {
                'confidence': self.confidence_threshold,
                'entropy': self.entropy_threshold,
                'perplexity': self.perplexity_threshold
            }
        }