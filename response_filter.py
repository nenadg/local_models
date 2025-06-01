import re
import time
import hashlib
import numpy as np
from typing import Dict, Optional, Tuple, List, Union, Any
from datetime import datetime

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
        confidence_threshold: float = 0.45,  # Lowered from 0.7
        entropy_threshold: float = 3.5,      # Increased from 2.5
        perplexity_threshold: float = 25.0,  # Increased from 15.0
        continuation_phrases: Optional[List[str]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        window_size: int = 5,
        question_classifier=None,
        debug_mode: bool = False,
        use_relative_filtering: bool = True,  # New parameter
        pattern_detection_weight: float = 0.6,  # New parameter
        token_count_threshold: int = 60  # Increased from 30
    ):
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.perplexity_threshold = perplexity_threshold
        self.user_context = user_context or {}
        self.window_size = window_size
        self.question_classifier = question_classifier
        self.debug_mode = debug_mode
        self.use_relative_filtering = use_relative_filtering
        self.pattern_detection_weight = pattern_detection_weight
        self.token_count_threshold = token_count_threshold

        # Initialize tracking variables
        self.context_history = []
        self.confidence_window = []
        self.last_content_assessment = {}
        self.enable_aggressive_filtering = False  # Changed from True

        # For confidence baseline tracking
        self.baseline_confidence = 0.5
        self.confidence_values = []
        self.confidence_variance = 0.1

        # OPTIMIZATION: Precompile regex patterns
        self.duplicate_line_pattern = re.compile(r'(.+)\n\1(\n\1)+')
        self.char_repeat_pattern = re.compile(r'(.)\1{8,}')
        self.empty_code_blocks = re.compile(r'(```\s*[a-z]*\s*```\s*){2,}')
        self.repeated_dollar_signs = re.compile(r'(\$\s*\$\s*\$\s*\$+)')
        self.dollar_n_pattern = re.compile(r'(\$\s*n\s*\$\s*\$\s*n+)')
        self.file_permission_fragments = re.compile(r'(r-[-rwxs]{8,})')
        self.txt_fragments = re.compile(r'(txt,txt,txt)')
        self.comma_separated_words = re.compile(r'(\w+,){5,}')
        self.repeated_connectors = re.compile(r'(and,and|or,or|the,the)')

        # OPTIMIZATION: Store the last analysis results
        self._last_analysis = {
            'text_length': 0,
            'has_repetitive_patterns': False,
            'detected_patterns': [],
            'timestamp': 0
        }

        # OPTIMIZATION: Cache for analyzed text segments
        self._analyzed_segments = {}

        # Continuation phrases unchanged...
        self.continuation_phrases = continuation_phrases or [
            "please continue",
            "continue anyway",
            "speculate anyway",
            "continue",
            "go on"
            # Other phrases remain the same...
        ]

        self.overflow_patterns = [
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
        ]

        # Pattern detection thresholds remain mostly the same...
        self.pattern_config = {
            'repeated_phrases': {
                'threshold': 3,
                'min_length': 5,
                'max_distance': 100,
            },
            'repeated_characters': {
                'threshold': 6,        # Max repeated characters
                'ignore_in_code': True # Ignore in code blocks
            },
            'inconsistency': {
                'contradictions': True, # Look for self-contradictions
            },
            'overflow_patterns': self.overflow_patterns,
            # Coherence assessment parameters
            'coherence': {
                'min_sentence_length': 5,         # Minimum words for a valid sentence
                'max_malformed_ratio': 0.3,       # Max ratio of malformed sentences allowed
                'subject_verb_check': True,       # Check for subject-verb structure
            }
        }

    def get_time(self) -> str:
        """Get formatted timestamp for logging."""
        return datetime.now().strftime("[%d/%m/%y %H:%M:%S] [ResponseFilter] ")

    def log(self, message: str) -> None:
        """Log messages if debug mode is enabled."""
        if self.debug_mode:
            print(f"[ResponseFilter] {message}")

    def calculate_aggressive_content_entropy(self, content: str) -> Dict[str, float]:
        """
        Aggressive content-based semantic entropy calculation that's more sensitive to problematic patterns.

        Args:
            content: Text content to analyze

        Returns:
            Dictionary with entropy metrics and pattern indicators
        """
        # Quick check for very short content
        tokens = content.split()
        if len(tokens) < 3:
            return {
                "entropy": 0.5,  # Low entropy for very short content
                "severity": 0.3,
                "pattern_score": 0.0,
                "repetition_score": 0.0
            }

        # Base semantic entropy calculation
        base_entropy = self.calculate_semantic_entropy(content)

        # Aggressive pattern detection
        severity = 0.0
        pattern_score = 0.0

        # 1. Repetition detection - aggressively penalize repeated words/phrases
        word_freq = {}
        bigram_freq = {}

        # Count word and bigram frequencies
        for i, token in enumerate(tokens):
            # Word frequency
            token_lower = token.lower()
            word_freq[token_lower] = word_freq.get(token_lower, 0) + 1

            # Bigram frequency
            if i < len(tokens) - 1:
                bigram = f"{token_lower} {tokens[i+1].lower()}"
                bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1

        # Calculate repetition scores
        repetition_score = 0.0

        # Word repetition penalty (exclude common words)
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}
        for word, count in word_freq.items():
            if word not in common_words and count > 2:
                repetition_score += (count - 2) * 0.1

        # Bigram repetition penalty (stronger signal)
        for bigram, count in bigram_freq.items():
            if count > 1:
                repetition_score += (count - 1) * 0.3

        # 2. Pattern complexity detection using POS tags
        pos_transition_score = 0.0

        try:
            import nltk
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('punkt', quiet=True)

            pos_tags = nltk.pos_tag(tokens)

            # Analyze POS transitions (low diversity = potential issue)
            pos_transitions = {}
            for i in range(len(pos_tags) - 1):
                transition = f"{pos_tags[i][1][:2]}->{pos_tags[i+1][1][:2]}"
                pos_transitions[transition] = pos_transitions.get(transition, 0) + 1

            # Low POS transition diversity indicates repetitive structure
            unique_transitions = len(pos_transitions)
            total_transitions = len(pos_tags) - 1

            if total_transitions > 0:
                transition_diversity = unique_transitions / total_transitions
                # Penalize low diversity
                if transition_diversity < 0.3:
                    pos_transition_score = (0.3 - transition_diversity) * 2.0

            # Check for specific problematic POS patterns
            pos_sequence = [tag[1] for tag in pos_tags]

            # Pattern: Too many consecutive same POS tags
            max_consecutive = 0
            current_consecutive = 1

            for i in range(1, len(pos_sequence)):
                if pos_sequence[i][:2] == pos_sequence[i-1][:2]:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 1

            if max_consecutive > 4:
                pattern_score += (max_consecutive - 4) * 0.2

        except ImportError:
            # Fallback: Use simple pattern detection
            # Look for repeated sentence structures
            sentences = content.split('.')
            sentence_patterns = {}

            for sentence in sentences:
                if sentence.strip():
                    # Create simple pattern from sentence structure
                    words = sentence.strip().split()
                    if len(words) > 2:
                        pattern = f"{len(words)}-{words[0].lower()}-{words[-1].lower()}"
                        sentence_patterns[pattern] = sentence_patterns.get(pattern, 0) + 1

            # Penalize repeated sentence patterns
            for pattern, count in sentence_patterns.items():
                if count > 1:
                    pattern_score += (count - 1) * 0.3

        # 3. Content quality indicators
        quality_issues = 0.0

        # Check for incomplete sentences
        if not content.rstrip().endswith(('.', '!', '?')):
            quality_issues += 0.2

        # Check for very short sentences (potential fragmentation)
        sentences = re.split(r'[.!?]+', content)
        very_short_sentences = sum(1 for s in sentences if 0 < len(s.split()) < 3)
        if len(sentences) > 0:
            short_sentence_ratio = very_short_sentences / len(sentences)
            if short_sentence_ratio > 0.3:
                quality_issues += short_sentence_ratio * 0.5

        # Check for excessive punctuation or special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?]', content)) / max(1, len(content))
        if special_char_ratio > 0.1:
            quality_issues += special_char_ratio * 2.0

        # 4. Entropy adjustments based on content characteristics

        # Adjust base entropy for problematic patterns
        adjusted_entropy = base_entropy

        # Low entropy with high repetition is worse
        if base_entropy < 1.0 and repetition_score > 0.5:
            adjusted_entropy *= 0.7  # Reduce entropy to flag as more problematic
            severity += 0.3

        # High entropy with quality issues indicates garbled content
        if base_entropy > 2.0 and quality_issues > 0.3:
            adjusted_entropy *= 1.5  # Increase entropy to flag as problematic
            severity += 0.4

        # 5. Calculate final severity score
        severity = min(1.0, severity + (
            repetition_score * 0.3 +
            pos_transition_score * 0.2 +
            pattern_score * 0.3 +
            quality_issues * 0.2
        ))

        # 6. Special case detection for known problematic patterns

        # Check for context overflow patterns
        overflow_patterns = [
            r'(-rw-r--r--.*?root.*?\d+[KMG]?)',  # File listings
            r'(\$\s*\$\s*\$\s*\$+)',              # Repeated $ signs
            r'(txt,txt,txt)',                     # Repeated words with commas
            r'(\w+\.\w+\.\w+\.\w+)',              # Repeated dotted patterns
        ]

        for pattern in overflow_patterns:
            if re.search(pattern, content):
                severity = max(severity, 0.8)
                pattern_score += 0.5
                break

        # return adjusted_entropy
        return {
            "entropy": adjusted_entropy,
            "base_entropy": base_entropy,
            "severity": severity,
            "repetition_score": min(1.0, repetition_score),
            "pattern_score": min(1.0, pattern_score),
            "quality_issues": min(1.0, quality_issues),
            "pos_transition_score": pos_transition_score,
            "token_count": len(tokens)
        }

    def calculate_semantic_entropy(self, content: str) -> float:
        """
        Calculate semantic entropy of content using the SE = -∑ₓ P(c|x) log P(c|x) formula.

        Args:
            content: Text content to analyze

        Returns:
            Semantic entropy value (0.0-2.5 typically)
        """
        # Tokenize the content (simple word splitting as fallback)
        tokens = content.split()

        if len(tokens) < 5:
            return 0.0  # Not enough tokens to calculate meaningful entropy

        # Create semantic clusters (we'll use POS tags as a simple proxy for semantic roles)
        try:
            import nltk
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('punkt', quiet=True)

            # Get POS tags
            pos_tags = nltk.pos_tag(tokens)

            # Group by POS category
            clusters = {}
            for _, pos in pos_tags:
                # Simplify tags to broader categories
                if pos.startswith('N'):  # Nouns
                    category = 'noun'
                elif pos.startswith('V'):  # Verbs
                    category = 'verb'
                elif pos.startswith('J'):  # Adjectives
                    category = 'adj'
                elif pos.startswith('R'):  # Adverbs
                    category = 'adv'
                else:
                    category = 'other'

                clusters[category] = clusters.get(category, 0) + 1
        except ImportError:
            # Fallback if nltk not available - use word length as a crude proxy
            clusters = {}
            for token in tokens:
                length_category = min(5, len(token) // 3)  # Group by length category
                clusters[length_category] = clusters.get(length_category, 0) + 1

        # Calculate probabilities and entropy
        total = len(tokens)
        probabilities = [count/total for count in clusters.values()]

        # Calculate entropy using the formula SE = -∑ₓ P(c|x) log P(c|x)
        import math
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

        return entropy

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

        return normalized_metrics

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
        Optimized extraction of repeated meaningful phrases in the text.

        Args:
            text: Text to analyze

        Returns:
            List of repeated phrases
        """
        # OPTIMIZATION: Use a hash of the text as cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # Check cache for this text
        if text_hash in self._analyzed_segments:
            return self._analyzed_segments[text_hash]

        # OPTIMIZATION: Skip if text is too short
        if len(text) < 100:
            return []

        repeated = []

        # Get minimum length for phrases to check
        min_len = self.pattern_config['repeated_phrases']['min_length']

        # OPTIMIZATION: Extract words once
        words = re.findall(r'\b\w+\b', text.lower())

        # OPTIMIZATION: Only check n-grams of reasonable sizes
        # For longer text, checking every n-gram size is too expensive
        if len(words) > 1000:
            n_values = [3, 5, 7]  # Check only a few representative sizes
        else:
            # For shorter text, we can check more n-gram sizes
            n_values = range(min_len, min(10, len(words) // 3))

        # Check for repeated phrases (n-grams)
        for n in n_values:
            # OPTIMIZATION: Use dictionary for faster lookups
            ngrams = {}
            ngram_positions = {}

            # OPTIMIZATION: Use sliding window to build n-grams
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                if len(ngram.strip()) < min_len * 2:  # Skip very short phrases
                    continue

                if ngram in ngrams:
                    ngrams[ngram] += 1

                    # Only track positions if this might be a repeated n-gram
                    if ngrams[ngram] == 2:
                        ngram_positions[ngram] = [i - n, i]  # First and current position
                    elif ngrams[ngram] > 2:
                        ngram_positions[ngram].append(i)
                else:
                    ngrams[ngram] = 1

            # OPTIMIZATION: Only process n-grams that occurred multiple times
            for ngram, count in ngrams.items():
                if count >= self.pattern_config['repeated_phrases']['threshold']:
                    if ngram not in repeated:
                        repeated.append(ngram)

                    # OPTIMIZATION: Limit the total phrases we collect
                    if len(repeated) >= 5:
                        # Save results to cache and return
                        self._analyzed_segments[text_hash] = repeated
                        return repeated

        # Save results to cache
        self._analyzed_segments[text_hash] = repeated

        # Limit cache size
        if len(self._analyzed_segments) > 50:
            # Remove oldest entries
            old_keys = list(self._analyzed_segments.keys())[:-20]
            for key in old_keys:
                del self._analyzed_segments[key]

        return repeated

    def analyze_content_quality(self, response: str, incremental: bool = True) -> Dict[str, Any]:
        """
        Analyze the quality of the generated content with optimized performance.

        Args:
            response: Generated text response
            incremental: Whether to use incremental analysis for performance

        Returns:
            Dictionary with content quality assessment
        """
        # OPTIMIZATION: Check if we can reuse cached analysis
        if hasattr(self, '_last_quality_analysis') and incremental:
            last_analysis = getattr(self, '_last_quality_analysis', {})
            last_length = last_analysis.get('text_length', 0)

            if len(response) <= last_length:
                # Text got shorter or hasn't changed, return cached result
                return last_analysis.get('results', {'is_low_quality': False, 'quality_score': 1.0})
        else:
            last_length = 0

        # Initialize quality assessment
        results = {
            'is_low_quality': False,
            'quality_issues': [],
            'quality_score': 1.0,  # Start with perfect score
        }

        # Skip if response is too short
        if len(response.strip()) < 50:
            # OPTIMIZATION: Cache this result
            self._last_quality_analysis = {
                'text_length': len(response),
                'results': results,
                'timestamp': time.time()
            }
            return results

        # OPTIMIZATION: Only analyze new content if possible
        if incremental and last_length > 0:
            # Get the new portion of text
            new_text = response[last_length:]

            # If the new text is very short, do a quick basic check
            if len(new_text) < 100:
                # Quick check for obvious quality issues
                if '...' in new_text * 3 or '   ' in new_text:
                    results['is_low_quality'] = True
                    results['quality_issues'].append('basic_formatting_issues')
                    results['quality_score'] -= 0.2

                    # OPTIMIZATION: Cache this result
                    self._last_quality_analysis = {
                        'text_length': len(response),
                        'results': results,
                        'timestamp': time.time()
                    }
                    return results

        # OPTIMIZATION: Use cached sentence splitting
        # Split text into sentences if we haven't already
        if hasattr(self, '_cached_sentences') and len(response) == self._cached_sentences_length:
            sentences = self._cached_sentences
        else:
            # OPTIMIZATION: More efficient sentence splitting
            sentences = [s.strip() for s in re.split(r'[.!?]\s+', response) if s.strip()]
            self._cached_sentences = sentences
            self._cached_sentences_length = len(response)

        # Check sentence quality - but only process if we have enough sentences
        if len(sentences) < 3:
            # Not enough sentences to meaningfully analyze
            return results

        # OPTIMIZATION: Only check a sample of sentences for larger text
        if len(sentences) > 20:
            # Check first 5, last 5, and 10 random sentences from the middle
            sample_indices = list(range(5))  # First 5
            sample_indices.extend(range(len(sentences)-5, len(sentences)))  # Last 5

            # Add 10 random sentences from the middle
            middle_indices = list(range(5, len(sentences)-5))
            if middle_indices:
                import random
                random_count = min(10, len(middle_indices))
                sample_indices.extend(random.sample(middle_indices, random_count))

            # Create a sample of sentences
            sentence_sample = [sentences[i] for i in sorted(sample_indices)]
        else:
            # For shorter text, check all sentences
            sentence_sample = sentences

        # OPTIMIZATION: Precompile regex pattern for subject-verb structure
        if not hasattr(self, '_subject_verb_pattern'):
            self._subject_verb_pattern = re.compile(r'\b[A-Z][a-z]+\b.*?\b[a-z]+s\b|\b[A-Z][a-z]+\b.*?\b[a-z]+ed\b')

        # Check for malformed sentences
        malformed_count = 0
        for sentence in sentence_sample:
            # Skip very short sentences
            min_sentence_length = self.pattern_config['coherence']['min_sentence_length']
            if len(sentence.split()) < min_sentence_length:
                continue

            # Check for subject-verb structure if enabled
            if self.pattern_config['coherence']['subject_verb_check']:
                # Use precompiled pattern
                has_subject_verb = self._subject_verb_pattern.search(sentence)
                if not has_subject_verb:
                    malformed_count += 1

        # Calculate malformed ratio based on sample, extrapolate to full text
        if sentence_sample:
            malformed_ratio = malformed_count / len(sentence_sample)

            if malformed_ratio > self.pattern_config['coherence']['max_malformed_ratio']:
                results['is_low_quality'] = True
                results['quality_issues'].append('high_malformed_sentence_ratio')
                results['malformed_ratio'] = malformed_ratio

                # Reduce quality score
                results['quality_score'] -= malformed_ratio

        # OPTIMIZATION: Analyze text structure more efficiently
        # Only run this analysis on longer text
        if len(sentences) >= 10:
            structure_result = self._analyze_text_structure_optimized(response, sentences)
            if structure_result['structure_deterioration']:
                results['is_low_quality'] = True
                results['quality_issues'].append('structure_deterioration')

                # Reduce quality score
                results['quality_score'] -= 0.3

        # Ensure quality score stays in range [0, 1]
        results['quality_score'] = max(0.0, min(1.0, results['quality_score']))

        # OPTIMIZATION: Cache this result
        self._last_quality_analysis = {
            'text_length': len(response),
            'results': results,
            'timestamp': time.time()
        }

        return results

    def _analyze_text_structure_optimized(self, text: str, sentences: List[str]) -> Dict[str, Any]:
        """
        Optimized analysis of text structure for signs of deterioration.

        Args:
            text: Full text to analyze
            sentences: Pre-split sentences (for efficiency)

        Returns:
            Dictionary with structure analysis
        """
        result = {
            'structure_deterioration': False,
            'section_quality': []
        }

        # OPTIMIZATION: If we have very few sentences, skip detailed analysis
        if len(sentences) < 5:
            return result

        # OPTIMIZATION: For very long texts, analyze sections rather than sentences
        if len(sentences) > 20:
            # Divide into beginning, middle, and end sections
            section_size = max(5, len(sentences) // 5)
            beginning = sentences[:section_size]
            middle = sentences[len(sentences)//2 - section_size//2:len(sentences)//2 + section_size//2]
            end = sentences[-section_size:]

            # Analyze each section
            sections_to_analyze = [beginning, middle, end]
            section_names = ["beginning", "middle", "end"]
        else:
            # For shorter texts, look at individual sentences
            # Group sentences into small coherent sections
            sections_to_analyze = []
            current_section = []

            for sentence in sentences:
                current_section.append(sentence)
                if len(current_section) >= 3:
                    sections_to_analyze.append(current_section)
                    current_section = []

            # Add any remaining sentences
            if current_section:
                sections_to_analyze.append(current_section)

            section_names = [f"section_{i+1}" for i in range(len(sections_to_analyze))]

        # Track quality metrics for each section
        section_metrics = []

        # OPTIMIZATION: Precompile regex patterns
        word_pattern = re.compile(r'\b\w+\b')
        capitalization_pattern = re.compile(r'\b[A-Z][a-z]+\b')

        # Analyze each section
        for i, section in enumerate(sections_to_analyze):
            # Join section sentences
            section_text = ' '.join(section)

            # Skip very short sections
            if len(section_text) < 20:
                continue

            # Calculate simple quality metrics
            words = word_pattern.findall(section_text)
            unique_ratio = len(set(words)) / max(1, len(words))

            # Check structural indicators
            has_punctuation = '.' in section_text or ',' in section_text
            has_capitalization = bool(capitalization_pattern.search(section_text))

            # Calculate quality score
            quality = (unique_ratio * 0.7) + (0.15 if has_punctuation else 0) + (0.15 if has_capitalization else 0)

            # Store metrics
            section_metrics.append({
                'name': section_names[i],
                'quality': quality,
                'unique_ratio': unique_ratio,
            })

        # Store quality scores
        result['section_quality'] = [section['quality'] for section in section_metrics]

        # Check for degradation pattern
        if len(result['section_quality']) >= 3:
            # Check if quality consistently declines
            first_third = result['section_quality'][0]
            last_third = result['section_quality'][-1]

            if last_third < first_third * 0.6:  # More than 40% drop
                result['structure_deterioration'] = True
                result['quality_decline'] = first_third - last_third

        return result

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
        Get domain-specific thresholds based on the query with calibrated values for TinyLlama.
        """
        # Default thresholds - calibrated for TinyLlama
        thresholds = {
            'confidence': self.confidence_threshold,  # Default is now 0.45
            'entropy': self.entropy_threshold,        # Default is now 3.5
            'perplexity': self.perplexity_threshold,  # Default is now 25.0
            'memory_weight': 0.6                      # Increased baseline memory weight
        }

        # Get domain settings if question classifier available
        if self.question_classifier and query:
            try:
                domain_settings = self.question_classifier.get_domain_settings(query)
                domain = domain_settings.get('domain', 'unknown')

                # Apply domain-specific thresholds calibrated for TinyLlama
                if domain == 'arithmetic':
                    # Still stricter for math but more reasonable
                    thresholds['confidence'] = 0.55  # Down from 0.75
                    thresholds['entropy'] = 2.8      # Up from 1.8
                    thresholds['perplexity'] = 15.0  # Up from 8.0
                    thresholds['memory_weight'] = 0.4  # Up from 0.2
                elif domain == 'factual':
                    # Stricter for factual knowledge but calibrated
                    thresholds['confidence'] = 0.50  # Down from 0.70
                    thresholds['entropy'] = 3.0      # Up from 2.0
                    thresholds['perplexity'] = 18.0  # Up from 10.0
                    thresholds['memory_weight'] = 0.9  # Up from 0.8
                elif domain == 'translation':
                    # Translation needs more flexibility
                    thresholds['confidence'] = 0.40  # Down from 0.60
                    thresholds['entropy'] = 3.8      # Up from 2.8
                    thresholds['perplexity'] = 20.0  # Up from 12.0
                    thresholds['memory_weight'] = 0.8  # Up from 0.7
                elif domain == 'conceptual':
                    # Balanced with more memory reliance
                    thresholds['confidence'] = 0.45  # Down from 0.65
                    thresholds['entropy'] = 3.5      # Up from 2.3
                    thresholds['perplexity'] = 22.0  # Up from 12.0
                    thresholds['memory_weight'] = 0.6  # Up from 0.4
                elif domain == 'procedural':
                    # Balanced with more memory reliance
                    thresholds['confidence'] = 0.45  # Down from 0.65
                    thresholds['entropy'] = 3.5      # Up from 2.3
                    thresholds['perplexity'] = 22.0  # Up from 12.0
                    thresholds['memory_weight'] = 0.7  # Up from 0.6

                self.log(f"Using domain-specific thresholds for domain: {domain}")
            except Exception as e:
                self.log(f"Error getting domain settings: {e}")

        # Apply pattern-based adjustments based on query, but with reduced impact
        if query:
            # Technical queries - smaller adjustment
            if re.search(r'\b(?:API|code|programming|function|algorithm|technical)\b', query, re.IGNORECASE):
                thresholds['entropy'] += 0.2  # Down from 0.3

            # Factual queries - smaller adjustment
            if re.search(r'\bwhat is\b|\bwho is\b|\bwhen did\b|\bwhere is\b', query, re.IGNORECASE):
                thresholds['confidence'] += 0.03  # Down from 0.05

            # Mathematical queries - smaller adjustment
            if re.search(r'\bcalculate\b|\bcompute\b|\bsolve\b|\bhow many\b', query, re.IGNORECASE):
                thresholds['confidence'] += 0.03  # Down from 0.05

        return thresholds

    def analyze_confidence_patterns(self, token_confidences: List[float]) -> Dict[str, Any]:
        """
        Analyze confidence patterns to detect issues like sudden drops or repetitive patterns.

        Args:
            token_confidences: List of token confidence values

        Returns:
            Dictionary with pattern analysis results
        """
        if len(token_confidences) < 10:
            return {"pattern_detected": False, "severity": 0.0, "pattern_type": None}

        results = {
            "pattern_detected": False,
            "severity": 0.0,
            "pattern_type": None,
            "details": {}
        }

        # Calculate baseline metrics
        mean_confidence = sum(token_confidences) / len(token_confidences)
        # Update baseline confidence for this model (running average)
        self.baseline_confidence = 0.9 * self.baseline_confidence + 0.1 * mean_confidence

        # Calculate variance
        variance = sum((c - mean_confidence) ** 2 for c in token_confidences) / len(token_confidences)
        self.confidence_variance = 0.9 * self.confidence_variance + 0.1 * variance

        # 1. Check for sudden confidence drops (potential hallucinations)
        for i in range(1, len(token_confidences)):
            drop = token_confidences[i-1] - token_confidences[i]
            if drop > 0.3 and token_confidences[i] < mean_confidence * 0.7:
                results["pattern_detected"] = True
                results["pattern_type"] = "sudden_drop"
                results["severity"] = max(results["severity"], drop)
                results["details"]["drop_position"] = i
                results["details"]["drop_size"] = drop

        # 2. Check for consistent degradation (context window issues)
        if len(token_confidences) > 20:
            first_third = sum(token_confidences[:len(token_confidences)//3]) / (len(token_confidences)//3)
            last_third = sum(token_confidences[-len(token_confidences)//3:]) / (len(token_confidences)//3)

            degradation = first_third - last_third
            if degradation > 0.2 and last_third < mean_confidence * 0.8:
                results["pattern_detected"] = True
                results["pattern_type"] = "degradation"
                results["severity"] = max(results["severity"], degradation)
                results["details"]["degradation"] = degradation

        # 3. Check for repetitive patterns (common in stuck outputs)
        if len(token_confidences) > 15:
            # Look for repeating patterns of 3-5 tokens
            for pattern_length in range(3, 6):
                if len(token_confidences) < pattern_length * 2:
                    continue

                # Check for repeating subsequences
                for i in range(len(token_confidences) - pattern_length * 2):
                    pattern1 = token_confidences[i:i+pattern_length]
                    pattern2 = token_confidences[i+pattern_length:i+pattern_length*2]

                    # Calculate similarity between patterns
                    similarity = 1.0 - sum(abs(a-b) for a, b in zip(pattern1, pattern2)) / pattern_length

                    if similarity > 0.85:
                        results["pattern_detected"] = True
                        results["pattern_type"] = "repetition"
                        results["severity"] = max(results["severity"], similarity - 0.85)
                        results["details"]["repetition_pos"] = i
                        results["details"]["pattern_length"] = pattern_length

        return results

    def is_likely_hallucination(self, metrics, semantic_component):
        """
        Determine if response is likely a hallucination based on model type.
        """
        # Check if we're using Gemma model
        is_gemma = hasattr(self, 'user_context') and 'model_name' in self.user_context and 'gemma' in self.user_context['model_name'].lower()

        if is_gemma:
            # Gemma models need different thresholds
            return (
                metrics['entropy'] > 3.0 or          # Higher threshold for Gemma
                semantic_component > 0.60 or         # Higher threshold for Gemma
                metrics['severity'] > 0.25 or        # Higher threshold for Gemma
                (metrics['entropy'] > 2.5 and        # Higher combined threshold
                 metrics['repetition_score'] > 0.35)
            )
        else:
            # Original thresholds for other models
            return (
                metrics['entropy'] > 2.2 or
                semantic_component > 0.45 or
                metrics['severity'] > 0.15 or
                (metrics['entropy'] > 1.8 and
                 metrics['repetition_score'] > 0.25)
            )

    def should_filter(
        self,
        metrics: Dict[str, float],
        response: str,
        query: Optional[str] = None,
        tokens_generated: int = 0
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Determine if a response should be filtered using pattern detection and relaxed thresholds.
        """
        has_reason = False
        return_reason = False
        reason = ""

        # Skip aggressive filtering for very short responses
        if tokens_generated < self.token_count_threshold:
            # Skip comprehensive filtering for short responses to allow model to build confidence
            normalized_metrics = self.normalize_confidence_metrics(metrics)
            confidence = normalized_metrics.get('normalized_confidence',
                                              normalized_metrics.get('confidence', 0.5))

            # Only filter extremely low confidence early output
            if confidence < 0.25:  # Reduced from 0.3
                has_reason = True
                reason = "extremely_low_confidence"
                return_reason = True

            # Otherwise let it continue
            has_reason = True
            reason = "acceptable"
            return_reason = False

        # Apply normalization to confidence metrics
        normalized_metrics = self.normalize_confidence_metrics(metrics)

        # Get normalized confidence values
        confidence = normalized_metrics.get('normalized_confidence', normalized_metrics.get('confidence', 0.5))

        entropy = float(normalized_metrics.get('entropy', 2.0))

        perplexity = float(normalized_metrics.get('perplexity', 10.0))

        # Get domain-specific thresholds
        thresholds = self.get_domain_thresholds(query)

        # Get token confidence values for pattern detection
        token_confidences = []
        if hasattr(self, 'confidence_window') and self.confidence_window:
            token_confidences = self.confidence_window
        elif 'token_probabilities' in metrics:
            token_confidences = metrics['token_probabilities']

        # Detect context overflow first (highest priority)
        overflow_results = self.detect_context_overflow(response)
        if overflow_results['is_overflow']:
            has_reason = True
            reason = f"context_overflow_{overflow_results['overflow_patterns']}"
            return_reason = True

        # Analyze confidence patterns if we have enough tokens
        pattern_results = self.analyze_confidence_patterns(token_confidences) if token_confidences else {"pattern_detected": False}

        # Analyze content quality
        quality_results = self.analyze_content_quality(response)

        # Save content assessment for future reference
        self.last_content_assessment = quality_results

        # Check if response is supported by memory (if available in context)
        memory_supported = False
        memory_count = 0
        memory_details = []

        if hasattr(self, 'user_context') and self.user_context:
            memory_details = self.user_context.get('memory_details', [])
            memory_count = len(memory_details)

            # Check for high-quality memories (threshold: 0.5 instead of 0.7)
            for memory in memory_details:
                similarity = memory.get('similarity', 0)
                if similarity > 0.5:  # More lenient threshold
                    memory_supported = True
                    break

            # If no high-similarity matches, check if we have any reasonable matches
            if not memory_supported and memory_details:
                # Consider it "limited" support if we have any memories above 0.3
                max_similarity = max([m.get('similarity', 0) for m in memory_details], default=0)
                memory_supported = max_similarity > 0.3

        # Calculate semantic entropy - NEW!
        semantic_entropy_calculation = self.calculate_aggressive_content_entropy(response) # self.calculate_semantic_entropy(response)
        semantic_entropy = semantic_entropy_calculation.get('entropy', 0.0)

        # FIXED: Calculate a better bounded uncertainty score with weighted components
        confidence_component = max(0.0, min(1.0, (1 - confidence) * 0.35))
        quality_component = max(0.0, min(1.0, (1 - quality_results['quality_score']) * 0.25))
        entropy_component = max(0.0, min(1.0, (entropy / thresholds['entropy']) * 0.15))
        perplexity_component = max(0.0, min(1.0, (perplexity / thresholds['perplexity']) * 0.15))

        # NEW! Add semantic entropy component
        # semantic_entropy = pattern_results.get("semantic_entropy", 0.0)
        max_expected_entropy = 1.16 # 2.32  # Theoretical max for 5 equal clusters
        semantic_component = max(0.0, min(1.0, (semantic_entropy / max_expected_entropy) * 0.2))

        pattern_component = max(0.0, min(1.0, pattern_results['severity'] * self.pattern_detection_weight))
        memory_bonus = -0.1 if memory_supported else 0

        # Sum the components and clamp to [0.0, 1.0]
        uncertainty_score = max(0.0, min(1.0,
            confidence_component +
            quality_component +
            entropy_component +
            perplexity_component +
            semantic_component +
            pattern_component +
            memory_bonus
        ))

        is_likely_hallucination = self.is_likely_hallucination(semantic_entropy_calculation, semantic_component)

        # Adjust filtering threshold based on query domain
        uncertainty_threshold = 0.85 # was 0.7 (initial-0.55)

        details = {
            'uncertainty_score': uncertainty_score,
            'confidence': confidence,
            'entropy': entropy,
            'perplexity': perplexity,
            'quality_score': quality_results['quality_score'],
            'entropy_ratio': entropy / thresholds['entropy'],
            'perplexity_ratio': perplexity / thresholds['perplexity'],
            'pattern_detected': pattern_results.get('pattern_detected', False),
            'memory_supported': memory_supported,
            'memory_count': memory_count,
            'memory_details': memory_details,  # Explicitly pass all memory details
            'thresholds': thresholds,
            'overflow_results': overflow_results,
            'is_likely_hallucination': is_likely_hallucination
        }

        # Use semantic entropy as an indicator of pattern issues
        if semantic_entropy > 2.0 or is_likely_hallucination:  # High entropy indicates uncertainty
            return True, "high_semantic_uncertainty", details

        if has_reason:
            return return_reason, reason, details

        # Pattern-based filtering overrides threshold-based if enabled
        if self.use_relative_filtering and pattern_results["pattern_detected"]:
            pattern_type = pattern_results["pattern_type"]
            severity = pattern_results["severity"]

            if severity > 0.5: # (initial-0.3:  # Only filter for significant pattern issues)
                return True, f"pattern_detected_{pattern_type}", details

        # Still check quality but with more lenient threshold
        if quality_results['is_low_quality'] and quality_results['quality_score'] < 0.3:  # was 0.4 Reduced from 0.5
            return True, f"low_quality_{','.join(quality_results['quality_issues'])}", details

        # Extreme cases for absolute thresholds - much lower than before
        if confidence < thresholds['confidence'] * 0.6:  # 40% below threshold (from 30%)
            return True, "very_low_confidence", details

        if entropy > thresholds['entropy'] * 1.6:  # 60% above threshold (from 40%)
            return True, "extremely_high_entropy", details

        if perplexity > thresholds['perplexity'] * 1.8:  # 80% above threshold (from 50%)
            return True, "extremely_high_perplexity", details

        # Final check with uncertainty score
        if uncertainty_score > uncertainty_threshold:
            return True, "high_uncertainty", details

        # If we get here, the response passes all checks
        return False, "acceptable", details

    def get_confidence_indicator(self, metrics: Dict[str, Any], width: int = 30) -> str:
        """
        Generate a visual confidence indicator similar to context window usage bar.

        Args:
            metrics: Dictionary with confidence metrics
            width: Width of the indicator bar

        Returns:
            String with formatted confidence indicator
        """
        # Get key metrics
        confidence = min(1.0, metrics.get('confidence', 0.5))
        quality = min(1.0, metrics.get('quality_score', 0.7))

        # FIXED: Ensure uncertainty is properly bounded to prevent reliability > 100%
        uncertainty = metrics.get('uncertainty_score', 0.5)
        # Clamp uncertainty to the range [0.0, 1.0]
        uncertainty = max(0.0, min(1.0, uncertainty))
        reliability = 1.0 - uncertainty

        memory_supported = metrics.get('memory_supported', False)
        memory_count = metrics.get('memory_count', 0)
        memory_details = metrics.get('memory_details', [])

        # Calculate best memory similarity if we have memory details
        max_similarity = 0
        if memory_details:
            max_similarity = max([m.get('similarity', 0) for m in memory_details], default=0)

        # Function to generate a bar
        def generate_bar(value, width):
            # Ensure value is properly bounded between 0.0 and 1.0
            bounded_value = max(0.0, min(1.0, value))
            filled = int(bounded_value * width)
            return '█' * filled + '░' * (width - filled)

        # Choose colors (using ANSI color codes)
        def color_for_value(value):
            if value > 0.7:
                return "\033[32m"  # Green
            elif value > 0.5:
                return "\033[33m"  # Yellow
            else:
                return "\033[31m"  # Red

        reset = "\033[0m"

        # Create indicators
        conf_color = color_for_value(confidence)
        conf_bar = generate_bar(confidence, width)

        qual_color = color_for_value(quality)
        qual_bar = generate_bar(quality, width)

        # Inverse for uncertainty (lower is better)
        uncert_color = color_for_value(reliability)
        uncert_bar = generate_bar(reliability, width)

        # Memory support indicator
        mem_color = "\033[32m" if memory_supported else "\033[33m"
        mem_status = "None"

        if memory_count > 0:
            if max_similarity > 0.8:
                mem_color = "\033[32m"  # Green
                mem_status = f"Excellent ({memory_count} items, {max_similarity:.1%})"
            elif max_similarity > 0.6:
                mem_color = "\033[32m"  # Green
                mem_status = f"Good ({memory_count} items, {max_similarity:.1%})"
            elif max_similarity > 0.4:
                mem_color = "\033[33m"  # Yellow
                mem_status = f"Moderate ({memory_count} items, {max_similarity:.1%})"
            else:
                mem_color = "\033[33m"  # Yellow
                mem_status = f"Limited ({memory_count} items, {max_similarity:.1%})"

        # Format percentages as integers between 0-100%
        confidence_pct = int(confidence * 100)
        quality_pct = int(quality * 100)
        reliability_pct = int(reliability * 100)

        # Assemble the complete indicator
        indicator = (
            f"Response Metrics:\n"
            f"Confidence:     {conf_color}{conf_bar}{reset} {confidence_pct}%\n"
            f"Quality:        {qual_color}{qual_bar}{reset} {quality_pct}%\n"
            f"Reliability:    {uncert_color}{uncert_bar}{reset} {reliability_pct}%\n"
            f"Memory Support: {mem_color}{mem_status}{reset}"
        )

        return indicator
