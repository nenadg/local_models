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
        fallback_messages: Optional[List[str]] = None,
        continuation_phrases: Optional[List[str]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        sharpening_factor: float = 0.3,
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
        self.sharpening_factor = sharpening_factor
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

        # Default fallback messages unchanged...
        self.fallback_messages = fallback_messages or [
            "I don't have sufficient information to answer this question reliably.",
            "I'm uncertain about this topic and don't want to provide potentially incorrect information.",
            "I'm not confident in my ability to provide a good answer to this question.",
            "I'm not able to provide a satisfactory answer about this topic.",
        ]

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

    def _calculate_text_similarity(self, response: str, memories: List[Dict[str, Any]]) -> float:
        """Calculate text-based similarity as a fallback."""
        import re

        # Extract key terms from response
        response_terms = set(re.findall(r'\b[A-Za-z]{3,}\b', response.lower()))

        # Calculate term overlap with each memory
        similarities = []
        for memory in memories:
            memory_content = memory.get('content', '').lower()
            memory_terms = set(re.findall(r'\b[A-Za-z]{3,}\b', memory_content))

            # Calculate Jaccard similarity
            if response_terms and memory_terms:
                intersection = len(response_terms.intersection(memory_terms))
                union = len(response_terms.union(memory_terms))
                similarity = intersection / union

                # Apply memory's original similarity weight if available
                memory_similarity = memory.get('similarity', 1.0)
                weighted_similarity = similarity * memory_similarity

                similarities.append(weighted_similarity)

        # Return max similarity if any, otherwise 0
        return max(similarities) if similarities else 0.0

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

    def detect_repetitive_patterns(self, response: str, incremental_only: bool = False) -> dict:
        """
        Optimized detection of problematic repetitive patterns in the response.

        Args:
            response: Generated text response
            incremental_only: Whether to only check the new portion since last check

        Returns:
            Dictionary with pattern detection results
        """
        # Initialize results
        results = {
            'has_repetitive_patterns': False,
            'detected_patterns': []
        }

        # OPTIMIZATION: Skip if too short to have meaningful patterns
        if len(response) < 20:
            return results

        # OPTIMIZATION: Process only new content when possible
        current_length = len(response)
        if incremental_only and self._last_analysis['text_length'] > 0:
            # Only analyze new content if we've analyzed this text before
            if current_length <= self._last_analysis['text_length']:
                # No new content or text got shorter somehow
                return self._last_analysis

            # Get only the new portion of the text
            new_portion = response[self._last_analysis['text_length']:]

            # If new portion is small, do a quick check first
            if len(new_portion) < 50:
                # Quick check for obvious repetitive characters
                if self.char_repeat_pattern.search(new_portion):
                    results['has_repetitive_patterns'] = True
                    results['detected_patterns'].append('repeated_characters')
                    self._last_analysis = results
                    self._last_analysis['text_length'] = current_length
                    self._last_analysis['timestamp'] = time.time()
                    return results

        # OPTIMIZATION: Check the most problematic patterns first
        # This allows for early exit if problems are found

        # 1. Check for garbage patterns that frequently appear in broken responses
        # Using pre-compiled regexes (much faster)
        patterns_to_check = [
            (self.empty_code_blocks, 'empty_code_blocks'),
            (self.repeated_dollar_signs, 'repeated_dollar_signs'),
            (self.dollar_n_pattern, 'dollar_n_pattern'),
            (self.file_permission_fragments, 'file_permission_fragments'),
            (self.txt_fragments, 'txt_fragments'),
            (self.comma_separated_words, 'comma_separated_words'),
            (self.repeated_connectors, 'repeated_connectors')
        ]

        for pattern, pattern_name in patterns_to_check:
            # OPTIMIZATION: Use search instead of findall when we only need to know if matches exist
            if pattern.search(response):
                results['has_repetitive_patterns'] = True
                results['detected_patterns'].append(pattern_name)

                # OPTIMIZATION: Early exit once a pattern is found
                if self.debug_mode:
                    self.log(f"Detected pattern: {pattern_name}")

                # Update last analysis
                self._last_analysis = results
                self._last_analysis['text_length'] = current_length
                self._last_analysis['timestamp'] = time.time()
                return results

        # 2. Check for repetitive characters (very common in stuck outputs)
        char_repeat_matches = self.char_repeat_pattern.findall(response)
        if char_repeat_matches:
            results['has_repetitive_patterns'] = True
            results['detected_patterns'].append('repeated_characters')
            results['repeated_characters'] = char_repeat_matches

            # Update last analysis
            self._last_analysis = results
            self._last_analysis['text_length'] = current_length
            self._last_analysis['timestamp'] = time.time()
            return results

        # 3. Check for duplicate lines (compute-intensive, so do it last)
        # OPTIMIZATION: Use regex to find repeated lines more efficiently
        duplicate_matches = self.duplicate_line_pattern.findall(response)
        if duplicate_matches:
            results['has_repetitive_patterns'] = True
            results['detected_patterns'].append('duplicate_lines')
            results['duplicate_lines'] = duplicate_matches

        # OPTIMIZATION: Only run expensive n-gram analysis if nothing found yet
        # and the text is long enough to possibly have such patterns
        if not results['has_repetitive_patterns'] and len(response) > 200:
            phrases = self._extract_repeated_phrases(response)
            if phrases and len(phrases) >= self.pattern_config['repeated_phrases']['threshold']:
                results['has_repetitive_patterns'] = True
                results['detected_patterns'].append('repetitive_phrases')
                results['repetitive_phrases'] = phrases[:3]  # OPTIMIZATION: Only store a few examples

        # Update last analysis
        self._last_analysis = results
        self._last_analysis['text_length'] = current_length
        self._last_analysis['timestamp'] = time.time()

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
        return (
            metrics['entropy'] > 2.2 or          # Increased from 1.65
            semantic_component > 0.45 or         # Increased from 0.28
            metrics['severity'] > 0.15 or        # Increased from 0.05
            (metrics['entropy'] > 1.8 and        # Increased from 1.5
             metrics['repetition_score'] > 0.25)  # Increased from 0.1
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
        confidence = normalized_metrics.get('sharpened_confidence',
                                           normalized_metrics.get('normalized_confidence',
                                                                normalized_metrics.get('confidence', 0.5)))

        entropy = float(normalized_metrics.get('sharpened_entropy',
                                             normalized_metrics.get('entropy', 2.0)))

        perplexity = float(normalized_metrics.get('sharpened_perplexity',
                                                normalized_metrics.get('perplexity', 10.0)))

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

        # Preserve memory information from the metrics dictionary
        if 'memory_results' in self.user_context:
            details['memory_details'] = self.user_context['memory_results']
            details['memory_count'] = len(self.user_context['memory_results'])

            # Check for memory support
            if self.user_context['memory_results']:
                max_similarity = max([m.get('similarity', 0) for m in self.user_context['memory_results']], default=0)
                details['memory_supported'] = max_similarity > 0.3

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