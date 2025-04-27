"""
WebKnowledgeEnhancer for TinyLlama Chat
Provides web search capabilities to augment model responses with real-time information.
Compatible with UnifiedMemoryManager.
"""
import torch
import requests
import re
import time
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from datetime import datetime
import concurrent.futures

class WebKnowledgeEnhancer:
    """
    Enhances LLM knowledge by retrieving and processing web search results,
    working with the UnifiedMemoryManager.
    """

    def __init__(
        self,
        memory_manager=None,
        chat=None,
        confidence_threshold: float = 0.65,
        sharpening_factor: float = 0.3,
        max_results: int = 8,
        search_engine: str = "duckduckgo"
    ):
        """
        Initialize the web knowledge enhancer.

        Args:
            memory_manager: The UnifiedMemoryManager instance
            chat: Reference to the chat system
            confidence_threshold: Threshold below which to trigger web search
            sharpening_factor: Factor for sharpening vector comparisons
            max_results: Maximum number of search results to process
            search_engine: Search engine to use ('duckduckgo' or 'google')
        """
        self.memory_manager = memory_manager
        self.chat = chat
        self.confidence_threshold = confidence_threshold
        self.sharpening_factor = sharpening_factor
        self.max_results = max_results
        self.search_engine = search_engine

        # User agents to rotate through for requests
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
        ]

        # Session for requests
        self.session = requests.Session()

        # Cache for search results
        self.search_cache = {}
        self.cache_ttl = 3600  # 1 hour cache TTL

        # Stats
        self.total_searches = 0
        self.successful_searches = 0
        self.cache_hits = 0

        # Try to load NLP components if installed
        try:
            import spacy
            from spacy.lang.en.stop_words import STOP_WORDS
            self.nlp = spacy.load("en_core_web_sm")
            self.custom_stop_words = STOP_WORDS.union({"show", "find", "list", "give", "me", "please",
                                                 "that", "are", "code", "example", "write", "create",
                                                 "make", "do", "how", "can", "would", "should"})
            self.nlp_available = True
        except ImportError:
            self.nlp_available = False

    def get_time(self):
        """Get formatted timestamp for logging"""
        return datetime.now().strftime("[%d/%m/%y %H:%M:%S]") + ' [Web]'

    def should_enhance_with_web_knowledge(self,
                                         query: str,
                                         confidence_data: Dict[str, float],
                                         domain: Optional[str] = None) -> bool:
        """
        Determine if the response should be enhanced with web knowledge.

        Args:
            query: The user query
            confidence_data: Confidence metrics from the model
            domain: Optional domain classification of the query

        Returns:
            Boolean indicating whether to use web knowledge
        """
        # Get confidence score
        confidence = confidence_data.get('confidence', 1.0)

        # Adjust threshold based on domain
        adjusted_threshold = self.confidence_threshold

        if domain == "factual":
            # For factual queries, be more aggressive about using web knowledge
            adjusted_threshold = max(0.75, self.confidence_threshold)
        elif domain == "procedural":
            # For procedural queries, be somewhat more aggressive
            adjusted_threshold = max(0.7, self.confidence_threshold)
        elif domain == "conceptual":
            # For conceptual questions, be slightly less aggressive
            adjusted_threshold = min(0.6, self.confidence_threshold)

        # Check for time-sensitive indicators
        time_indicators = [
            "latest", "current", "recent", "today", "news",
            "2023", "2024", "2025", "update", "new"
        ]

        has_time_indicator = any(indicator in query.lower() for indicator in time_indicators)

        # If confidence is below threshold or query contains time indicators
        if confidence < adjusted_threshold or has_time_indicator:
            # Skip personal or subjective questions
            personal_indicators = [
                "your opinion", "do you think", "do you like", "would you",
                "personal", "you feel", "your favorite"
            ]
            if any(indicator in query.lower() for indicator in personal_indicators):
                return False

            return True

        return False

    def search_web(self, query: str, num_results: int = None) -> List[Dict[str, Any]]:
        """
        Search the web for information related to the query.

        Args:
            query: The search query
            num_results: Number of results to retrieve

        Returns:
            List of search result dictionaries
        """
        if num_results is None:
            num_results = self.max_results

        # Check cache first
        cache_key = f"{self.search_engine}:{query}:{num_results}"
        if cache_key in self.search_cache:
            timestamp, results = self.search_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                self.cache_hits += 1
                print(f"{self.get_time()} Cache hit for query: '{query}'")
                return results

        # Increment search counter
        self.total_searches += 1

        # Select search method based on engine
        if self.search_engine.lower() == "duckduckgo":
            results = self._search_duckduckgo(query, num_results)
        elif self.search_engine.lower() == "google":
            results = self._search_google(query, num_results)
        else:
            raise ValueError(f"Unsupported search engine: {self.search_engine}")

        # Cache results
        self.search_cache[cache_key] = (time.time(), results)

        # Update success counter
        if results:
            self.successful_searches += 1

        return results

    def _search_duckduckgo(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """
        Search DuckDuckGo for results.

        Args:
            query: Search query
            num_results: Number of results to retrieve

        Returns:
            List of search result dictionaries
        """
        # Prepare request
        encoded_query = quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        # Random user agent
        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        try:
            # Make request
            response = self.session.get(url, headers=headers, timeout=10)

            # Check response
            if response.status_code != 200:
                print(f"{self.get_time()} Error: DuckDuckGo returned status code {response.status_code}")
                return []

            print(f"{self.get_time()} Searching {url}")

            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract results
            results = []
            result_elements = soup.select('.result')

            for element in result_elements[:num_results]:
                try:
                    # Extract title
                    title_element = element.select_one('.result__a')
                    title = title_element.get_text(strip=True) if title_element else "No title"

                    # Extract URL
                    href = title_element.get('href', '') if title_element else ""
                    if href.startswith('/'):
                        continue  # Skip internal links

                    # Clean URL (DuckDuckGo uses redirects)
                    if "uddg=" in href:
                        url_match = re.search(r'uddg=([^&]+)', href)
                        if url_match:
                            href = requests.utils.unquote(url_match.group(1))

                    # Extract snippet
                    snippet_element = element.select_one('.result__snippet')
                    snippet = snippet_element.get_text(strip=True) if snippet_element else ""

                    # Append result
                    results.append({
                        'title': title,
                        'url': href,
                        'snippet': snippet,
                        'source': 'duckduckgo'
                    })
                except Exception as e:
                    print(f"{self.get_time()} Error extracting result: {e}")

            print(f"{self.get_time()} Found {len(results)} results from DuckDuckGo for query: '{query}'")
            return results

        except Exception as e:
            print(f"{self.get_time()} Error searching DuckDuckGo: {e}")
            return []

    def _search_google(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """
        Search Google for results.

        Args:
            query: Search query
            num_results: Number of results to retrieve

        Returns:
            List of search result dictionaries
        """
        # Prepare request
        encoded_query = quote_plus(query)
        url = f"https://www.google.com/search?q={encoded_query}&num={num_results}"

        # Random user agent
        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        try:
            # Add random delay to avoid detection
            time.sleep(random.uniform(1.0, 3.0))

            # Make request
            response = self.session.get(url, headers=headers, timeout=10)

            # Check response
            if response.status_code != 200:
                print(f"{self.get_time()} Error: Google returned status code {response.status_code}")
                return []

            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract results (simplified - Google's HTML structure changes frequently)
            results = []

            # Find result divs
            result_elements = soup.select('div.g')

            for element in result_elements[:num_results]:
                try:
                    # Extract title
                    title_element = element.select_one('h3')
                    title = title_element.get_text(strip=True) if title_element else "No title"

                    # Extract URL
                    link_element = element.select_one('a')
                    href = link_element.get('href', '') if link_element else ""

                    # Clean URL
                    if href.startswith('/url?q='):
                        url_match = re.search(r'/url\?q=([^&]+)', href)
                        if url_match:
                            href = requests.utils.unquote(url_match.group(1))

                    # Skip internal Google links
                    if not href or href.startswith('https://google.com'):
                        continue

                    # Extract snippet
                    snippet_element = element.select_one('div.VwiC3b')
                    snippet = snippet_element.get_text(strip=True) if snippet_element else ""

                    # Append result
                    results.append({
                        'title': title,
                        'url': href,
                        'snippet': snippet,
                        'source': 'google'
                    })
                except Exception as e:
                    print(f"{self.get_time()} Error extracting result: {e}")

            print(f"{self.get_time()} Found {len(results)} results from Google for query: '{query}'")
            return results

        except Exception as e:
            print(f"{self.get_time()} Error searching Google: {e}")
            return []

    def fetch_content(self, url: str, timeout: int = 10) -> Optional[str]:
        """
        Fetch content from a URL.

        Args:
            url: URL to fetch content from
            timeout: Request timeout in seconds

        Returns:
            Plain text content or None if failed
        """
        # Random user agent
        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        try:
            # Add random delay to avoid detection
            time.sleep(random.uniform(0.5, 1.5))

            # Make request
            response = self.session.get(url, headers=headers, timeout=timeout)

            # Check response
            if response.status_code != 200:
                print(f"{self.get_time()} Error: URL {url} returned status code {response.status_code}")
                return None

            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script, style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.extract()

            # Get text and clean it
            text = soup.get_text(separator=' ')

            # Clean whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            # Truncate if too long
            if len(text) > 10000:
                text = text[:10000] + "..."

            return text

        except Exception as e:
            print(f"{self.get_time()} Error fetching content from {url}: {e}")
            return None

    def create_seo_friendly_query(self, query: str) -> str:
        """
        Create an SEO-friendly search query.

        Args:
            query: The original user query

        Returns:
            SEO-friendly search query
        """
        # Extract entities first (important terms)
        entities = self._extract_entities(query)

        # If NLP is available, use it for better extraction
        if self.nlp_available:
            return self._nlp_query_extraction(query, entities)
        else:
            # Fallback to simpler extraction
            return self._simple_query_extraction(query, entities)

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract important entities from text.

        Args:
            text: Text to extract entities from

        Returns:
            List of extracted entities
        """
        # Extract quoted text (highest priority)
        quoted = re.findall(r'"([^"]+)"', text)
        entities = quoted.copy()

        # Extract proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b', text)
        entities.extend(proper_nouns)

        # Extract years and numbers
        years = re.findall(r'\b((?:19|20)\d{2})\b', text)
        entities.extend(years)

        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.lower() not in seen:
                seen.add(entity.lower())
                unique_entities.append(entity)

        return unique_entities

    def _nlp_query_extraction(self, query: str, entities: List[str]) -> str:
        """
        Extract key terms using NLP.

        Args:
            query: Original query
            entities: Extracted entities

        Returns:
            SEO-friendly query
        """
        doc = self.nlp(query)

        # Extract keywords (nouns and important words)
        keywords = []
        for token in doc:
            if token.pos_ in {"ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"} and token.text.lower() not in self.custom_stop_words:
                if len(token.text) > 1:
                    keywords.append(token.text)

        # Combine entities and keywords
        all_terms = entities.copy()
        for keyword in keywords:
            if keyword.lower() not in [term.lower() for term in all_terms]:
                all_terms.append(keyword)

        # Limit to 8 terms max
        return " ".join(all_terms[:8])

    def _simple_query_extraction(self, query: str, entities: List[str]) -> str:
        """
        Extract key terms without NLP.

        Args:
            query: Original query
            entities: Extracted entities

        Returns:
            SEO-friendly query
        """
        # Simple stop words list
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "and", "or", "but", "if", "then", "else", "when", "where", "what",
            "how", "who", "why", "which", "do", "does", "did", "done", "am", "is",
            "are", "was", "were", "will", "would", "should", "can", "could", "for",
            "to", "of", "in", "on", "at", "by", "with", "about", "please", "help",
            "me", "my", "your", "his", "her", "their", "this", "that", "these", "those"
        }

        # Split query into words and filter out stop words
        words = query.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        # Combine entities and keywords
        all_terms = entities.copy()
        for keyword in keywords:
            if keyword.lower() not in [term.lower() for term in all_terms]:
                all_terms.append(keyword)

        # Limit to 8 terms max
        return " ".join(all_terms[:8])

    def enhance_response(self,
                        query: str,
                        confidence_data: Dict[str, float],
                        domain: Optional[str] = None,
                        process_urls: bool = False,
                        messages: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Enhanced: Improve response generation with web knowledge using batch processing.

        Args:
            query: User query
            confidence_data: Confidence metrics from the model
            domain: Optional domain classification
            process_urls: Whether to fetch full content from URLs
            messages: Optional conversation history

        Returns:
            Web enhancement data
        """
        try:
            # Check if memory manager is available
            if not self.memory_manager:
                print(f"{self.get_time()} Memory manager not available")
                return {
                    'enhanced': False,
                    'reason': 'web_knowledge_disabled',
                    'web_results': []
                }

            # Create an SEO-friendly query for better search results
            seo_friendly_query = self.create_seo_friendly_query(query)
            print(f"{self.get_time()} SEO-friendly query: {seo_friendly_query}")

            # Search the web
            search_results = self.search_web(seo_friendly_query)
            print(f"{self.get_time()} Search returned {len(search_results)} results")

            if not search_results:
                return {
                    'enhanced': False,
                    'reason': 'no_search_results',
                    'web_results': []
                }

            # Process results to fetch full content if requested
            if process_urls and search_results:
                print(f"{self.get_time()} Fetching full content from URLs in batch")

                # Get top URLs to process (limit to 5 for efficiency)
                urls_to_process = [result['url'] for result in search_results[:5] if 'url' in result]

                if urls_to_process:
                    # Use batch fetching for better performance
                    url_contents = self.fetch_content_batch(urls_to_process)

                    # Add content to results
                    processed_results = []
                    for result in search_results:
                        if 'url' in result and result['url'] in url_contents:
                            content = url_contents[result['url']]
                            if content:
                                # Add full content to result
                                result_with_content = result.copy()
                                result_with_content['content'] = content
                                processed_results.append(result_with_content)
                            else:
                                # Keep original result if content fetch failed
                                processed_results.append(result)
                        else:
                            # Keep results without URLs
                            processed_results.append(result)

                    if processed_results:
                        search_results = processed_results
                        print(f"{self.get_time()} Processed {len(processed_results)} results with content")

            # Check embedding function
            has_embeddings = (hasattr(self.memory_manager, 'embedding_function') and
                             self.memory_manager.embedding_function is not None)

            # Calculate relevance scores if embeddings are available
            if has_embeddings:
                try:
                    print(f"{self.get_time()} Ranking results by similarity")
                    scored_results = self._rank_results_by_similarity(query, search_results)

                    # Filter to top results
                    filtered_results = scored_results[:self.max_results]
                    print(f"{self.get_time()} Ranked and filtered to {len(filtered_results)} results")
                except Exception as e:
                    print(f"{self.get_time()} Error ranking results: {e}")
                    # Fall back to search order
                    filtered_results = search_results[:self.max_results]
            else:
                # If no embedding function, just use search order
                print(f"{self.get_time()} No embedding function, using search order")
                filtered_results = search_results[:self.max_results]

                # Add default similarity scores
                for result in filtered_results:
                    result['similarity'] = 0.7  # Default high-ish score

            # Ensure all results have string content
            for result in filtered_results:
                for key in ['title', 'snippet', 'url']:
                    if key in result and not isinstance(result[key], str):
                        result[key] = str(result[key])

            # Return enhancement data
            enhancement_data = {
                'enhanced': True,
                'reason': 'web_search',
                'web_results': filtered_results,
                'query': query,
                'seo_friendly_query': seo_friendly_query
            }

            return enhancement_data

        except Exception as e:
            print(f"{self.get_time()} Error in enhance_response: {e}")
            import traceback
            traceback.print_exc()

            # Return minimal data
            return {
                'enhanced': False,
                'reason': 'error',
                'error_message': str(e),
                'web_results': []
            }

    def _rank_results_by_similarity(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank search results by semantic similarity to the query with enhanced batch processing.

        Args:
            query: Original query
            results: Search results to rank

        Returns:
            Ranked results with similarity scores
        """
        # Generate query embedding
        try:
            query_embedding = self.memory_manager.embedding_function(query)
        except Exception as e:
            print(f"{self.get_time()} Error generating query embedding: {e}")
            # If we can't generate embeddings, just return results in original order
            for result in results:
                result['similarity'] = 0.5
            return results[:self.max_results]

        # Prepare texts for batch embedding
        texts_to_embed = []
        for result in results:
            text_to_embed = f"{result['title']} {result['snippet']}"
            texts_to_embed.append(text_to_embed)

        if not texts_to_embed:
            return []

        # Use batch embedding for better performance
        try:
            print(f"{self.get_time()} Generating {len(texts_to_embed)} result embeddings in batch")

            # Check if memory manager has batch embedding function
            if hasattr(self.memory_manager, 'batch_embedding_function'):
                result_embeddings = self.memory_manager.batch_embedding_function(texts_to_embed)
            else:
                # Fall back to individual embedding
                result_embeddings = []
                for text in texts_to_embed:
                    try:
                        embedding = self.memory_manager.embedding_function(text)
                        result_embeddings.append(embedding)
                    except Exception as e:
                        print(f"{self.get_time()} Error generating result embedding: {e}")
                        # Use a zero embedding as fallback
                        result_embeddings.append(np.zeros(self.memory_manager.embedding_dim))

            # Calculate similarities and prepare scored results
            scored_results = []

            for i, (result, embedding) in enumerate(zip(results, result_embeddings)):
                # Calculate cosine similarity
                similarity = self._calculate_similarity(query_embedding, embedding)

                # Ensure similarity is a Python float
                if hasattr(similarity, 'item'):
                    similarity = float(similarity.item())
                else:
                    similarity = float(similarity)

                # Apply sharpening if enabled
                if self.sharpening_factor > 0:
                    similarity = self._apply_sharpening(similarity, self.sharpening_factor)

                # Add similarity to result
                result_with_score = result.copy()
                result_with_score['similarity'] = similarity
                scored_results.append(result_with_score)

            # Sort by similarity
            scored_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)

            return scored_results

        except Exception as e:
            print(f"{self.get_time()} Error in batch similarity ranking: {e}")
            # Fall back to sequential processing
            return self._rank_results_sequentially(query_embedding, results)

    def _rank_results_sequentially(self, query_embedding: np.ndarray, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank results sequentially as a fallback if batch processing fails.

        Args:
            query_embedding: Query embedding
            results: Search results to rank

        Returns:
            Ranked results
        """
        scored_results = []

        for result in results:
            # Create text for embedding
            text_to_embed = f"{result['title']} {result['snippet']}"

            try:
                # Generate embedding
                result_embedding = self.memory_manager.embedding_function(text_to_embed)

                # Calculate similarity
                similarity = self._calculate_similarity(query_embedding, result_embedding)

                # Ensure similarity is a Python float
                if hasattr(similarity, 'item'):
                    similarity = float(similarity.item())
                else:
                    similarity = float(similarity)

                # Apply sharpening
                if self.sharpening_factor > 0:
                    similarity = self._apply_sharpening(similarity, self.sharpening_factor)

                # Add to results
                result_with_score = result.copy()
                result_with_score['similarity'] = similarity
                scored_results.append(result_with_score)

            except Exception as e:
                print(f"{self.get_time()} Error ranking result: {e}")
                # Add with default similarity
                result_copy = result.copy()
                result_copy['similarity'] = 0.5
                scored_results.append(result_copy)

        # Sort by similarity
        scored_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        return scored_results

    def fetch_content_batch(self, urls: List[str], timeout: int = 10) -> Dict[str, Optional[str]]:
        """
        Fetch content from multiple URLs in parallel.

        Args:
            urls: List of URLs to fetch content from
            timeout: Request timeout in seconds

        Returns:
            Dictionary mapping URLs to their content (or None if failed)
        """
        # Import for concurrent fetching
        import concurrent.futures

        # Random user agent function
        def get_random_user_agent():
            return random.choice(self.user_agents)

        # Function to fetch a single URL
        def fetch_url(url):
            try:
                # Add random delay to avoid detection
                time.sleep(random.uniform(0.5, 1.5))

                # Random user agent
                headers = {
                    'User-Agent': get_random_user_agent(),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }

                # Make request
                response = self.session.get(url, headers=headers, timeout=timeout)

                # Check response
                if response.status_code != 200:
                    print(f"{self.get_time()} Error: URL {url} returned status code {response.status_code}")
                    return url, None

                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')

                # Remove script, style elements
                for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    script.extract()

                # Get text and clean it
                text = soup.get_text(separator=' ')

                # Clean whitespace
                text = re.sub(r'\s+', ' ', text).strip()

                # Truncate if too long
                if len(text) > 10000:
                    text = text[:10000] + "..."

                return url, text

            except Exception as e:
                print(f"{self.get_time()} Error fetching content from {url}: {e}")
                return url, None

        # Process URLs in parallel
        max_workers = min(8, len(urls))  # Limit to 8 workers maximum
        results = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit fetching tasks
            future_to_url = {executor.submit(fetch_url, url): url for url in urls}

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_url):
                url, content = future.result()
                results[url] = content

        return results

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity (0-1)
        """
        try:
            # Handle multidimensional arrays (ensure we have 1D arrays)
            if embedding1.ndim > 1:
                embedding1 = embedding1.reshape(-1)
            if embedding2.ndim > 1:
                embedding2 = embedding2.reshape(-1)

            # Normalize
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            # Avoid division by zero
            if norm1 < 1e-10 or norm2 < 1e-10:
                return 0.0

            embedding1_normalized = embedding1 / norm1
            embedding2_normalized = embedding2 / norm2

            # Calculate cosine similarity
            similarity = np.dot(embedding1_normalized, embedding2_normalized)

            # Convert to scalar if it's still an array
            if isinstance(similarity, np.ndarray):
                if similarity.size == 1:
                    similarity = similarity.item()
                else:
                    # If we somehow got a multi-element array, use the mean
                    similarity = similarity.mean().item()

            # Ensure we return a Python float
            return float(similarity)
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            # Return a default value on error
            return 0.5

    def _apply_sharpening(self, similarity: float, sharpening_factor: float) -> float:
        """
        Apply non-linear sharpening to similarity scores to increase contrast.

        Args:
            similarity: Raw similarity score (0.0-1.0)
            sharpening_factor: Strength of sharpening effect (0.0-1.0)

        Returns:
            Sharpened similarity score
        """
        # Convert to scalar float if it's a numpy type
        if hasattr(similarity, 'item'):
            similarity = similarity.item()

        # Skip if no sharpening requested
        if sharpening_factor <= 0:
            return similarity

        # Apply non-linear sharpening
        if float(similarity) > 0.6:
            # Boost high similarities (more confident matches)
            boost = (float(similarity) - 0.6) * sharpening_factor * 2.0
            sharpened = min(1.0, float(similarity) + boost)
        elif float(similarity) < 0.4:
            # Reduce low similarities (less confident matches)
            reduction = (0.4 - float(similarity)) * sharpening_factor * 2.0
            sharpened = max(0.0, float(similarity) - reduction)
        else:
            # Middle range - moderate effect
            deviation = (float(similarity) - 0.5) * sharpening_factor
            sharpened = 0.5 + deviation

        return float(sharpened)

    def format_web_results_for_context(self, enhancement_data: Dict[str, Any]) -> str:
        """
        Format web results for inclusion in the context.

        Args:
            enhancement_data: Web enhancement data

        Returns:
            Formatted string for context
        """
        if not enhancement_data.get('enhanced', False):
            return ""

        web_results = enhancement_data.get('web_results', [])

        if not web_results:
            return ""

        # Extract entities for better context
        entities = self._extract_entities(web_results[0].get('title', '') + " " + web_results[0].get('snippet', ''))
        entity_str = '", "'.join(entities) if entities else ""

        # Format results with clear hierarchy
        context = "WEB SEARCH RESULTS\n"
        context += "----------------\n"
        context += "Please synthesize an answer using the following information:\n\n"

        # Group results by relevance
        primary_results = []
        secondary_results = []

        for result in web_results:
            # Get similarity if available
            similarity = result.get('similarity', 0.5)

            if similarity > 0.6:
                primary_results.append(result)
            else:
                secondary_results.append(result)

        # Add primary results first
        if primary_results:
            context += "PRIMARY INFORMATION:\n"

            for i, result in enumerate(primary_results[:3]):
                title = result.get('title', 'Untitled')
                snippet = result.get('snippet', 'No description available')
                url = result.get('url', '')

                # Format with clear structure
                context += f"{i+1}. [{title}]\n"
                context += f"   {snippet}\n"

                # Add source domain
                if url:
                    domain = url.split('/')[2] if len(url.split('/')) > 2 else url
                    context += f"   Source: {domain}\n"

                context += "\n"

        # Add secondary results
        remaining_slots = max(0, 5 - len(primary_results))
        if secondary_results and remaining_slots > 0:
            context += "SUPPORTING INFORMATION:\n"

            for i, result in enumerate(secondary_results[:remaining_slots]):
                title = result.get('title', 'Untitled')
                snippet = result.get('snippet', 'No description available')

                # More concise format for secondary results
                context += f"{i+1}. {title}: {snippet}\n"

            context += "\n"

        # Add instructions for using the information
        context += "INSTRUCTIONS:\n"
        context += "- Synthesize a direct answer from the information above\n"

        if entities:
            context += f'- Focus on information about "{entity_str}"\n'

        context += "- If the information appears contradictory, acknowledge this\n"
        context += "- If search results don't contain relevant information, state this clearly\n"

        return context

    def add_web_results_to_memory(self, user_id: str, query: str, enhancement_data: Dict[str, Any]) -> int:
        """
        Add web search results to memory.

        Args:
            user_id: User ID
            query: Original query
            enhancement_data: Web enhancement data

        Returns:
            Number of items added to memory
        """
        if not enhancement_data.get('enhanced', False) or not self.memory_manager:
            return 0

        web_results = enhancement_data.get('web_results', [])

        # Prepare batch items for bulk addition
        batch_items = []

        for result in web_results:
            # Skip low relevance results
            similarity = result.get('similarity', 0.5)
            if similarity < 0.4:
                continue

            # Create memory text
            memory_text = f"{result['title']} - {result['snippet']}"

            # Add source URL
            if 'url' in result:
                memory_text += f" Source: {result['url']}"

            # Create metadata
            metadata = {
                'source_query': query,
                'url': result.get('url', ''),
                'similarity': similarity,
                'web_timestamp': time.time()
            }

            # Add to batch
            batch_items.append({
                "content": memory_text,
                "memory_type": "web_knowledge",
                "source": "web_search",
                "metadata": metadata
            })

        # Add all items at once
        if not batch_items:
            return 0

        # Use bulk add method if available
        if hasattr(self.memory_manager, 'add_bulk'):
            added_ids = self.memory_manager.add_bulk(batch_items)
            return sum(1 for item_id in added_ids if item_id is not None)
        else:
            # Fallback to individual add
            added_count = 0
            for item in batch_items:
                item_id = self.memory_manager.add(
                    content=item["content"],
                    memory_type=item["memory_type"],
                    source=item["source"],
                    metadata=item["metadata"]
                )
                if item_id:
                    added_count += 1
            return added_count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about web knowledge enhancement.

        Returns:
            Dictionary of statistics
        """
        return {
            'total_searches': self.total_searches,
            'successful_searches': self.successful_searches,
            'cache_hits': self.cache_hits,
            'cache_size': len(self.search_cache),
            'success_rate': self.successful_searches / max(1, self.total_searches),
            'cache_hit_rate': self.cache_hits / max(1, self.total_searches)
        }