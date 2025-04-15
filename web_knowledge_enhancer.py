"""
Web Knowledge Enhancer for the TinyLlama Chat system.
Provides search engine scraping functionality to augment model responses with real-time web information.
"""

import requests
import re
import time
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

class WebKnowledgeEnhancer:
    """
    Enhances LLM knowledge by retrieving and processing web search results,
    converting them to vectors for comparison with user queries.
    """
    
    def __init__(
        self, 
        memory_manager=None,
        chat=None,
        confidence_threshold: float = 0.65,
        vector_sharpening_factor: float = 0.3, 
        max_results: int = 8,
        search_engine: str = "duckduckgo",
        embedding_function=None,
        user_agents: List[str] = None
    ):
        """
        Initialize the web knowledge enhancer.
        
        Args:
            memory_manager: The memory manager instance for embedding and storage
            confidence_threshold: Threshold below which to trigger web search
            vector_sharpening_factor: Factor for sharpening vector comparisons
            max_results: Maximum number of search results to process
            search_engine: Search engine to use ('duckduckgo' or 'google')
            embedding_function: Optional custom embedding function
            user_agents: List of user agents to rotate through for requests
        """
        self.memory_manager = memory_manager
        self.chat = chat
        self.confidence_threshold = confidence_threshold
        self.vector_sharpening_factor = vector_sharpening_factor
        self.max_results = max_results
        self.search_engine = search_engine
        self.embedding_function = embedding_function
        
        # Default user agents to rotate
        self.user_agents = user_agents or [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
        ]
        
        # Session for requests
        self.session = requests.Session()
        
        # Cache for search results to avoid excessive requests
        self.search_cache = {}
        self.cache_ttl = 3600  # 1 hour cache TTL
        
        # Stats
        self.total_searches = 0
        self.successful_searches = 0
        self.cache_hits = 0
    
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
            # Additional check to avoid searching for personal or subjective questions
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

        if hasattr(self, '_last_search_query') and self._last_search_query == query:
            print(f"[Web] WARNING: Reusing same query as previous request: '{query}'")
        self._last_search_query = query

        if cache_key in self.search_cache:
            timestamp, results = self.search_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                self.cache_hits += 1
                print(f"[Web] Cache hit for query: '{query}'")
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
                print(f"[Web] Error: DuckDuckGo returned status code {response.status_code}")
                return []
                
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
                    print(f"[Web] Error extracting DuckDuckGo result: {e}")
            
            print(f"[Web] Found {len(results)} results from DuckDuckGo for query: '{query}'")
            return results
            
        except Exception as e:
            print(f"[Web] Error searching DuckDuckGo: {e}")
            return []
    
    def _search_google(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """
        Search Google for results (simplified implementation).
        
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
                print(f"[Web] Error: Google returned status code {response.status_code}")
                return []
                
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract results (simplified - Google's HTML structure changes frequently)
            results = []
            
            # Find result divs (this selector may need updates as Google changes its HTML)
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
                    print(f"[Web] Error extracting Google result: {e}")
            
            print(f"[Web] Found {len(results)} results from Google for query: '{query}'")
            return results
            
        except Exception as e:
            print(f"[Web] Error searching Google: {e}")
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
                print(f"[Web] Error: URL {url} returned status code {response.status_code}")
                return None
                
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script, style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
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
            print(f"[Web] Error fetching content from {url}: {e}")
            return None
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if self.embedding_function:
            # Use provided embedding function
            return self.embedding_function(text)
        elif self.memory_manager and hasattr(self.memory_manager, 'generate_embedding'):
            # Use memory manager's embedding function
            return self.memory_manager.generate_embedding(text)
        else:
            # Fallback to random embedding (should be replaced in production)
            print("[Web] Warning: No embedding function available, using random embeddings")
            return np.random.random(384)  # Common embedding dimension
    
    def compare_vectors(self, query_vector: np.ndarray, result_vectors: List[Tuple[np.ndarray, Dict]]) -> List[Dict]:
        """
        Compare query vector with result vectors and score by relevance.
        
        Args:
            query_vector: Vector representation of the query
            result_vectors: List of (vector, metadata) tuples
            
        Returns:
            List of results with similarity scores
        """
        scored_results = []
        
        # Normalize query vector
        query_vector_norm = query_vector / np.linalg.norm(query_vector)
        
        for vec, metadata in result_vectors:
            # Normalize result vector
            vec_norm = vec / np.linalg.norm(vec)
            
            # Calculate cosine similarity
            similarity = np.dot(query_vector_norm, vec_norm)
            
            # Apply sharpening if enabled
            if self.vector_sharpening_factor > 0:
                # Enhanced similarity with sharpening
                if similarity > 0.7:
                    # Boost high similarities
                    similarity = similarity + (similarity - 0.7) * self.vector_sharpening_factor
                elif similarity < 0.3:
                    # Penalize low similarities
                    similarity = similarity - (0.3 - similarity) * self.vector_sharpening_factor
                    
                # Ensure valid range
                similarity = max(0.0, min(1.0, similarity))
            
            # Add to results with similarity score
            result = metadata.copy()
            result['similarity'] = float(similarity)
            scored_results.append(result)
        
        # Sort by similarity score
        scored_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return scored_results
    
    def compare_vectors_enhanced(
        self,
        query_vector: np.ndarray,
        result_vectors: List[Tuple[np.ndarray, Dict]],
        entities: List[str] = None,
        min_threshold: float = 0.4,
        diversity_factor: float = 0.1
    ) -> List[Dict]:
        """
        Enhanced vector comparison with entity-boosting, diversity promotion,
        and improved similarity calculation.

        Args:
            query_vector: Vector representation of the query
            result_vectors: List of (vector, metadata) tuples
            entities: List of extracted entities from the query
            min_threshold: Minimum similarity threshold
            diversity_factor: How much to penalize similar results (0-1)

        Returns:
            List of scored and ranked results
        """
        if not entities:
            entities = []

        # Convert entities to lowercase for matching
        entities_lower = [e.lower() for e in entities]

        # Initial scoring pass
        scored_results = []

        # Normalize query vector
        query_vector_norm = query_vector / np.linalg.norm(query_vector)

        for vec, metadata in result_vectors:
            # Normalize result vector
            vec_norm = vec / np.linalg.norm(vec)

            # Calculate cosine similarity
            similarity = np.dot(query_vector_norm, vec_norm)

            # Skip if below threshold
            if similarity < min_threshold:
                continue

            # Calculate entity matching score
            entity_score = 0.0
            result_text = (metadata.get('title', '') + ' ' + metadata.get('snippet', '')).lower()

            # Count entity matches and prioritize exact matches
            matched_entities = 0
            for entity in entities_lower:
                if entity in result_text:
                    matched_entities += 1
                    # Proximity boost for entities near the beginning
                    pos = result_text.find(entity)
                    if pos >= 0:
                        proximity_factor = max(0.0, 1.0 - (pos / min(100, len(result_text))))
                        entity_score += 0.1 * proximity_factor

            # Calculate entity coverage proportion
            if entities:
                entity_coverage = matched_entities / len(entities)
                entity_score += entity_coverage * 0.2

            # Apply sharpening if enabled
            if self.vector_sharpening_factor > 0:
                # Enhanced similarity with sharpening
                if similarity > 0.7:
                    # Boost high similarities
                    similarity_boost = (similarity - 0.7) * self.vector_sharpening_factor
                    similarity = similarity + similarity_boost
                elif similarity < 0.3:
                    # Penalize low similarities
                    similarity_penalty = (0.3 - similarity) * self.vector_sharpening_factor
                    similarity = similarity - similarity_penalty

                # Ensure valid range
                similarity = max(0.0, min(1.0, similarity))

            # Combine with entity score - entity matching can significantly boost results
            combined_score = similarity + entity_score

            # Add to results with similarity score
            result = metadata.copy()
            result['similarity'] = float(combined_score)
            result['base_similarity'] = float(similarity)
            result['entity_score'] = float(entity_score)
            result['matched_entities'] = matched_entities

            scored_results.append(result)

        # If no results passed threshold, return empty list
        if not scored_results:
            return []

        # Sort by combined score
        scored_results.sort(key=lambda x: x['similarity'], reverse=True)

        # Apply diversity penalty in second pass (avoid similar results)
        if len(scored_results) > 1 and diversity_factor > 0:
            # Start with first result
            diversified_results = [scored_results[0]]
            remaining = scored_results[1:]

            while remaining and len(diversified_results) < len(scored_results):
                # Calculate diversity penalty based on similarity to already included results
                for candidate in remaining:
                    candidate_text = f"{candidate.get('title', '')} {candidate.get('snippet', '')}"

                    # Calculate text overlap penalty
                    max_overlap = 0.0
                    for included in diversified_results:
                        included_text = f"{included.get('title', '')} {included.get('snippet', '')}"

                        # Simple Jaccard similarity for text overlap
                        candidate_words = set(candidate_text.lower().split())
                        included_words = set(included_text.lower().split())

                        if not candidate_words or not included_words:
                            continue

                        overlap = len(candidate_words.intersection(included_words)) / len(candidate_words.union(included_words))
                        max_overlap = max(max_overlap, overlap)

                    # Apply diversity penalty
                    penalty = max_overlap * diversity_factor
                    candidate['diversity_penalty'] = float(penalty)
                    candidate['adjusted_similarity'] = max(0.0, candidate['similarity'] - penalty)

                # Re-sort remaining by adjusted score
                remaining.sort(key=lambda x: x['adjusted_similarity'], reverse=True)

                # Take best and continue
                if remaining:
                    next_result = remaining.pop(0)
                    diversified_results.append(next_result)

            return diversified_results
        else:
            return scored_results

    def enhance_response(self, 
                        query: str,
                        confidence_data: Dict[str, float],
                        domain: Optional[str] = None,
                        process_urls: bool = False,
                        messages: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Enhanced version of the response enhancement function with improved
        query processing and result filtering.

        Args:
            query: User query
            confidence_data: Model confidence metrics
            domain: Optional domain classification
            process_urls: Whether to fetch full content from URLs
            messages: Optional conversation history

        Returns:
            Dictionary with enhancement data
        """
        # Check if enhancement is needed
        if not self.should_enhance_with_web_knowledge(query, confidence_data, domain):
            return {
                'enhanced': False,
                'reason': 'confidence_sufficient',
                'web_results': []
            }

        # Extract entities first - these are critical for relevance
        entities = self._extract_entities(query)

        # Extract seo_friendly_query from query
        seo_friendly_query = self.create_seo_friendly_sentence(query, messages)
        seo_friendly_query = self._strip_preambles_strictly(seo_friendly_query)

        # Generate embedding for original query
        query_vector = self.generate_embedding(query)

        # Generate embedding for seo_friendly_query
        keyword_vector = self.generate_embedding(seo_friendly_query)

        # Make sure entities are included in the search
        if entities and not any(entity.lower() in seo_friendly_query.lower() for entity in entities):
            search_query = f"{seo_friendly_query} {' '.join(entities)}"
            search_query = ' '.join(search_query.split()[:10])  # Limit length
        else:
            search_query = seo_friendly_query

        # Search the web
        print(f"[Web] Enhanced search query: {search_query}")
        search_results = self.search_web(search_query)

        if not search_results:
            return {
                'enhanced': False,
                'reason': 'no_search_results',
                'web_results': []
            }

        # Process results in parallel if fetching full content
        if process_urls:
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Fetch content from URLs in parallel
                future_to_result = {
                    executor.submit(self.fetch_content, result['url']): result
                    for result in search_results[:5]  # Limit to top 5 for performance
                }

                # Process completed futures
                for future in future_to_result:
                    result = future_to_result[future]
                    try:
                        content = future.result()
                        if content:
                            # Add full content to result
                            result['content'] = content
                    except Exception as e:
                        print(f"[Web] Error processing URL {result['url']}: {e}")

        # Create result vectors
        result_vectors = []

        for result in search_results:
            # Determine text to embed (include title for better precision)
            text_to_embed = f"{result['title']} {result['snippet']}"

            # Generate embedding
            embedding = self.generate_embedding(text_to_embed)

            # Add to result vectors
            result_vectors.append((embedding, result))

        # Use enhanced vector comparison
        scored_results = self.compare_vectors_enhanced(
            query_vector,
            result_vectors,
            entities=entities,
            min_threshold=0.35,
            diversity_factor=0.15
        )

        # Filter results by minimum similarity
        filtered_results = scored_results[:self.max_results]

        return {
            'enhanced': True,
            'reason': 'low_confidence',
            'query_vector': query_vector,
            'web_results': filtered_results,
            'all_results_count': len(search_results),
            'filtered_count': len(filtered_results),
            'entities': entities
        }

    def compare_vectors_with_dual_verification(
        self,
        query_vector: np.ndarray,
        keyword_vector: np.ndarray,
        result_vectors: List[Tuple[np.ndarray, Dict]],
        query_weight: float = 0.7,
        keyword_weight: float = 0.3,
        min_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Compare result vectors against both query and keywords vectors.

        Args:
            query_vector: Vector representation of the original query
            keyword_vector: Vector representation of the keywords
            result_vectors: List of (vector, metadata) tuples
            query_weight: Weight for query similarity (0-1)
            keyword_weight: Weight for keyword similarity (0-1)
            min_threshold: Minimum similarity threshold

        Returns:
            List of results with combined similarity scores
        """
        scored_results = []

        # Normalize vectors
        query_vector_norm = query_vector / np.linalg.norm(query_vector)
        keyword_vector_norm = keyword_vector / np.linalg.norm(keyword_vector)

        for vec, metadata in result_vectors:
            # Normalize result vector
            vec_norm = vec / np.linalg.norm(vec)

            # Calculate similarities
            query_similarity = np.dot(query_vector_norm, vec_norm)
            keyword_similarity = np.dot(keyword_vector_norm, vec_norm)

            # Calculate weighted combined similarity
            combined_similarity = (query_weight * query_similarity) + (keyword_weight * keyword_similarity)

            # Apply sharpening if enabled
            if self.vector_sharpening_factor > 0:
                # Enhanced similarity with sharpening
                if combined_similarity > 0.7:
                    # Boost high similarities
                    combined_similarity = combined_similarity + (combined_similarity - 0.7) * self.vector_sharpening_factor
                elif combined_similarity < 0.3:
                    # Penalize low similarities
                    combined_similarity = combined_similarity - (0.3 - combined_similarity) * self.vector_sharpening_factor

                # Ensure valid range
                combined_similarity = max(0.0, min(1.0, combined_similarity))

            # Only include results above minimum threshold
            if combined_similarity >= min_threshold:
                # Add to results with similarity score
                result = metadata.copy()
                result['similarity'] = float(combined_similarity)
                result['query_similarity'] = float(query_similarity)
                result['keyword_similarity'] = float(keyword_similarity)
                scored_results.append(result)

        # Sort by similarity score
        scored_results.sort(key=lambda x: x['similarity'], reverse=True)

        return scored_results
    
    def format_web_results_for_context(self, enhancement_data: Dict[str, Any], max_results: int = 5) -> str:
        """
        Enhanced formatting of web results for better context integration
        with clearer guidance for the model.

        Args:
            enhancement_data: Web enhancement data
            max_results: Maximum number of results to include

        Returns:
            Formatted string for context
        """
        if not enhancement_data.get('enhanced', False):
            return ""

        web_results = enhancement_data.get('web_results', [])

        if not web_results:
            return ""

        # Extract entities for special handling
        entities = enhancement_data.get('entities', [])
        entity_str = '", "'.join(entities) if entities else ""

        # Format results with clearer hierarchy and instructions
        context = "WEB SEARCH RESULTS\n"
        context += "----------------\n"
        context += "Please synthesize a comprehensive answer using the following information from the web.\n\n"

        # Group results by relevance tier
        primary_sources = []
        supporting_sources = []

        for result in web_results:
            if result.get('similarity', 0) > 0.7 or result.get('matched_entities', 0) > 0:
                primary_sources.append(result)
            else:
                supporting_sources.append(result)

        # Add primary sources first with clear labeling
        if primary_sources:
            context += "PRIMARY INFORMATION:\n"

            for i, result in enumerate(primary_sources[:3]):  # Limit to top 3 primary sources
                similarity = result.get('similarity', 0.0)
                title = result.get('title', 'Untitled')
                snippet = result.get('snippet', 'No description available')
                url = result.get('url', '')

                # Format with clear importance indicators
                context += f"{i+1}. [{title}]\n"
                context += f"   {snippet}\n"

                # Include key metadata that might help the model assess source quality
                if url:
                    domain = url.split('/')[2] if len(url.split('/')) > 2 else url
                    context += f"   Source: {domain}\n"

                context += "\n"

        # Add supporting sources if available
        remaining_slots = max(0, max_results - len(primary_sources))
        if supporting_sources and remaining_slots > 0:
            context += "SUPPORTING INFORMATION:\n"

            for i, result in enumerate(supporting_sources[:remaining_slots]):
                title = result.get('title', 'Untitled')
                snippet = result.get('snippet', 'No description available')

                # Format supporting sources more concisely
                context += f"{i+1}. {title}: {snippet}\n"

            context += "\n"

        # Add explicit instructions to help the model use the information effectively
        context += "INSTRUCTIONS:\n"
        context += "- Synthesize a direct answer from the information above\n"

        if entities:
            # For entity-specific queries, add focused guidance
            context += f'- Focus on information about "{entity_str}"\n'

        context += "- If the information appears contradictory, acknowledge this\n"
        context += "- If search results don't contain relevant information, state this clearly\n"
        context += "- Do not hallucinate information not present in the search results\n"
        
        return context
    
    def add_web_results_to_memory(self, 
                                user_id: str, 
                                query: str, 
                                enhancement_data: Dict[str, Any],
                                min_similarity: float = 0.5) -> int:
        """
        Add relevant web results to memory for future use.
        
        Args:
            user_id: User identifier
            query: Original query
            enhancement_data: Web enhancement data
            min_similarity: Minimum similarity threshold for memory storage
            
        Returns:
            Number of memories added
        """
        if not enhancement_data.get('enhanced', False) or not self.memory_manager:
            return 0
            
        web_results = enhancement_data.get('web_results', [])
        added_count = 0
        
        for result in web_results:
            # Skip low relevance results
            if result.get('similarity', 0.0) < min_similarity:
                continue
                
            # Create memory entry
            memory_text = f"{result['title']} - {result['snippet']}"
            
            # Add source URL
            if 'url' in result:
                memory_text += f" Source: {result['url']}"
                
            # Add to memory with web metadata
            try:
                memories_added = self.memory_manager.add_memory(
                    user_id,
                    query,
                    memory_text,
                    memory_type="web_knowledge",
                    attributes={
                        'source': result.get('source', 'web'),
                        'url': result.get('url', ''),
                        'similarity': result.get('similarity', 0.0),
                        'web_timestamp': time.time()
                    }
                )
                
                added_count += memories_added
            except Exception as e:
                print(f"[Web] Error adding web result to memory: {e}")
        
        if added_count > 0:
            print(f"[Web] Added {added_count} web knowledge items to memory")
            
        return added_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about web knowledge enhancement"""
        return {
            'total_searches': self.total_searches,
            'successful_searches': self.successful_searches,
            'cache_hits': self.cache_hits,
            'cache_size': len(self.search_cache),
            'success_rate': self.successful_searches / max(1, self.total_searches),
            'cache_hit_rate': self.cache_hits / max(1, self.total_searches)
        }

    def create_seo_friendly_sentence(self, query: str, messages=None, max_words: int = 8) -> str:
        """
        Create an SEO-friendly search query with comprehensive fallbacks and entity extraction.

        Args:
            query: The user query
            messages: Optional conversation history
            max_words: Maximum words in the search query (increased from 5 to 8)

        Returns:
            SEO-friendly search query
        """
        # Extract entities (titles, proper nouns) first - these are highest priority
        entities = self._extract_entities(query)

        # Log entities found for debugging
        if entities:
            print(f"[Web] Extracted entities: {entities}")

        # Try multiple methods and build a composite query

        # First try to extract from fractal memory for similar past queries
        fractal_query = self._extract_from_fractal_memory(query)
        if fractal_query:
            print(f"[Web] Using fractal memory query: {fractal_query}")

            # Make sure entity is included
            if entities and not any(entity.lower() in fractal_query.lower() for entity in entities):
                enhanced_query = f"{fractal_query} {' '.join(entities)}"
                enhanced_query = ' '.join(enhanced_query.split()[:max_words])
                return enhanced_query

            return fractal_query

        # Try pattern matching, which is more context-aware
        pattern_query = self._extract_with_patterns(query)
        if pattern_query:
            print(f"[Web] Using pattern-based query: {pattern_query}")

            # Make sure entity is included
            if entities and not any(entity.lower() in pattern_query.lower() for entity in entities):
                enhanced_query = f"{pattern_query} {' '.join(entities)}"
                enhanced_query = ' '.join(enhanced_query.split()[:max_words])
                return enhanced_query

            return pattern_query

        # Last resort - use rule-based key term extraction
        key_terms = self._extract_key_terms(query, max_words)

        # Ensure entities are included in key terms
        if entities and not any(entity.lower() in key_terms.lower() for entity in entities):
            # Prioritize entities and combine with key terms
            terms = key_terms.split()
            combined_terms = entities + [term for term in terms if term.lower() not in [e.lower() for e in entities]]
            key_terms = ' '.join(combined_terms[:max_words])

        print(f"[Web] Using enhanced key terms extraction: {key_terms}")
        return key_terms

    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract important entities from a query (titles, proper nouns, etc.)

        Args:
            query: The user query

        Returns:
            List of extracted entities
        """
        import re

        entities = []

        # Extract quoted text - highest priority
        quoted_text = re.findall(r'"([^"]+)"', query)
        quoted_text.extend(re.findall(r"'([^']+)'", query))
        entities.extend(quoted_text)

        # Extract proper nouns (sequences of capitalized words)
        proper_noun_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        proper_nouns = re.findall(proper_noun_pattern, query)
        for noun in proper_nouns:
            # Check if it's not just a capitalized first word of sentence
            words = query.split()
            if noun in words and words.index(noun) > 0:
                entities.append(noun)

        # Extract years - important for temporal context
        years = re.findall(r'\b(19\d\d|20\d\d)\b', query)
        entities.extend(years)

        # Extract currency codes and specific items
        currency_codes = re.findall(r'\b([A-Z]{3})\b', query)  # 3-letter currency codes

        # Filter currency codes - only include known ones
        known_currencies = {'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD', 'BAM'}
        filtered_currencies = [code for code in currency_codes if code in known_currencies]
        entities.extend(filtered_currencies)

        return entities

    def _extract_from_fractal_memory(self, query: str) -> Optional[str]:
        """
        Enhanced function to extract from fractal memory with retry mechanisms.

        Args:
            query: The current query

        Returns:
            Extracted search terms or None if not found
        """
        try:
            if not hasattr(self, 'memory_manager') or not self.memory_manager:
                print("[Web] Memory manager not available for fractal retrieval")
                return None

            # Ensure we have a user ID
            current_user_id = getattr(self.chat, 'current_user_id', 'default_user') if hasattr(self, 'chat') else 'default_user'
            print(f"[Web] Attempting fractal memory retrieval for user: {current_user_id}")

            # Create fractal-enabled store if not already using one
            store = self.memory_manager._get_user_store(current_user_id)

            if not hasattr(store, 'fractal_enabled') or not store.fractal_enabled:
                print("[Web] Fractal embeddings not enabled in vector store")
                return None

            # Generate embedding for query
            query_embedding = self.memory_manager.generate_embedding(query)

            # Log diagnostics
            print(f"[Web] Fractal search enabled: {getattr(store, 'fractal_enabled', False)}")
            print(f"[Web] Store total documents: {len(getattr(store, 'documents', []))}")

            # Conduct a broader, multi-level fractal search with lower threshold to find anything related
            print("[Web] Executing multi-level fractal search")
            search_results = store.search(
                query_embedding,
                top_k=5,
                min_similarity=0.60,  # Lower threshold to find more potential matches
                multi_level_search=True
            )

            print(f"[Web] Fractal search returned {len(search_results)} results")

            # Process results
            for result in search_results:
                similarity = result.get('similarity', 0)
                print(f"[Web] Fractal result: sim={similarity:.2f}, level={result.get('level', 0)}")

                metadata = result.get('metadata', {})

                # Check for search terms in metadata
                if 'search_term' in metadata:
                    print(f"[Web] Found search term in memory: {metadata['search_term']}")
                    return metadata['search_term']

                # Check for web knowledge memories
                if metadata.get('memory_type') == 'web_knowledge':
                    content = result.get('text', '')
                    # Extract potential search terms
                    first_sentence = content.split('.')[0].strip()
                    if len(first_sentence.split()) <= 10:
                        print(f"[Web] Extracted search term from web memory: {first_sentence}")
                        return first_sentence

                # Look for source queries that might be similar
                if 'source_query' in metadata:
                    source_query = metadata['source_query']
                    # Only use if reasonably short
                    if len(source_query.split()) <= 15:
                        # Extract key terms from the source query instead of using it directly
                        extracted_terms = self._extract_key_terms(source_query, 7)
                        print(f"[Web] Using terms from similar query: {extracted_terms}")
                        return extracted_terms

            print("[Web] No useful search terms found in fractal memory")
            return None
        except Exception as e:
            print(f"[Web] Error in fractal memory extraction: {e}")
            return None

    def _extract_with_patterns(self, query: str) -> Optional[str]:
        """
        Extract search terms using improved pattern matching with entity awareness.

        Args:
            query: The user query

        Returns:
            Extracted search terms or None if not matched
        """
        import re

        # First extract any entities in the query - critical for accuracy
        entities = self._extract_entities(query)
        entity_text = ' '.join(entities) if entities else ''

        # Normalize query - remove quotes since we already extracted them
        clean_query = re.sub(r'["\']+', '', query.lower())

        # Enhanced patterns with better entity handling
        patterns = [
            # Movie/book character patterns
            (r'(?:who|what) (?:was|is|plays|played) (?:the)? (.+?) (?:in|from) (?:the)? ((?:movie|film|book|novel|show|series).*)',
             r'\1 \2 character', 0.95),

            # Movie/character with year
            (r'(?:who|what) (?:was|is|plays|played) (?:the)? (.+?) (?:in|from) (?:the)? (.+?) ((?:19|20)\d\d)',
             r'\1 \2 \3 character', 0.95),

            # Main character specifically
            (r'(?:who|what) (?:was|is) (?:the)? (main|lead|primary|principal) character (?:in|from) (.+)',
             r'\2 main character', 0.95),

            # Specific actor query
            (r'who (?:plays|played) (.+?) (?:in|from) (.+)',
             r'\1 \2 actor', 0.95),

            # When was something established/founded
            (r'when was (?:the)? (.+?) (?:established|founded|created|formed|setup|started|introduced)',
             r'\1 established date history', 0.9),

            # General patterns with proper context preservation
            (r'who (?:is|was) (.+)', r'\1 biography', 0.85),
            (r'when did (.+)', r'\1 date when', 0.85),
            (r'where is (.+)', r'\1 location where', 0.85),
            (r'what is(?: an?)? (.+)', r'\1 definition', 0.85),
            (r'how (?:do|to|can) I? (.+)', r'\1 how to', 0.85),
        ]

        # Check for matches with confidence
        best_match = None
        best_confidence = 0

        for pattern, replacement, confidence in patterns:
            match = re.search(pattern, clean_query)
            if match and confidence > best_confidence:
                try:
                    result = re.sub(pattern, replacement, clean_query)
                    # Clean up any extra spaces
                    result = ' '.join(result.split())
                    best_match = result
                    best_confidence = confidence
                except Exception:
                    continue

        # If we have a match and entities, ensure entities are included
        if best_match and entity_text and not any(entity.lower() in best_match.lower() for entity in entities):
            best_match = f"{best_match} {entity_text}"

        # Limit to 8 words
        if best_match:
            best_match = ' '.join(best_match.split()[:8])

        return best_match

    def _extract_key_terms(self, query: str, max_terms: int = 8) -> str:
        """
        Extract key terms with improved entity recognition and prioritization.

        Args:
            query: User query
            max_terms: Maximum number of terms

        Returns:
            Space-separated key terms
        """
        import re
        from collections import Counter

        # First extract entities - they get highest priority
        entities = self._extract_entities(query)

        # Remove entities from query to avoid duplication
        clean_query = query
        for entity in entities:
            clean_query = clean_query.replace(entity, '')

        # Normalize and clean query
        query_lower = clean_query.lower()
        query_clean = re.sub(r'[^\w\s]', ' ', query_lower)
        words = query_clean.split()

        # Extended stop words list (use the same list as in the previous implementation)
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'up', 'down',
            'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'from',
            'of', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
            'what', 'who', 'where', 'when', 'why', 'how', 'which', 'do', 'does', 'did',
            'can', 'could', 'would', 'should', 'will', 'shall', 'may', 'might', 'must',
            'me', 'my', 'mine', 'your', 'yours', 'his', 'her', 'hers', 'their', 'them',
            'i', 'we', 'us', 'you', 'he', 'she', 'it', 'they', 'am', 'this', 'that',
            'these', 'those', 'had', 'has', 'have', 'having', 'get', 'gets', 'got',
            'just', 'more', 'most', 'other', 'some', 'such', 'only', 'than', 'too',
            'very', 'really', 'okay', 'ok', 'say', 'says', 'said', 'show', 'tell',
            'know', 'think', 'see', 'look', 'try', 'help', 'explain', 'describe',
            'hello', 'hi', 'hey', 'please', 'thanks', 'thank', 'answer'
        }

        # Get word frequencies for remaining words
        word_counts = Counter([word for word in words if word not in stop_words and len(word) > 2])

        # Score the words
        scored_words = {}
        for word, count in word_counts.items():
            # Position score - words at start are more important (0.5-1.0)
            try:
                position = words.index(word)
                position_score = 1.0 - (0.5 * position / max(1, len(words)))
            except ValueError:
                position_score = 0.5

            # Length score - longer words often more important (0.0-0.5)
            length_score = min(0.5, len(word) / 20)

            # Final score
            scored_words[word] = (count * position_score + length_score)

        # Get top terms
        top_words = [word for word, _ in sorted(scored_words.items(), key=lambda x: x[1], reverse=True)]

        # Combine entities and top words, ensuring no duplication
        all_terms = entities.copy()
        for word in top_words:
            if word.lower() not in [term.lower() for term in all_terms]:
                all_terms.append(word)

        # Check if we have enough terms
        if len(all_terms) < 2 and top_words:
            # If extremely few terms, add some more from top words
            additional_terms = [w for w in words if w not in stop_words
                               and w not in [t.lower() for t in all_terms]
                               and len(w) > 2][:max_terms - len(all_terms)]
            all_terms.extend(additional_terms)

        # Return limited terms
        return ' '.join(all_terms[:max_terms])


    def _clean_search_term(self, response: str, original_query: str) -> str:
        """Aggressively clean a search term response."""
        # Remove any text before a colon
        if ":" in response:
            response = response.split(":", 1)[1].strip()

        # Remove quotes
        response = response.replace('"', '').replace("'", "")

        # Remove common prefixes and labels
        prefixes = ["search term", "search query", "query", "keywords", "search for",
                    "i would search for", "search", "term", "look for"]

        response_lower = response.lower()
        for prefix in prefixes:
            if response_lower.startswith(prefix):
                response = response[len(prefix):].strip()
                # Remove any leading punctuation
                response = response.lstrip(',:;.- ')

        # If it's just repeating the original query, extract key terms
        if response.lower() == original_query.lower():
            return self._extract_key_terms(original_query)

        return response.strip()

    def _extract_conversation_context(self, messages: List[Dict]) -> str:
        """Extract focused context from conversation history."""
        if not messages or len(messages) <= 2:
            return ""

        # Check if current query contains math expression
        current_query = messages[-1].get("content", "").strip() if messages[-1].get("role") == "user" else ""
        contains_math = bool(re.search(r'\d+\s*[\+\-\*\/]\s*\d+', current_query))

        # Take only the last 2-3 exchanges for relevance
        relevant_messages = messages[-5:] if len(messages) > 5 else messages[1:]

        # Extract subjects and entities
        context = []
        entities = set()

        for msg in relevant_messages:
            content = msg.get("content", "").strip()
            if not content:
                continue

            # Skip previous mathematical expressions if current query contains math
            if contains_math and re.search(r'\d+\s*[\+\-\*\/]\s*\d+', content) and content != current_query:
                continue

            # Extract potential entities (capitalized words)
            words = content.split()
            for word in words:
                clean_word = word.strip(",.?!():;'\"-")
                if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                    entities.add(clean_word)

            # Add truncated content
            if len(content) > 100:
                first_sentence = content.split('.')[0]
                context.append(first_sentence[:100])
            else:
                context.append(content)

        # Format the context
        formatted_context = "\n".join(context[-2:])  # Just latest exchanges

        # Add extracted entities if they exist
        if entities:
            entity_text = ", ".join(list(entities)[:5])  # Limit to 5 most recent
            formatted_context += f"\nEntities mentioned: {entity_text}"

        return formatted_context

        # Format the context
        formatted_context = "\n".join(context[-2:])  # Just latest exchanges

        # Add extracted entities if they exist
        if entities:
            entity_text = ", ".join(list(entities)[:5])  # Limit to 5 most recent
            formatted_context += f"\nEntities mentioned: {entity_text}"

        return formatted_context



    def _strip_preambles_strictly(self, text: str) -> str:
        """
        Aggressively strip any preambles or formatting from the model's output.

        Args:
            text: Text to clean

        Returns:
            Cleaned text with preambles removed
        """
        # Strip common prefixes
        prefixes = [
            "SEO-Friendly Query:", "SEO Friendly Query:", "SEO Query:",
            "Search Term:", "Query:", "Keywords:", "Search Keywords:",
            "Generated Search Term:"
        ]

        # Check for each prefix and remove it
        for prefix in prefixes:
            if prefix.lower() in text.lower():
                parts = text.lower().split(prefix.lower(), 1)
                if len(parts) > 1:
                    text = parts[1].strip()

        # Remove section headers that might appear
        lines = text.split('\n')
        filtered_lines = []
        for line in lines:
            if not line.endswith(':') and not line.startswith('#'):
                filtered_lines.append(line)

        text = ' '.join(filtered_lines)

        # Remove quotation marks
        text = text.replace('"', '').replace("'", "")

        # Remove any remaining non-query content
        keywords_marker = "keywords:"
        if keywords_marker in text.lower():
            text = text.lower().split(keywords_marker)[0].strip()

        return text.strip()