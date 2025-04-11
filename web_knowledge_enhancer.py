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
    
    def enhance_response(self, 
                        query: str, 
                        confidence_data: Dict[str, float],
                        domain: Optional[str] = None,
                        process_urls: bool = False) -> Dict[str, Any]:
        """
        Enhance model response with web knowledge.
        
        Args:
            query: User query
            confidence_data: Model confidence metrics
            domain: Optional domain classification
            process_urls: Whether to fetch full content from URLs
            
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
        
        # Generate embedding for query
        query_vector = self.generate_embedding(query)
        
        # Search the web
        search_results = self.search_web(query)
        
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
            # Determine text to embed based on available content
            if process_urls and 'content' in result:
                # Use title + full content if available
                text_to_embed = f"{result['title']} {result['content']}"
            else:
                # Use title + snippet
                text_to_embed = f"{result['title']} {result['snippet']}"
            
            # Generate embedding
            embedding = self.generate_embedding(text_to_embed)
            
            # Add to result vectors
            result_vectors.append((embedding, result))
        
        # Compare vectors
        scored_results = self.compare_vectors(query_vector, result_vectors)
        
        # Filter results by minimum similarity
        filtered_results = [
            result for result in scored_results 
            if result['similarity'] >= 0.3  # Minimum similarity threshold
        ]
        
        return {
            'enhanced': True,
            'reason': 'low_confidence',
            'query_vector': query_vector,
            'web_results': filtered_results[:self.max_results],
            'all_results_count': len(search_results),
            'filtered_count': len(filtered_results)
        }
    
    def format_web_results_for_context(self, enhancement_data: Dict[str, Any], max_results: int = 5) -> str:
        """
        Format web results for inclusion in the context.
        
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
            
        # Format results for context
        context = "INFORMATION FROM WEB SEARCH:\n\n"
        
        for i, result in enumerate(web_results[:max_results]):
            similarity = result.get('similarity', 0.0)
            relevance_indicator = "★★★" if similarity > 0.7 else ("★★" if similarity > 0.5 else "★")
            
            # Add result with relevance indicator
            context += f"{i+1}. {relevance_indicator} {result['title']}\n"
            context += f"   {result['snippet']}\n"
            
            # Add source URL (shortened)
            url = result['url']
            if len(url) > 70:
                url = url[:67] + "..."
            context += f"   Source: {url}\n\n"
        
        context += f"[End of web search results. Found {len(web_results)} relevant sources.]\n"
        
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