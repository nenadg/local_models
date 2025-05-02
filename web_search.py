"""
Web search integration for local LLM chat with memory enhancement.
Provides web scraping, content extraction, and memory integration.
"""

import re
import time
import json
import requests
import numpy as np
import hashlib
import threading
import traceback
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse

class WebSearchManager:
    """
    Manages web search operations and integration with memory system.
    Uses a stateless approach for simple integration with existing components.
    """
    
    def __init__(
        self,
        memory_manager=None,
        question_classifier=None,
        similarity_enhancement_factor: float = 0.3,
        search_url_template: str = "https://duckduckgo.com/html/?q={query}",
        max_search_results: int = 10,
        max_content_chars: int = 8000,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
    ):
        """
        Initialize the web search manager.
        
        Args:
            memory_manager: Optional MemoryManager for storing results
            question_classifier: Optional QuestionClassifier for keyword generation
            similarity_enhancement_factor: Factor for similarity enhancement
            search_url_template: Template for search URL
            max_search_results: Maximum number of search results to process
            max_content_chars: Maximum characters to extract per page
            user_agent: User agent string for requests
        """
        self.memory_manager = memory_manager
        self.question_classifier = question_classifier
        self.similarity_enhancement_factor = similarity_enhancement_factor
        self.search_url_template = search_url_template
        self.max_search_results = max_search_results
        self.max_content_chars = max_content_chars
        self.user_agent = user_agent
        
        # Headers for requests
        self.headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Cache for recent searches
        self._search_cache = {}
        self._cache_lock = threading.RLock()
        self._cache_expiry_seconds = 300  # Cache results for 5 minutes
        
    def get_time(self) -> str:
        """Get formatted timestamp for logging."""
        return datetime.now().strftime("[%d/%m/%y %H:%M:%S]") + ' [WebSearch]'
    
    def search(
        self, 
        query: str, 
        include_content: bool = True,
        max_results: Optional[int] = None,
        max_content_length: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform a web search for the given query.
        
        Args:
            query: The search query
            include_content: Whether to fetch full page content
            max_results: Maximum number of results to return
            max_content_length: Maximum content length to extract
            
        Returns:
            List of search result dictionaries
        """
        search_terms = self._generate_search_keywords(query)
        print(f"{self.get_time()} Searching for: {search_terms}")
        
        # Check cache first
        cache_key = f"search_{hashlib.md5(search_terms.encode()).hexdigest()}"
        with self._cache_lock:
            if cache_key in self._search_cache:
                cached_data = self._search_cache[cache_key]
                cache_time = cached_data.get('timestamp', 0)
                if time.time() - cache_time < self._cache_expiry_seconds:
                    print(f"{self.get_time()} Using cached results")
                    return cached_data.get('results', [])
        
        # Set limits
        max_results = max_results or self.max_search_results
        max_content_length = max_content_length or self.max_content_chars
        
        # Get search results
        search_url = self.search_url_template.format(query=quote_plus(search_terms))
        try:
            print(f"{self.get_time()} Fetching search results...")
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parse search results
            results = self._parse_search_results(response.text, max_results)
            
            # Fetch content if requested
            if include_content and results:
                print(f"{self.get_time()} Fetching content for {len(results)} results...")
                self._fetch_content_for_results(results, max_content_length)
            
            # Cache results
            with self._cache_lock:
                self._search_cache[cache_key] = {
                    'timestamp': time.time(),
                    'results': results
                }
                
                # Clean up old cache entries
                self._clean_cache()
            
            return results
            
        except Exception as e:
            print(f"{self.get_time()} Error during search: {e}")
            traceback.print_exc()
            return []
    
    def _parse_search_results(self, html: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Parse search results from HTML.
        
        Args:
            html: Search results HTML
            max_results: Maximum results to return
            
        Returns:
            List of result dictionaries
        """
        results = []
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Handle DuckDuckGo results
            result_elements = soup.select('.result')
            if not result_elements:
                # Try alternative selectors for other engines
                result_elements = soup.select('.g') or soup.select('.result__body') or soup.select('.web-result')
            
            for i, element in enumerate(result_elements):
                if i >= max_results:
                    break
                    
                # Try to extract title, url, and snippet with fallbacks
                title_elem = (element.select_one('.result__title') or 
                             element.select_one('.result-title') or 
                             element.select_one('h3') or 
                             element.select_one('h2'))
                
                link_elem = (element.select_one('.result__url') or 
                           element.select_one('a.result__a') or 
                           element.select_one('a'))
                
                snippet_elem = (element.select_one('.result__snippet') or 
                               element.select_one('.result-snippet') or 
                               element.select_one('.snippet'))
                
                # Extract data with fallbacks
                title = title_elem.get_text(strip=True) if title_elem else "No title"
                
                url = None
                if link_elem and link_elem.has_attr('href'):
                    url = link_elem['href']
                    # Clean tracking parameters if present
                    url = self._clean_url(url)
                
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                
                # Skip if no valid URL
                if not url or not url.startswith('http'):
                    continue
                
                # Create result object
                result = {
                    'title': title,
                    'url': url,
                    'snippet': snippet,
                    'source': self._get_domain(url),
                    'timestamp': time.time(),
                    'fetched_content': False,
                    'content': "",
                    'tokens': []  # Will be filled later if needed
                }
                
                results.append(result)
                
            print(f"{self.get_time()} Extracted {len(results)} search results")
            
        except Exception as e:
            print(f"{self.get_time()} Error parsing search results: {e}")
            traceback.print_exc()
            
        return results
    
    def _fetch_content_for_results(self, results: List[Dict[str, Any]], max_length: int):
        """
        Fetch and extract content for search results.
        
        Args:
            results: List of search result dictionaries
            max_length: Maximum content length to extract
        """
        for result in results:
            url = result.get('url')
            if not url:
                continue
                
            try:
                # Fetch the page
                print(f"{self.get_time()} Fetching content from: {url}")
                response = requests.get(url, headers=self.headers, timeout=8)
                response.raise_for_status()
                
                # Extract content
                content = self._extract_main_content(response.text, url, max_length)
                
                # Store in result
                result['content'] = content
                result['fetched_content'] = True
                result['fetch_time'] = time.time()
                
                # Add tokens using a simple approach
                # In a real implementation, you'd use the tokenizer from your existing code
                result['tokens'] = content.split()
                
            except Exception as e:
                print(f"{self.get_time()} Error fetching content for {url}: {e}")
    
    def _extract_main_content(self, html: str, url: str, max_length: int) -> str:
        """
        Extract main content from webpage.
        
        Args:
            html: Web page HTML
            url: URL of the page
            max_length: Maximum content length
            
        Returns:
            Extracted main content
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script, style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'header']):
                element.decompose()
            
            # Try to find main content using common selectors
            main_content = None
            
            # Try article first
            for selector in ['article', 'main', '.content', '#content', '.post', '.article', '.main-content']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            # If no main content found, use body
            if not main_content:
                main_content = soup.body
            
            if not main_content:
                return ""
            
            # Extract paragraphs
            paragraphs = main_content.find_all('p')
            
            # If no paragraphs found, use div text
            if not paragraphs:
                paragraphs = main_content.find_all('div')
            
            # Get text from paragraphs
            content = "\n\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            
            # Truncate if needed
            if len(content) > max_length:
                content = content[:max_length] + "..."
            
            return content
        
        except Exception as e:
            print(f"{self.get_time()} Error extracting content: {e}")
            return ""
    
    def _generate_search_keywords(self, query: str) -> str:
        """
        Generate optimized search keywords from natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            Optimized search keywords
        """
        # Get domain information if classifier is available
        domain = 'unknown'
        if self.question_classifier:
            try:
                domain_settings = self.question_classifier.get_domain_settings(query)
                domain = domain_settings.get('domain', 'unknown')
            except Exception as e:
                print(f"{self.get_time()} Error getting domain: {e}")
        
        # Clean the query
        clean_query = re.sub(r'\b(what|how|who|when|where|why|is|are|do|does|can|could|would|should)\b', '', query.lower())
        clean_query = re.sub(r'\b(the|a|an|in|on|at|by|for|of|with|about)\b', '', clean_query)
        clean_query = re.sub(r'\s+', ' ', clean_query).strip()
        
        # Apply domain-specific rules
        if domain == 'factual':
            # For factual queries, keep entity names and key terms
            keywords = self._extract_key_terms(query)
        elif domain in ['procedural', 'conceptual']:
            # For how-to or concept queries, add "guide" or "explained"
            base_terms = self._extract_key_terms(query)
            if "how to" in query.lower() or "how do" in query.lower():
                keywords = f"{base_terms} guide tutorial"
            else:
                keywords = f"{base_terms} explained definition"
        else:
            # Default approach
            keywords = self._extract_key_terms(query)
        
        # Add time markers for current/recent information
        if re.search(r'\b(current|latest|recent|now|today|update)\b', query.lower()):
            current_year = datetime.now().year
            current_month = datetime.now().strftime("%B")
            keywords = f"{keywords} {current_month} {current_year}"
        
        return keywords
    
    def _extract_key_terms(self, query: str) -> str:
        """
        Extract key terms from a query using simple NLP techniques.
        
        Args:
            query: Query text
            
        Returns:
            String of key terms
        """
        # Remove question words and common stop words
        clean_query = re.sub(r'\b(what|how|who|when|where|why|is|are|do|does|can|could|would|should)\b', '', query.lower())
        clean_query = re.sub(r'\b(the|a|an|in|on|at|by|for|of|with|about)\b', '', clean_query)
        clean_query = re.sub(r'[^\w\s]', '', clean_query)  # Remove punctuation
        clean_query = re.sub(r'\s+', ' ', clean_query).strip()  # Normalize whitespace
        
        # Split into words
        words = clean_query.split()
        
        # Filter out very short words
        words = [w for w in words if len(w) > 2]
        
        # Find potential entities (capitalized words in original query)
        entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
        
        # Combine terms
        if entities:
            # Preserve the case of entities
            combined_terms = " ".join(entities) + " " + " ".join(words)
        else:
            combined_terms = " ".join(words)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in combined_terms.split():
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
        
        # Combine and limit length
        result = " ".join(unique_terms)
        
        return result
    
    def _clean_url(self, url: str) -> str:
        """
        Clean tracking parameters from URL.
        
        Args:
            url: URL to clean
            
        Returns:
            Cleaned URL
        """
        try:
            # Parse URL
            parsed = urlparse(url)
            
            # Remove common tracking parameters
            tracking_params = ['utm_source', 'utm_medium', 'utm_campaign', 
                              'utm_term', 'utm_content', 'fbclid', 'gclid',
                              'ref', 'source', 'ref_src']
            
            # Get query parameters
            if parsed.query:
                # Parse query parameters
                params = dict(pair.split('=') for pair in parsed.query.split('&'))
                
                # Remove tracking parameters
                for param in tracking_params:
                    if param in params:
                        del params[param]
                
                # Rebuild query string
                query = '&'.join(f"{k}={v}" for k, v in params.items())
                
                # Rebuild URL
                cleaned_parts = list(parsed)
                cleaned_parts[4] = query
                cleaned_url = urlparse("").geturl()
            else:
                cleaned_url = url
                
            return cleaned_url
            
        except Exception:
            # If parsing fails, return original URL
            return url
    
    def _get_domain(self, url: str) -> str:
        """
        Extract domain name from URL.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Domain name
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
                
            return domain
            
        except Exception:
            return "unknown"
    
    def _clean_cache(self):
        """Clean expired entries from cache."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, data in self._search_cache.items():
            cache_time = data.get('timestamp', 0)
            if current_time - cache_time > self._cache_expiry_seconds:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._search_cache[key]
    
    def rank_results_by_relevance(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Rank search results by relevance to query.
        
        Args:
            results: List of search results
            query: Original query
            
        Returns:
            Ranked list of results
        """
        if not results:
            return []
            
        # Calculate relevance scores
        for result in results:
            # Start with base score
            score = 0.0
            
            # Score based on title match
            title = result.get('title', '').lower()
            query_terms = query.lower().split()
            
            # Title exact match bonus
            if query.lower() in title:
                score += 0.4
            
            # Title term match
            title_term_matches = sum(1 for term in query_terms if term in title)
            score += 0.1 * (title_term_matches / max(1, len(query_terms)))
            
            # Content relevance if available
            content = result.get('content', '')
            if content:
                # Simple term frequency scoring
                content_term_matches = sum(1 for term in query_terms if term in content.lower())
                score += 0.05 * (content_term_matches / max(1, len(query_terms)))
                
                # Length normalization
                content_len = len(content)
                if content_len > 100:
                    score += 0.1  # Favor non-trivial content
            
            # Apply similarity enhancement
            if self.similarity_enhancement_factor > 0:
                enhanced_score = self._enhance_similarity(score)
                result['relevance_score'] = enhanced_score
            else:
                result['relevance_score'] = score
        
        # Sort by relevance score
        ranked_results = sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return ranked_results
    
    def _enhance_similarity(self, similarity: float) -> float:
        """
        Apply non-linear enhancement to similarity scores.
        
        Args:
            similarity: Raw similarity score (0.0-1.0)
            
        Returns:
            Enhanced similarity score
        """
        # Skip if no enhancement requested
        if self.similarity_enhancement_factor <= 0:
            return similarity
        
        # Apply non-linear enhancement
        if similarity > 0.6:
            # Boost high similarities (more confident matches)
            boost = (similarity - 0.6) * self.similarity_enhancement_factor * 2.0
            enhanced = min(1.0, similarity + boost)
        elif similarity < 0.4:
            # Reduce low similarities (less confident matches)
            reduction = (0.4 - similarity) * self.similarity_enhancement_factor * 2.0
            enhanced = max(0.0, similarity - reduction)
        else:
            # Middle range - moderate effect
            deviation = (similarity - 0.5) * self.similarity_enhancement_factor
            enhanced = 0.5 + deviation
        
        return enhanced
    
    def search_and_add_to_memory(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search and add relevant results to memory.
        
        Args:
            query: Search query
            max_results: Maximum results to add to memory
            
        Returns:
            List of memory items added
        """
        if not self.memory_manager:
            print(f"{self.get_time()} No memory manager available")
            return []
        
        # Perform search
        results = self.search(query, include_content=True)
        
        # Rank by relevance
        ranked_results = self.rank_results_by_relevance(results, query)
        
        # Add top results to memory
        added_items = []
        
        for result in ranked_results[:max_results]:
            if not result.get('content'):
                continue
                
            # Create metadata
            metadata = {
                'source': 'web_search',
                'url': result.get('url'),
                'title': result.get('title'),
                'domain': result.get('source'),
                'query': query,
                'timestamp': time.time()
            }
            
            # Add to memory
            try:
                item_id = self.memory_manager.add(
                    content=result.get('content'),
                    metadata=metadata
                )
                
                if item_id:
                    result['memory_id'] = item_id
                    added_items.append(result)
                    print(f"{self.get_time()} Added result to memory: {result.get('title')}")
            except Exception as e:
                print(f"{self.get_time()} Error adding to memory: {e}")
        
        return added_items
    
    def format_for_context(self, results: List[Dict[str, Any]], query: str) -> str:
        """
        Format search results for inclusion in context.
        
        Args:
            results: List of search results
            query: Original query
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        # Rank results if not already ranked
        if not any('relevance_score' in r for r in results):
            results = self.rank_results_by_relevance(results, query)
        
        # Build formatted output
        output = "WEB SEARCH RESULTS:\n\n"
        
        for i, result in enumerate(results[:5]):  # Limit to top 5
            title = result.get('title', 'No title')
            url = result.get('url', '')
            domain = result.get('source', 'unknown')
            score = result.get('relevance_score', 0)
            content = result.get('content', result.get('snippet', ''))
            
            # Truncate content to a reasonable length
            if content:
                content = content.strip()
                if len(content) > 300:
                    content = content[:297] + "..."
            
            # Format result
            output += f"[{i+1}] {title} ({domain})\n"
            if content:
                output += f"{content}\n"
            output += f"Source: {url}\n\n"
        
        output += "Use the information above to help answer the query if relevant.\n"
        return output
    
    def should_search_web(self, query: str) -> bool:
        """
        Determine if a web search is needed for this query.
        
        Args:
            query: Query text
            
        Returns:
            True if web search is recommended
        """
        # Check for explicit search requests
        if re.search(r'\b(search|look up|google|find online|web search)\b', query.lower()):
            return True
        
        # Check for current information requests
        if re.search(r'\b(latest|current|recent|today|now|update|news)\b', query.lower()):
            return True
        
        # Check for specific facts that might need verification
        if re.search(r'\b(how many|population|statistics|percentage|price|rate|date)\b', query.lower()):
            return True
        
        # Check for domain-specific triggers using classifier
        if self.question_classifier:
            try:
                domain_settings = self.question_classifier.get_domain_settings(query)
                domain = domain_settings.get('domain', 'unknown')
                confidence = domain_settings.get('domain_confidence', 0)
                
                # Domains that might benefit from web search
                if domain in ['factual'] and confidence > 0.6:
                    return True
            except Exception:
                pass
        
        # Default - no need for search
        return False