"""
Google Custom Search API client for integration with TinyLlama Chat.
Simplified version without OCR dependency.
"""

import os
import json
import requests
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

class GoogleSearchClient:
    """
    Client for Google Custom Search API.
    """

    def __init__(
        self,
        api_key: str = None,
        cx_id: str = None,
        ocr_extractor = None,  # Kept for compatibility but not used
        memory_manager = None,
        cache_dir: str = "./cache",
        cache_duration: int = 3600,  # 1 hour cache by default
        similarity_enhancement_factor: float = 0.3
    ):
        """
        Initialize the Google Custom Search client.

        Args:
            api_key: Google API key
            cx_id: Google Custom Search Engine ID
            ocr_extractor: OCR extraction module (not used)
            memory_manager: Memory manager for storing results
            cache_dir: Directory for caching search results
            cache_duration: Duration in seconds to cache results
            similarity_enhancement_factor: Factor for similarity enhancement
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.cx_id = cx_id or os.environ.get("GOOGLE_CX_ID")
        self.memory_manager = memory_manager
        self.cache_dir = cache_dir
        self.cache_duration = cache_duration
        self.similarity_enhancement_factor = similarity_enhancement_factor

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Track statistics
        self.search_stats = {
            "api_calls": 0,
            "cache_hits": 0,
            "memory_additions": 0
        }

    def get_time(self) -> str:
        """Get formatted timestamp for logging."""
        return datetime.now().strftime("[%d/%m/%y %H:%M:%S]") + ' [GoogleSearch]'

    def search(
        self,
        query: str,
        num_results: int = 10,
        include_content: bool = True,
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Perform a Google search using the Custom Search API.

        Args:
            query: Search query
            num_results: Number of results to return (max 10 per API call)
            include_content: Whether to fetch content snippets (no actual page content)
            force_refresh: Force refresh cached results

        Returns:
            List of search result dictionaries
        """
        if not self.api_key or not self.cx_id:
            print(f"{self.get_time()} API key or CX ID not set")
            return []

        # Check cache first
        cache_key = f"gsearch_{hashlib.md5(query.encode()).hexdigest()}"
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        if not force_refresh and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)

                # Check cache validity
                if time.time() - cached_data.get('timestamp', 0) < self.cache_duration:
                    print(f"{self.get_time()} Using cached search results for: {query}")
                    self.search_stats["cache_hits"] += 1
                    return cached_data.get('results', [])
            except Exception as e:
                print(f"{self.get_time()} Error reading cache: {e}")

        # Prepare API request
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.cx_id,
            "q": query,
            "num": min(10, num_results)  # API limit is 10 per request
        }

        try:
            print(f"{self.get_time()} Performing Google search for: {query}")
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()

            # Track API usage
            self.search_stats["api_calls"] += 1

            # Parse response
            data = response.json()

            # Extract search results
            results = []
            if 'items' in data:
                for item in data['items']:
                    # Extract snippet as content (simpler approach without OCR)
                    snippet = item.get('snippet', '')

                    # Add any additional content from Google's API
                    if 'pagemap' in item and 'metatags' in item['pagemap']:
                        meta_description = item['pagemap']['metatags'][0].get('og:description', '')
                        if meta_description and len(meta_description) > len(snippet):
                            snippet = meta_description

                    result = {
                        'title': item.get('title', 'No title'),
                        'url': item.get('link', ''),
                        'snippet': snippet,
                        'source': self._extract_domain(item.get('link', '')),
                        'timestamp': time.time(),
                        'content': snippet,  # Use snippet as content
                        'fetched_content': True  # Mark as fetched since we're using the snippet
                    }
                    results.append(result)

            # Cache results
            try:
                cache_data = {
                    'timestamp': time.time(),
                    'query': query,
                    'results': results
                }
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2)
            except Exception as e:
                print(f"{self.get_time()} Error caching results: {e}")

            return results

        except Exception as e:
            print(f"{self.get_time()} Error performing search: {e}")
            return []

    def get_url_content(self, url: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get content from a specific URL (simplified to just return metadata and snippet).

        Args:
            url: URL to get information about
            force_refresh: Force refresh cached content

        Returns:
            Dictionary with URL metadata
        """
        # Check cache first for this URL
        cache_key = f"url_{hashlib.md5(url.encode()).hexdigest()}"
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        if not force_refresh and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)

                # Check cache validity
                if time.time() - cached_data.get('timestamp', 0) < self.cache_duration:
                    print(f"{self.get_time()} Using cached content for: {url}")
                    return cached_data
            except Exception as e:
                print(f"{self.get_time()} Error reading cache: {e}")

        try:
            print(f"{self.get_time()} Getting information about: {url}")

            # Perform a search for the exact URL to get metadata
            search_url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.api_key,
                "cx": self.cx_id,
                "q": f"site:{self._extract_domain(url)} {url}",
                "num": 1
            }

            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()

            # Track API usage
            self.search_stats["api_calls"] += 1

            # Parse response
            data = response.json()

            # Try to find the URL in results
            if 'items' in data and len(data['items']) > 0:
                item = data['items'][0]
                title = item.get('title', 'Unknown title')
                snippet = item.get('snippet', 'No content available')

                # Add any additional content from Google's API
                if 'pagemap' in item and 'metatags' in item['pagemap']:
                    meta_description = item['pagemap']['metatags'][0].get('og:description', '')
                    if meta_description and len(meta_description) > len(snippet):
                        snippet = meta_description
            else:
                # Use basic information if not found
                title = f"Page at {url}"
                snippet = "No content available through search API."

            # Create result
            result = {
                'url': url,
                'title': title,
                'content': snippet,
                'timestamp': time.time()
            }

            # Cache result
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
            except Exception as e:
                print(f"{self.get_time()} Error caching content: {e}")

            return result

        except Exception as e:
            print(f"{self.get_time()} Error getting URL info: {e}")
            return {
                'url': url,
                'title': f"Page at {url}",
                'content': f"Error retrieving content: {e}",
                'timestamp': time.time()
            }

    def _extract_domain(self, url: str) -> str:
        """
        Extract domain name from URL.

        Args:
            url: URL to extract domain from

        Returns:
            Domain name
        """
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc

            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]

            return domain

        except Exception:
            return "unknown"

    def search_and_add_to_memory(
        self,
        query: str,
        max_results: int = 5,
        content_threshold: int = 20  # Minimum content length to add to memory
    ) -> List[Dict[str, Any]]:
        """
        Search and add relevant results to memory.

        Args:
            query: Search query
            max_results: Maximum number of results to add to memory
            content_threshold: Minimum content length to consider adding to memory

        Returns:
            List of memory items added
        """
        if not self.memory_manager:
            print(f"{self.get_time()} No memory manager available")
            return []

        # Perform search
        results = self.search(query, num_results=max_results)

        # Add relevant results to memory
        added_items = []

        for result in results:
            content = result.get('content', '') or result.get('snippet', '')

            # Skip results with insufficient content
            if len(content) < content_threshold:
                continue

            # Create metadata
            metadata = {
                'source': 'google_search',
                'url': result.get('url'),
                'title': result.get('title'),
                'domain': result.get('source'),
                'query': query,
                'timestamp': time.time()
            }

            # Add to memory
            try:
                item_id = self.memory_manager.add(
                    content=content,
                    metadata=metadata
                )

                if item_id:
                    result['memory_id'] = item_id
                    added_items.append(result)
                    print(f"{self.get_time()} Added to memory: {result.get('title')}")
                    self.search_stats["memory_additions"] += 1
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

        # Build formatted output
        output = "WEB SEARCH RESULTS:\n\n"

        for i, result in enumerate(results[:5]):  # Limit to top 5
            title = result.get('title', 'No title')
            url = result.get('url', '')
            domain = result.get('source', 'unknown')
            snippet = result.get('snippet', '')

            # Format result
            output += f"[{i+1}] {title} ({domain})\n"
            if snippet:
                output += f"{snippet}\n"
            output += f"Source: {url}\n\n"

        output += "Use the information above to help answer the query if relevant.\n"
        return output

    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        return {
            "api_calls": self.search_stats["api_calls"],
            "cache_hits": self.search_stats["cache_hits"],
            "memory_additions": self.search_stats["memory_additions"]
        }