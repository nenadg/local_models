#!/usr/bin/env python3
"""
Minimal web search integration for TinyLlama Chat.
Uses DuckDuckGo HTML version for search and python-readability for content extraction.
"""

import re
import requests
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from urllib.parse import quote_plus, urlparse
from readability import parse

from bs4 import BeautifulSoup
import hashlib

# Import from existing system
from memory_utils import classify_content, save_to_memory
from question_classifier import QuestionClassifier


class WebSearchIntegration:
    """
    Minimal web search integration that fetches and processes web content.
    """
    
    def __init__(self, memory_manager=None, question_classifier=None):
        """
        Initialize web search integration.
        
        Args:
            memory_manager: MemoryManager instance for storing web results
            question_classifier: QuestionClassifier instance for categorizing content
        """
        self.memory_manager = memory_manager
        self.question_classifier = question_classifier or QuestionClassifier()
        
        # DuckDuckGo HTML search endpoint
        self.search_url = "https://html.duckduckgo.com/html?t=h_&q="  # "https://html.duckduckgo.com/html/"
        
        # Headers to appear as a regular browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Cache for search results to avoid redundant fetches
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        
    def get_time(self) -> str:
        """Get formatted timestamp for logging."""
        return datetime.now().strftime("[%d/%m/%y %H:%M:%S] [WebSearch]")
        
    def should_search(self, query: str) -> Tuple[bool, str]:
        """
        Determine if a query needs web search.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (should_search, reason)
        """
        query_lower = query.lower()
        
        # Keywords that strongly indicate need for current info
        current_indicators = [
            'latest', 'current', 'today', 'now', 'recent', 'update',
            'news', '2024', '2025', 'price', 'weather', 'score',
            'happening', 'status', 'real-time', 'live'
        ]
        
        # Check for current info indicators
        for indicator in current_indicators:
            if indicator in query_lower:
                return True, f"current_info_indicator:{indicator}"
                
        # Check for questions about unknown entities
        if re.search(r'\b(who|what|where|when|how)\s+(is|are|was|were)\s+\w+', query_lower):
            # This might be asking about something specific
            return True, "specific_entity_query"
            
        # Check for URLs in query
        if re.search(r'https?://\S+', query):
            return True, "url_in_query"
            
        # Questions asking to search or look up
        if re.search(r'\b(search|look up|find|google|check)\b', query_lower):
            return True, "explicit_search_request"
            
        return False, "no_search_needed"
        
    def search_duckduckgo(self, query: str, num_results: int = 10) -> List[Dict[str, str]]:
        """
        Search DuckDuckGo HTML version for results.
        
        Args:
            query: Search query
            num_results: Maximum number of results to return
            
        Returns:
            List of search results with title, url, and snippet
        """
        # Check cache first
        cache_key = hashlib.md5(f"{query}:{num_results}".encode()).hexdigest()
        if cache_key in self.cache:
            cached_time, cached_results = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                print(f"{self.get_time()} Using cached search results for: {query}")
                return cached_results
                
        print(f"{self.get_time()} Searching DuckDuckGo for: {query}")
        
        try:
            # Make search request
            params = {'q': query}
            response = requests.post(self.search_url + "+".join(params["q"].split(" ")), data=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            with open('ddg_response.html', 'w') as f:
                f.write(response.text)


            # Parse HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            # Find result divs
            for result_div in soup.find_all('div', class_='result'):
                if len(results) >= num_results:
                    break
                    
                # Extract title and URL
                title_elem = result_div.find('h2', class_='result__title')
                if not title_elem:
                    continue
                    
                link_elem = title_elem.find('a', class_='result__a')
                if not link_elem or not link_elem.get('href'):
                    continue
                    
                # Extract snippet
                snippet_elem = result_div.find('div', class_='result__snippet')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                
                # Clean up the data
                title = link_elem.get_text(strip=True)
                url = link_elem['href']
                
                # Skip if no valid URL
                if not url.startswith(('http://', 'https://')):
                    continue
                    
                results.append({
                    'title': title,
                    'url': url,
                    'snippet': snippet,
                    'source': 'duckduckgo'
                })
                
            # Cache results
            self.cache[cache_key] = (time.time(), results)
            
            print(f"{self.get_time()} Found {len(results)} search results")
            return results
            
        except Exception as e:
            print(f"{self.get_time()} Error searching DuckDuckGo: {e}")
            return []
            
    def extract_content(self, url: str, timeout: int = 10) -> Optional[Dict[str, str]]:
        """
        Extract readable content from a URL using python-readability.
        
        Args:
            url: URL to extract content from
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with title, content, and metadata or None if failed
        """
        print(f"{self.get_time()} Extracting content from: {url}")
        
        try:
            # Fetch the page
            response = requests.get(url, headers=self.headers, timeout=timeout)
            response.raise_for_status()
            
            # Use readability to extract content
            doc = parse(response.text)
            
            # Get article content
            article_html = doc.summary()
            article_title = doc.title()
            
            # Convert HTML to plain text
            soup = BeautifulSoup(article_html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get text content
            content = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit content length to avoid token limits
            max_content_length = 2000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
                
            return {
                'title': article_title,
                'content': content,
                'url': url,
                'domain': urlparse(url).netloc,
                'extracted_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"{self.get_time()} Error extracting content from {url}: {e}")
            return None
            
    def search_and_extract(self, query: str, max_pages: int = 3) -> List[Dict[str, str]]:
        """
        Search for query and extract content from top results.
        
        Args:
            query: Search query
            max_pages: Maximum number of pages to extract content from
            
        Returns:
            List of extracted content dictionaries
        """
        # Search for results
        search_results = self.search_duckduckgo(query)
        
        if not search_results:
            return []
            
        # Extract content from top results
        extracted_content = []
        
        for i, result in enumerate(search_results[:max_pages]):
            url = result['url']
            
            # Skip certain domains that are hard to extract
            skip_domains = ['youtube.com', 'twitter.com', 'instagram.com', 'facebook.com']
            if any(domain in url for domain in skip_domains):
                continue
                
            # Extract content
            content_data = self.extract_content(url)
            
            if content_data:
                # Add search metadata
                content_data['search_rank'] = i + 1
                content_data['search_snippet'] = result['snippet']
                extracted_content.append(content_data)
                
        return extracted_content
        
    def save_to_memory(self, content_data: Dict[str, str], query: str) -> bool:
        """
        Save extracted web content to memory system.
        
        Args:
            content_data: Extracted content dictionary
            query: Original search query
            
        Returns:
            Success status
        """
        if not self.memory_manager:
            return False
            
        try:
            # Format content for memory
            formatted_content = (
                f"Web search result for '{query}': "
                f"{content_data['title']} - {content_data['content']}"
            )
            
            # Classify content
            classification = classify_content(formatted_content, self.question_classifier)
            
            # Add web-specific metadata
            classification['source_type'] = 'web'
            classification['source_url'] = content_data['url']
            classification['source_domain'] = content_data['domain']
            classification['search_rank'] = content_data.get('search_rank', 0)
            classification['extracted_at'] = content_data['extracted_at']
            
            # Save to memory
            result = save_to_memory(
                memory_manager=self.memory_manager,
                content=formatted_content,
                classification=classification
            )
            
            return result.get('saved', False)
            
        except Exception as e:
            print(f"{self.get_time()} Error saving to memory: {e}")
            return False
            
    def process_query(self, query: str) -> Dict[str, any]:
        """
        Main method to process a query with web search if needed.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with search results and status
        """
        result = {
            'searched': False,
            'results_found': 0,
            'saved_to_memory': 0,
            'content': []
        }
        
        # Check if search is needed
        should_search, reason = self.should_search(query)
        
        if not should_search:
            return result
            
        print(f"{self.get_time()} Web search triggered: {reason}")
        result['searched'] = True
        
        # Perform search and extraction
        extracted_content = self.search_and_extract(query)
        result['results_found'] = len(extracted_content)
        result['content'] = extracted_content
        
        # Save to memory if available
        if self.memory_manager:
            for content_data in extracted_content:
                if self.save_to_memory(content_data, query):
                    result['saved_to_memory'] += 1
                    
            print(f"{self.get_time()} Saved {result['saved_to_memory']} results to memory")
            
        return result


# Integration hook for local_ai.py
def integrate_web_search(chat_instance):
    """
    Integrate web search into existing MemoryEnhancedChat instance.
    
    Args:
        chat_instance: MemoryEnhancedChat instance
    """
    # Create web search instance
    web_search = WebSearchIntegration(
        memory_manager=chat_instance.memory_manager,
        question_classifier=chat_instance.question_classifier
    )
    
    # Add to chat instance
    chat_instance.web_search = web_search
    
    # Monkey-patch the chat method to include web search
    original_chat = chat_instance.chat
    
    def enhanced_chat(messages, **kwargs):
        # Get user query
        user_query = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
        
        if user_query and chat_instance.enable_memory:
            # Process web search if needed
            web_results = web_search.process_query(user_query)
            
            if web_results['searched'] and web_results['results_found'] > 0:
                print(f"{chat_instance.get_time()} Web search found {web_results['results_found']} results")
                
                # The results are already saved to memory by process_query
                # Just proceed with normal chat which will retrieve them
                
        # Call original chat method
        return original_chat(messages, **kwargs)
        
    # Replace chat method
    chat_instance.chat = enhanced_chat
    
    print(f"{chat_instance.get_time()} Web search integration enabled")
    
    return web_search