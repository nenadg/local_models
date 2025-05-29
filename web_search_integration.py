#!/usr/bin/env python3
"""
Minimal web search integration for TinyLlama Chat using Google Custom Search.
Uses Google Custom Search API for search and python-readability for content extraction.
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
import os

# Import from existing system
from memory_utils import classify_content, save_to_memory
from question_classifier import QuestionClassifier


class WebSearchIntegration:
    """
    Minimal web search integration using Google Custom Search API.
    """

    def __init__(self, memory_manager=None, question_classifier=None,
                 api_key=None, search_engine_id=None):
        """
        Initialize web search integration.

        Args:
            memory_manager: MemoryManager instance for storing web results
            question_classifier: QuestionClassifier instance for categorizing content
            api_key: Google Custom Search API key
            search_engine_id: Google Custom Search Engine ID
        """
        self.memory_manager = memory_manager
        self.question_classifier = question_classifier or QuestionClassifier()

        # Google Custom Search configuration
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY')
        self.search_engine_id = search_engine_id or os.environ.get('GOOGLE_CX_ID')

        if not self.api_key or not self.search_engine_id:
            print(f"{self.get_time()} Warning: Google API key or Search Engine ID not set")
            print(f"{self.get_time()} Set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables")

        # Google Custom Search API endpoint
        self.search_url = "https://www.googleapis.com/customsearch/v1"

        # Headers for content fetching
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Cache for search results to avoid redundant API calls
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

    def search_google(self, query: str, num_results: int = 10) -> List[Dict[str, str]]:
        """
        Search Google using Custom Search API.

        Args:
            query: Search query
            num_results: Maximum number of results to return

        Returns:
            List of search results with title, url, and snippet
        """
        # Check if API is configured
        if not self.api_key or not self.search_engine_id:
            print(f"{self.get_time()} Google Custom Search API not configured")
            return []

        # Check cache first
        cache_key = hashlib.md5(f"{query}:{num_results}".encode()).hexdigest()
        if cache_key in self.cache:
            cached_time, cached_results = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                print(f"{self.get_time()} Using cached search results for: {query}")
                return cached_results

        print(f"{self.get_time()} Searching Google for: {query}")

        try:
            # Prepare API request
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(num_results, 10)  # Google API max is 10 per request
            }

            # Make search request
            response = requests.get(self.search_url, params=params, timeout=10)
            response.raise_for_status()

            # Parse JSON response
            data = response.json()

            results = []

            # Process search results
            if 'items' in data:
                for item in data['items']:
                    # Extract required fields
                    title = item.get('title', 'No title')
                    url = item.get('link', '')
                    snippet = item.get('snippet', '')

                    # Skip if no valid URL
                    if not url or not url.startswith(('http://', 'https://')):
                        continue

                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet,
                        'source': 'google'
                    })

            # Check if we got any results
            total_results = data.get('searchInformation', {}).get('totalResults', '0')
            print(f"{self.get_time()} Found {len(results)} results (total available: {total_results})")

            # Cache results
            self.cache[cache_key] = (time.time(), results)

            return results

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"{self.get_time()} API rate limit exceeded")
            else:
                print(f"{self.get_time()} HTTP error searching Google: {e}")
            return []
        except Exception as e:
            print(f"{self.get_time()} Error searching Google: {e}")
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

            # Debug: Check response
            print(f"{self.get_time()} Got response, status: {response.status_code}, length: {len(response.text)}")

            # Use readability to parse content
            from readability import parse
            article = parse(response.text)

            # Debug: Check what we got
            print(f"{self.get_time()} Article type: {type(article)}")
            print(f"{self.get_time()} Has title: {hasattr(article, 'title')}, Has content: {hasattr(article, 'content')}")

            # Extract title from Article object
            title = article.title if hasattr(article, 'title') else 'No title found'

            # Extract content from Article object
            content = article.content if hasattr(article, 'content') else ''

            if content is None:
                content = ''

            # Debug: Check extracted content
            print(f"{self.get_time()} Title: {title[:50]}...")
            print(f"{self.get_time()} Content length before processing: {len(content)}")

            # If content is HTML, extract text
            if content and '<' in content:
                soup = BeautifulSoup(content, 'html.parser')

                # Remove script and style elements
                for element in soup(['script', 'style']):
                    element.decompose()

                # Get text content
                content = soup.get_text()

                # Clean up whitespace
                lines = (line.strip() for line in content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                content = ' '.join(chunk for chunk in chunks if chunk)

            # Debug: Final content check
            print(f"{self.get_time()} Final content length: {len(content)}")

            # Limit content length to avoid token limits
            max_content_length = 2000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."

            # Check if we have valid content
            if not content or len(content.strip()) < 50:
                print(f"{self.get_time()} Content too short or empty, skipping")
                return None

            result = {
                'title': title,
                'content': content,
                'url': url,
                'domain': urlparse(url).netloc,
                'extracted_at': datetime.now().isoformat()
            }

            print(f"{self.get_time()} Successfully extracted content")
            return result

        except Exception as e:
            print(f"{self.get_time()} Error extracting content from {url}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def search_and_extract(self, query: str, max_pages: int = 3) -> List[Dict[str, str]]:
        """
        Search for query and extract content from top results.
        """
        # Search for results
        search_results = self.search_google(query)

        if not search_results:
            return []

        # Extract content from top results
        extracted_content = []
        processed_count = 0

        # Prioritize Wikipedia results
        wikipedia_results = [r for r in search_results if 'wikipedia.org' in r['url']]
        other_results = [r for r in search_results if 'wikipedia.org' not in r['url']]

        # Process Wikipedia first, then others
        ordered_results = wikipedia_results + other_results

        for i, result in enumerate(ordered_results):
            if processed_count >= max_pages:
                break

            url = result['url']

            # Skip certain domains that are hard to extract or low quality
            skip_domains = ['youtube.com', 'twitter.com', 'instagram.com', 'facebook.com', 'linkedin.com', 'reddit.com']
            if any(domain in url.lower() for domain in skip_domains):
                print(f"{self.get_time()} Skipping social media/forum domain: {url}")
                continue

            # Extract content
            content_data = self.extract_content(url)

            if content_data:
                # Check if we got meaningful content
                if len(content_data['content']) < 100 or content_data['title'] in ['The heart of the internet', 'Reddit']:
                    print(f"{self.get_time()} Skipping low-quality content from: {url}")
                    continue

                # Add search metadata
                content_data['search_rank'] = i + 1
                content_data['search_snippet'] = result['snippet']
                content_data['search_title'] = result['title']
                extracted_content.append(content_data)
                processed_count += 1
                
        return extracted_content

    def save_to_memory(self, content_data: Dict[str, str], query: str) -> bool:
        """
        Save extracted web content to memory system in a format similar to memory_importer.
        """
        if not self.memory_manager:
            print(f"{self.get_time()} No memory manager available")
            return False

        try:
            # Use the search title if the extracted title is generic
            title = content_data['title']
            if title in ['The heart of the internet', 'Reddit', 'No title found']:
                title = content_data.get('search_title', title)

            # Extract the main content
            content = content_data['content']
            if len(content) < 100:
                print(f"{self.get_time()} Content too short, not saving")
                return False

            # Parse content into meaningful chunks/facts
            # Split content into sentences or meaningful segments
            sentences = re.split(r'(?<=[.!?])\s+', content)

            saved_count = 0

            # Save the main answer as a fact
            if query.lower().startswith(('who is', 'what is', 'where is', 'when is')):
                # Create a clear answer format
                subject = query.replace('?', '').strip()

                # Extract the first meaningful sentence as the main answer
                main_answer = None
                for sentence in sentences[:5]:  # Check first 5 sentences
                    if len(sentence) > 30 and not sentence.startswith(('From Wikipedia', 'This article', 'For other')):
                        main_answer = sentence
                        break

                if main_answer:
                    # Format as a clear fact
                    fact_content = f"{subject}: {main_answer}"

                    # Classify and save
                    classification = classify_content(fact_content, self.question_classifier)
                    classification['source_type'] = 'web'
                    classification['source_url'] = content_data['url']
                    classification['source_domain'] = content_data['domain']

                    result = save_to_memory(
                        memory_manager=self.memory_manager,
                        content=fact_content,
                        classification=classification
                    )

                    if result.get('saved', False):
                        saved_count += 1
                        print(f"{self.get_time()} Saved main fact: {fact_content[:100]}...")

            # Save additional key facts from the content
            # Look for sentences that contain key information
            key_patterns = [
                r'(?:is|was|are|were)\s+(?:a|an|the)?\s*([^.]+)',  # Definitions
                r'(?:born|died|founded|created|developed)\s+(?:in|on)?\s*([^.]+)',  # Dates/events
                r'(?:known for|famous for|notable for)\s*([^.]+)',  # Achievements
            ]

            facts_saved = 0
            max_facts = 5  # Limit number of facts per source

            for sentence in sentences:
                if facts_saved >= max_facts:
                    break

                # Skip very short or metadata sentences
                if len(sentence) < 50 or sentence.startswith(('From Wikipedia', 'This article', 'For other', 'See also')):
                    continue

                # Check if sentence contains key information
                for pattern in key_patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        # Clean up the sentence
                        clean_sentence = sentence.strip()
                        if clean_sentence.endswith('.'):
                            clean_sentence = clean_sentence[:-1]

                        # Add source attribution
                        fact_with_source = f"{clean_sentence} (Source: {content_data['domain']})"

                        # Classify and save
                        classification = classify_content(fact_with_source, self.question_classifier)
                        classification['source_type'] = 'web'
                        classification['source_url'] = content_data['url']
                        classification['source_domain'] = content_data['domain']
                        classification['original_query'] = query

                        result = save_to_memory(
                            memory_manager=self.memory_manager,
                            content=fact_with_source,
                            classification=classification
                        )

                        if result.get('saved', False):
                            saved_count += 1
                            facts_saved += 1
                            print(f"{self.get_time()} Saved fact: {fact_with_source[:100]}...")

                        break  # Only save once per sentence

            # Also save a summary version with key information
            if saved_count == 0:  # Fallback if no facts were extracted
                # Create a summary
                summary = f"Information about {query}: {title}. {sentences[0] if sentences else content[:200]}..."

                classification = classify_content(summary, self.question_classifier)
                classification['source_type'] = 'web'
                classification['source_url'] = content_data['url']
                classification['source_domain'] = content_data['domain']

                result = save_to_memory(
                    memory_manager=self.memory_manager,
                    content=summary,
                    classification=classification
                )

                if result.get('saved', False):
                    saved_count += 1

            print(f"{self.get_time()} Total facts saved from this source: {saved_count}")
            return saved_count > 0

        except Exception as e:
            print(f"{self.get_time()} Error saving to memory: {e}")
            import traceback
            traceback.print_exc()
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
def integrate_web_search(chat_instance, api_key=None, search_engine_id=None):
    """
    Integrate web search into existing MemoryEnhancedChat instance.

    Args:
        chat_instance: MemoryEnhancedChat instance
        api_key: Google Custom Search API key (optional, will use env var)
        search_engine_id: Google Custom Search Engine ID (optional, will use env var)
    """
    # Create web search instance
    web_search = WebSearchIntegration(
        memory_manager=chat_instance.memory_manager,
        question_classifier=chat_instance.question_classifier,
        api_key=api_key,
        search_engine_id=search_engine_id
    )

    # Add to chat instance
    chat_instance.web_search = web_search

    print(f"{chat_instance.get_time()} Web search integration enabled with Google Custom Search")
    
    return web_search