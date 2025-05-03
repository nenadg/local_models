"""
Simplified web integration for local LLM chat system.
Focuses on Google Custom Search API integration with memory enhancement.
"""

import re
import time
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

class WebIntegration:
    """
    Integrates web search capabilities with local LLM chat using Google Custom Search API.
    Works with existing MemoryEnhancedChat class.
    """

    def __init__(
        self,
        memory_enhanced_chat,
        similarity_enhancement_factor: float = 0.3,
        max_web_results: int = 5,
        add_to_memory: bool = True,
        api_key: Optional[str] = None,
        cx_id: Optional[str] = None,
        cache_dir: str = "./cache"
    ):
        """
        Initialize web integration with Google Custom Search.

        Args:
            memory_enhanced_chat: Existing MemoryEnhancedChat instance
            similarity_enhancement_factor: Factor for similarity enhancement
            max_web_results: Maximum web results to include
            add_to_memory: Whether to add results to memory
            api_key: Google API key (will use environment variable if None)
            cx_id: Google Custom Search Engine ID (will use environment variable if None)
            cache_dir: Directory for caching search results
        """
        self.chat = memory_enhanced_chat

        # Initialize Google Search client
        from google_search_client import GoogleSearchClient
        self.search_client = GoogleSearchClient(
            api_key=api_key,
            cx_id=cx_id,
            ocr_extractor=None,  # No OCR extractor needed
            memory_manager=memory_enhanced_chat.memory_manager if add_to_memory else None,
            cache_dir=cache_dir,
            similarity_enhancement_factor=similarity_enhancement_factor
        )

        # Initialize the WebSearchManager
        from web_search import WebSearchManager
        self.web_search = WebSearchManager(
            memory_manager=memory_enhanced_chat.memory_manager if add_to_memory else None,
            question_classifier=memory_enhanced_chat.question_classifier,
            similarity_enhancement_factor=similarity_enhancement_factor
        )

        # Connect the WebSearchManager to the GoogleSearchClient
        self.web_search.set_search_client(self.search_client)

        # Settings
        self.max_web_results = max_web_results
        self.add_to_memory = add_to_memory
        self.web_enabled = True

        # Stats
        self.web_stats = {
            "searches": 0,
            "items_added": 0,
            "direct_url_accesses": 0
        }

        print(f"{self.get_time()} Web integration initialized with Google Custom Search API")

    def get_time(self) -> str:
        """Get formatted timestamp for logging."""
        return datetime.now().strftime("[%d/%m/%y %H:%M:%S]") + ' [WebIntegration]'

    def toggle_web_search(self) -> bool:
        """Toggle web search on/off."""
        self.web_enabled = not self.web_enabled
        return self.web_enabled

    def enhance_conversation_with_web(
        self,
        messages: List[Dict[str, str]],
        query: str,
        force_search: bool = False
    ) -> List[Dict[str, str]]:
        """
        Enhance conversation with web search results.

        Args:
            messages: Current conversation messages
            query: Current user query
            force_search: Force web search even if not needed

        Returns:
            Enhanced messages with web context
        """
        # Skip if web search is disabled
        if not self.web_enabled:
            return messages

        # Check if web search is needed
        should_search = force_search or self.web_search.should_search_web(query)

        if not should_search:
            return messages

        # Track statistics
        self.web_stats["searches"] += 1

        # IMPORTANT FIX: Don't use web_search.search() as it calls search_client.search again
        # Instead, call search_client.search directly with the processed keywords

        # Generate optimized search keywords
        optimized_query = self.web_search._generate_search_keywords(query)
        print(f"{self.get_time()} Using optimized query: {optimized_query}")

        # Get search results directly from search client
        results = self.search_client.search(
            query=optimized_query,
            num_results=self.max_web_results,
            include_content=True
        )

        # Skip if no results
        if not results:
            return messages

        # Add to memory if enabled
        if self.add_to_memory and self.chat.memory_manager:
            added_items = self.search_client.search_and_add_to_memory(
                query=query,
                max_results=self.max_web_results
            )

            if added_items:
                self.web_stats["items_added"] += len(added_items)
                print(f"{self.get_time()} Added {len(added_items)} web search results to memory")

        # Create enhanced messages
        enhanced_messages = messages.copy()

        # Format results for context
        web_context = self.search_client.format_for_context(results, query)
        
        # Extract system message
        system_message = enhanced_messages[0]["content"]
        
        # Check if already contains web context
        if "WEB SEARCH RESULTS:" in system_message:
            # Replace existing web context
            parts = system_message.split("WEB SEARCH RESULTS:")
            system_prefix = parts[0]
            
            # Check if there's a memory context we need to preserve
            if "MEMORY CONTEXT:" in system_prefix:
                memory_parts = system_prefix.split("MEMORY CONTEXT:")
                system_content = memory_parts[0].strip()
                memory_context = "MEMORY CONTEXT:" + memory_parts[1]
                
                # Create enhanced system message
                enhanced_system = f"{system_content}\n\n{memory_context}\n\n{web_context}"
            else:
                # No memory context
                enhanced_system = f"{system_prefix.strip()}\n\n{web_context}"
        
        elif "MEMORY CONTEXT:" in system_message:
            # Contains memory context but no web context
            parts = system_message.split("MEMORY CONTEXT:")
            system_content = parts[0].strip()
            memory_context = "MEMORY CONTEXT:" + parts[1]
            
            # Create enhanced system message
            enhanced_system = f"{system_content}\n\n{memory_context}\n\n{web_context}"
            
        else:
            # No memory or web context yet
            enhanced_system = f"{system_message}\n\n{web_context}"
        
        # Update system message
        enhanced_messages[0]["content"] = enhanced_system
        
        return enhanced_messages

    def process_web_command(self, query: str) -> Optional[str]:
        """
        Process web search commands from user.

        Args:
            query: User query

        Returns:
            Response message or None if not a web command
        """
        # Check for web search command
        if query.startswith("!web"):
            # Extract search query
            search_query = query[4:].strip()

            if not search_query:
                return "Please provide a search query. Example: !web latest climate reports"

            # IMPORTANT FIX: Use a single search operation
            # Generate optimized keywords
            optimized_query = self.web_search._generate_search_keywords(search_query)
            print(f"{self.get_time()} Using optimized query: {optimized_query}")

            # Perform search using the optimized query directly from search client
            results = self.search_client.search(
                query=optimized_query,
                num_results=7,  # More results for direct search
                include_content=True
            )

            if not results:
                return f"No results found for: {search_query}"

            # Format response
            response = f"Search results for: {search_query}\n\n"

            for i, result in enumerate(results):
                title = result.get('title', 'No title')
                url = result.get('url', '')
                domain = result.get('source', '')
                snippet = result.get('snippet', '')

                response += f"{i+1}. {title} ({domain})\n"
                if snippet:
                    response += f"   {snippet}\n"
                response += f"   {url}\n\n"

            # Add to memory if enabled - reuse existing results instead of searching again
            if self.add_to_memory and self.chat.memory_manager:
                added = 0
                for result in results:
                    content = result.get('content', '')
                    if not content:
                        continue

                    # Create metadata
                    metadata = {
                        'source': 'web_search',
                        'url': result.get('url', ''),
                        'title': result.get('title', ''),
                        'domain': result.get('source', ''),
                        'query': search_query,
                        'timestamp': time.time()
                    }

                    # Add to memory
                    try:
                        item_id = self.chat.memory_manager.add(
                            content=content,
                            metadata=metadata
                        )

                        if item_id:
                            added += 1
                    except Exception:
                        pass

                if added > 0:
                    self.web_stats["items_added"] += added
                    response += f"Added {added} search results to memory.\n"

            return response

        # URL command handling remains the same
        elif query.startswith("!url:"):
            # This part is unchanged since it doesn't involve duplicate searches
            url = query[5:].strip()

            if not url:
                return "Please provide a URL. Example: !url: https://example.com"

            # Normalize URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            # Track statistics
            self.web_stats["direct_url_accesses"] += 1

            # Get content
            result = self.search_client.get_url_content(url)

            if not result or not result.get('content'):
                return f"Could not extract content from: {url}"

            # Format response
            response = f"Content extracted from: {url}\n\n"
            response += f"Title: {result.get('title', 'Unknown title')}\n\n"

            # Show content preview
            content = result.get('content', '')
            if len(content) > 500:
                content_preview = content[:500] + "...\n(content truncated, added to memory for retrieval)"
            else:
                content_preview = content

            response += f"Content preview:\n{content_preview}\n\n"

            # Add to memory if enabled
            if self.add_to_memory and self.chat.memory_manager:
                try:
                    # Create metadata
                    metadata = {
                        'source': 'direct_url',
                        'url': url,
                        'title': result.get('title', 'Unknown title'),
                        'domain': self._extract_domain(url),
                        'timestamp': time.time()
                    }

                    # Add to memory
                    item_id = self.chat.memory_manager.add(
                        content=content,
                        metadata=metadata
                    )

                    if item_id:
                        self.web_stats["items_added"] += 1
                        response += f"Content added to memory (ID: {item_id}).\n"
                except Exception as e:
                    response += f"Error adding to memory: {str(e)}\n"
            
            return response
            
        # Not a web command
        return None

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

    def get_stats(self) -> Dict[str, Any]:
        """Get web search statistics."""
        # Get stats from search client
        client_stats = self.search_client.get_stats() if hasattr(self.search_client, 'get_stats') else {}

        return {
            "searches_performed": self.web_stats["searches"],
            "items_added_to_memory": self.web_stats["items_added"],
            "direct_url_accesses": self.web_stats["direct_url_accesses"],
            "api_calls": client_stats.get("api_calls", 0),
            "cache_hits": client_stats.get("cache_hits", 0),
            "web_enabled": self.web_enabled
        }