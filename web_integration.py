"""
Streamlined web integration for local LLM chat system.
Eliminates duplicate API calls and focuses on Google Custom Search API.
"""

import re
import time
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

class WebIntegration:
    """
    Integrates web search capabilities with local LLM chat using Google Custom Search API.
    Optimized to prevent duplicate API calls.
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

        # Import keyword generation, but don't create full WebSearchManager
        # This avoids redundancy and multiple search paths
        from web_search import WebSearchManager
        self.keyword_generator = WebSearchManager(
            memory_manager=None,  # Don't need memory access here
            question_classifier=memory_enhanced_chat.question_classifier,
            similarity_enhancement_factor=similarity_enhancement_factor
        )

        # Initialize Google Search client
        from google_search_client import GoogleSearchClient
        self.search_client = GoogleSearchClient(
            api_key=api_key,
            cx_id=cx_id,
            memory_manager=memory_enhanced_chat.memory_manager if add_to_memory else None,
            cache_dir=cache_dir,
            similarity_enhancement_factor=similarity_enhancement_factor
        )

        # Settings
        self.max_web_results = max_web_results
        self.add_to_memory = add_to_memory
        self.web_enabled = True

        # Initialize OCR capabilities
        self.ocr_enabled = True  # Enable by default
        try:
            # Import OCR extractor
            from web_ocr import WebOCRExtractor

            self.ocr_extractor = WebOCRExtractor(
                memory_manager=memory_enhanced_chat.memory_manager,
                similarity_enhancement_factor=similarity_enhancement_factor,
                temp_dir="./temp"
            )
            print(f"{self.get_time()} OCR extraction capabilities initialized")
        except Exception as e:
            print(f"{self.get_time()} Could not initialize OCR: {e}")
            self.ocr_enabled = False
            self.ocr_extractor = None

        # Update web_stats to include ocr_extractions
        self.web_stats = {
            "searches": 0,
            "items_added": 0,
            "direct_url_accesses": 0,
            "ocr_extractions": 0  # Add this to track OCR extractions
        }

        print(f"{self.get_time()} Web integration initialized with Google Custom Search API")

    def get_time(self) -> str:
        """Get formatted timestamp for logging."""
        return datetime.now().strftime("[%d/%m/%y %H:%M:%S]") + ' [WebIntegration]'

    def toggle_web_search(self) -> bool:
        """Toggle web search on/off."""
        self.web_enabled = not self.web_enabled
        return self.web_enabled

    def toggle_ocr(self) -> bool:
        """Toggle OCR extraction on/off."""
        self.ocr_enabled = not self.ocr_enabled
        print(f"{self.get_time()} OCR extraction {'enabled' if self.ocr_enabled else 'disabled'}")
        return self.ocr_enabled

    def should_search_web(self, query: str) -> bool:
        """
        Determine if a web search is needed for this query.

        Args:
            query: Query text

        Returns:
            True if web search is recommended
        """
        # Reuse the logic from WebSearchManager without creating the whole class
        return self.keyword_generator.should_search_web(query)

    def enhance_conversation_with_web(
        self,
        messages: List[Dict[str, str]],
        query: str,
        force_search: bool = False
    ) -> List[Dict[str, str]]:
        """
        Enhance conversation with web search results.
        Optimized to prevent duplicate API calls.

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

        # Check for OCR flags in query
        use_ocr = self.ocr_enabled  # Default to current setting
        if '--ocr' in query:
            use_ocr = True
            # Clean query for better search
            query = query.replace('--ocr', '').strip()
        elif '--no-ocr' in query:
            use_ocr = False
            # Clean query for better search
            query = query.replace('--no-ocr', '').strip()

        # Check if web search is needed
        should_search = force_search or self.should_search_web(query)

        if not should_search:
            return messages

        # Generate optimized search keywords
        optimized_query = self.keyword_generator._generate_search_keywords(query)
        print(f"{self.get_time()} Using optimized query: {optimized_query}")

        # Perform search
        self.web_stats["searches"] += 1
        results = self.search_client.search(
            query=optimized_query,
            num_results=self.max_web_results,
            include_content=True
        )

        # Skip if no results
        if not results:
            return messages

        print("--------USE OCR", use_ocr and hasattr(self, 'ocr_extractor') and self.ocr_extractor)
        # Use OCR to enhance content if enabled
        if use_ocr and hasattr(self, 'ocr_extractor') and self.ocr_extractor:
            try:
                # Import OCR helper
                from web_ocr import run_async

                # Enhance top results with OCR
                for i, result in enumerate(results[:2]):  # Limit to top 2 to save time
                    url = result.get('url')
                    if not url:
                        continue

                    print(f"{self.get_time()} Enhancing result with OCR: {url}")
                    ocr_result = run_async(self.ocr_extractor.extract_page_content(url))

                    if ocr_result and ocr_result.get('full_content'):
                        # Replace snippet with full content
                        result['content'] = ocr_result.get('full_content')
                        result['ocr_enhanced'] = True
                        self.web_stats["ocr_extractions"] += 1
            except Exception as e:
                print(f"{self.get_time()} Error enhancing with OCR: {e}")

        # Add to memory if enabled - REUSE results to avoid duplicate API calls
        if self.add_to_memory and self.chat.memory_manager:
            added_items = self.search_client.search_and_add_to_memory(
                query=query,
                max_results=self.max_web_results,
                results=results  # Pass existing results to avoid duplicate API calls
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
        Optimized to prevent duplicate API calls.

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

            # Generate optimized search keywords
            optimized_query = self.keyword_generator._generate_search_keywords(search_query)
            print(f"{self.get_time()} Using optimized query: {optimized_query}")

            # Perform search
            self.web_stats["searches"] += 1
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

            # Add to memory if enabled - REUSE results to avoid duplicate API calls
            if self.add_to_memory and self.chat.memory_manager:
                added_items = self.search_client.search_and_add_to_memory(
                    query=search_query,
                    max_results=7,
                    results=results  # Pass existing results to avoid duplicate API calls
                )

                if added_items:
                    self.web_stats["items_added"] += len(added_items)
                    response += f"Added {len(added_items)} search results to memory.\n"

            return response

        # Check for direct URL command
        elif query.startswith("!url:"):
            # Extract URL
            url = query[5:].strip()

            if not url:
                return "Please provide a URL. Example: !url: https://example.com"

            # Normalize URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            # Track statistics
            self.web_stats["direct_url_accesses"] += 1

            # Use OCR extractor if available and enabled
            if self.ocr_enabled and hasattr(self, 'ocr_extractor') and self.ocr_extractor:
                try:
                    # Import helper from web_ocr
                    from web_ocr import run_async

                    print(f"{self.get_time()} Extracting content with OCR: {url}")
                    ocr_result = run_async(self.ocr_extractor.extract_page_content(url))

                    if ocr_result and ocr_result.get('full_content'):
                        # Track OCR extractions
                        self.web_stats["ocr_extractions"] += 1

                        # Format response
                        response = f"Content extracted from: {url}\n\n"
                        response += f"Title: {ocr_result.get('title', 'Unknown title')}\n\n"

                        # Show content preview
                        content = ocr_result.get('full_content', '')
                        if len(content) > 500:
                            content_preview = content[:500] + "...\n(content truncated, added to memory for retrieval)"
                        else:
                            content_preview = content

                        response += f"Content preview:\n{content_preview}\n\n"

                        # Add to memory if enabled
                        if self.add_to_memory and self.chat.memory_manager:
                            added_items = run_async(self.ocr_extractor.add_to_memory(url, query))

                            if added_items:
                                self.web_stats["items_added"] += len(added_items)
                                response += f"Added {len(added_items)} content sections to memory.\n"

                        return response
                except Exception as e:
                    print(f"{self.get_time()} Error using OCR extraction: {e}")

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
            "ocr_extractions": self.web_stats.get("ocr_extractions", 0),  # Get with default in case it doesn't exist
            "api_calls": client_stats.get("api_calls", 0),
            "cache_hits": client_stats.get("cache_hits", 0),
            "session_cache_size": client_stats.get("session_cache_size", 0),
            "web_enabled": self.web_enabled,
            "ocr_enabled": getattr(self, 'ocr_enabled', False)
        }