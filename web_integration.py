"""
Web integration for local LLM chat system.
Provides seamless integration of web search with existing memory system.
"""

import re
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from web_search import WebSearchManager

class WebIntegration:
    """
    Integrates web search capabilities with local LLM chat.
    Works with existing MemoryEnhancedChat class.
    """
    
    def __init__(
        self,
        memory_enhanced_chat,
        similarity_enhancement_factor: float = 0.3,
        max_web_results: int = 5,
        add_to_memory: bool = True,
        ocr_enabled: bool = True
    ):
        """
        Initialize web integration.
        
        Args:
            memory_enhanced_chat: Existing MemoryEnhancedChat instance
            similarity_enhancement_factor: Factor for similarity enhancement
            max_web_results: Maximum web results to include
            add_to_memory: Whether to add results to memory
            ocr_enabled: Whether to enable OCR fallback extraction
        """
        self.chat = memory_enhanced_chat
        
        # Initialize web search manager
        self.web_search = WebSearchManager(
            memory_manager=memory_enhanced_chat.memory_manager if add_to_memory else None,
            question_classifier=memory_enhanced_chat.question_classifier,
            similarity_enhancement_factor=similarity_enhancement_factor
        )

        # Settings
        self.max_web_results = max_web_results
        self.add_to_memory = add_to_memory
        self.web_enabled = True
        self.ocr_enabled = ocr_enabled  # Add this line
        self.web_stats = {
            "searches": 0,
            "items_added": 0,
            "ocr_extractions": 0  # Add this line
        }

    def toggle_ocr(self) -> bool:
        """Toggle OCR extraction on/off."""
        self.ocr_enabled = not self.ocr_enabled
        return self.ocr_enabled

    def toggle_web_search(self) -> bool:
        """Toggle web search on/off."""
        self.web_enabled = not self.web_enabled
        return self.web_enabled
    
    def enhance_conversation_with_web(
        self,
        messages: List[Dict[str, str]],
        query: str,
        force_search: bool = False,
        use_ocr: Optional[bool] = None
    ) -> List[Dict[str, str]]:
        """
        Enhance conversation with web search results.
        
        Args:
            messages: Current conversation messages
            query: Current user query
            force_search: Force web search even if not needed
            use_ocr: Override default OCR setting
            
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

        # Determine if OCR should be used
        use_ocr_value = self.ocr_enabled if use_ocr is None else use_ocr
        
        # Get search results
        results = self.web_search.search(
            query=query,
            include_content=True,
            max_results=self.max_web_results
        )

        # Fetch content with OCR support if enabled
        if results and use_ocr_value:
            for result in results:
                if not result.get('content') and result.get('url'):
                    # No content fetched yet, try with OCR
                    self.web_search._fetch_content_with_fallback(result)
                    if result.get('extraction_method') == 'ocr':
                        self.web_stats["ocr_extractions"] += 1
        
        # Skip if no results
        if not results:
            return messages
        
        # Add to memory if enabled
        if self.add_to_memory and self.chat.memory_manager:
            added_items = self.web_search.search_and_add_to_memory(
                query=query,
                max_results=self.max_web_results
            )

            if added_items:
                self.web_stats["items_added"] += len(added_items)
                print(f"Added {len(added_items)} web search results to memory")
        
        # Create enhanced messages
        enhanced_messages = messages.copy()
        
        # Format results for context
        web_context = self.web_search.format_for_context(results, query)
        
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
        Process web search command from user.
        
        Args:
            query: User query
            
        Returns:
            Response message or None if not a web command
        """
        # Check for web command
        if query.startswith("!web"):
            # Extract search query
            search_query = query[4:].strip()
            
            if not search_query:
                return "Please provide a search query. Example: !web latest climate reports"
            
            # Check for OCR flag
            use_ocr = self.ocr_enabled

            if search_query.startswith("--no-ocr "):
                use_ocr = False
                search_query = search_query[9:].strip()
            elif search_query.startswith("--ocr "):
                use_ocr = True
                search_query = search_query[6:].strip()

            # Perform search
            results = self.web_search.search(
                query=search_query,
                include_content=True,
                max_results=7  # More results for direct search
            )

            # Fetch content with OCR support if enabled
            if results and use_ocr:
                ocr_count = 0
                for result in results:
                    if not result.get('content') and result.get('url'):
                        # No content fetched yet, try with OCR
                        self.web_search._fetch_content_with_fallback(result)
                        if result.get('extraction_method') == 'ocr':
                            ocr_count += 1
                            self.web_stats["ocr_extractions"] += 1
            
            if not results:
                return f"No results found for: {search_query}"
            
            # Format response
            response = f"Search results for: {search_query}\n\n"
            
            for i, result in enumerate(results):
                title = result.get('title', 'No title')
                url = result.get('url', '')
                domain = result.get('source', '')
                snippet = result.get('snippet', '')
                
                # Indicate OCR enhancement
                extraction_type = " [OCR]" if result.get('extraction_method') == 'ocr' else ""

                response += f"{i+1}. {title} ({domain}){extraction_type}\n"
                if snippet:
                    response += f"   {snippet}\n"
                response += f"   {url}\n\n"
            
            # Add to memory if enabled
            if self.add_to_memory and self.chat.memory_manager:
                added = 0
                for result in results:
                    content = result.get('content', '')
                    if not content:
                        continue
                    
                    # Create metadata
                    metadata = {
                        'source': 'web_search',
                        'extraction_method': result.get('extraction_method', 'html'),
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
            
        # Not a web command
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get web search statistics."""
        return {
            "searches_performed": self.web_stats["searches"],
            "items_added_to_memory": self.web_stats["items_added"],
            "ocr_extractions": self.web_stats["ocr_extractions"],
            "web_enabled": self.web_enabled,
            "ocr_enabled": self.ocr_enabled
        }

    def format_for_context(self, results: List[Dict[str, Any]], query: str) -> str:
        """
        Format search results for inclusion in context.
        Enhanced to show OCR extraction information.

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
            results = self.web_search.rank_results_by_relevance(results, query)

        # Build formatted output
        output = "WEB SEARCH RESULTS:\n\n"

        for i, result in enumerate(results[:5]):  # Limit to top 5
            title = result.get('title', 'No title')
            url = result.get('url', '')
            domain = result.get('source', 'unknown')
            score = result.get('relevance_score', 0)
            content = result.get('content', result.get('snippet', ''))

            # Add extraction method indicator
            extraction_method = "[OCR]" if result.get('extraction_method') == 'ocr' else ""

            # Truncate content to a reasonable length
            if content:
                content = content.strip()
                if len(content) > 300:
                    content = content[:297] + "..."

            # Format result
            output += f"[{i+1}] {title} ({domain}) {extraction_method}\n"
            if content:
                output += f"{content}\n"
            output += f"Source: {url}\n\n"

        output += "Use the information above to help answer the query if relevant.\n"
        return output