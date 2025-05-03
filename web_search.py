"""
Simplified web search for local LLM chat with Google Custom Search API integration.
Only handles search keyword generation and interfaces with google_search_client.py.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime

class WebSearchManager:
    """
    Simplified web search manager that only handles search keyword generation
    and interfaces with GoogleSearchClient for actual searches.
    """

    def __init__(
        self,
        memory_manager=None,
        question_classifier=None,
        similarity_enhancement_factor: float = 0.3
    ):
        """
        Initialize the web search manager.

        Args:
            memory_manager: Optional MemoryManager for storing results
            question_classifier: Optional QuestionClassifier for keyword generation
            similarity_enhancement_factor: Factor for similarity enhancement
        """
        self.memory_manager = memory_manager
        self.question_classifier = question_classifier
        self.similarity_enhancement_factor = similarity_enhancement_factor

        # Initialize Google Search client when needed
        self.search_client = None

    def get_time(self) -> str:
        """Get formatted timestamp for logging."""
        return datetime.now().strftime("[%d/%m/%y %H:%M:%S]") + ' [WebSearch]'

    def set_search_client(self, search_client) -> None:
        """
        Set the Google Search client.

        Args:
            search_client: GoogleSearchClient instance
        """
        self.search_client = search_client

    def search(
        self,
        query: str,
        include_content: bool = True,
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform a web search for the given query using GoogleSearchClient.

        Args:
            query: The search query
            include_content: Whether to fetch full page content
            max_results: Maximum number of results to return

        Returns:
            List of search result dictionaries
        """
        if not self.search_client:
            print(f"{self.get_time()} No search client available")
            return []

        # Generate optimized search keywords
        search_terms = self._generate_search_keywords(query)
        print(f"{self.get_time()} Searching for: {search_terms}")

        # Delegate search to GoogleSearchClient
        return self.search_client.search(
            query=search_terms,
            num_results=max_results or 10,  # Default to 10 results if not specified
            include_content=include_content
        )

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

        if not self.search_client:
            print(f"{self.get_time()} No search client available")
            return []

        # Delegate to search client
        return self.search_client.search_and_add_to_memory(
            query=query,
            max_results=max_results
        )

    def format_for_context(self, results: List[Dict[str, Any]], query: str) -> str:
        """
        Format search results for inclusion in context.

        Args:
            results: List of search results
            query: Original query

        Returns:
            Formatted context string
        """
        if not self.search_client:
            print(f"{self.get_time()} No search client available")
            return ""

        # Delegate to search client
        return self.search_client.format_for_context(results, query)