"""
Memory type handlers for the unified memory system.
Provides specialized processing for different types of memories.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

class MemoryTypeHandler:
    """Base class for memory type handlers."""
    
    def __init__(self, memory_manager=None):
        """
        Initialize the memory type handler.
        
        Args:
            memory_manager: Reference to the unified memory manager
        """
        self.memory_manager = memory_manager
    
    def process_for_storage(self, content: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Process content and metadata before storage.
        
        Args:
            content: The content to process
            metadata: The metadata to process
            
        Returns:
            Processed content and metadata
        """
        return content, metadata
    
    def process_for_retrieval(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process results during retrieval.
        
        Args:
            results: The search results to process
            
        Returns:
            Processed results
        """
        return result

class KnowledgeMemoryHandler(MemoryTypeHandler):
    """Handler for knowledge memories (facts, definitions, procedures, etc.)."""

    def process_for_storage(self, content: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Process knowledge content and metadata before storage."""
        # Detect knowledge subtype
        knowledge_subtype = self._detect_knowledge_subtype(content)
        if knowledge_subtype:
            metadata["knowledge_subtype"] = knowledge_subtype

        # Extract entities if not already present
        if "entities" not in metadata:
            entities = self._extract_entities(content)
            if entities:
                metadata["entities"] = entities

        # Estimate confidence if not provided
        if "confidence" not in metadata:
            metadata["confidence"] = self._estimate_confidence(content)

        return content, metadata

    def process_for_retrieval(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process knowledge results during retrieval."""
        # Group by knowledge subtypes
        grouped_results = {}

        for result in results:
            metadata = result.get("metadata", {})
            subtype = metadata.get("knowledge_subtype", "general")

            if subtype not in grouped_results:
                grouped_results[subtype] = []

            grouped_results[subtype].append(result)

        # Sort each group by similarity
        for subtype, group in grouped_results.items():
            group.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        # Build final list with diversity of knowledge types
        processed_results = []

        # First add corrections and facts (high priority)
        for priority_type in ["correction", "fact", "definition"]:
            if priority_type in grouped_results:
                # Add up to 3 from each priority type
                processed_results.extend(grouped_results[priority_type][:3])

        # Then add a mix of other types
        used_count = len(processed_results)
        remaining_slots = 10 - used_count  # Allow up to 10 total

        if remaining_slots > 0:
            other_types = [t for t in grouped_results.keys()
                          if t not in ["correction", "fact", "definition"]]

            # Add one from each remaining type until we fill slots
            for subtype in other_types:
                if grouped_results[subtype] and remaining_slots > 0:
                    processed_results.append(grouped_results[subtype][0])
                    remaining_slots -= 1

            # If we still have slots, add more from any type
            if remaining_slots > 0:
                flat_remaining = []
                for subtype in grouped_results:
                    # Skip items we've already added
                    skip_count = 3 if subtype in ["correction", "fact", "definition"] else 1
                    if len(grouped_results[subtype]) > skip_count:
                        flat_remaining.extend(grouped_results[subtype][skip_count:])

                # Sort by similarity
                flat_remaining.sort(key=lambda x: x.get("similarity", 0), reverse=True)

                # Add up to remaining slots
                processed_results.extend(flat_remaining[:remaining_slots])

        return processed_results

    def format_for_prompt(self, results: List[Dict[str, Any]], header: str = None) -> str:
        """Format knowledge results for inclusion in the prompt."""
        if not results:
            return ""

        # Group by knowledge subtypes
        corrections = []
        facts = []
        definitions = []
        procedures = []
        other = []

        for result in results:
            metadata = result.get("metadata", {})
            subtype = metadata.get("knowledge_subtype", "")

            if subtype == "correction" or metadata.get("is_correction", False):
                corrections.append(result)
            elif subtype == "fact":
                facts.append(result)
            elif subtype == "definition":
                definitions.append(result)
            elif subtype == "procedure":
                procedures.append(result)
            else:
                other.append(result)

        lines = []

        # Add corrections first (highest priority)
        if corrections:
            lines.append("IMPORTANT CORRECTIONS (You MUST apply these):")
            for result in corrections:
                lines.append(f"- {result['content']}")
            lines.append("")

        # Combine facts and definitions
        if facts or definitions:
            lines.append("FACTUAL INFORMATION:")

            # Add facts
            for result in facts:
                lines.append(f"- {result['content']}")

            # Add definitions
            for result in definitions:
                # Format slightly differently for definitions
                content = result['content']
                if ":" not in content:
                    # Try to format as term: definition
                    metadata = result.get("metadata", {})
                    term = metadata.get("term", "")
                    if term and not content.startswith(term):
                        content = f"{term}: {content}"

                lines.append(f"- {content}")

            lines.append("")

        # Add procedures
        if procedures:
            lines.append("PROCEDURAL KNOWLEDGE:")
            for result in procedures:
                lines.append(f"- {result['content']}")
            lines.append("")

        # Add other knowledge
        if other and not (facts or definitions or procedures):
            lines.append("OTHER RELEVANT KNOWLEDGE:")
            for result in other:
                lines.append(f"- {result['content']}")

        return "\n".join(lines)

    def _detect_knowledge_subtype(self, content: str) -> str:
        """Detect the subtype of knowledge content."""
        content_lower = content.lower()

        # Check for corrections
        correction_indicators = ["correction", "incorrect", "wrong", "mistake", "error", "actually", "instead"]
        if any(indicator in content_lower for indicator in correction_indicators):
            return "correction"

        # Check for definitions
        definition_indicators = ["is defined as", "refers to", "means", "is a term", "definition"]
        if any(indicator in content_lower for indicator in definition_indicators) or ": " in content:
            return "definition"

        # Check for procedures
        procedure_indicators = ["step", "procedure", "process", "how to", "method", "steps to", "instructions"]
        if any(indicator in content_lower for indicator in procedure_indicators):
            return "procedure"

        # Check for facts (common patterns)
        fact_indicators = ["is a", "was a", "are", "were", "has", "had", "located in", "discovered", "founded"]
        if any(indicator in content_lower for indicator in fact_indicators):
            return "fact"

        # Default to general knowledge
        return "general"

    def _extract_entities(self, content: str) -> List[str]:
        """Extract key entities from knowledge content."""
        # Simple extraction of capitalized phrases
        entities = []
        import re

        # Find proper nouns (capitalized words)
        matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', content)

        # Filter out common non-entity capitalized words
        common_non_entities = {"I", "The", "A", "An", "This", "That", "These", "Those"}

        for match in matches:
            if match not in common_non_entities and len(match) > 1:
                entities.append(match)

        return entities[:5]  # Limit to top 5 entities

    def _estimate_confidence(self, content: str) -> float:
        """Estimate confidence in knowledge content based on heuristics."""
        # Default medium confidence
        confidence = 0.5

        # Adjust based on certainty indicators
        certainty_indicators = ["certainly", "definitely", "absolutely", "always", "never", "fact", "proven"]
        uncertainty_indicators = ["maybe", "perhaps", "possibly", "might", "could", "sometimes", "often"]

        content_lower = content.lower()

        # Boost for certainty indicators
        for indicator in certainty_indicators:
            if indicator in content_lower:
                confidence += 0.1
                break  # Only boost once for certainty

        # Reduce for uncertainty indicators
        for indicator in uncertainty_indicators:
            if indicator in content_lower:
                confidence -= 0.1
                break  # Only reduce once for uncertainty

        # Boost for specific details like dates, numbers, proper nouns
        if re.search(r'\b\d{4}\b', content):  # Years
            confidence += 0.05

        if re.search(r'\b\d+(?:\.\d+)?\b', content):  # Numbers
            confidence += 0.05

        if re.search(r'\b[A-Z][a-z]+\b', content):  # Proper nouns
            confidence += 0.05

        # Ensure in valid range
        return max(0.1, min(0.9, confidence))

class WebKnowledgeHandler(MemoryTypeHandler):
    """Handler for web-sourced memories with source validation."""

    def process_for_storage(self, content: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Process web knowledge content and metadata before storage."""
        # Extract URL if present in content but not in metadata
        if "url" not in metadata:
            import re
            url_match = re.search(r'Source: (https?://[^\s]+)', content)
            if url_match:
                metadata["url"] = url_match.group(1)

                # Clean up content by removing the source URL
                content = re.sub(r'Source: https?://[^\s]+', '', content).strip()

        # Add source domain if URL is available
        if "url" in metadata and "domain" not in metadata:
            import re
            domain_match = re.match(r'https?://(?:www\.)?([^/]+)', metadata["url"])
            if domain_match:
                metadata["domain"] = domain_match.group(1)

        # Estimate freshness if timestamp is available
        if "web_timestamp" in metadata:
            current_time = datetime.now().timestamp()
            age_days = (current_time - metadata["web_timestamp"]) / (60 * 60 * 24)

            # Simple freshness score (1.0 = fresh, 0.0 = stale)
            freshness = max(0.0, min(1.0, 1.0 - (age_days / 90)))  # 90 day scale
            metadata["freshness"] = freshness

        # Set knowledge type if not present
        if "knowledge_subtype" not in metadata:
            knowledge_subtype = self._detect_knowledge_subtype(content)
            metadata["knowledge_subtype"] = knowledge_subtype

        return content, metadata

    def process_for_retrieval(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process web knowledge results during retrieval."""
        # Prioritize fresh content
        for result in results:
            metadata = result.get("metadata", {})

            # Calculate recency boost (0.0-0.2)
            freshness = metadata.get("freshness", 0.5)
            recency_boost = freshness * 0.2

            # Apply boost to similarity
            original_similarity = result.get("similarity", 0.0)
            result["similarity"] = min(1.0, original_similarity + recency_boost)
            result["recency_boost"] = recency_boost

        # Re-sort by adjusted similarity
        results.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

        # Deduplicate by domain
        domains_seen = set()
        deduplicated = []

        for result in results:
            metadata = result.get("metadata", {})
            domain = metadata.get("domain", "")

            # Always keep high-similarity results
            if result.get("similarity", 0.0) > 0.7 or not domain or domain not in domains_seen:
                deduplicated.append(result)
                if domain:
                    domains_seen.add(domain)

        return deduplicated

    def format_for_prompt(self, results: List[Dict[str, Any]], header: str = None) -> str:
        """Format web knowledge results for inclusion in the prompt."""
        if not results:
            return ""

        header = header or "WEB KNOWLEDGE"
        lines = [f"{header}:"]

        # Add primary sources (high confidence) first
        primary_sources = [r for r in results if r.get("similarity", 0.0) > 0.7]
        if primary_sources:
            lines.append("\nPRIMARY SOURCES:")
            for i, result in enumerate(primary_sources[:3]):
                metadata = result.get("metadata", {})
                domain = metadata.get("domain", "")
                domain_str = f" from {domain}" if domain else ""

                lines.append(f"- {result['content']}{domain_str}")
            lines.append("")

        # Add secondary sources
        secondary_sources = [r for r in results if r.get("similarity", 0.0) <= 0.7]
        if secondary_sources:
            lines.append("\nADDITIONAL INFORMATION:")
            for result in secondary_sources[:5]:
                lines.append(f"- {result['content']}")

        return "\n".join(lines)

    def _detect_knowledge_subtype(self, content: str) -> str:
        """Detect the subtype of web knowledge content."""
        content_lower = content.lower()

        # Check for news
        news_indicators = ["today", "yesterday", "last week", "reported", "announced", "news", "according to"]
        if any(indicator in content_lower for indicator in news_indicators):
            return "news"

        # Check for product information
        product_indicators = ["product", "pricing", "costs", "features", "specs", "model", "version"]
        if any(indicator in content_lower for indicator in product_indicators):
            return "product"

        # Default to general web information
        return "web_info"

# Registry of handlers for different memory types
MEMORY_TYPE_HANDLERS = {
    "conversation": ConversationMemoryHandler,
    "command": CommandMemoryHandler,
    "knowledge": KnowledgeMemoryHandler,
    "web_knowledge": WebKnowledgeHandler,
    # Default handler for any unregistered types
    "default": MemoryTypeHandler
}

def get_handler_for_type(memory_type: str, memory_manager=None) -> MemoryTypeHandler:
    """
    Get the appropriate handler for a memory type.

    Args:
        memory_type: Type of memory
        memory_manager: Reference to memory manager

    Returns:
        Handler instance for the memory type
    """
    # Split to get base type (e.g., "command_tabular" -> "command")
    base_type = memory_type.split("_")[0] if "_" in memory_type else memory_type

    # Get handler class
    handler_class = MEMORY_TYPE_HANDLERS.get(base_type, MEMORY_TYPE_HANDLERS["default"])

    # Create and return instance
    return handler_class(memory_manager)s

    def format_for_prompt(self, results: List[Dict[str, Any]], header: str = None) -> str:
        """
        Format results for inclusion in the prompt.

        Args:
            results: The search results to format
            header: Optional header to include

        Returns:
            Formatted string
        """
        if not results:
            return ""

        lines = []
        if header:
            lines.append(f"{header}:")

        for result in results:
            lines.append(f"- {result['content']}")

        return "\n".join(lines)

class ConversationMemoryHandler(MemoryTypeHandler):
    """Handler for conversation memories."""

    def process_for_storage(self, content: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Process conversation content and metadata before storage."""
        # Clean up content
        content = content.strip()

        # Add additional metadata
        metadata["conversation_timestamp"] = datetime.now().isoformat()
        metadata["word_count"] = len(content.split())

        return content, metadata

    def process_for_retrieval(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process conversation results during retrieval."""
        # Group by conversation clusters
        clusters = self._cluster_by_conversation(results)

        # Select representative results from each cluster
        processed_results = []
        for cluster in clusters:
            # Take the highest similarity result from each cluster
            cluster.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            processed_results.append(cluster[0])

        return processed_results

    def _cluster_by_conversation(self, results: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Cluster results by conversation to avoid redundancy.

        Args:
            results: List of search results

        Returns:
            List of clusters (each cluster is a list of results)
        """
        # Extract timestamps and sort
        timestamped_results = []
        for result in results:
            timestamp = result.get("metadata", {}).get("conversation_timestamp")
            if timestamp:
                timestamped_results.append((timestamp, result))

        # Sort by timestamp
        timestamped_results.sort(key=lambda x: x[0])

        # Cluster by time proximity (5 minute window)
        clusters = []
        current_cluster = []
        last_timestamp = None

        for timestamp, result in timestamped_results:
            if not current_cluster:
                # First item in cluster
                current_cluster.append(result)
                last_timestamp = timestamp
            elif self._timestamp_diff(timestamp, last_timestamp) < 300:  # 5 minutes in seconds
                # Within time window of previous item
                current_cluster.append(result)
                last_timestamp = timestamp
            else:
                # Start new cluster
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = [result]
                last_timestamp = timestamp

        # Add final cluster
        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def _timestamp_diff(self, ts1: str, ts2: str) -> float:
        """Calculate difference between ISO timestamps in seconds."""
        try:
            dt1 = datetime.fromisoformat(ts1)
            dt2 = datetime.fromisoformat(ts2)
            return abs((dt1 - dt2).total_seconds())
        except Exception:
            return float("inf")  # Return infinity if timestamps can't be parsed

class CommandMemoryHandler(MemoryTypeHandler):
    """Handler for command memories with special handling for tabular data."""

    def process_for_storage(self, content: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Process command content and metadata before storage."""
        # Detect if content is tabular
        is_tabular = self._is_tabular_data(content)
        metadata["is_tabular"] = is_tabular

        # Extract command from metadata if available
        command = metadata.get("command", "")
        if command:
            metadata["command_type"] = self._detect_command_type(command)

        # For tabular data, add structured representation
        if is_tabular:
            # Extract header and rows
            lines = content.strip().split("\n")
            if len(lines) >= 2:
                header = lines[0]
                rows = lines[1:]

                columns = self._extract_columns_from_header(header)
                metadata["columns"] = columns
                metadata["row_count"] = len(rows)

                # Parse a sample of rows
                parsed_rows = []
                for i, row in enumerate(rows[:5]):  # Limit to first 5 rows
                    parsed_row = self._extract_row_data(row, columns, header)
                    if parsed_row:
                        parsed_rows.append(parsed_row)

                if parsed_rows:
                    metadata["sample_data"] = parsed_rows

        return content, metadata

    def process_for_retrieval(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process command results during retrieval."""
        # Prioritize exact command matches
        command_matches = {}

        for result in results:
            metadata = result.get("metadata", {})
            command = metadata.get("command", "")

            if command:
                # Use command as key for deduplication
                if command not in command_matches or result.get("similarity", 0) > command_matches[command].get("similarity", 0):
                    command_matches[command] = result

        # Add exact matches first
        processed_results = list(command_matches.values())

        # Add remaining results
        for result in results:
            metadata = result.get("metadata", {})
            command = metadata.get("command", "")

            if not command or command not in command_matches:
                processed_results.append(result)

        return processed_results

    def format_for_prompt(self, results: List[Dict[str, Any]], header: str = None) -> str:
        """Format command results for inclusion in the prompt."""
        if not results:
            return ""

        header = header or "COMMAND OUTPUT INFORMATION"
        lines = [f"{header}:"]

        # Group by command type
        tabular_results = []
        general_results = []

        for result in results:
            metadata = result.get("metadata", {})
            if metadata.get("is_tabular", False):
                tabular_results.append(result)
            else:
                general_results.append(result)

        # Format tabular results first
        if tabular_results:
            lines.append("\nTABULAR DATA:")
            for result in tabular_results[:3]:  # Limit to 3 tabular results
                metadata = result.get("metadata", {})
                command = metadata.get("command", "")
                lines.append(f"- From command '{command}':")

                # Add column information if available
                columns = metadata.get("columns", [])
                if columns:
                    lines.append(f"  Columns: {', '.join(columns)}")

                # Add row count if available
                row_count = metadata.get("row_count", 0)
                if row_count:
                    lines.append(f"  Rows: {row_count}")

                # Add sample data if available
                sample_data = metadata.get("sample_data", [])
                if sample_data:
                    lines.append("  Sample data:")
                    for i, row in enumerate(sample_data[:2]):  # Limit to 2 sample rows
                        row_str = ", ".join(f"{k}: {v}" for k, v in row.items())
                        lines.append(f"    Row {i+1}: {row_str}")

                # Add a separator between tabular results
                lines.append("")

        # Format general results
        if general_results:
            lines.append("\nGENERAL COMMAND INFORMATION:")
            for result in general_results[:5]:  # Limit to 5 general results
                lines.append(f"- {result['content']}")

        return "\n".join(lines)

    def _is_tabular_data(self, output: str) -> bool:
        """Detect if the output is likely in a tabular format."""
        if not output:
            return False

        lines = output.strip().split('\n')
        if len(lines) < 2:
            return False

        # Check for consistent column structure in first few lines
        first_line_spaces = [i for i, char in enumerate(lines[0]) if char.isspace()]

        # Need at least some spaces for columns
        if not first_line_spaces:
            return False

        # Check if second line also has spaces at similar positions
        second_line_spaces = [i for i, char in enumerate(lines[1]) if char.isspace()]

        # Count matching space positions (with some tolerance)
        matches = 0
        for pos1 in first_line_spaces:
            for pos2 in second_line_spaces:
                if abs(pos1 - pos2) <= 3:  # Allow 3 char tolerance
                    matches += 1
                    break

        # If we have a good number of matching spaces, likely tabular
        return matches >= 3

    def _detect_command_type(self, command: str) -> str:
        """Detect the type of command."""
        if any(cmd in command for cmd in ["ls", "dir"]):
            return "file_listing"
        elif any(cmd in command for cmd in ["grep", "find", "locate"]):
            return "search"
        elif any(cmd in command for cmd in ["df", "du", "free", "top", "ps"]):
            return "system_metrics"
        elif any(cmd in command for cmd in ["cat", "less", "more", "head", "tail"]):
            return "file_content"
        elif any(cmd in command for cmd in ["uname", "hostname", "whoami", "id"]):
            return "system_info"
        else:
            return "general_command"

    def _extract_columns_from_header(self, header_row: str) -> list:
        """Extract column names from a header row."""
        # Simple version: split by whitespace and filter empty strings
        return [col for col in header_row.split() if col.strip()]

    def _extract_row_data(self, row: str, columns: list, header_row: str) -> dict:
        """
        Extract structured data from a row based on column positions.

        Args:
            row: Row string
            columns: List of column names
            header_row: Original header row for position reference

        Returns:
            Dictionary mapping column names to values
        """
        result = {}

        # Try to split by whitespace alignment first
        row_parts = row.split()

        # If we have exactly the same number of parts as columns, direct mapping
        if len(row_parts) == len(columns):
            for i, col in enumerate(columns):
                result[col] = row_parts[i]
            return result

        # For more complex alignment, try position-based extraction
        try:
            # Find column boundaries based on header
            col_positions = []
            current_pos = 0
            for col in columns:
                col_pos = header_row.find(col, current_pos)
                if col_pos == -1:
                    break
                col_positions.append(col_pos)
                current_pos = col_pos + len(col)

            # Add end position
            col_positions.append(len(header_row) + 1)

            # Extract column values using positions
            for i in range(len(col_positions) - 1):
                start = col_positions[i]
                end = col_positions[i+1]

                # Make sure row is long enough
                if start < len(row):
                    value = row[start:min(end, len(row))].strip()
                    result[columns[i]] = value
        except Exception:
            # Fall back to simple splitting if position-based fails
            if not result and len(row_parts) > 0:
                # Just assign values to columns until we run out of either
                for i in range(min(len(columns), len(row_parts))):
                    result[columns[i]] = row_parts[i]

                # If we have more row parts than columns, append remaining to the last column
                if len(row_parts) > len(columns) and len(columns) > 0:
                    last_col = columns[-1]
                    result[last_col] = " ".join([result.get(last_col, "")] + row_parts[len(columns):])

        return result