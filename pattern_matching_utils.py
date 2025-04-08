"""
Utility functions for pattern matching across the local_models project.
Centralizes common text extraction, pattern detection, and classification functions.
"""

import re
from typing import List, Dict, Tuple, Optional, Any

# Command detection utilities
def is_command_related_query(query: str) -> bool:
    """
    Detect if a query is related to command execution or output.

    Args:
        query: User query

    Returns:
        Boolean indicating if query is command-related
    """
    # Look for explicit command markers
    if re.search(r'[!`]{\s*[^}]+\s*}', query):
        return True

    # Look for command references
    command_references = [
        r'\b(command|cmd|terminal|shell|bash|console)\b',
        r'\b(output|result) of\b',
        r'\b(ran|executed|typed|entered)\b',
        r'\b(df|ls|ps|top|grep|find|cat)\b'
    ]

    for pattern in command_references:
        if re.search(pattern, query, re.IGNORECASE):
            return True

    return False

def extract_command_context(query: str) -> Optional[str]:
    """
    Extract command context from a query.

    Args:
        query: User query

    Returns:
        Command context string or None
    """
    # Check for explicit command references
    cmd_match = re.search(r'[!`]{\s*([^}]+)\s*}', query)
    if cmd_match:
        return cmd_match.group(1).strip()

    # Look for command mentions
    cmd_mention = re.search(r'(?:command|cmd|ran|executed|typed|entered)\s+[`"]([^`"]+)[`"]', query)
    if cmd_mention:
        return cmd_mention.group(1).strip()

    # Extract command name references
    for cmd in ['df', 'ls', 'ps', 'top', 'grep', 'find', 'cat', 'du']:
        # Look for cmd or cmd -flags pattern
        cmd_pattern = rf'\b({cmd}(?:\s+-\w+)?)\b'
        cmd_ref = re.search(cmd_pattern, query)
        if cmd_ref:
            return cmd_ref.group(1).strip()

    return None

def extract_command_type(command_context: str) -> str:
    """
    Extract command type from context string.

    Args:
        command_context: Context about the command

    Returns:
        Command type string
    """
    # Extract the actual command if possible
    command_match = re.search(r'`([^`]+)`', command_context)
    if command_match:
        command = command_match.group(1).strip()
    else:
        command = command_context.strip()

    # Detect command type
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

def is_tabular_command(command_context: str) -> bool:
    """
    Check if a command typically produces tabular output.

    Args:
        command_context: Command context string

    Returns:
        Boolean indicating if command likely produces tabular output
    """
    tabular_commands = [
        "df", "du", "ls -l", "ps", "top", "free",
        "netstat", "ip addr", "mount", "docker ps",
        "systemctl list", "apt list", "pip list"
    ]

    # Check if any tabular command is in the context
    return any(cmd in command_context for cmd in tabular_commands)

def is_tabular_data(output: str) -> bool:
    """
    Detect if the output is likely in a tabular format.

    Args:
        output: Command output text

    Returns:
        Boolean indicating if output is tabular
    """
    if not output:
        return False

    lines = output.strip().split('\n')
    if len(lines) < 2:
        return False

    # Check for common tabular format indicators

    # 1. Check if lines have consistent column-like structure
    # Extract positions of whitespace gaps in the first two lines
    if len(lines) >= 2:
        # Get whitespace positions in first line (potential header)
        first_line_spaces = [i for i, char in enumerate(lines[0]) if char.isspace()]
        # Get clusters of whitespace positions (columns are separated by multiple spaces)
        column_separators = []
        current_cluster = []
        for pos in first_line_spaces:
            if not current_cluster or pos == current_cluster[-1] + 1:
                current_cluster.append(pos)
            else:
                if len(current_cluster) >= 2:  # Only consider gaps of 2+ spaces
                    column_separators.append((current_cluster[0], current_cluster[-1]))
                current_cluster = [pos]

        # Check if second line also has gaps at similar positions
        if column_separators:
            if len(lines) >= 2:
                second_line_spaces = [i for i, char in enumerate(lines[1]) if char.isspace()]
                second_line_clusters = []
                current_cluster = []
                for pos in second_line_spaces:
                    if not current_cluster or pos == current_cluster[-1] + 1:
                        current_cluster.append(pos)
                    else:
                        if len(current_cluster) >= 2:
                            second_line_clusters.append((current_cluster[0], current_cluster[-1]))
                        current_cluster = [pos]

                # If gap patterns are similar, likely tabular
                if len(column_separators) > 0 and len(second_line_clusters) > 0:
                    matches = 0
                    for sep1 in column_separators:
                        for sep2 in second_line_clusters:
                            if abs(sep1[0] - sep2[0]) <= 3 and abs(sep1[1] - sep2[1]) <= 3:
                                matches += 1
                                break

                    if matches >= min(len(column_separators), len(second_line_clusters)) * 0.5:
                        return True

    # 2. Check for common table border characters
    has_borders = any(all(c in line for c in ['+', '-', '+']) for line in lines)
    if has_borders:
        return True

    # 3. Check for consistent pipe separators
    pipe_separated = all('|' in line for line in lines[:min(5, len(lines))])
    if pipe_separated:
        return True

    # 4. Known tabular commands
    tabular_commands = ["df", "du", "ls -l", "ps", "top", "free", "netstat", "ip addr", "mount"]
    if any(cmd in output.lower() for cmd in tabular_commands):
        return True

    return False

# Data extraction utilities
def extract_columns_from_header(header_row: str) -> list:
    """
    Extract column names from a header row using whitespace patterns.

    Args:
        header_row: Header row string

    Returns:
        List of column names
    """
    # Simple version: split by whitespace and filter out empty strings
    columns = [col for col in header_row.split() if col.strip()]
    return columns

def extract_column_references(query: str) -> List[str]:
    """
    Extract column name references from a query.

    Args:
        query: User query

    Returns:
        List of column names referenced
    """
    # Common column names in tabular data commands
    common_columns = [
        "filesystem", "size", "used", "avail", "use%", "mounted", "mount",
        "pid", "user", "cpu", "mem", "command", "name", "device",
        "type", "total", "free", "shared", "buff/cache", "available"
    ]

    # Extract column references (case insensitive)
    query_lower = query.lower()
    referenced_columns = []

    for col in common_columns:
        # Look for column name mentions
        if col.lower() in query_lower:
            # Check if it's actually a column reference
            # (surrounded by spaces or punctuation, not part of another word)
            pattern = r'(?:^|\W)(' + re.escape(col) + r')(?:$|\W)'
            if re.search(pattern, query_lower, re.IGNORECASE):
                referenced_columns.append(col)

    return referenced_columns

def extract_row_references(query: str) -> List[int]:
    """
    Extract row number references from a query.

    Args:
        query: User query

    Returns:
        List of row indices referenced
    """
    # Look for patterns like "first row", "row 3", "third line", etc.
    row_indices = []

    # Numeric references
    num_matches = re.findall(r'(?:row|line)\s+(\d+)', query.lower())
    for match in num_matches:
        try:
            # Convert to 0-based index
            idx = int(match) - 1
            if idx >= 0:
                row_indices.append(idx)
        except ValueError:
            pass

    # Textual references
    text_map = {
        'first': 0, 'second': 1, 'third': 2, 'fourth': 3, 'fifth': 4,
        'last': -1, 'final': -1
    }

    for text, idx in text_map.items():
        if f"{text} row" in query.lower() or f"{text} line" in query.lower():
            row_indices.append(idx)

    return row_indices

def extract_arithmetic_expression(query: str) -> Optional[str]:
    """
    Extract an arithmetic expression from a query.

    Args:
        query: The query string

    Returns:
        Extracted expression or None
    """
    # Look for patterns like "what is 5 + 3" or "calculate 10 - 7"
    # or just "5 + 3"
    patterns = [
        r'(\d+\s*[\+\-\*\/]\s*\d+)',
        r'(?:what is|calculate|compute|evaluate|result of).*?(\d+\s*[\+\-\*\/]\s*\d+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            # Get the expression and clean it up
            expr = match.group(1).strip()
            # Replace any unicode math symbols with Python operators
            expr = expr.replace('×', '*').replace('÷', '/')
            # Remove any spaces
            expr = expr.replace(' ', '')
            return expr

    return None

# Domain-specific pattern detection
def is_mapping_data(command: str, output: str) -> bool:
    """Detect if content contains mapping data (like transliteration tables)"""
    # Look for patterns indicating mapping data
    mapping_indicators = [
        r'\w+\s*→\s*\w+',  # word → word
        r'\w+\s*->\s*\w+',  # word -> word
        r'mapping|map|translation|conversion',  # Keywords
        r'table|chart|correspondence'  # Structure indicators
    ]

    # Check if file is likely a mapping table
    if any(re.search(pattern, output, re.IGNORECASE) for pattern in mapping_indicators):
        return True

    # Check if filename indicates mapping
    if "map" in command or "translation" in command:
        return True

    return False

def is_mathematical_content(command: str, output: str) -> bool:
    """Detect if content contains mathematical information"""
    # Look for patterns indicating mathematical content
    math_indicators = [
        r'=\s*\d+',  # Equations with numbers
        r'[+\-*/^]',  # Mathematical operators
        r'log|sin|cos|tan|sqrt',  # Mathematical functions
        r'theorem|formula|equation',  # Mathematical terms
        r'∫|∑|∏|√|π|∞|≠|≤|≥'  # Mathematical symbols
    ]

    # Check file content for math indicators
    if any(re.search(pattern, output) for pattern in math_indicators):
        return True

    # Check filename for math indicators
    math_files = ["math", "calc", "formula", "equation", "log", "algorithm"]
    if any(term in command.lower() for term in math_files):
        return True

    return False

def is_entity_description(command: str, output: str) -> bool:
    """Detect if content contains entity descriptions"""
    # Look for patterns indicating entity descriptions
    entity_indicators = [
        r'^[A-Z][a-z]+(?:\s+[a-z]+){0,3}(?:\s+[A-Z][a-z]+)*\s+is\s+a',  # Entity is a...
        r'\([A-Z]{2,}\)',  # Acronyms in parentheses
        r'species|genus|family|class|order|type',  # Taxonomic terms
        r'was\s+born|founded|established|discovered|created',  # Historical statements
        r'located\s+in|found\s+in|native\s+to',  # Location statements
        r'\d+\s*(?:cm|m|km|ft|year|kg)'  # Measurements/quantitative properties
    ]

    # Check for entity description patterns
    if any(re.search(pattern, output) for pattern in entity_indicators):
        return True

    return False

# Command matching utility
def commands_match(context: str, command: str) -> bool:
    """
    Check if a command context matches a command string.

    Args:
        context: Command context
        command: Command string

    Returns:
        Boolean indicating if they match
    """
    # Extract main command (first word)
    context_cmd = context.split()[0] if context else ""
    command_cmd = command.split()[0] if command else ""

    # Direct match
    if context == command:
        return True

    # Main command match
    if context_cmd and context_cmd == command_cmd:
        return True

    # Check if one contains the other
    if context in command or command in context:
        return True

    return False

def extract_example_pairs(content: str) -> List[Tuple[str, str]]:
    """Extract example input/output pairs from content"""
    examples = []

    # Look for example section
    example_section_match = re.search(r'(?:example|examples|word examples)[:\s]+(.*?)(?=(?:^#)|$)',
                                    content, re.I | re.DOTALL | re.MULTILINE)

    if example_section_match:
        example_text = example_section_match.group(1)
        # Extract pairs in format: "source → target"
        pairs = re.findall(r'"([^"]+)"\s*(?:→|->)\s*"([^"]+)"', example_text)
        if pairs:
            examples.extend(pairs)
        else:
            # Try alternate format: source → target
            alt_pairs = re.findall(r'(\w+)\s*(?:→|->)\s*([^\s(]+)', example_text)
            examples.extend(alt_pairs)

    return examples

def extract_mapping_category(source_item: str, full_content: str) -> str:
    """Extract category for a mapping entry based on content patterns"""
    lowercase_content = full_content.lower()

    # Check section headers near this item
    lines = full_content.split('\n')
    for i, line in enumerate(lines):
        if source_item in line.lower():
            # Look at up to 10 lines before this one for a header
            for j in range(max(0, i-10), i):
                if re.match(r'^#\s+(.+)', lines[j]):
                    header = re.match(r'^#\s+(.+)', lines[j]).group(1)
                    # Clean up the header
                    header = re.sub(r'consonants|vowels|characters|mapping', '', header, flags=re.I)
                    return header.strip()

    # Check common categories
    if re.match(r'^[aeiou]', source_item):
        return "vowels"
    elif len(source_item) == 1:
        return "single characters"
    elif "consonant" in lowercase_content and source_item[0] in lowercase_content:
        for line in lowercase_content.split('\n'):
            if f"{source_item[0]} consonants" in line:
                return f"{source_item[0].upper()} consonants"

    # Default to character type
    if len(source_item) == 1:
        return "single characters"
    elif len(source_item) == 2 and source_item[1] in "aeiou":
        return f"{source_item[0].upper()} consonants"
    else:
        return "combinations"

def clean_duplicate_memories(memory_text: str) -> str:
    """Remove duplicate or highly similar memories from the text"""
    if not memory_text:
        return ""

    # Split into sections
    sections = re.split(r'(IMPORTANT CORRECTIONS|FACTUAL INFORMATION|OTHER RELEVANT INFORMATION):', memory_text)

    if len(sections) <= 1:
        return memory_text

    cleaned_sections = []
    for i in range(0, len(sections), 2):
        if i+1 < len(sections):
            header = sections[i]
            content = sections[i+1]

            # Split content into bullet points
            bullets = content.split('\n- ')

            # Remove duplicates while preserving order
            seen = set()
            unique_bullets = []

            for bullet in bullets:
                # Create a simplified key for comparison (lowercase, punctuation removed)
                simplified = re.sub(r'[^\w\s]', '', bullet.lower())
                simplified = ' '.join(simplified.split())  # Normalize whitespace

                if simplified and simplified not in seen:
                    seen.add(simplified)
                    unique_bullets.append(bullet)
                elif not simplified:
                    unique_bullets.append(bullet)  # Keep formatting paragraphs
            else:
                unique_bullets.append(bullet)  # Keep empty lines

            # Rebuild content
            cleaned_content = '\n- '.join(unique_bullets)
            if i == 0:  # First section doesn't need the header
                cleaned_sections.append(header + ":" + cleaned_content)
            else:
                cleaned_sections.append(header + ":" + cleaned_content)
        else:
            cleaned_sections.append(sections[i])

    return ''.join(cleaned_sections)