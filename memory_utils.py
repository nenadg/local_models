"""
Memory utilities for consistent processing of knowledge across components.
Provides shared functions for content classification, metadata generation, and memory operations.
"""

import re
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

def classify_content(content: str, question_classifier=None) -> Dict[str, Any]:
    """
    Classify content into knowledge categories and subcategories.
    
    Args:
        content: Text content to classify
        question_classifier: Optional QuestionClassifier instance
        
    Returns:
        Dictionary with classification results
    """
    result = {
        'main_category': 'unknown',
        'subcategory': None,
        'confidence': 0.5
    }
    
    # Skip empty content
    if not content or not content.strip():
        return result
    
    # Use question classifier if available
    if question_classifier:
        try:
            category, confidence = question_classifier.classify(content)
            result['main_category'] = category
            result['confidence'] = confidence
            
            # Determine subcategory
            if category in question_classifier.subcategories:
                subcategory, subcategory_confidence = question_classifier.identify_subcategory(content, category)
                if subcategory:
                    result['subcategory'] = subcategory
                    result['subcategory_confidence'] = subcategory_confidence
        except Exception as e:
            print(f"Error classifying content: {e}")
    
    # Fall back to basic heuristics if category is still unknown
    if result['main_category'] == 'unknown':
        # Simple heuristics for common patterns
        if re.search(r'\b(what is|who is|where is|when did)\b', content.lower()):
            result['main_category'] = 'declarative'
        elif re.search(r'\b(how to|how do I|steps to)\b', content.lower()):
            result['main_category'] = 'procedural_knowledge'
        elif re.search(r'\b(theory|principle|concept|framework)\b', content.lower()):
            result['main_category'] = 'conceptual_knowledge'
        elif re.search(r'\b(manual|guide|documentation|book|publication)\b', content.lower()):
            result['main_category'] = 'explicit'
        elif re.search(r'\b(experience|feeling|what it\'s like)\b', content.lower()):
            result['main_category'] = 'experiential'
        elif re.search(r'\b(opinion|think|suggest|recommend|advise)\b', content.lower()):
            result['main_category'] = 'tacit'
        elif re.search(r'\b(context|environment|setting|circumstances)\b', content.lower()):
            result['main_category'] = 'contextual'
    
    return result

def generate_memory_metadata(content: str, classification: Optional[Dict[str, Any]] = None, 
                            source: str = "unknown") -> Dict[str, Any]:
    """
    Generate metadata for memory storage based on content classification.
    
    Args:
        content: Text content
        classification: Classification results or None
        source: Source of the memory (e.g., "user_query", "command_output")
        
    Returns:
        Dictionary with metadata
    """
    # Default metadata
    metadata = {
        "source": source,
        "source_hint": source,  # Duplicate for backward compatibility
        "timestamp": datetime.now().timestamp(),
        "type": "unknown"
    }
    
    # Apply classification if available
    if classification:
        main_category = classification.get("main_category", "unknown")
        metadata["main_category"] = main_category
        
        # Set type based on main category
        if main_category in ['declarative', 'factual', 'explicit']:
            metadata["type"] = "fact"
        elif main_category in ['procedural', 'procedural_knowledge']:
            metadata["type"] = "procedure"
        elif main_category in ['conceptual', 'conceptual_knowledge']:
            metadata["type"] = "concept"
        elif main_category in ['experiential', 'tacit']:
            metadata["type"] = "experience"
        elif main_category == 'contextual':
            metadata["type"] = "context"
        
        # Add subcategory if available
        if classification.get("subcategory"):
            metadata["subcategory"] = classification["subcategory"]
            metadata["source_hint"] = classification["subcategory"]  # For compatibility
        else:
            metadata["source_hint"] = main_category  # For compatibility
        
        metadata["classification_confidence"] = classification.get("confidence", 0.5)
    
    # Add content-based metadata
    # Length categorization
    content_length = len(content)
    if content_length < 100:
        metadata["content_length"] = "short"
    elif content_length < 500:
        metadata["content_length"] = "medium"
    else:
        metadata["content_length"] = "long"
    
    # Content hash for identification
    content_hash = hashlib.md5(content.encode()).hexdigest()[:10]
    metadata["content_hash"] = content_hash
    
    # Extract topics based on keywords
    topics = extract_topics(content)
    if topics:
        metadata["topics"] = topics
    
    return metadata

def extract_topics(text: str) -> List[str]:
    """
    Extract potential topics from text for improved retrieval.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of extracted topics
    """
    topics = []
    
    # Look for key noun phrases
    noun_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    matches = re.findall(noun_pattern, text)
    
    # Filter to reasonable length phrases
    for match in matches:
        if 4 <= len(match) <= 30 and match not in topics:
            topics.append(match.lower())
    
    # Look for key terminology with specific domain-based patterns
    domain_patterns = [
        # Technology
        r'\b(artificial intelligence|machine learning|python|javascript|programming|database|cloud computing)\b',
        # Science
        r'\b(physics|chemistry|biology|astronomy|mathematics|geology|quantum mechanics)\b',
        # Health
        r'\b(medicine|health|disease|treatment|symptoms|diagnosis|wellness|nutrition)\b',
        # Business
        r'\b(finance|marketing|management|entrepreneurship|investment|economics|strategy)\b',
        # Arts
        r'\b(literature|music|painting|sculpture|dance|theater|photography|film)\b'
    ]
    
    for pattern in domain_patterns:
        matches = re.findall(pattern, text.lower())
        topics.extend([m for m in matches if m not in topics])
    
    # Limit the number of topics
    return topics[:5]

def format_content_for_storage(content: str, classification: Optional[Dict[str, Any]] = None) -> str:
    """
    Format content for optimal storage and retrieval.
    
    Args:
        content: Original content
        classification: Classification results or None
        
    Returns:
        Formatted content
    """
    formatted = content.strip()
    
    # Ensure proper punctuation
    if formatted and not formatted.endswith(('.', '!', '?')):
        formatted += '.'
    
    # Format based on category if available
    if classification:
        main_category = classification.get("main_category", "unknown")
        
        if main_category == "procedural_knowledge":
            # Ensure procedural content has clear structure
            if not re.search(r'\b(steps|first|begin|start|how to)\b', formatted.lower()):
                formatted = f"How to: {formatted}"
        
        elif main_category == "declarative" and len(formatted) < 200:
            # Ensure declarative content is stated clearly for short entries
            if not re.search(r'\b(is|are|was|were|has|have|had)\b', formatted.lower()):
                # Try to make it more definitional
                formatted = f"This information states that {formatted}"
                
        elif main_category == "conceptual_knowledge" and len(formatted) < 200:
            # Ensure concept is clearly indicated
            if not re.search(r'\b(concept|theory|principle|framework)\b', formatted.lower()):
                formatted = f"The concept explained: {formatted}"
    
    return formatted

def save_to_memory(memory_manager, content: str, classification: Optional[Dict[str, Any]] = None, 
                 source: str = "unknown", related_content: Optional[str] = None) -> Dict[str, Any]:
    """
    Save content to memory with appropriate metadata.
    
    Args:
        memory_manager: MemoryManager instance
        content: Content to save
        classification: Classification results or None
        source: Source of the memory
        related_content: Additional related content to save
        
    Returns:
        Dictionary with results
    """
    results = {
        "saved": False,
        "memory_id": None,
        "additional_ids": []
    }
    
    # Skip empty content
    if not content or not content.strip():
        return results
    
    try:
        # Format content
        formatted_content = format_content_for_storage(content, classification)
        
        # Generate metadata
        metadata = generate_memory_metadata(formatted_content, classification, source)
        
        # Add to memory
        memory_id = memory_manager.add(
            content=formatted_content,
            metadata=metadata
        )
        
        if memory_id:
            results["saved"] = True
            results["memory_id"] = memory_id
            results["metadata"] = metadata
            
            # Save related content if provided
            if related_content and related_content.strip():
                # Generate classification for related content
                if classification and "question_classifier" in classification:
                    related_classification = classify_content(
                        related_content, 
                        classification["question_classifier"]
                    )
                else:
                    related_classification = None
                
                # Create related metadata
                related_metadata = generate_memory_metadata(
                    related_content, 
                    related_classification, 
                    source
                )
                related_metadata["related_to"] = memory_id
                
                # Add to memory
                related_id = memory_manager.add(
                    content=related_content,
                    metadata=related_metadata
                )
                
                if related_id:
                    results["additional_ids"].append(related_id)
    
    except Exception as e:
        results["error"] = str(e)
    
    return results

def format_memories_by_category(memories: List[Dict[str, Any]], main_category: str) -> str:
    """
    Format memories based on the main knowledge category.
    
    Args:
        memories: List of retrieved memories
        main_category: Main knowledge category
        
    Returns:
        Formatted memory text
    """
    if not memories:
        return ""
        
    output = "MEMORY CONTEXT:\n\n"
    
    # Group memories by category and subcategory
    categorized = {}
    
    for memory in memories:
        metadata = memory.get("metadata", {})
        mem_category = metadata.get("main_category", metadata.get("source_hint", "unknown"))
        mem_subcategory = metadata.get("subcategory", "general")
        
        if mem_category not in categorized:
            categorized[mem_category] = {}
            
        if mem_subcategory not in categorized[mem_category]:
            categorized[mem_category][mem_subcategory] = []
            
        categorized[mem_category][mem_subcategory].append(memory)
    
    # Format based on query category
    if main_category in ['declarative', 'factual']:
        # For declarative queries, prioritize factual information
        output += "RELEVANT FACTS:\n"
        
        # Add declarative memories first if available
        if 'declarative' in categorized or 'factual' in categorized:
            fact_categories = [cat for cat in categorized if cat in ['declarative', 'factual']]
            for cat in fact_categories:
                for subcategory, subcategory_memories in categorized[cat].items():
                    for memory in subcategory_memories:
                        content = memory.get("content", "").strip()
                        similarity = memory.get("similarity", 0)
                        output += f"- [{similarity:.2f}] {content}\n\n"
        
        # Add other categories with headers
        other_categories = [cat for cat in categorized if cat not in ['declarative', 'factual']]
        if other_categories:
            for category in other_categories:
                readable_category = category.replace('_', ' ').capitalize()
                output += f"\nADDITIONAL {readable_category.upper()} INFORMATION:\n"
                
                for subcategory, subcategory_memories in categorized[category].items():
                    for memory in subcategory_memories:
                        content = memory.get("content", "").strip()
                        similarity = memory.get("similarity", 0)
                        output += f"- [{similarity:.2f}] {content}\n\n"
                        
    elif main_category in ['procedural', 'procedural_knowledge']:
        # For procedural queries, prioritize step-by-step information
        output += "HOW-TO INFORMATION:\n"
        
        # Add procedural memories first if available
        proc_categories = [cat for cat in categorized if cat in ['procedural', 'procedural_knowledge']]
        if proc_categories:
            for cat in proc_categories:
                for subcategory, subcategory_memories in categorized[cat].items():
                    for memory in subcategory_memories:
                        content = memory.get("content", "").strip()
                        similarity = memory.get("similarity", 0)
                        output += f"- [{similarity:.2f}] {content}\n\n"
        
        # Add other categories with headers
        other_categories = [cat for cat in categorized if cat not in ['procedural', 'procedural_knowledge']]
        if other_categories:
            for category in other_categories:
                readable_category = category.replace('_', ' ').capitalize()
                output += f"\nADDITIONAL {readable_category.upper()} INFORMATION:\n"
                
                for subcategory, subcategory_memories in categorized[category].items():
                    for memory in subcategory_memories:
                        content = memory.get("content", "").strip()
                        similarity = memory.get("similarity", 0)
                        output += f"- [{similarity:.2f}] {content}\n\n"
    
    elif main_category in ['conceptual', 'conceptual_knowledge']:
        # For conceptual queries, prioritize theories and principles
        output += "CONCEPTUAL INFORMATION:\n"
        
        # Add conceptual memories first
        concept_categories = [cat for cat in categorized if cat in ['conceptual', 'conceptual_knowledge']]
        if concept_categories:
            for cat in concept_categories:
                for subcategory, subcategory_memories in categorized[cat].items():
                    for memory in subcategory_memories:
                        content = memory.get("content", "").strip()
                        similarity = memory.get("similarity", 0)
                        output += f"- [{similarity:.2f}] {content}\n\n"
        
        # Add other categories that might provide supporting info
        other_categories = [cat for cat in categorized if cat not in ['conceptual', 'conceptual_knowledge']]
        if other_categories:
            for category in other_categories:
                readable_category = category.replace('_', ' ').capitalize()
                output += f"\nSUPPORTING {readable_category.upper()} INFORMATION:\n"
                
                for subcategory, subcategory_memories in categorized[category].items():
                    for memory in subcategory_memories:
                        content = memory.get("content", "").strip()
                        similarity = memory.get("similarity", 0)
                        output += f"- [{similarity:.2f}] {content}\n\n"
    
    # Handle other categories with similar logic...
    elif main_category == 'experiential':
        output += "EXPERIENCE-BASED INFORMATION:\n"
        # Similar formatting for experiential memories
        
    elif main_category == 'tacit':
        output += "INSIGHTS AND OPINIONS:\n"
        # Similar formatting for tacit knowledge
        
    elif main_category == 'explicit':
        output += "DOCUMENTED INFORMATION:\n"
        # Similar formatting for explicit knowledge
        
    elif main_category == 'contextual':
        output += "CONTEXTUAL INFORMATION:\n"
        # Similar formatting for contextual knowledge
    
    else:
        # Default formatting for unknown or other categories
        for category, subcategories in categorized.items():
            readable_category = category.replace('_', ' ').capitalize()
            output += f"\n{readable_category.upper()} INFORMATION:\n"
            
            for subcategory, subcategory_memories in subcategories.items():
                for memory in subcategory_memories:
                    content = memory.get("content", "").strip()
                    similarity = memory.get("similarity", 0)
                    output += f"- [{similarity:.2f}] {content}\n\n"
    
    # Add guidance based on category
    if main_category in ['declarative', 'factual']:
        output += "\nUSE THE FACTUAL INFORMATION ABOVE TO PROVIDE AN ACCURATE ANSWER.\n"
    elif main_category in ['procedural', 'procedural_knowledge']:
        output += "\nUSE THE PROCEDURAL STEPS ABOVE TO PROVIDE CLEAR INSTRUCTIONS.\n"
    elif main_category in ['conceptual', 'conceptual_knowledge']:
        output += "\nUSE THE CONCEPTUAL INFORMATION ABOVE TO EXPLAIN THE UNDERLYING PRINCIPLES.\n"
    else:
        output += "\nUSE THE ABOVE INFORMATION TO HELP ANSWER THE QUERY IF RELEVANT.\n"
    
    return output

def extract_key_statements(text: str) -> List[str]:
    """
    Extract key factual statements from text for memory storage.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of extracted statements
    """
    statements = []
    
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        # Skip short sentences
        if len(sentence) < 15:
            continue
            
        # Skip sentences that are questions
        if sentence.endswith('?'):
            continue
            
        # Look for factual declarative sentences
        if re.match(r'^[A-Z].*?(?:is|are|was|were|has|have)\b', sentence):
            # Check if it contains an actual assertion
            if " is " in sentence or " are " in sentence or " was " in sentence or " have " in sentence:
                if sentence not in statements:
                    statements.append(sentence)
                    
        # Look for procedural statements
        if re.search(r'\b(?:first|then|next|finally|step|process)\b', sentence, re.IGNORECASE):
            if sentence not in statements:
                statements.append(sentence)
                    
        # Look for conceptual explanations
        if re.search(r'\b(?:concept|theory|principle|means|refers to)\b', sentence, re.IGNORECASE):
            if sentence not in statements:
                statements.append(sentence)
                    
        # Special handling for date/time information
        if re.search(r'\b(?:today|current date|the date|time is)\b', sentence.lower()):
            if re.search(r'\b(?:is|was)\b', sentence.lower()):
                if sentence not in statements:
                    statements.append(sentence)

    # Limit to most important statements (longer sentences typically contain more information)
    sorted_statements = sorted(statements, key=len, reverse=True)
    return sorted_statements[:5]