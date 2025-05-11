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
            category, confidence, subcategory, subcategory_confidence = question_classifier.classify(content)
            result['main_category'] = category
            result['confidence'] = confidence
            result['subcategory'] = subcategory
            result['subcategory_confidence'] = subcategory_confidence

            # # Determine subcategory
            # if category in question_classifier.subcategories:
            #     subcategory, subcategory_confidence = question_classifier.identify_subcategory(content, category)
            #     if subcategory:
            #         result['subcategory'] = subcategory
            #         result['subcategory_confidence'] = subcategory_confidence
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

def generate_memory_metadata(content: str, classification: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate metadata for memory storage based on content classification.
    
    Args:
        content: Text content
        classification: Classification results or None
        
    Returns:
        Dictionary with metadata
    """
    # Default metadata
    metadata = {
        "timestamp": datetime.now().timestamp(),
        "type": "unknown"
    }
    
    # Apply classification if available
    if classification:
        main_category = classification.get("main_category", "unknown")
        subcategory = classification.get("subcategory", "unknown")
        metadata["main_category"] = main_category
        metadata["subcategory"] = subcategory
        
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
        
        metadata["classification_confidence"] = classification.get("confidence", 0.5)
        metadata["subcategory_confidence"] = classification.get("subcategory_confidence", 0.5)
    
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
    metadata["timestamp"] = datetime.now().timestamp()

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

def save_to_memory(memory_manager, content: str, classification: Optional[Dict[str, Any]] = None, related_content: Optional[str] = None) -> Dict[str, Any]:
    """
    Save content to memory with appropriate metadata.
    
    Args:
        memory_manager: MemoryManager instance
        content: Content to save
        classification: Classification results or None
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
        metadata = generate_memory_metadata(formatted_content, classification)
        
        # print("CLASSIFICATION", classification)
        # Add to memory
        memory_id = memory_manager.add(
            content=formatted_content,
            metadata=metadata
        )
        
        # if memory_id:
        #     results["saved"] = True
        #     results["memory_id"] = memory_id
        #     results["metadata"] = metadata
            
        #     # Save related content if provided
        #     if related_content and related_content.strip():
        #         # Generate classification for related content
        #         if classification and "question_classifier" in classification:
        #             related_classification = classify_content(
        #                 related_content,
        #                 classification["question_classifier"]
        #             )
        #         else:
        #             related_classification = None
                
        #         # Create related metadata
        #         related_metadata = generate_memory_metadata(
        #             related_content,
        #             related_classification
        #         )
        #         related_metadata["related_to"] = memory_id
                
        #         # Add to memory
        #         related_id = memory_manager.add(
        #             content=related_content,
        #             metadata=related_metadata
        #         )
                
        #         if related_id:
        #             results["additional_ids"].append(related_id)
    
    except Exception as e:
        results["error"] = str(e)
    
    return results

def format_memories_by_category(memories: List[Dict[str, Any]], main_category: str, subcategory) -> str:
    """
    Format memories based on the main knowledge category with improved relevance sorting.

    Args:
        memories: List of retrieved memories
        main_category: Main knowledge category
        subcategory: Subcategory for filtering

    Returns:
        Formatted memory text
    """
    if not memories:
        return ""

    output = "\nUse the following retrieved memories\n\n"

    output += "Consider a memory relevant if: \n\
        1. It's in the list under the HIGHLY RELEVANT INFORMATION \n \
        2. The ones with higher score are to be used first \n \
        3. Other memories in ADDITIONAL [category] INFORMATION are here to support the memories selected in 1. and 2.\n"

    # Sort memories by relevance first - this is critical for proper presentation
    memories = sorted(memories, key=lambda x: x.get('similarity', 0), reverse=True)

    # Find the highest similarity score for reference
    max_similarity = max([m.get('similarity', 0) for m in memories]) if memories else 0

    # Define a relevance threshold relative to max similarity
    # Documents with similarity > 75% of max are considered highly relevant
    relevance_threshold = max_similarity # * 0.75

    # Group memories by relevance first, then by category
    highly_relevant = []
    relevant_by_category = {}

    # Identify search terms from query content (if available)
    search_terms = []
    for memory in memories[:1]:  # Look at highest match
        if "who" in memory.get("content", "").lower() and "what" in memory.get("content", "").lower():
            # Extract search terms from question
            content = memory.get("content", "").lower()
            search_parts = re.findall(r'(?:who|what|when|where|why|how).*?([\w\s]+)(?:\?|$)', content)
            if search_parts:
                search_terms = [term.strip() for term in search_parts[0].split()]

    # Process each memory
    for memory in memories:
        metadata = memory.get("metadata", {})
        category = metadata.get("main_category", "unknown")
        content = memory.get("content", "").strip()
        similarity = memory.get("similarity", 0)

        # Check for direct keyword matches
        content_relevance = 0
        for term in search_terms:
            if term and len(term) > 3 and term.lower() in content.lower():
                content_relevance += 0.2  # Boost for each search term match

        # Consider a memory highly relevant if:
        # 1. It's above the relative similarity threshold OR
        # 2. It has direct keyword matches that push it above threshold
        effective_similarity = similarity + content_relevance

        if effective_similarity >= relevance_threshold:
            # This is a highly relevant memory regardless of category
            highly_relevant.append(memory)
        else:
            # Group by category for secondary relevance
            if category not in relevant_by_category:
                relevant_by_category[category] = []
            relevant_by_category[category].append(memory)

    # Output highly relevant memories first, regardless of category
    if highly_relevant:
        output += "\nHIGHLY RELEVANT INFORMATION:\n"
        for memory in highly_relevant:
            content = memory.get("content", "").strip()
            item_id = memory.get("id", "")[-6:] if memory.get("id") else ""
            similarity = memory.get("similarity", 0)
            output += f"- [{similarity:.2f}] {content}\n\n"

    # Then output remaining memories by category
    for category_name, category_memories in relevant_by_category.items():
        if not category_memories:
            continue

        # Skip if we've already shown all memories in this category as highly relevant
        if all(memory in highly_relevant for memory in category_memories):
            continue

        # Format category name for display
        readable_category = category_name.replace("_", " ").title()
        output += f"\nADDITIONAL {readable_category.upper()} INFORMATION:\n"

        # Show memories not already shown as highly relevant
        for memory in category_memories:
            if memory in highly_relevant:
                continue

            content = memory.get("content", "").strip()
            item_id = memory.get("id", "")[-6:] if memory.get("id") else ""
            similarity = memory.get("similarity", 0)
            output += f"- [{similarity:.2f}] {content}\n\n"

    # Add guidance based on query category
    output += "\nUSE THE INFORMATION ABOVE TO ANSWER THE QUERY, PRIORITIZING THE HIGHLY RELEVANT INFORMATION.\n"

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