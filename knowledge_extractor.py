"""
Knowledge extraction module for TinyLlama Chat system.
Extracts structured knowledge from unstructured text using pattern matching and NLP techniques.
"""

import re
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from pattern_matching_utils import (
    extract_example_pairs, extract_mapping_category, clean_duplicate_memories,
    is_entity_description, is_mathematical_content, is_mapping_data
)

class KnowledgeExtractor:
    """
    Extracts structured knowledge from unstructured text.
    
    The KnowledgeExtractor identifies and extracts different types of knowledge:
    - Facts: Declarative statements ("X is Y")
    - Definitions: Concept explanations
    - Procedures: Step-by-step instructions
    - Relationships: Connections between entities
    - Mappings: Translations or equivalences
    """
    
    def __init__(
        self, 
        embedding_function=None,
        pattern_confidence_threshold: float = 0.7,
        enable_fractal_validation: bool = True
    ):
        """
        Initialize the knowledge extractor.
        
        Args:
            embedding_function: Function to generate embeddings from text
            pattern_confidence_threshold: Confidence threshold for pattern matches
            enable_fractal_validation: Whether to use fractal embeddings for validation
        """
        self.embedding_function = embedding_function
        self.pattern_confidence_threshold = pattern_confidence_threshold
        self.enable_fractal_validation = enable_fractal_validation
        
        # Initialize knowledge type patterns
        self._initialize_patterns()
        
        # Track statistics
        self.extraction_stats = {
            "total_processed": 0,
            "facts_extracted": 0,
            "definitions_extracted": 0,
            "procedures_extracted": 0,
            "relationships_extracted": 0,
            "mappings_extracted": 0,
            "low_confidence_skipped": 0
        }
    
    def _initialize_patterns(self):
        """Initialize patterns for different knowledge types."""
        # Fact patterns - declarative statements about entities
        self.fact_patterns = [
            # "X is Y" pattern
            (r"([A-Z][a-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})\s+is\s+(?:an?|the)\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,10})", 0.8),
            # "X was Y" pattern (historical facts)
            (r"([A-Z][a-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})\s+was\s+(?:an?|the)\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,10})", 0.8),
            # "X has Y" pattern (properties)
            (r"([A-Z][a-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})\s+has\s+(?:an?|the)?\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,10})", 0.75),
            # Numerical facts
            (r"([A-Z][a-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})\s+(?:is|was|has)\s+([0-9,.]+(?:\s+[a-z]+){0,3})", 0.85),
            # Location facts
            (r"([A-Z][a-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})\s+is\s+(?:located|found|situated)\s+in\s+([A-Z][a-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})", 0.85),
            # Date facts
            (r"([A-Z][a-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})\s+(?:occurred|happened|took place|began|started|ended)\s+(?:on|in|during)\s+([A-Za-z0-9]+(?:\s+[0-9]+,?\s+[0-9]+)?)", 0.8)
        ]
        
        # Definition patterns - explaining concepts
        self.definition_patterns = [
            # "X is defined as Y" pattern
            (r"([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})\s+is\s+defined\s+as\s+([^.;:!?]+)", 0.9),
            # "X refers to Y" pattern
            (r"([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})\s+refers\s+to\s+([^.;:!?]+)", 0.85),
            # "X is a term that Y" pattern
            (r"([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,3})\s+is\s+a\s+(?:term|concept|word)\s+that\s+([^.;:!?]+)", 0.85),
            # "X is a type of Y" pattern
            (r"([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})\s+is\s+a\s+type\s+of\s+([^.;:!?]+)", 0.8),
            # Technical definitions
            (r"(?:In|Within)\s+([a-zA-Z0-9]+(?:\s+[a-zA-Z0-9]+){0,3}),\s+([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})\s+(?:is|refers to|means|represents)\s+([^.;:!?]+)", 0.75)
        ]
        
        # Procedural patterns - step-by-step instructions
        self.procedure_patterns = [
            # Numbered steps
            (r"(?:Step|Phase|Part)\s+([0-9]+)(?:\s*:)?\s+([^.;:!?]+)", 0.9),
            # Bulleted steps
            (r"(?:•|-|\*|\+)\s+([^.;:!?]+)", 0.8),
            # "First, second, etc." pattern
            (r"(?:First|Second|Third|Fourth|Fifth|Finally|Lastly)(?:,|:)\s+([^.;:!?]+)", 0.85),
            # "To do X, do Y" pattern
            (r"To\s+([a-z]+(?:\s+[a-z]+){0,5}),\s+(?:you\s+(?:need|should|must|can)\s+)?([^.;:!?]+)", 0.75),
            # "X can be done by Y" pattern
            (r"([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})\s+can\s+be\s+(?:done|performed|achieved|accomplished)\s+by\s+([^.;:!?]+)", 0.7)
        ]
        
        # Relationship patterns - connections between entities
        self.relationship_patterns = [
            # "X is related to Y" pattern
            (r"([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})\s+is\s+(?:related|connected|linked)\s+to\s+([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})", 0.8),
            # "X causes Y" pattern
            (r"([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})\s+(?:causes|leads to|results in|produces)\s+([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})", 0.85),
            # "X depends on Y" pattern
            (r"([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})\s+(?:depends|relies|is dependent|is based)\s+on\s+([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})", 0.8),
            # "X is part of Y" pattern
            (r"([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})\s+is\s+(?:part of|a component of|an element of|a subset of)\s+([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})", 0.85),
            # "X and Y are" pattern
            (r"([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})\s+and\s+([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,5})\s+are\s+([^.;:!?]+)", 0.7)
        ]
        
        # Mapping patterns - translations or equivalences
        self.mapping_patterns = [
            # "X in Y is Z" pattern (common for translations)
            (r"\"?([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,3})\"?\s+in\s+([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,2})\s+is\s+\"?([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,3})\"?", 0.9),
            # "X is called Y in Z" pattern
            (r"\"?([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,3})\"?\s+is\s+(?:called|known as|termed)\s+\"?([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,3})\"?\s+in\s+([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,2})", 0.85),
            # "X → Y" or "X -> Y" pattern
            (r"\"?([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,3})\"?\s*(?:→|->)\s*\"?([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,3})\"?", 0.95),
            # "X corresponds to Y" pattern
            (r"\"?([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,3})\"?\s+corresponds\s+to\s+\"?([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,3})\"?", 0.8),
            # "X equals Y" pattern
            (r"\"?([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,3})\"?\s+(?:equals|=|is equivalent to)\s+\"?([A-Za-z0-9]+(?:\s+[a-zA-Z0-9]+){0,3})\"?", 0.85)
        ]
        
    def extract_knowledge(self, text: str, source: str = None, domain: str = None) -> List[Dict[str, Any]]:
        """
        Extract all types of knowledge from text.
        
        Args:
            text: Input text to process
            source: Optional source identifier
            domain: Optional domain identifier
            
        Returns:
            List of extracted knowledge items
        """
        self.extraction_stats["total_processed"] += 1

        if len(text.split()) <= 3:
            return self.extract_knowledge_for_short_query(text, domain)
        
        # Clean the text (remove excessive whitespace, etc.)
        cleaned_text = self._clean_text(text)
        
        # Split into sentences for better processing
        sentences = self._split_into_sentences(cleaned_text)
        
        # Extract each type of knowledge
        facts = self._extract_facts(sentences, source, domain)
        definitions = self._extract_definitions(sentences, source, domain)
        procedures = self._extract_procedures(cleaned_text, source, domain)
        relationships = self._extract_relationships(sentences, source, domain)
        mappings = self._extract_mappings(cleaned_text, source, domain)
        
        # Combine all extracted knowledge
        all_knowledge = facts + definitions + procedures + relationships + mappings
        
        # Update statistics
        self.extraction_stats["facts_extracted"] += len(facts)
        self.extraction_stats["definitions_extracted"] += len(definitions)
        self.extraction_stats["procedures_extracted"] += len(procedures)
        self.extraction_stats["relationships_extracted"] += len(relationships)
        self.extraction_stats["mappings_extracted"] += len(mappings)
        
        return all_knowledge
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing excessive whitespace, etc."""
        # Replace multiple spaces with a single space
        cleaned = re.sub(r'\s+', ' ', text)
        # Trim whitespace
        cleaned = cleaned.strip()
        # Fix common spacing issues around punctuation
        cleaned = re.sub(r'\s+([.,;:!?])', r'\1', cleaned)
        return cleaned
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for processing."""
        # Simple sentence splitting (can be enhanced with NLP libraries)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_facts(self, sentences: List[str], source: str, domain: str) -> List[Dict[str, Any]]:
        """Extract factual knowledge from sentences."""
        facts = []
        
        for sentence in sentences:
            for pattern, confidence in self.fact_patterns:
                matches = re.findall(pattern, sentence)
                for match in matches:
                    # Skip if confidence is below threshold
                    if confidence < self.pattern_confidence_threshold:
                        self.extraction_stats["low_confidence_skipped"] += 1
                        continue
                    
                    # Check if match is a tuple or single string
                    if isinstance(match, tuple):
                        subject = match[0].strip()
                        predicate = "is"  # Default predicate
                        obj = match[1].strip()
                    else:
                        # Skip if we can't parse correctly
                        continue
                    
                    # Create fact object
                    fact = {
                        "type": "fact",
                        "content": {
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj,
                            "context": sentence
                        },
                        "metadata": {
                            "source": source,
                            "domain": domain,
                            "confidence": confidence,
                            "extraction_pattern": pattern,
                            "timestamp": self._get_timestamp()
                        }
                    }
                    
                    facts.append(fact)
        
        return facts
    
    def _extract_definitions(self, sentences: List[str], source: str, domain: str) -> List[Dict[str, Any]]:
        """Extract definitions from sentences."""
        definitions = []
        
        for sentence in sentences:
            for pattern, confidence in self.definition_patterns:
                matches = re.findall(pattern, sentence)
                for match in matches:
                    # Skip if confidence is below threshold
                    if confidence < self.pattern_confidence_threshold:
                        self.extraction_stats["low_confidence_skipped"] += 1
                        continue
                    
                    # Check if match is a tuple or single string
                    if isinstance(match, tuple):
                        term = match[0].strip()
                        definition = match[1].strip()
                    else:
                        # Skip if we can't parse correctly
                        continue
                    
                    # Create definition object
                    definition_obj = {
                        "type": "definition",
                        "content": {
                            "term": term,
                            "definition": definition,
                            "context": sentence
                        },
                        "metadata": {
                            "source": source,
                            "domain": domain,
                            "confidence": confidence,
                            "extraction_pattern": pattern,
                            "timestamp": self._get_timestamp()
                        }
                    }
                    
                    definitions.append(definition_obj)
        
        return definitions
    
    def _extract_procedures(self, text: str, source: str, domain: str) -> List[Dict[str, Any]]:
        """Extract procedural knowledge from text."""
        procedures = []
        
        # First check for numbered lists or bullet points
        step_patterns = [
            r'(?:Step|Phase|Stage)\s+(\d+)(?:\s*:)?\s+([^\n]+)',  # Step 1: Do something
            r'(\d+)[.)\]]\s+([^\n]+)',                           # 1. Do something or 1) Do something
            r'(?:•|-|\*)\s+([^\n]+)'                            # • Do something or - Do something
        ]
        
        steps = []
        
        # Try to extract procedural patterns
        for pattern in step_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # If we found structured steps
                try:
                    if len(matches[0]) == 2:  # Numbered step
                        steps = [(int(m[0]), m[1].strip()) for m in matches]
                        steps.sort(key=lambda x: x[0])  # Sort by step number
                        steps = [step[1] for step in steps]  # Keep only the instructions
                    else:  # Bulleted step
                        steps = [m.strip() for m in matches]
                except (ValueError, IndexError):
                    steps = []
                
                if steps:
                    break  # Stop if we found steps
        
        # If we found steps, create a procedure object
        if steps:
            # Try to find a title for the procedure
            title_match = re.search(r'(?:How to|Steps to|Procedure for|Instructions for)\s+([^\n.]+)', text)
            title = title_match.group(1).strip() if title_match else "Procedure"
            
            procedure = {
                "type": "procedure",
                "content": {
                    "title": title,
                    "steps": steps,
                    "context": text[:100]  # First 100 chars for context
                },
                "metadata": {
                    "source": source,
                    "domain": domain,
                    "confidence": 0.85,  # Higher confidence for structured steps
                    "step_count": len(steps),
                    "timestamp": self._get_timestamp()
                }
            }
            
            procedures.append(procedure)
            
        # Also try to extract procedural sentences
        sentences = self._split_into_sentences(text)
        for sentence in sentences:
            for pattern, confidence in self.procedure_patterns:
                matches = re.findall(pattern, sentence)
                if matches and confidence >= self.pattern_confidence_threshold:
                    # Found a procedural sentence
                    procedure = {
                        "type": "procedure",
                        "content": {
                            "title": "Procedure",
                            "steps": [m.strip() if isinstance(m, str) else m[0].strip() for m in matches],
                            "context": sentence
                        },
                        "metadata": {
                            "source": source,
                            "domain": domain,
                            "confidence": confidence,
                            "extraction_pattern": pattern,
                            "timestamp": self._get_timestamp()
                        }
                    }
                    
                    procedures.append(procedure)
        
        return procedures
    
    def _extract_relationships(self, sentences: List[str], source: str, domain: str) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        relationships = []
        
        for sentence in sentences:
            for pattern, confidence in self.relationship_patterns:
                matches = re.findall(pattern, sentence)
                for match in matches:
                    # Skip if confidence is below threshold
                    if confidence < self.pattern_confidence_threshold:
                        self.extraction_stats["low_confidence_skipped"] += 1
                        continue
                    
                    # Check if match is a tuple or single string
                    if isinstance(match, tuple):
                        if len(match) >= 2:
                            entity1 = match[0].strip()
                            entity2 = match[1].strip()
                            
                            # Try to determine relationship type
                            rel_type = "general"
                            if "causes" in pattern or "leads to" in pattern:
                                rel_type = "causal"
                            elif "part of" in pattern:
                                rel_type = "part-whole"
                            elif "depends" in pattern:
                                rel_type = "dependency"
                            
                            # Create relationship object
                            relationship = {
                                "type": "relationship",
                                "content": {
                                    "entity1": entity1,
                                    "entity2": entity2,
                                    "relationship_type": rel_type,
                                    "context": sentence
                                },
                                "metadata": {
                                    "source": source,
                                    "domain": domain,
                                    "confidence": confidence,
                                    "extraction_pattern": pattern,
                                    "timestamp": self._get_timestamp()
                                }
                            }
                            
                            relationships.append(relationship)
        
        return relationships
    
    def _extract_mappings(self, text: str, source: str, domain: str) -> List[Dict[str, Any]]:
        """Extract mappings (translations, equivalences)."""
        mappings = []
        
        # First try to extract mapping table if it exists
        if is_mapping_data(source if source else "", text):
            # Extract pairs using the existing function
            pairs = extract_example_pairs(text)
            if pairs:
                # Create a mapping object
                mapping = {
                    "type": "mapping",
                    "content": {
                        "title": domain if domain else "Mapping",
                        "pairs": [{"from": pair[0], "to": pair[1]} for pair in pairs],
                        "context": text[:100]  # First 100 chars for context
                    },
                    "metadata": {
                        "source": source,
                        "domain": domain,
                        "confidence": 0.9,  # High confidence for table-based mappings
                        "mapping_count": len(pairs),
                        "timestamp": self._get_timestamp()
                    }
                }
                
                mappings.append(mapping)
        
        # Also extract individual mapping sentences
        sentences = self._split_into_sentences(text)
        for sentence in sentences:
            for pattern, confidence in self.mapping_patterns:
                matches = re.findall(pattern, sentence)
                for match in matches:
                    # Skip if confidence is below threshold
                    if confidence < self.pattern_confidence_threshold:
                        self.extraction_stats["low_confidence_skipped"] += 1
                        continue
                    
                    # Check if match is a tuple or single string
                    if isinstance(match, tuple):
                        if len(match) >= 2:
                            from_value = match[0].strip().strip('"\'')
                            to_value = match[1].strip().strip('"\'')
                            
                            # Create mapping object
                            mapping = {
                                "type": "mapping",
                                "content": {
                                    "title": "Translation" if "in" in pattern else "Equivalence",
                                    "pairs": [{"from": from_value, "to": to_value}],
                                    "context": sentence
                                },
                                "metadata": {
                                    "source": source,
                                    "domain": domain,
                                    "confidence": confidence,
                                    "extraction_pattern": pattern,
                                    "timestamp": self._get_timestamp()
                                }
                            }
                            
                            # Add language info if available for translations
                            if len(match) >= 3 and "in" in pattern:
                                mapping["content"]["language"] = match[2].strip()
                            
                            mappings.append(mapping)
        
        return mappings
    
    def batch_extract(self, texts: List[str], sources: List[str] = None, domain: str = None) -> List[Dict[str, Any]]:
        """
        Extract knowledge from multiple texts.
        
        Args:
            texts: List of texts to process
            sources: Optional list of source identifiers
            domain: Optional domain identifier
            
        Returns:
            List of extracted knowledge items
        """
        all_knowledge = []
        
        for i, text in enumerate(texts):
            # Get source if available
            source = sources[i] if sources and i < len(sources) else None
            
            # Extract knowledge from this text
            knowledge = self.extract_knowledge(text, source, domain)
            all_knowledge.extend(knowledge)
        
        return all_knowledge
    
    def validate_with_embeddings(self, knowledge_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate knowledge using embeddings (requires embedding_function).
        
        Args:
            knowledge_items: List of knowledge items to validate
            
        Returns:
            List of validated knowledge items with updated confidence
        """
        if not self.embedding_function or not self.enable_fractal_validation:
            # Return original items if we can't validate
            return knowledge_items
        
        # Get embeddings for all knowledge items
        knowledge_texts = []
        for item in knowledge_items:
            # Convert knowledge item to text representation
            if item["type"] == "fact":
                text = f"{item['content']['subject']} {item['content']['predicate']} {item['content']['object']}"
            elif item["type"] == "definition":
                text = f"{item['content']['term']} is defined as {item['content']['definition']}"
            elif item["type"] == "procedure":
                text = f"{item['content']['title']}: {' '.join(item['content']['steps'])}"
            elif item["type"] == "relationship":
                text = f"{item['content']['entity1']} relates to {item['content']['entity2']}"
            elif item["type"] == "mapping":
                pairs_text = ", ".join([f"{pair['from']} -> {pair['to']}" for pair in item['content']['pairs']])
                text = f"Mapping: {pairs_text}"
            else:
                text = str(item)
            
            knowledge_texts.append(text)
        
        # Generate embeddings
        embeddings = [self.embedding_function(text) for text in knowledge_texts]
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    # Normalize embeddings
                    emb1 = embeddings[i] / np.linalg.norm(embeddings[i])
                    emb2 = embeddings[j] / np.linalg.norm(embeddings[j])
                    # Calculate cosine similarity
                    similarity_matrix[i][j] = np.dot(emb1, emb2)
        
        # Update confidence based on validations
        validated_items = []
        for i, item in enumerate(knowledge_items):
            # Find similar items (support)
            similar_indices = [j for j in range(len(embeddings)) 
                              if i != j and similarity_matrix[i][j] > 0.8]
            
            # Adjust confidence based on supporting evidence
            base_confidence = item["metadata"]["confidence"]
            if similar_indices:
                # Boost confidence if we have supporting items
                support_boost = min(0.15, 0.05 * len(similar_indices))
                new_confidence = min(0.99, base_confidence + support_boost)
                
                # Add validation metadata
                item["metadata"]["validation"] = {
                    "method": "embedding_similarity",
                    "original_confidence": base_confidence,
                    "supporting_items": len(similar_indices),
                    "confidence_boost": support_boost
                }
            else:
                # Slightly reduce confidence for unsupported items
                new_confidence = max(0.1, base_confidence - 0.05)
                
                # Add validation metadata
                item["metadata"]["validation"] = {
                    "method": "embedding_similarity",
                    "original_confidence": base_confidence,
                    "supporting_items": 0,
                    "confidence_reduction": 0.05
                }
            
            # Update confidence
            item["metadata"]["confidence"] = new_confidence
            validated_items.append(item)
        
        return validated_items
    
    def get_extraction_stats(self) -> Dict[str, int]:
        """
        Get statistics about extraction.
        
        Returns:
            Dictionary of extraction statistics
        """
        return self.extraction_stats
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def save_knowledge_to_file(self, knowledge_items: List[Dict[str, Any]], filepath: str) -> bool:
        """
        Save extracted knowledge to a JSON file.
        
        Args:
            knowledge_items: List of knowledge items to save
            filepath: Path to save file
            
        Returns:
            Success boolean
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(knowledge_items, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving knowledge to file: {e}")
            return False
    
    def load_knowledge_from_file(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load knowledge items from a JSON file.
        
        Args:
            filepath: Path to load file from
            
        Returns:
            List of knowledge items
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                knowledge_items = json.load(f)
            return knowledge_items
        except Exception as e:
            print(f"Error loading knowledge from file: {e}")
            return []

    def extract_knowledge_for_short_query(self, text: str, domain: str = None) -> List[Dict[str, Any]]:
        """Special handling for short queries that don't match standard patterns."""

        # For very short input
        if len(text.split()) <= 3:
            # Try to look up the term directly
            try:
                # Create a broader search query
                search_query = f"what is {text}" if len(text.split()) == 1 else text

                # Try to find information about this term on the web
                if hasattr(self, 'embedding_function') and self.embedding_function:
                    try:
                        from web_knowledge_enhancer import WebKnowledgeEnhancer
                        web_enhancer = WebKnowledgeEnhancer(
                            embedding_function=self.embedding_function,
                            search_engine="duckduckgo"
                        )

                        # Use web search to get information
                        search_results = web_enhancer.search_web(search_query, num_results=3)

                        if search_results:
                            # Extract text from search results
                            texts = [result.get('snippet', '') for result in search_results]
                            source = "web_search:" + search_query
                            # Use the standard extraction on this richer text
                            return self.extract_knowledge("\n".join(texts), source, domain)
                    except ImportError:
                        pass
            except Exception as e:
                print(f"Error in extract_knowledge_for_short_query: {e}")

        # If we couldn't get more information, just do standard extraction
        return self.extract_knowledge(text, source="direct_query", domain=domain)