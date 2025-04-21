"""
Knowledge validation module for TinyLlama Chat system.
Validates knowledge items using fractal embeddings and confidence metrics.
"""

import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from datetime import datetime

class KnowledgeValidator:
    """
    Validation system for ensuring consistency and quality of knowledge.
    
    The KnowledgeValidator uses fractal embeddings and confidence metrics
    to validate knowledge items, detect contradictions, and identify outliers.
    """
    
    def __init__(
        self,
        embedding_function: Optional[Callable] = None,
        validation_threshold: float = 0.7,
        contradiction_threshold: float = 0.7,
        outlier_threshold: float = 0.4,
        enable_fractal_validation: bool = True,
        vector_store=None
    ):
        """
        Initialize the knowledge validator.
        
        Args:
            embedding_function: Function to generate embeddings
            validation_threshold: Minimum similarity for validation
            contradiction_threshold: Similarity threshold for contradictions
            outlier_threshold: Maximum similarity for outliers
            enable_fractal_validation: Whether to use fractal validation
            vector_store: Optional vector store for knowledge lookup
        """
        self.embedding_function = embedding_function
        self.validation_threshold = validation_threshold
        self.contradiction_threshold = contradiction_threshold
        self.outlier_threshold = outlier_threshold
        self.enable_fractal_validation = enable_fractal_validation
        self.vector_store = vector_store
        
        # Track validation statistics
        self.validation_stats = {
            "total_validations": 0,
            "validated_items": 0,
            "contradictions_found": 0,
            "outliers_found": 0,
            "low_confidence_items": 0
        }
    
    def validate_item(self, knowledge_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single knowledge item.
        
        Args:
            knowledge_item: Knowledge item to validate
            
        Returns:
            Validated knowledge item with validation metadata
        """
        self.validation_stats["total_validations"] += 1
        
        # Initialize validation metadata
        validation_result = {
            "validated": False,
            "confidence": knowledge_item.get("metadata", {}).get("confidence", 0.5),
            "contradictions": [],
            "supporting_evidence": [],
            "is_outlier": False,
            "validation_method": "none",
            "timestamp": datetime.now().isoformat()
        }
        
        # Skip validation if no embedding function or vector store available
        if not self.embedding_function or not self.vector_store:
            knowledge_item["validation"] = validation_result
            return knowledge_item
        
        # Convert knowledge item to text
        item_text = self._knowledge_item_to_text(knowledge_item)
        
        # Generate embedding
        try:
            item_embedding = self.embedding_function(item_text)
        except Exception as e:
            print(f"Error generating embedding for validation: {e}")
            knowledge_item["validation"] = validation_result
            return knowledge_item
        
        # Search for related items in vector store
        if self.enable_fractal_validation and hasattr(self.vector_store, 'enhanced_fractal_search'):
            # Use enhanced fractal search
            search_results = self.vector_store.enhanced_fractal_search(
                query_embedding=item_embedding,
                top_k=10,
                min_similarity=0.3,  # Lower threshold to catch both supporting and contradicting
                apply_sharpening=True,
                multi_level_search=True
            )
        else:
            # Fall back to standard search
            search_results = self.vector_store.search(
                query_embedding=item_embedding,
                top_k=10,
                min_similarity=0.3
            )
        
        # Analyze search results for validation
        supporting_evidence = []
        contradictions = []
        similar_count = 0
        
        for result in search_results:
            similarity = result.get('similarity', 0)
            metadata = result.get('metadata', {})
            result_item = metadata.get('knowledge_item', {})
            
            # Skip if this is the same item (exact match)
            if self._is_same_item(knowledge_item, result_item):
                continue
            
            # Check if item is potentially contradictory
            if similarity >= self.contradiction_threshold and self._is_potential_contradiction(knowledge_item, result_item):
                contradictions.append({
                    "item": result_item,
                    "similarity": similarity,
                    "text": result.get('text', '')
                })
            
            # Check if item provides supporting evidence
            if similarity >= self.validation_threshold:
                supporting_evidence.append({
                    "item": result_item,
                    "similarity": similarity,
                    "text": result.get('text', '')
                })
                similar_count += 1
        
        # Determine if item is an outlier (no or very few similar items)
        is_outlier = similar_count <= 1
        
        # Calculate adjusted confidence based on supporting evidence and contradictions
        base_confidence = knowledge_item.get("metadata", {}).get("confidence", 0.5)
        adjusted_confidence = base_confidence
        
        # Increase confidence with supporting evidence
        if supporting_evidence:
            support_boost = min(0.3, 0.05 * len(supporting_evidence))
            adjusted_confidence += support_boost
        
        # Decrease confidence with contradictions
        if contradictions:
            contradiction_penalty = min(0.5, 0.1 * len(contradictions))
            adjusted_confidence -= contradiction_penalty
        
        # Decrease confidence for outliers
        if is_outlier:
            adjusted_confidence -= 0.2
        
        # Keep confidence in valid range
        adjusted_confidence = max(0.01, min(0.99, adjusted_confidence))
        
        # Update validation result
        validation_result.update({
            "validated": True,
            "confidence": adjusted_confidence,
            "original_confidence": base_confidence,
            "contradictions": [c["text"] for c in contradictions],
            "contradiction_count": len(contradictions),
            "supporting_evidence": [s["text"] for s in supporting_evidence],
            "supporting_count": len(supporting_evidence),
            "is_outlier": is_outlier,
            "validation_method": "fractal_embedding" if self.enable_fractal_validation else "embedding"
        })
        
        # Update validation statistics
        self.validation_stats["validated_items"] += 1
        if contradictions:
            self.validation_stats["contradictions_found"] += 1
        if is_outlier:
            self.validation_stats["outliers_found"] += 1
        if adjusted_confidence < 0.4:
            self.validation_stats["low_confidence_items"] += 1
        
        # Add validation to knowledge item
        knowledge_item["validation"] = validation_result
        
        return knowledge_item
    
    def validate_items(self, knowledge_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate multiple knowledge items.
        
        Args:
            knowledge_items: List of knowledge items to validate
            
        Returns:
            List of validated knowledge items
        """
        validated_items = []
        
        for item in knowledge_items:
            validated_item = self.validate_item(item)
            validated_items.append(validated_item)
        
        return validated_items
    
    def filter_validated_items(self, knowledge_items: List[Dict[str, Any]], min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Filter knowledge items based on validation results.
        
        Args:
            knowledge_items: List of knowledge items to filter
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of knowledge items
        """
        filtered_items = []
        
        for item in knowledge_items:
            # Get validation result
            validation = item.get("validation", {})
            confidence = validation.get("confidence", item.get("metadata", {}).get("confidence", 0))
            
            # Include if confidence is above threshold and not contradicted
            if confidence >= min_confidence and validation.get("contradiction_count", 0) == 0:
                filtered_items.append(item)
        
        return filtered_items
    
    def get_validation_stats(self) -> Dict[str, int]:
        """
        Get statistics about validation.
        
        Returns:
            Dictionary of validation statistics
        """
        return self.validation_stats
    
    def _knowledge_item_to_text(self, item: Dict[str, Any]) -> str:
        """
        Convert a knowledge item to text for embedding.
        
        Args:
            item: Knowledge item
            
        Returns:
            Text representation
        """
        item_type = item.get("type", "unknown")
        content = item.get("content", {})
        
        if item_type == "fact":
            return f"{content.get('subject', '')} {content.get('predicate', 'is')} {content.get('object', '')}. {content.get('context', '')}"
        
        elif item_type == "definition":
            return f"{content.get('term', '')} is defined as {content.get('definition', '')}. {content.get('context', '')}"
        
        elif item_type == "procedure":
            steps = content.get('steps', [])
            steps_text = ". ".join(steps)
            return f"{content.get('title', 'Procedure')}: {steps_text}. {content.get('context', '')}"
        
        elif item_type == "relationship":
            return f"{content.get('entity1', '')} relates to {content.get('entity2', '')} as {content.get('relationship_type', 'related')}. {content.get('context', '')}"
        
        elif item_type == "mapping":
            pairs = content.get('pairs', [])
            pairs_text = ", ".join([f"{pair.get('from', '')} maps to {pair.get('to', '')}" for pair in pairs])
            return f"Mapping {content.get('title', '')}: {pairs_text}. {content.get('context', '')}"
        
        else:
            # Fallback for unknown types
            return json.dumps(item)
    
    def _is_same_item(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> bool:
        """
        Check if two knowledge items are the same.
        
        Args:
            item1: First knowledge item
            item2: Second knowledge item
            
        Returns:
            Boolean indicating if items are the same
        """
        # Check if one item is empty or None
        if not item1 or not item2:
            return False
        
        # Check if items have the same type
        if item1.get("type") != item2.get("type"):
            return False
        
        # Get content
        content1 = item1.get("content", {})
        content2 = item2.get("content", {})
        
        # Check based on type
        item_type = item1.get("type")
        
        if item_type == "fact":
            return (
                content1.get("subject", "").lower() == content2.get("subject", "").lower() and
                content1.get("object", "").lower() == content2.get("object", "").lower()
            )
        
        elif item_type == "definition":
            return content1.get("term", "").lower() == content2.get("term", "").lower()
        
        elif item_type == "procedure":
            return content1.get("title", "").lower() == content2.get("title", "").lower()
        
        elif item_type == "relationship":
            return (
                (content1.get("entity1", "").lower() == content2.get("entity1", "").lower() and
                 content1.get("entity2", "").lower() == content2.get("entity2", "").lower()) or
                (content1.get("entity1", "").lower() == content2.get("entity2", "").lower() and
                 content1.get("entity2", "").lower() == content2.get("entity1", "").lower())
            )
        
        elif item_type == "mapping":
            # Compare based on title for mappings
            return content1.get("title", "").lower() == content2.get("title", "").lower()
        
        return False
    
    def _is_potential_contradiction(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> bool:
        """
        Check if two knowledge items potentially contradict each other.
        
        Args:
            item1: First knowledge item
            item2: Second knowledge item
            
        Returns:
            Boolean indicating potential contradiction
        """
        # Check if one item is empty or None
        if not item1 or not item2:
            return False
        
        # Only facts and definitions can contradict each other
        item1_type = item1.get("type")
        item2_type = item2.get("type")
        
        if item1_type != item2_type or (item1_type != "fact" and item1_type != "definition"):
            return False
        
        # Get content
        content1 = item1.get("content", {})
        content2 = item2.get("content", {})


        # Add web source verification
        # If both items are from web sources, check which one is more recent
        source1 = item1.get("metadata", {}).get("source", "")
        source2 = item2.get("metadata", {}).get("source", "")

        if "web" in source1.lower() and "web" in source2.lower():
            # Check timestamps
            timestamp1 = item1.get("metadata", {}).get("timestamp", "")
            timestamp2 = item2.get("metadata", {}).get("timestamp", "")

            if timestamp1 and timestamp2:
                # If the second item is newer, consider it corrective
                # This helps handle the case where new information contradicts old
                if timestamp2 > timestamp1:
                    return False  # Not a contradiction, but a correction
        
        if item1_type == "fact":
            # Facts contradict if they have the same subject but different objects
            return (
                content1.get("subject", "").lower() == content2.get("subject", "").lower() and
                content1.get("predicate", "").lower() == content2.get("predicate", "").lower() and
                content1.get("object", "").lower() != content2.get("object", "").lower()
            )
        
        elif item1_type == "definition":
            # Definitions contradict if they have the same term but significantly different definitions
            term1 = content1.get("term", "").lower()
            term2 = content2.get("term", "").lower()
            def1 = content1.get("definition", "").lower()
            def2 = content2.get("definition", "").lower()
            
            if term1 == term2:
                # Check if definitions are significantly different
                # Here we do a simple word overlap check, but more sophisticated
                # methods could be used for determining semantic contradiction
                words1 = set(def1.split())
                words2 = set(def2.split())
                
                # Calculate Jaccard similarity
                if not words1 or not words2:
                    return False
                    
                overlap = len(words1.intersection(words2))
                union = len(words1.union(words2))
                
                # Low word overlap suggests contradiction
                return overlap / union < 0.3
        
        return False
    
    def generate_validation_report(self, knowledge_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report for knowledge items.
        
        Args:
            knowledge_items: List of knowledge items to analyze
            
        Returns:
            Validation report dictionary
        """
        # Validate items if not already validated
        validated_items = []
        for item in knowledge_items:
            if "validation" not in item:
                validated_item = self.validate_item(item)
                validated_items.append(validated_item)
            else:
                validated_items.append(item)
        
        # Analyze validation results
        total_items = len(validated_items)
        high_confidence = sum(1 for item in validated_items if item.get("validation", {}).get("confidence", 0) >= 0.7)
        medium_confidence = sum(1 for item in validated_items if 0.4 <= item.get("validation", {}).get("confidence", 0) < 0.7)
        low_confidence = sum(1 for item in validated_items if item.get("validation", {}).get("confidence", 0) < 0.4)
        
        contradicted_items = sum(1 for item in validated_items if item.get("validation", {}).get("contradiction_count", 0) > 0)
        outlier_items = sum(1 for item in validated_items if item.get("validation", {}).get("is_outlier", False))
        
        # Group by type
        type_stats = {}
        for item in validated_items:
            item_type = item.get("type", "unknown")
            if item_type not in type_stats:
                type_stats[item_type] = {
                    "count": 0,
                    "high_confidence": 0,
                    "contradicted": 0,
                    "outliers": 0
                }
            
            type_stats[item_type]["count"] += 1
            
            validation = item.get("validation", {})
            if validation.get("confidence", 0) >= 0.7:
                type_stats[item_type]["high_confidence"] += 1
            if validation.get("contradiction_count", 0) > 0:
                type_stats[item_type]["contradicted"] += 1
            if validation.get("is_outlier", False):
                type_stats[item_type]["outliers"] += 1
        
        # Create report
        report = {
            "total_items": total_items,
            "high_confidence_items": high_confidence,
            "medium_confidence_items": medium_confidence,
            "low_confidence_items": low_confidence,
            "contradicted_items": contradicted_items,
            "outlier_items": outlier_items,
            "confidence_distribution": {
                "high": high_confidence / total_items if total_items > 0 else 0,
                "medium": medium_confidence / total_items if total_items > 0 else 0,
                "low": low_confidence / total_items if total_items > 0 else 0
            },
            "type_statistics": type_stats,
            "validation_method": "fractal_embedding" if self.enable_fractal_validation else "embedding",
            "timestamp": datetime.now().isoformat()
        }
        
        return report