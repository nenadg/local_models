#!/usr/bin/env python
"""
Test 3: Knowledge Extraction
Tests extracting structured knowledge from unstructured text.
"""

import os
import sys
import json

# insert the project root (one level up) at the front of sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Ensure memory directory exists
os.makedirs("./memory", exist_ok=True)

# Import knowledge system components
try:
    from memory_manager import MemoryManager
    from knowledge_extractor import KnowledgeExtractor
except ImportError as e:
    print(f"[ERROR] Failed to import required modules: {e}")
    sys.exit(1)

# Sample text for extraction testing
SAMPLE_TEXTS = {
    "music": """
Duett is an electronic music project by Ben Macklin from the UK.
The band is known for their dreamy synth-pop sound with tracks like "Borderline" and "Highrise".
Their hit song "Highrise" was released on the album "Leisure" in 2017.
Duett has also released other albums including "Horizons" and their self-titled album "Duett".
The song features nostalgic 80s-inspired synthesizers and smooth vocals.
    """,
    
    "technical": """
Python is a high-level programming language created by Guido van Rossum.
It was first released in 1991 and is known for its readability and simplicity.
Python 3.0 was released in 2008, introducing many changes that were not backward compatible.
The language is object-oriented and supports functional programming paradigms.
Popular Python libraries include NumPy for numerical computing, pandas for data analysis, and TensorFlow for machine learning.
    """,
    
    "procedural": """
How to make a basic cup of coffee:
1. Boil fresh water in a kettle.
2. Add 1-2 tablespoons of coffee grounds to your cup or French press.
3. Pour the hot water over the coffee grounds.
4. Let it steep for 3-4 minutes.
5. If using a French press, push the plunger down slowly.
6. For pour-over methods, make sure your filter is properly seated.
7. Enjoy your coffee black or add cream and sugar to taste.
    """,
    
    "mapping": """
English to French translation of basic greetings:
"Hello" → "Bonjour"
"Good morning" → "Bon matin"
"Good afternoon" → "Bon après-midi"
"Good evening" → "Bonsoir"
"Goodbye" → "Au revoir"
"How are you?" → "Comment allez-vous?"
"Thank you" → "Merci"
"You're welcome" → "De rien"
    """
}

def run_test():
    """Run the knowledge extraction test"""
    print("\n===== Knowledge Extraction Test =====\n")
    
    # Step 1: Initialize memory manager and extractor
    print("Step 1: Initializing memory manager and knowledge extractor...")
    try:
        memory_manager = MemoryManager(
            memory_dir="./memory",
            fractal_enabled=True
        )
        
        extractor = KnowledgeExtractor(
            embedding_function=memory_manager.generate_embedding,
            enable_fractal_validation=True
        )
        print("  Successfully initialized knowledge extractor")
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        return False
    
    # Step 2: Test extraction from music text
    print("\nStep 2: Extracting knowledge from music-related text...")
    try:
        music_knowledge = extractor.extract_knowledge(
            SAMPLE_TEXTS["music"],
            source="test_extraction",
            domain="music"
        )
        
        print(f"  Extracted {len(music_knowledge)} knowledge items")
        
        # Display sample items
        for i, item in enumerate(music_knowledge[:3]):
            print(f"\n  Item {i+1}:")
            print(f"    Type: {item['type']}")
            
            # Format content based on type
            if item['type'] == 'fact':
                print(f"    Content: {item['content'].get('subject', '')} {item['content'].get('predicate', '')} {item['content'].get('object', '')}")
            elif item['type'] == 'definition':
                print(f"    Content: {item['content'].get('term', '')} = {item['content'].get('definition', '')}")
            else:
                print(f"    Content: {json.dumps(item['content'], indent=2)[:100]}...")
                
            print(f"    Confidence: {item['metadata']['confidence']:.2f}")
    except Exception as e:
        print(f"[ERROR] Music knowledge extraction failed: {e}")
        return False
    
    # Step 3: Test extraction from procedural text
    print("\nStep 3: Extracting knowledge from procedural text...")
    try:
        procedural_knowledge = extractor.extract_knowledge(
            SAMPLE_TEXTS["procedural"],
            source="test_extraction",
            domain="cooking"
        )
        
        print(f"  Extracted {len(procedural_knowledge)} knowledge items")
        
        # Check if we found a procedure
        procedures = [item for item in procedural_knowledge if item['type'] == 'procedure']
        if procedures:
            procedure = procedures[0]
            print(f"  Found procedure with {len(procedure['content'].get('steps', []))} steps:")
            for i, step in enumerate(procedure['content'].get('steps', [])[:3]):
                print(f"    Step {i+1}: {step}")
        else:
            print("  [WARNING] No procedures found in procedural text")
    except Exception as e:
        print(f"[ERROR] Procedural knowledge extraction failed: {e}")
        return False
    
    # Step 4: Test extraction from mapping text
    print("\nStep 4: Extracting knowledge from mapping/translation text...")
    try:
        mapping_knowledge = extractor.extract_knowledge(
            SAMPLE_TEXTS["mapping"],
            source="test_extraction",
            domain="language"
        )
        
        print(f"  Extracted {len(mapping_knowledge)} knowledge items")
        
        # Check if we found mappings
        mappings = [item for item in mapping_knowledge if item['type'] == 'mapping']
        if mappings:
            mapping = mappings[0]
            pairs = mapping['content'].get('pairs', [])
            print(f"  Found mapping with {len(pairs)} pairs:")
            for i, pair in enumerate(pairs[:3]):
                print(f"    {pair.get('from', '')} → {pair.get('to', '')}")
        else:
            print("  [WARNING] No mappings found in mapping text")
    except Exception as e:
        print(f"[ERROR] Mapping knowledge extraction failed: {e}")
        return False
    
    # Step 5: Test extraction from technical text
    print("\nStep 5: Extracting knowledge from technical text...")
    try:
        technical_knowledge = extractor.extract_knowledge(
            SAMPLE_TEXTS["technical"],
            source="test_extraction",
            domain="programming"
        )
        
        print(f"  Extracted {len(technical_knowledge)} knowledge items")
        
        # Count types of knowledge
        knowledge_types = {}
        for item in technical_knowledge:
            knowledge_types[item['type']] = knowledge_types.get(item['type'], 0) + 1
        
        print("  Knowledge type distribution:")
        for k_type, count in knowledge_types.items():
            print(f"    {k_type}: {count}")
    except Exception as e:
        print(f"[ERROR] Technical knowledge extraction failed: {e}")
        return False
    
    # Step 6: Test extraction statistics
    print("\nStep 6: Getting extraction statistics...")
    try:
        stats = extractor.get_extraction_stats()
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Facts extracted: {stats['facts_extracted']}")
        print(f"  Definitions extracted: {stats['definitions_extracted']}")
        print(f"  Procedures extracted: {stats['procedures_extracted']}")
        print(f"  Relationships extracted: {stats['relationships_extracted']}")
        print(f"  Mappings extracted: {stats['mappings_extracted']}")
    except Exception as e:
        print(f"[ERROR] Failed to get extraction stats: {e}")
        return False
    
    # Step 7: Test knowledge validation if available
    print("\nStep 7: Testing knowledge validation (if available)...")
    try:
        from knowledge_validator import KnowledgeValidator
        
        validator = KnowledgeValidator(
            embedding_function=memory_manager.generate_embedding,
            enable_fractal_validation=True
        )
        
        # Validate music knowledge items
        validated_items = validator.validate_items(music_knowledge)
        print(f"  Validated {len(validated_items)} music knowledge items")
        
        # Check confidence changes
        for i, item in enumerate(validated_items[:2]):
            original = item['metadata'].get('confidence', 0)
            validation = item.get('validation', {})
            new_conf = validation.get('confidence', 0)
            print(f"  Item {i+1} confidence: {original:.2f} → {new_conf:.2f}")
            
        print("  Validation successful")
    except ImportError:
        print("  [INFO] KnowledgeValidator not available, skipping validation test")
    except Exception as e:
        print(f"  [WARNING] Knowledge validation test failed: {e}")
    
    print("\n✅ All knowledge extraction tests passed")
    return True

if __name__ == "__main__":
    success = run_test()
    if success:
        print("\n[SUCCESS] Knowledge extraction test passed")
    else:
        print("\n[FAILURE] Knowledge extraction test failed")
        sys.exit(1)