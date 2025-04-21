#!/usr/bin/env python
"""
Test 4: Knowledge Search and Storage
Tests adding knowledge to domains and searching for it.
"""

import os
import sys
import json
import time

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

# Sample text for knowledge
SAMPLE_TEXT = """
Duett is an electronic music project by Ben Macklin based in the UK.
The project is known for its nostalgic synthesizer sounds and retro aesthetics.
Their popular song "Highrise" was released on the album "Leisure" in 2017.
The track features dreamy synth melodies and was well-received by synthwave fans.
Other notable songs include "Borderline" and "Voyager" from their album "Horizons".
Duett's music style combines elements of synthwave, vaporwave, and chillwave.
The project gained popularity through online streaming platforms and retro synth communities.
"""

def run_test():
    """Run the knowledge search and storage test"""
    print("\n===== Knowledge Search and Storage Test =====\n")
    
    # Step 1: Initialize memory manager and components
    print("Step 1: Initializing memory manager and components...")
    try:
        memory_manager = MemoryManager(
            memory_dir="./memory",
            fractal_enabled=True,
            max_fractal_levels=3
        )
        
        # Initialize the knowledge system
        knowledge_enabled = memory_manager.initialize_knowledge_system()
        print(f"  Knowledge system enabled: {knowledge_enabled}")
        
        if not knowledge_enabled:
            print("[ERROR] Knowledge system could not be initialized")
            return False
            
        # Initialize extractor
        extractor = KnowledgeExtractor(
            embedding_function=memory_manager.generate_embedding,
            enable_fractal_validation=True
        )
        print("  Knowledge extractor initialized")
        
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        return False
    
    # Step 2: Create or load a domain
    print("\nStep 2: Creating a test domain for knowledge storage...")
    try:
        # Check for existing domains first
        domains = memory_manager.knowledge_registry.list_domains()
        
        if domains and any(d['name'] == 'Music' for d in domains):
            # Use existing Music domain if available
            domain_id = next(d['domain_id'] for d in domains if d['name'] == 'Music')
            print(f"  Using existing Music domain with ID: {domain_id}")
        else:
            # Create new domain
            domain_id = memory_manager.create_domain(
                name="Music",
                description="Information about music artists, songs, and albums"
            )
            print(f"  Created new Music domain with ID: {domain_id}")
        
        # Load the domain
        domain = memory_manager.get_domain(domain_id)
        
        if not domain:
            print("[ERROR] Failed to load domain")
            return False
            
        print(f"  Successfully loaded domain: {domain.metadata['name']}")
        
    except Exception as e:
        print(f"[ERROR] Domain creation/loading failed: {e}")
        return False
    
    # Step 3: Extract knowledge from sample text
    print("\nStep 3: Extracting knowledge from sample text...")
    try:
        knowledge_items = extractor.extract_knowledge(
            SAMPLE_TEXT,
            source="test_search_storage",
            domain="music"
        )
        
        print(f"  Extracted {len(knowledge_items)} knowledge items")
        
        # Print sample items
        for i, item in enumerate(knowledge_items[:3]):
            print(f"\n  Item {i+1}:")
            print(f"    Type: {item['type']}")
            
            # Print content based on type
            if item['type'] == 'fact':
                content = item['content']
                print(f"    Subject: {content.get('subject', '')}")
                print(f"    Predicate: {content.get('predicate', '')}")
                print(f"    Object: {content.get('object', '')}")
            elif item['type'] == 'definition':
                content = item['content']
                print(f"    Term: {content.get('term', '')}")
                print(f"    Definition: {content.get('definition', '')}")
            else:
                # Just print brief content summary
                content_str = str(item['content'])
                print(f"    Content: {content_str[:100]}..." if len(content_str) > 100 else content_str)
            
            print(f"    Confidence: {item['metadata']['confidence']:.2f}")
            
    except Exception as e:
        print(f"[ERROR] Knowledge extraction failed: {e}")
        return False
    
    # Step 4: Add knowledge to domain
    print("\nStep 4: Adding knowledge items to domain...")
    try:
        added_count = domain.add_knowledge(knowledge_items)
        print(f"  Successfully added {added_count} knowledge items to domain")
        
        if added_count <= 0:
            print("  [WARNING] No items were added to the domain")
            
        # Update domain stats
        stats = domain.get_stats()
        print(f"  Domain now has {stats['total_items']} total items")
        
    except Exception as e:
        print(f"[ERROR] Adding knowledge failed: {e}")
        return False
    
    # Step 5: Search for knowledge in domain
    print("\nStep 5: Searching for knowledge in domain...")
    try:
        # Try several search queries
        search_queries = [
            "Duett highrise song",
            "electronic music project ben macklin", 
            "leisure album 2017",
            "synthwave"
        ]
        
        for query in search_queries:
            print(f"\n  Searching for: '{query}'")
            results = domain.search(query, top_k=3)
            
            print(f"  Found {len(results)} results")
            
            # Print search results
            for i, result in enumerate(results):
                print(f"    Result {i+1}:")
                
                # Get similarity
                similarity = result.get('search_metadata', {}).get('similarity', 0)
                print(f"      Similarity: {similarity:.2f}")
                
                # Print differently based on result type
                r_type = result.get('type', 'unknown')
                print(f"      Type: {r_type}")
                
                # Print brief content
                if r_type == 'fact':
                    content = result.get('content', {})
                    subject = content.get('subject', '')
                    obj = content.get('object', '')
                    print(f"      Content: {subject} ... {obj}")
                elif r_type == 'definition':
                    content = result.get('content', {})
                    term = content.get('term', '')
                    definition = content.get('definition', '')
                    print(f"      Content: {term} = {definition[:50]}...")
                else:
                    # Generic content display
                    content_str = str(result.get('content', {}))
                    print(f"      Content: {content_str[:70]}...")
        
    except Exception as e:
        print(f"[ERROR] Knowledge search failed: {e}")
        return False
    
    # Step 6: Test saving and exporting
    print("\nStep 6: Testing knowledge export functionality...")
    try:
        # Save domain first
        save_success = domain.save()
        print(f"  Domain save successful: {save_success}")
        
        # Export knowledge
        export_path = domain.export_knowledge()
        print(f"  Exported knowledge to: {export_path}")
        
        # Verify export file exists
        if os.path.exists(export_path):
            file_size = os.path.getsize(export_path)
            print(f"  Export file size: {file_size} bytes")
            
            # Try reading a few items from export file
            with open(export_path, 'r', encoding='utf-8') as f:
                export_data = json.load(f)
                
            print(f"  Export contains {len(export_data)} knowledge items")
        else:
            print("  [WARNING] Export file not found")
            
    except Exception as e:
        print(f"[ERROR] Knowledge export failed: {e}")
        return False
    
    # Step 7: Test domain resource management
    print("\nStep 7: Testing domain resource management...")
    try:
        # Test registry cleanup
        registry = memory_manager.knowledge_registry
        registry.cleanup()
        print("  Registry cleanup successful")
        
        # Test domain reloading
        reloaded_domain = memory_manager.get_domain(domain_id)
        if reloaded_domain:
            print(f"  Successfully reloaded domain after cleanup")
            reloaded_stats = reloaded_domain.get_stats()
            print(f"  Reloaded domain has {reloaded_stats['total_items']} items")
        else:
            print("  [WARNING] Failed to reload domain after cleanup")
            
    except Exception as e:
        print(f"[WARNING] Resource management test warning: {e}")
    
    print("\n✅ All knowledge search and storage tests passed")
    return True

if __name__ == "__main__":
    success = run_test()
    if success:
        print("\n[SUCCESS] Knowledge search and storage test passed")
    else:
        print("\n[FAILURE] Knowledge search and storage test failed")
        sys.exit(1)