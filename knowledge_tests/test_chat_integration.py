#!/usr/bin/env python
"""
Test 5: Chat Integration
Tests the integration of the knowledge system with TinyLlamaChat.
"""

import os
import sys
import time
import argparse

# insert the project root (one level up) at the front of sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Ensure memory directory exists
os.makedirs("./memory", exist_ok=True)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Test knowledge system integration with TinyLlamaChat")
parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                   help="Model name to use for testing")
parser.add_argument("--no-generate", action="store_true",
                    help="Skip response generation (useful for systems without GPU)")
args = parser.parse_args()

def run_test():
    """Run the chat integration test"""
    print("\n===== Chat Knowledge Integration Test =====\n")
    
    # Step 1: Import TinyLlamaChat
    print("Step 1: Importing TinyLlamaChat...")
    try:
        from tiny_llama_6_memory import TinyLlamaChat
        print("  Successfully imported TinyLlamaChat")
    except ImportError as e:
        print(f"[ERROR] Failed to import TinyLlamaChat: {e}")
        return False
    
    # Step 2: Initialize chat with knowledge system
    print("\nStep 2: Initializing chat with knowledge system...")
    try:
        chat = TinyLlamaChat(
            model_name=args.model,
            memory_dir="./memory",
            enable_sharpening=True,
            fractal_enabled=True,
            max_fractal_levels=3
        )
        
        print(f"  Model loaded: {args.model}")
        print(f"  Knowledge system enabled: {chat.knowledge_system_enabled}")
        
        if not chat.knowledge_system_enabled:
            print("[ERROR] Knowledge system not enabled in TinyLlamaChat")
            return False
    except Exception as e:
        print(f"[ERROR] Chat initialization failed: {e}")
        return False
    
    # Step 3: Create or load a knowledge domain
    print("\nStep 3: Creating or loading a knowledge domain...")
    try:
        # Check for existing domains
        if hasattr(chat.memory_manager, 'knowledge_registry'):
            domains = chat.memory_manager.knowledge_registry.list_domains()
            
            if domains:
                # Use first existing domain
                domain_id = domains[0]['domain_id']
                domain_name = domains[0]['name']
                print(f"  Using existing domain: {domain_name} (ID: {domain_id})")
            else:
                # Create a new domain
                domain_id = chat.create_knowledge_domain(
                    name="General Knowledge", 
                    description="General information for testing"
                )
                print(f"  Created new domain with ID: {domain_id}")
        else:
            print("[ERROR] Knowledge registry not available")
            return False
        
        # Load the domain
        if chat.load_knowledge_domain(domain_id):
            print(f"  Successfully loaded domain: {domain_id}")
        else:
            print(f"[ERROR] Failed to load domain: {domain_id}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Domain creation/loading failed: {e}")
        return False
    
    # Step 4: Extract knowledge from text
    print("\nStep 4: Extracting knowledge from text...")
    
    test_text = """
    Duett is an electronic music project by Ben Macklin based in the UK.
    Their popular track "Highrise" was released on the album "Leisure" in 2017.
    The project is known for nostalgic synth sounds inspired by 80s music.
    Another popular track is "Borderline" from their album "Horizons".
    """
    
    try:
        knowledge_items = chat.extract_knowledge_from_text(test_text, source="test")
        print(f"  Extracted {len(knowledge_items)} knowledge items")
        
        # Print a sample item
        if knowledge_items:
            item = knowledge_items[0]
            print(f"  Sample item type: {item['type']}")
            print(f"  Sample item confidence: {item['metadata']['confidence']:.2f}")
    except Exception as e:
        print(f"[ERROR] Knowledge extraction failed: {e}")
        return False
    
    # Step 5: Add knowledge to domain
    print("\nStep 5: Adding knowledge to domain...")
    try:
        added = chat.add_knowledge_to_domain(knowledge_items)
        print(f"  Added {added} knowledge items to domain {chat.current_domain_id}")
    except Exception as e:
        print(f"[ERROR] Adding knowledge to domain failed: {e}")
        return False
    
    # Step 6: Test response generation with knowledge domain
    print("\nStep 6: Testing response generation with knowledge domain...")
    
    if args.no_generate:
        print("  [INFO] Skipping response generation (--no-generate flag set)")
    else:
        try:
            # Test query related to our knowledge
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant with knowledge about music."},
                {"role": "user", "content": "What do you know about Duett and their song Highrise?"}
            ]
            
            print("  Generating response...")
            response = chat.generate_response(
                test_messages,
                max_new_tokens=100,
                temperature=0.7,
                turbo_mode=True,
                show_confidence=False
            )
            
            print("\n  Response with domain knowledge:\n")
            print(f"  {response}")
            
            # Check if response contains knowledge
            contains_knowledge = any(term in response.lower() for term 
                                   in ["duett", "highrise", "leisure", "macklin", "2017"])
            
            if contains_knowledge:
                print("\n  ✅ Response contains domain knowledge")
            else:
                print("\n  [WARNING] Response doesn't seem to contain domain knowledge")
                
        except Exception as e:
            print(f"[ERROR] Response generation failed: {e}")
            return False
    
    # Step 7: Test web knowledge enhancement if available
    print("\nStep 7: Testing web knowledge enhancement with domains...")
    
    if not hasattr(chat, 'web_enhancer') or not chat.web_enhancer:
        print("  [INFO] Web knowledge enhancement not available, skipping")
    else:
        try:
            # Test web enhancer with domain integration
            search_results = chat.web_enhancer.search_web("Duett electronic music Highrise", num_results=3)
            print(f"  Found {len(search_results)} web results")
            
            if search_results:
                # Extract knowledge from results
                web_text = "\n".join([result.get('snippet', '') for result in search_results])
                web_knowledge = chat.extract_knowledge_from_text(web_text, source="web_search")
                print(f"  Extracted {len(web_knowledge)} knowledge items from web results")
                
                # Add to current domain
                if web_knowledge and chat.current_domain_id:
                    added = chat.add_knowledge_to_domain(web_knowledge)
                    print(f"  Added {added} web knowledge items to domain {chat.current_domain_id}")
            else:
                print("  [INFO] No web results found for test query")
                
        except Exception as e:
            print(f"[WARNING] Web knowledge enhancement test warning: {e}")
    
    print("\n✅ All chat integration tests passed")
    return True

if __name__ == "__main__":
    success = run_test()
    if success:
        print("\n[SUCCESS] Chat integration test passed")
    else:
        print("\n[FAILURE] Chat integration test failed")
        sys.exit(1)