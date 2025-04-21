#!/usr/bin/env python
"""
Test 2: Knowledge Domain Management
Tests creating, loading, and managing knowledge domains.
"""

import os
import sys
import time

# insert the project root (one level up) at the front of sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Ensure memory directory exists
os.makedirs("./memory", exist_ok=True)

# Import knowledge system components
try:
    from memory_manager import MemoryManager
except ImportError as e:
    print(f"[ERROR] Failed to import memory_manager.py: {e}")
    sys.exit(1)

def run_test():
    """Run the domain management test"""
    print("\n===== Knowledge Domain Management Test =====\n")
    
    # Step 1: Initialize memory manager with knowledge system
    print("Step 1: Initializing memory manager with knowledge system...")
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
    except Exception as e:
        print(f"[ERROR] Memory manager initialization failed: {e}")
        return False
    
    # Step 2: Create a new domain
    print("\nStep 2: Creating a new knowledge domain...")
    try:
        domain_id = memory_manager.create_domain(
            name="Music Knowledge",
            description="Information about music artists, songs, and albums"
        )
        
        if domain_id:
            print(f"  Successfully created domain with ID: {domain_id}")
        else:
            print("[ERROR] Failed to create domain")
            return False
    except Exception as e:
        print(f"[ERROR] Domain creation failed: {e}")
        return False
    
    # Step 3: Load and verify the domain
    print("\nStep 3: Loading the created domain...")
    try:
        domain = memory_manager.get_domain(domain_id)
        
        if domain:
            print(f"  Successfully loaded domain: {domain.metadata['name']}")
            print(f"  Description: {domain.metadata['description']}")
            print(f"  Created at: {domain.metadata['created_at']}")
        else:
            print("[ERROR] Failed to load domain")
            return False
    except Exception as e:
        print(f"[ERROR] Domain loading failed: {e}")
        return False
    
    # Step 4: List all domains to verify registration
    print("\nStep 4: Listing all domains in registry...")
    try:
        domains = memory_manager.knowledge_registry.list_domains()
        print(f"  Found {len(domains)} domains in registry")
        
        # Check if our domain is in the list
        found = False
        for d in domains:
            print(f"    - {d['name']} (ID: {d['domain_id']})")
            if d['domain_id'] == domain_id:
                found = True
        
        if not found:
            print("[ERROR] Created domain not found in registry listing")
            return False
    except Exception as e:
        print(f"[ERROR] Domain listing failed: {e}")
        return False
    
    # Step 5: Get domain statistics
    print("\nStep 5: Getting domain statistics...")
    try:
        stats = domain.get_stats()
        print(f"  Domain ID: {stats['domain_id']}")
        print(f"  Name: {stats['name']}")
        print(f"  Total items: {stats['total_items']}")
        print(f"  Vector store info: {stats['vector_store']}")
    except Exception as e:
        print(f"[ERROR] Failed to get domain stats: {e}")
        return False
    
    # Step 6: Test domain saving
    print("\nStep 6: Testing domain save functionality...")
    try:
        save_success = domain.save()
        print(f"  Domain save successful: {save_success}")
    except Exception as e:
        print(f"[ERROR] Domain save failed: {e}")
        return False
    
    # Step 7: Create another domain for multi-domain testing
    print("\nStep 7: Creating a second domain for multi-domain testing...")
    try:
        second_domain_id = memory_manager.create_domain(
            name="Movies",
            description="Information about movies, actors, and directors"
        )
        
        if second_domain_id:
            print(f"  Successfully created second domain with ID: {second_domain_id}")
        else:
            print("[WARNING] Failed to create second domain")
    except Exception as e:
        print(f"[WARNING] Second domain creation failed: {e}")
    
    # Step 8: Test domain registry stats
    print("\nStep 8: Getting registry statistics...")
    try:
        registry_stats = memory_manager.knowledge_registry.get_stats()
        print(f"  Total domains: {registry_stats['total_domains']}")
        print(f"  Active domains: {registry_stats['active_domains']}")
        print(f"  Last updated: {registry_stats['last_updated']}")
    except Exception as e:
        print(f"[ERROR] Registry stats failed: {e}")
        return False
    
    print("\n✅ All domain management tests passed")
    return True

if __name__ == "__main__":
    success = run_test()
    if success:
        print("\n[SUCCESS] Knowledge domain management test passed")
    else:
        print("\n[FAILURE] Knowledge domain management test failed")
        sys.exit(1)