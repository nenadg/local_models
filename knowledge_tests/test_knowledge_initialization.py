#!/usr/bin/env python
"""
Test 1: Knowledge System Initialization
Verifies that the knowledge system components can be properly initialized.
"""

import os
import sys
import time

# insert the project root (one level up) at the front of sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Ensure memory directory exists
os.makedirs("./memory", exist_ok=True)

# Try to import knowledge system components
try:
    from memory_manager import MemoryManager
    from knowledge_domain import KnowledgeDomain
    from knowledge_extractor import KnowledgeExtractor
    from knowledge_registry import KnowledgeRegistry
    from knowledge_validator import KnowledgeValidator
    
    print("[SUCCESS] Successfully imported knowledge system components")
except ImportError as e:
    print(f"[ERROR] Failed to import knowledge system components: {e}")
    print("Make sure all required files are in the current directory or Python path")
    sys.exit(1)

def run_test():
    """Run the initialization test"""
    print("\n===== Knowledge System Initialization Test =====\n")
    
    # Step 1: Initialize memory manager with knowledge system
    print("Step 1: Initializing memory manager with knowledge system...")
    try:
        memory_manager = MemoryManager(
            memory_dir="./memory",
            fractal_enabled=True,
            max_fractal_levels=3
        )
        print("  Memory manager initialized")
        
        # Initialize the knowledge system
        knowledge_enabled = memory_manager.initialize_knowledge_system()
        print(f"  Knowledge system enabled: {knowledge_enabled}")
        
        if not knowledge_enabled:
            print("[ERROR] Knowledge system could not be initialized")
            return False
    except Exception as e:
        print(f"[ERROR] Memory manager initialization failed: {e}")
        return False
    
    # Step 2: Check registry availability
    print("\nStep 2: Checking knowledge registry...")
    if hasattr(memory_manager, 'knowledge_registry'):
        registry = memory_manager.knowledge_registry
        print(f"  Registry found: {type(registry).__name__}")
        
        # Try listing domains
        try:
            domains = registry.list_domains()
            print(f"  Found {len(domains)} existing domains")
            for domain in domains:
                print(f"    - {domain['name']} (ID: {domain['domain_id']})")
        except Exception as e:
            print(f"[WARNING] Could not list domains: {e}")
    else:
        print("[ERROR] Knowledge registry not available")
        return False
    
    # Step 3: Initialize individual components directly
    print("\nStep 3: Testing direct component initialization...")
    
    # Test KnowledgeDomain initialization
    try:
        domain = KnowledgeDomain(
            name="Test Domain",
            description="Test domain for initialization",
            base_directory="./memory/knowledge_domains",
            embedding_function=memory_manager.generate_embedding,
            embedding_dim=memory_manager.embedding_dim,
            enable_fractal=True,
            max_fractal_levels=3
        )
        print(f"  KnowledgeDomain initialized: ID={domain.domain_id}")
    except Exception as e:
        print(f"[ERROR] KnowledgeDomain initialization failed: {e}")
        return False
    
    # Test KnowledgeExtractor initialization
    try:
        extractor = KnowledgeExtractor(
            embedding_function=memory_manager.generate_embedding,
            enable_fractal_validation=True
        )
        print(f"  KnowledgeExtractor initialized")
    except Exception as e:
        print(f"[ERROR] KnowledgeExtractor initialization failed: {e}")
        return False
    
    # Test KnowledgeValidator initialization  
    try:
        validator = KnowledgeValidator(
            embedding_function=memory_manager.generate_embedding,
            enable_fractal_validation=True
        )
        print(f"  KnowledgeValidator initialized")
    except Exception as e:
        print(f"[ERROR] KnowledgeValidator initialization failed: {e}")
        return False
    
    print("\n✅ All components initialized successfully")
    return True

if __name__ == "__main__":
    success = run_test()
    if success:
        print("\n[SUCCESS] Knowledge system initialization test passed")
    else:
        print("\n[FAILURE] Knowledge system initialization test failed")
        sys.exit(1)