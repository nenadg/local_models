#!/usr/bin/env python
"""
Main test runner script.
Runs all knowledge system tests in sequence.
"""

import os
import sys
import subprocess
import argparse
import time


# insert the project root (one level up) at the front of sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Define all test scripts
TEST_SCRIPTS = [
    "test_knowledge_initialization.py",
    "test_knowledge_domain_management.py",
    "test_knowledge_extraction.py",
    "test_knowledge_search_and_storage.py",
    "test_chat_integration.py"
]

def run_tests(args):
    """Run all test scripts and report results"""
    print("\n===== KNOWLEDGE SYSTEM TESTS =====\n")
    
    # Make test scripts executable
    make_scripts_executable()
    
    results = {}
    total_tests = len(TEST_SCRIPTS)
    passed_tests = 0
    
    for i, script in enumerate(TEST_SCRIPTS, 1):
        test_name = script.replace(".py", "")
        print(f"\n[{i}/{total_tests}] Running {test_name}...\n")
        
        # Build command
        cmd = [f"./{script}"]
        
        # Add model argument to chat integration test
        if script == "test_chat_integration.py" and args.model:
            cmd.extend(["--model", args.model])
            
        # Add no-generate flag if specified
        if script == "test_chat_integration.py" and args.no_generate:
            cmd.append("--no-generate")
        
        try:
            start_time = time.time()
            
            # Run the test
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Output the test results
            print(result.stdout)
            
            if result.stderr:
                print(f"STDERR: {result.stderr}")
            
            # Check if test passed
            passed = result.returncode == 0
            results[test_name] = passed
            
            if passed:
                passed_tests += 1
                print(f"✅ {test_name} PASSED in {duration:.2f} seconds")
            else:
                print(f"❌ {test_name} FAILED in {duration:.2f} seconds")
                
            # Add small delay between tests
            time.sleep(1)
            
        except Exception as e:
            print(f"Error running {script}: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n===== TEST SUMMARY =====")
    print(f"Passed: {passed_tests}/{total_tests}\n")
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    # Return success if all tests passed
    return passed_tests == total_tests

def make_scripts_executable():
    """Make all test scripts executable"""
    for script in TEST_SCRIPTS:
        try:
            os.chmod(script, 0o755)  # rwxr-xr-x
        except Exception as e:
            print(f"Warning: Could not make {script} executable: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all knowledge system tests")
    parser.add_argument("--model", type=str, 
                       default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                       help="Model to use for chat integration test")
    parser.add_argument("--no-generate", action="store_true",
                       help="Skip response generation in chat integration test")
    
    args = parser.parse_args()
    
    success = run_tests(args)
    
    if success:
        print("\n✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)