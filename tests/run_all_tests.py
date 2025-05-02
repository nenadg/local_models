#!/usr/bin/env python3
"""
Run All Tests for Memory-Enhanced Chat

This script runs all test scripts for the memory-enhanced chat system:
1. Memory System Test - tests basic memory functionality
2. Enhanced Embeddings Test - tests and visualizes the enhanced embedding system
3. MCP Test - tests the Model Content Protocol file handling

Usage:
    python run_all_tests.py [--output-dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def run_test(script_name, output_dir, extra_args=None):
    """Run a test script with proper arguments"""
    cmd = [sys.executable, script_name]
    
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
        
    if extra_args:
        cmd.extend(extra_args)
        
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        # Run the process and stream output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
            
        # Wait for completion
        process.wait()
        
        duration = time.time() - start_time
        if process.returncode == 0:
            print(f"✅ Test completed successfully in {duration:.2f} seconds")
            return True
        else:
            print(f"❌ Test failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def main():
    """Main function to run all tests"""
    parser = argparse.ArgumentParser(description="Run all tests for the memory-enhanced chat system")
    parser.add_argument("--output-dir", type=str, default="./test_output",
                      help="Directory for test output")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print start message
    print_header("MEMORY-ENHANCED CHAT SYSTEM TESTS")
    print(f"Starting tests at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    
    # Create test-specific output directories
    memory_output = os.path.join(args.output_dir, "memory_test")
    enhanced_output = os.path.join(args.output_dir, "enhanced_test")
    mcp_output = os.path.join(args.output_dir, "mcp_test")
    
    for directory in [memory_output, enhanced_output, mcp_output]:
        os.makedirs(directory, exist_ok=True)
    
    # Track test results
    results = {}
    
    # Run Memory Test
    print_header("1. MEMORY SYSTEM TEST")
    memory_test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_test.py")
    results["memory"] = run_test(memory_test_path, memory_output)
    
    # Run Enhanced Embeddings Test
    print_header("2. ENHANCED EMBEDDINGS TEST")
    enhanced_test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enhanced_embeddings_test.py")
    results["enhanced"] = run_test(enhanced_test_path, enhanced_output)
    
    # Run MCP Test
    print_header("3. MODEL CONTENT PROTOCOL TEST")
    mcp_test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_test.py")
    results["mcp"] = run_test(mcp_test_path, mcp_output)
    
    # Print summary
    print_header("TEST SUMMARY")
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        all_passed = all_passed and passed
    
    overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
    print(f"\nOverall status: {overall_status}")
    print(f"Tests completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All output files were saved to: {os.path.abspath(args.output_dir)}")
    
    # Return appropriate exit code
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())