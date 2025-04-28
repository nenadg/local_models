#!/usr/bin/env python
"""
Fractal Memory Extreme Stress Test
----------------------------------
Tests the limits of the system by:
1. Creating numerous nearly identical items
2. Making adversarial queries
3. Forcing extremely large batch sizes
4. Running concurrent operations
5. Creating semantic confusion
"""

import os
import sys
import time
import random
import threading
import concurrent.futures
import numpy as np
from datetime import datetime

# Adjust path to include the local_models directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import memory system components
from unified_memory import UnifiedMemoryManager
from resource_manager import ResourceManager

# Test configuration
TEST_MEMORY_DIR = "./stress_test_memory"
NUM_NEAR_DUPLICATE_ITEMS = 200  # Create this many nearly identical items
NUM_CONCURRENT_OPERATIONS = 8   # Run this many operations in parallel
BATCH_SIZES_TO_TEST = [1, 8, 16, 32, 64, 128, 256, 512]  # Test these batch sizes
NUM_QUERIES = 50  # Number of queries to run
ADVERSARIAL_MIXING_RATIO = 0.7  # How much to mix domains in a query

# Clean test directory
def clean_directory():
    """Clean the test directory to start fresh"""
    # import shutil
    # if os.path.exists(TEST_MEMORY_DIR):
    #     shutil.rmtree(TEST_MEMORY_DIR)
    # os.makedirs(TEST_MEMORY_DIR, exist_ok=True)

# Generate nearly-identical but semantically distinct items
def generate_near_duplicates():
    """Generate items that are nearly identical but with subtle differences"""
    items = []
    
    # Base templates for different domains
    templates = {
        "history": "In {year}, the {event} occurred which led to significant changes in {location}.",
        "science": "The {theory} explains how {process} works by demonstrating the relationship between {entity1} and {entity2}.",
        "music": "The song '{title}' by {artist} was released in {year} and features {instrument} prominently.",
        "technology": "The {technology} was developed by {company} in {year} to solve the problem of {problem}."
    }
    
    # Generate variations that are extremely similar
    domains = list(templates.keys())
    for domain in domains:
        template = templates[domain]
        
        for i in range(NUM_NEAR_DUPLICATE_ITEMS // len(domains)):
            # Create variations with minimal differences
            if domain == "history":
                year = random.randint(1800, 2000)
                events = ["Treaty of " + chr(ord('A') + i % 26), "Battle of " + chr(ord('A') + i % 26)]
                event = events[i % len(events)]
                locations = ["Eastern " + chr(ord('A') + i % 26), "Western " + chr(ord('A') + i % 26)]
                location = locations[i % len(locations)]
                
                content = template.format(year=year, event=event, location=location)
                
            elif domain == "science":
                theories = ["Unified " + chr(ord('A') + i % 26) + " Theory", chr(ord('A') + i % 26) + " Principle"]
                theory = theories[i % len(theories)]
                processes = ["molecular " + chr(ord('A') + i % 26), "quantum " + chr(ord('A') + i % 26)]
                process = processes[i % len(processes)]
                entity1 = "particle " + chr(ord('A') + i % 26)
                entity2 = "wave " + chr(ord('A') + i % 26)
                
                content = template.format(theory=theory, process=process, entity1=entity1, entity2=entity2)
                
            elif domain == "music":
                titles = ["Symphony No." + str(i % 10), "Sonata No." + str(i % 10)]
                title = titles[i % len(titles)]
                artists = ["The " + chr(ord('A') + i % 26) + "s", chr(ord('A') + i % 26) + " Quartet"]
                artist = artists[i % len(artists)]
                year = 1950 + (i % 70)
                instruments = ["violin", "piano", "guitar", "drums"]
                instrument = instruments[i % len(instruments)]
                
                content = template.format(title=title, artist=artist, year=year, instrument=instrument)
                
            else:  # technology
                technologies = ["Advanced " + chr(ord('A') + i % 26) + " System", chr(ord('A') + i % 26) + " Framework"]
                technology = technologies[i % len(technologies)]
                companies = ["Tech" + chr(ord('A') + i % 26), chr(ord('A') + i % 26) + "Corp"]
                company = companies[i % len(companies)]
                year = 1980 + (i % 40)
                problems = ["efficiency in " + chr(ord('A') + i % 26), "optimization of " + chr(ord('A') + i % 26)]
                problem = problems[i % len(problems)]
                
                content = template.format(technology=technology, company=company, year=year, problem=problem)
            
            # Add to items list with appropriate metadata
            items.append({
                "content": content,
                "metadata": {
                    "domain": domain,
                    "index": i,
                    "variant": i % 5,
                    "created_at": time.time()
                }
            })
    
    return items

# Generate adversarial queries that mix domains and have ambiguous semantics
def generate_adversarial_queries():
    """Generate queries designed to confuse the system"""
    queries = []
    
    # Domain-specific keywords
    domain_keywords = {
        "history": ["event", "year", "occurred", "changes", "significant", "led"],
        "science": ["theory", "process", "explains", "relationship", "entity", "demonstrating"],
        "music": ["song", "artist", "released", "features", "instrument"],
        "technology": ["developed", "company", "solve", "problem", "technology"]
    }
    
    # 1. Mix domains in confusing ways
    for i in range(NUM_QUERIES // 4):
        # Pick 2-3 random domains
        num_domains = random.randint(2, 3)
        selected_domains = random.sample(list(domain_keywords.keys()), num_domains)
        
        # Mix keywords from these domains
        mixed_keywords = []
        for domain in selected_domains:
            domain_words = random.sample(domain_keywords[domain], 
                                       k=min(3, len(domain_keywords[domain])))
            mixed_keywords.extend(domain_words)
        
        # Create a confusing query
        random.shuffle(mixed_keywords)
        query = "Find information about " + " and ".join(mixed_keywords[:4]) + "."
        queries.append(query)
    
    # 2. Ambiguous entity references
    ambiguous_entities = [
        "Taylor", "Mercury", "Java", "Python", "Ruby", "Pearl", "Swift", 
        "Apple", "Windows", "Doors", "Keys", "Silverware"
    ]
    
    for i in range(NUM_QUERIES // 4):
        entity = random.choice(ambiguous_entities)
        query = f"Tell me about {entity} and its most important features."
        queries.append(query)
    
    # 3. Contradictory term pairs
    contradictory_pairs = [
        ("ancient", "modern"), ("quantum", "classical"), ("digital", "analog"),
        ("theoretical", "practical"), ("romantic", "classical"), ("fast", "slow")
    ]
    
    for i in range(NUM_QUERIES // 4):
        pair = random.choice(contradictory_pairs)
        query = f"Find the relationship between {pair[0]} and {pair[1]} techniques."
        queries.append(query)
    
    # 4. Extremely vague queries
    vague_queries = [
        "What about that thing?", "Find the information.", "The data from before.",
        "That event with the people.", "The one I mentioned earlier.", "Show me that again."
    ]
    
    queries.extend(vague_queries[:NUM_QUERIES // 4])
    
    return queries[:NUM_QUERIES]

# Stress test function for each thread
def stress_test_thread(thread_id, memory_manager, items, queries, batch_sizes):
    """Run stress test in a single thread"""
    results = {
        "thread_id": thread_id,
        "items_added": 0,
        "queries_processed": 0,
        "errors": [],
        "batch_size_results": {},
        "timings": [],
        "max_batch_size_achieved": 0
    }
    
    # Determine items for this thread
    items_per_thread = len(items) // NUM_CONCURRENT_OPERATIONS
    start_idx = thread_id * items_per_thread
    end_idx = start_idx + items_per_thread if thread_id < NUM_CONCURRENT_OPERATIONS - 1 else len(items)
    thread_items = items[start_idx:end_idx]
    
    # 1. Test adding items with various batch sizes
    for batch_size in batch_sizes:
        if batch_size > len(thread_items):
            continue
            
        # Create batches
        num_batches = len(thread_items) // batch_size
        
        batch_start_time = time.time()
        items_added = 0
        batch_errors = 0
        
        try:
            for i in range(num_batches):
                batch_items = thread_items[i*batch_size:(i+1)*batch_size]
                
                # Add batch with timing
                batch_timing_start = time.time()
                item_ids = memory_manager.add_bulk(batch_items, use_fractal=True)
                batch_timing_end = time.time()
                
                # Count successful adds
                items_added += sum(1 for item_id in item_ids if item_id is not None)
                
                results["timings"].append({
                    "operation": "add_bulk",
                    "batch_size": batch_size,
                    "duration": batch_timing_end - batch_timing_start,
                    "items_per_second": batch_size / (batch_timing_end - batch_timing_start)
                })
                
                # Sleep briefly to prevent overwhelming the system
                time.sleep(0.01)
            
            # Store results for this batch size
            results["batch_size_results"][batch_size] = {
                "items_added": items_added,
                "time_taken": time.time() - batch_start_time,
                "success_rate": items_added / (num_batches * batch_size) if num_batches > 0 else 0
            }
            
            # Update max batch size achieved
            if items_added > 0 and batch_size > results["max_batch_size_achieved"]:
                results["max_batch_size_achieved"] = batch_size
                
        except Exception as e:
            # Record the error
            results["errors"].append({
                "phase": "adding",
                "batch_size": batch_size,
                "error": str(e)
            })
            batch_errors += 1
            
            # Try next batch size
            continue
    
    # Total items added
    results["items_added"] = sum(result["items_added"] 
                               for result in results["batch_size_results"].values())
    
    # 2. Test querying with adversarial and mixed queries
    queries_per_thread = len(queries) // NUM_CONCURRENT_OPERATIONS
    thread_queries = queries[:queries_per_thread]  # Each thread gets a subset
    
    # Mix in some normal queries
    normal_queries = [
        f"Find information about {random.choice(['history', 'science', 'music', 'technology'])}",
        f"What is the relationship between {random.choice(['theory', 'event', 'song', 'technology'])} and {random.choice(['location', 'entity', 'artist', 'company'])}?",
        f"Tell me about events in {random.randint(1800, 2020)}"
    ]
    
    thread_queries.extend(normal_queries)
    random.shuffle(thread_queries)
    
    queries_processed = 0
    query_errors = 0
    
    for query in thread_queries:
        try:
            # Process query with timing
            query_timing_start = time.time()
            top_k = random.randint(3, 10)  # Random number of results to retrieve
            results_list = memory_manager.retrieve(query, top_k=top_k, min_similarity=0.2)
            query_timing_end = time.time()
            
            queries_processed += 1
            
            # Record timing
            results["timings"].append({
                "operation": "retrieve",
                "query": query,
                "top_k": top_k,
                "num_results": len(results_list),
                "duration": query_timing_end - query_timing_start,
                "results_per_second": len(results_list) / (query_timing_end - query_timing_start) if (query_timing_end - query_timing_start) > 0 else 0
            })
            
            # Sleep briefly to prevent overwhelming the system
            time.sleep(0.01)
            
        except Exception as e:
            # Record the error
            results["errors"].append({
                "phase": "querying",
                "query": query,
                "error": str(e)
            })
            query_errors += 1
    
    # Total queries processed
    results["queries_processed"] = queries_processed
    
    return results

def print_results_summary(all_results):
    """Print a summary of the stress test results"""
    total_items_added = sum(result["items_added"] for result in all_results)
    total_queries_processed = sum(result["queries_processed"] for result in all_results)
    total_errors = sum(len(result["errors"]) for result in all_results)
    
    max_batch_size = max(result["max_batch_size_achieved"] for result in all_results)
    
    # Calculate average timings
    add_timings = [timing for result in all_results 
                 for timing in result["timings"] if timing["operation"] == "add_bulk"]
    
    retrieve_timings = [timing for result in all_results 
                      for timing in result["timings"] if timing["operation"] == "retrieve"]
    
    avg_add_time = sum(timing["duration"] for timing in add_timings) / len(add_timings) if add_timings else 0
    avg_retrieve_time = sum(timing["duration"] for timing in retrieve_timings) / len(retrieve_timings) if retrieve_timings else 0
    
    avg_add_throughput = sum(timing["items_per_second"] for timing in add_timings) / len(add_timings) if add_timings else 0
    avg_retrieve_throughput = sum(timing["results_per_second"] for timing in retrieve_timings) / len(retrieve_timings) if retrieve_timings else 0
    
    print("\n" + "="*50)
    print("STRESS TEST RESULTS SUMMARY")
    print("="*50)
    print(f"Total items added: {total_items_added}/{NUM_NEAR_DUPLICATE_ITEMS}")
    print(f"Total queries processed: {total_queries_processed}/{NUM_QUERIES * NUM_CONCURRENT_OPERATIONS}")
    print(f"Total errors encountered: {total_errors}")
    print(f"Maximum successful batch size: {max_batch_size}")
    print(f"Average add operation time: {avg_add_time:.4f}s ({avg_add_throughput:.2f} items/s)")
    print(f"Average retrieve operation time: {avg_retrieve_time:.4f}s ({avg_retrieve_throughput:.2f} results/s)")
    
    # Print batch size results
    print("\nBatch Size Performance:")
    batch_sizes = sorted(all_results[0]["batch_size_results"].keys())
    
    for batch_size in batch_sizes:
        # Collect results for this batch size across all threads
        total_added = sum(result["batch_size_results"].get(batch_size, {}).get("items_added", 0) 
                         for result in all_results)
        
        success_rates = [result["batch_size_results"].get(batch_size, {}).get("success_rate", 0) 
                        for result in all_results 
                        if batch_size in result["batch_size_results"]]
        
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
        
        print(f"  Batch size {batch_size}: {total_added} items added, {avg_success_rate:.2%} success rate")
    
    # Print error types
    print("\nError Summary:")
    error_types = {}
    
    for result in all_results:
        for error in result["errors"]:
            error_str = error["error"]
            if error_str not in error_types:
                error_types[error_str] = 0
            error_types[error_str] += 1
    
    for error_str, count in error_types.items():
        print(f"  {error_str}: {count} occurrences")
        
    print("="*50)

def run_stress_test():
    """Run a complete fractal memory stress test"""
    print(f"Starting fractal memory stress test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test parameters:")
    print(f"  - Near duplicate items: {NUM_NEAR_DUPLICATE_ITEMS}")
    print(f"  - Concurrent operations: {NUM_CONCURRENT_OPERATIONS}")
    print(f"  - Batch sizes to test: {BATCH_SIZES_TO_TEST}")
    print(f"  - Adversarial queries: {NUM_QUERIES}")
    
    # Clean test directory
    clean_directory()
    
    # Create resource manager
    resource_manager = ResourceManager()
    
    # Create memory manager
    memory_manager = UnifiedMemoryManager(
        storage_path=TEST_MEMORY_DIR,
        embedding_function=lambda x: np.random.rand(384).astype(np.float32),  # Dummy for testing
        embedding_dim=384,
        use_fractal=True,
        max_fractal_levels=3,
        auto_save=True
    )
    
    # Generate test data
    print("Generating test data...")
    items = generate_near_duplicates()
    queries = generate_adversarial_queries()
    
    print(f"Generated {len(items)} items and {len(queries)} queries")
    
    # Run concurrent stress tests
    print(f"Running {NUM_CONCURRENT_OPERATIONS} concurrent test threads...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CONCURRENT_OPERATIONS) as executor:
        futures = []
        
        for i in range(NUM_CONCURRENT_OPERATIONS):
            future = executor.submit(
                stress_test_thread, 
                i, 
                memory_manager, 
                items, 
                queries, 
                BATCH_SIZES_TO_TEST
            )
            futures.append(future)
        
        # Wait for completion and collect results
        all_results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
                print(f"Thread {result['thread_id']} completed: added {result['items_added']} items, processed {result['queries_processed']} queries")
            except Exception as e:
                print(f"Thread failed with error: {e}")
    
    # Process and display results
    print_results_summary(all_results)
    
    # Clean up
    memory_manager.cleanup()
    
    print(f"Stress test completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    run_stress_test()