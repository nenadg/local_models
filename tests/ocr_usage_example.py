"""
Example usage of WebOCRExtractor with fixed asyncio handling.
"""
"""
Example usage of WebOCRExtractor.
"""
import os
import sys
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from web_ocr import (
    WebOCRExtractor,
    extract_page_content_sync,
    add_to_memory_sync,
    cleanup_sync,
    format_for_context_with_scoring
)

def main():
    # Initialize the extractor
    extractor = WebOCRExtractor(
        # Pass your memory_manager here if available
        memory_manager=None,
        similarity_enhancement_factor=0.3,
        # Path to Chrome executable (optional)
        chrome_executable=None
    )

    try:
        # URL to extract
        url = "https://example.com/sample-page"

        # Sample query
        query = "What is the main topic of this page?"

        # Extract content
        print("Extracting content...")
        result = extract_page_content_sync(extractor, url)

        # Print extracted sections
        print(f"\nExtracted {len(result['content_sections'])} sections from {result['title']}")

        # Print a sample of the content
        if result['content_sections']:
            print("\nSample content:")
            print(result['content_sections'][0][:200] + "...")

        # Format for context (using the fixed function with proper scoring)
        formatted_context = format_for_context_with_scoring(extractor, result, query)
        print("\nFormatted context for LLM:")
        print(formatted_context)
        
        # Add to memory
        print("\nAdding to memory...")
        added_items = add_to_memory_sync(extractor, url, query)
        
        print(f"Added {len(added_items)} items to memory")
        
    finally:
        # Clean up resources
        cleanup_sync(extractor)
        print("\nCleaned up resources")

if __name__ == "__main__":
    main()