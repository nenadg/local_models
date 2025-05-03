#!/usr/bin/env python3
"""
Test script for OCR-based web content extraction.
This allows you to test the OCR extraction capabilities directly from the command line.
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_time() -> str:
    """Get formatted timestamp for logging."""
    return datetime.now().strftime("[%d/%m/%y %H:%M:%S]")

def test_google_search(query, api_key=None, cx_id=None, max_results=5):
    """Test Google Custom Search with OCR extraction."""
    try:
        # Import our modules
        from web_ocr import WebOCRExtractor
        from google_search_client import GoogleSearchClient
        
        print(f"{get_time()} Initializing OCR extractor...")
        ocr_extractor = WebOCRExtractor(
            temp_dir="./temp",
            ocr_language="eng"
        )
        
        # Create Google Search client
        print(f"{get_time()} Creating Google Search client...")
        search_client = GoogleSearchClient(
            api_key=api_key,
            cx_id=cx_id,
            ocr_extractor=ocr_extractor,
            memory_manager=None,
            cache_dir="./cache"
        )
        
        # Perform search
        print(f"{get_time()} Searching for: {query}")
        results = search_client.search(
            query=query,
            num_results=max_results,
            include_content=True
        )
        
        # Display results
        if not results:
            print(f"{get_time()} No results found.")
            return False
        
        print(f"{get_time()} Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Title: {result.get('title', 'Unknown')}")
            print(f"URL: {result.get('url', '')}")
            print(f"Source: {result.get('source', 'unknown')}")
            
            # Check if content was extracted
            content = result.get('content', '')
            extraction_method = result.get('extraction_method', 'none')
            
            print(f"Extraction method: {extraction_method}")
            if content:
                content_preview = content[:200] + ("..." if len(content) > 200 else "")
                print(f"Content preview: {content_preview}")
                print(f"Content length: {len(content)} characters")
            else:
                print("No content extracted")
        
        print(f"\n{get_time()} Search complete.")
        return True
        
    except Exception as e:
        print(f"{get_time()} Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_url(url):
    """Test direct URL extraction with OCR."""
    try:
        # Import our modules
        from web_ocr import WebOCRExtractor, extract_page_content_sync
        
        print(f"{get_time()} Initializing OCR extractor...")
        ocr_extractor = WebOCRExtractor(
            temp_dir="./temp",
            ocr_language="eng"
        )
        
        # Extract content
        print(f"{get_time()} Extracting content from: {url}")
        result = extract_page_content_sync(ocr_extractor, url)
        
        # Display result
        print(f"\n--- Result ---")
        print(f"Title: {result.get('title', 'Unknown')}")
        print(f"URL: {result.get('url', '')}")
        
        # Check for content sections
        sections = result.get('content_sections', [])
        if sections:
            print(f"Extracted {len(sections)} content sections")
            for i, section in enumerate(sections[:3]):  # Show first 3 sections
                section_preview = section[:100] + ("..." if len(section) > 100 else "")
                print(f"\nSection {i+1} preview: {section_preview}")
            
            if len(sections) > 3:
                print(f"\n... and {len(sections) - 3} more sections")
        else:
            print("No content sections extracted")
        
        # Check full content
        full_content = result.get('full_content', '')
        if full_content:
            print(f"\nFull content length: {len(full_content)} characters")
            preview = full_content[:200] + ("..." if len(full_content) > 200 else "")
            print(f"Content preview: {preview}")
        
        print(f"\n{get_time()} Extraction complete.")
        
        # Ask if user wants to save content to file
        save = input("Save content to file? (y/n): ").lower().strip()
        if save == 'y':
            filename = f"content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Source: {result.get('title', 'Unknown')}\n")
                f.write(f"URL: {result.get('url', '')}\n\n")
                f.write(full_content)
            print(f"{get_time()} Content saved to: {filename}")
        
        return True
        
    except Exception as e:
        print(f"{get_time()} Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def ensure_dependencies():
    """Ensure that all required dependencies are installed."""
    try:
        missing = []
        
        # Check for pyppeteer
        try:
            import pyppeteer
        except ImportError:
            missing.append("pyppeteer")
        
        # Check for pytesseract
        try:
            import pytesseract
        except ImportError:
            missing.append("pytesseract")
        
        # Check for PIL
        try:
            from PIL import Image
        except ImportError:
            missing.append("Pillow")
        
        if missing:
            print(f"{get_time()} Missing dependencies: {', '.join(missing)}")
            install = input("Install missing dependencies? (y/n): ").lower().strip()
            if install == 'y':
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
                print(f"{get_time()} Dependencies installed. Please restart the script.")
                return False
            else:
                print(f"{get_time()} Cannot continue without required dependencies.")
                return False
        
        return True
        
    except Exception as e:
        print(f"{get_time()} Error checking dependencies: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test OCR-based web content extraction")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Test Google search with OCR extraction")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--api-key", help="Google API key (defaults to GOOGLE_API_KEY env variable)")
    search_parser.add_argument("--cx-id", help="Google Custom Search Engine ID (defaults to GOOGLE_CX_ID env variable)")
    search_parser.add_argument("--max-results", type=int, default=5, help="Maximum number of results to return")
    
    # URL command
    url_parser = subparsers.add_parser("url", help="Test direct URL extraction with OCR")
    url_parser.add_argument("url", help="URL to extract content from")
    
    args = parser.parse_args()
    
    # Create temp and cache directories
    os.makedirs("./temp", exist_ok=True)
    os.makedirs("./cache", exist_ok=True)
    
    # Ensure dependencies are installed
    if not ensure_dependencies():
        return 1
    
    if args.command == "search":
        # Get API key and CX ID from environment if not provided
        api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
        cx_id = args.cx_id or os.environ.get("GOOGLE_CX_ID")
        
        if not api_key or not cx_id:
            print(f"{get_time()} Google API credentials not found.")
            print("Please set GOOGLE_API_KEY and GOOGLE_CX_ID environment variables or provide them as arguments.")
            return 1
        
        # Run search test
        if not test_google_search(args.query, api_key, cx_id, args.max_results):
            return 1
            
    elif args.command == "url":
        # Run URL test
        if not test_direct_url(args.url):
            return 1
            
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())