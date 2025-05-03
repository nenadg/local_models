"""
Web OCR integration for local LLM chat system.
Provides screenshot-based extraction and OCR processing.
"""

import os
import time
import asyncio
import re
import tempfile
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# For puppeteer/pyppeteer
import pyppeteer
from pyppeteer import launch

# For OCR
import pytesseract
from PIL import Image

class WebOCRExtractor:
    """
    Extracts web content using screenshots and OCR instead of HTML parsing.
    Integrates with existing similarity enhancement system.
    """
    
    def __init__(
        self,
        memory_manager=None,
        similarity_enhancement_factor: float = 0.3,
        temp_dir: Optional[str] = None,
        chrome_executable: Optional[str] = None,
        ocr_language: str = "eng",
        screenshot_width: int = 1280,
        screenshot_height: int = 8000  # Very tall to capture most of the page
    ):
        """
        Initialize the OCR extractor.
        
        Args:
            memory_manager: Optional MemoryManager for storing results
            similarity_enhancement_factor: Factor for similarity enhancement
            temp_dir: Directory for storing temporary files
            chrome_executable: Path to Chrome executable (if not using bundled)
            ocr_language: Language for OCR
            screenshot_width: Width of screenshot
            screenshot_height: Maximum height of screenshot
        """
        self.memory_manager = memory_manager
        self.similarity_enhancement_factor = similarity_enhancement_factor
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.chrome_executable = chrome_executable
        self.ocr_language = ocr_language
        self.screenshot_width = screenshot_width
        self.screenshot_height = screenshot_height
        
        # Create temporary directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Cache for recent extracted content
        self._content_cache = {}
        self._browser = None
        
    async def init_browser(self):
        """Initialize browser once and reuse."""
        if self._browser is None:
            launch_args = {
                'headless': True,
                'args': [
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--disable-gpu',
                    f'--window-size={self.screenshot_width},{1000}'  # Initial window size
                ]
            }
            
            if self.chrome_executable:
                launch_args['executablePath'] = self.chrome_executable
                
            self._browser = await launch(**launch_args)
            
        return self._browser
    
    async def close_browser(self):
        """Close the browser if it's open."""
        if self._browser:
            await self._browser.close()
            self._browser = None
    
    def get_time(self) -> str:
        """Get formatted timestamp for logging."""
        return datetime.now().strftime("[%d/%m/%y %H:%M:%S]") + ' [WebOCR]'
    
    async def extract_page_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a web page using screenshot and OCR.
        
        Args:
            url: Page URL to extract
            
        Returns:
            Dictionary with extracted content and metadata
        """
        # Check cache first
        cache_key = f"ocr_{hashlib.md5(url.encode()).hexdigest()}"
        if cache_key in self._content_cache:
            print(f"{self.get_time()} Using cached OCR content for: {url}")
            return self._content_cache[cache_key]
        
        print(f"{self.get_time()} Capturing and processing: {url}")
        
        # Generate a unique filename for the screenshot
        screenshot_path = os.path.join(
            self.temp_dir, 
            f"screenshot_{hashlib.md5(url.encode()).hexdigest()}.png"
        )
        
        try:
            # Initialize browser if needed
            browser = await self.init_browser()
            
            # Create a new page
            page = await browser.newPage()
            
            # Set viewport size
            await page.setViewport({
                'width': self.screenshot_width,
                'height': 1000  # Initial height
            })
            
            print(f"{self.get_time()} [OCR] Seaching site: {url}")
            # Navigate to URL with timeout
            await page.goto(url, {'timeout': 30000, 'waitUntil': 'networkidle0'})
            
            # Wait for content to load
            await asyncio.sleep(2)
            
            # Get page title
            title = await page.title()
            
            # Get page scroll height for full screenshot
            scroll_height = await page.evaluate('document.body.scrollHeight')
            actual_height = min(scroll_height, self.screenshot_height)
            
            # Resize viewport to capture more content
            await page.setViewport({
                'width': self.screenshot_width,
                'height': actual_height
            })
            
            # Take screenshot of full page
            await page.screenshot({'path': screenshot_path, 'fullPage': True})
            
            # Close the page
            await page.close()

            await self.cleanup()
            
            # Process screenshot with OCR
            content_sections = self._process_screenshot_with_ocr(screenshot_path)
            
            # Create result
            result = {
                'url': url,
                'title': title,
                'content_sections': content_sections,
                'full_content': "\n\n".join(content_sections),
                'timestamp': time.time()
            }
            
            # Cache result
            self._content_cache[cache_key] = result
            
            # Return the result
            return result
            
        except Exception as e:
            print(f"{self.get_time()} Error capturing/processing {url}: {str(e)}")
            return {
                'url': url,
                'title': "Error capturing page",
                'content_sections': [],
                'full_content': f"Error: {str(e)}",
                'timestamp': time.time()
            }
        finally:
            # Cleanup screenshot file
            if os.path.exists(screenshot_path):
                try:
                    os.remove(screenshot_path)
                except Exception:
                    pass
    
    def _process_screenshot_with_ocr(self, screenshot_path: str) -> List[str]:
        """
        Process a screenshot with OCR to extract text sections.

        Args:
            screenshot_path: Path to screenshot image

        Returns:
            List of text sections extracted from the image
        """
        try:
            # Open image
            img = Image.open(screenshot_path)

            # Run OCR
            text = pytesseract.image_to_string(img, lang=self.ocr_language)

            # Split into sections
            sections = self._split_text_into_sections(text)

            # Clean sections
            cleaned_sections = [self._clean_section(section) for section in sections if section.strip()]

            # Structure sections better by adding semantic markers
            structured_sections = []
            for i, section in enumerate(cleaned_sections):
                # Try to identify section type
                if i == 0 and len(section) < 100:
                    # Likely a title or header
                    structured_sections.append(f"Title: {section}")
                elif re.match(r'^(\d+\.|\*|\-)\s', section):
                    # List item
                    structured_sections.append(f"List item: {section}")
                elif len(section.split()) < 15 and section.endswith(':'):
                    # Likely a subtitle or heading
                    structured_sections.append(f"Heading: {section}")
                else:
                    # Regular paragraph
                    structured_sections.append(f"Paragraph: {section}")

            return structured_sections if structured_sections else cleaned_sections

        except Exception as e:
            print(f"{self.get_time()} OCR processing error: {e}")
            return []
    
    def _split_text_into_sections(self, text: str) -> List[str]:
        """
        Split OCR text into logical sections based on whitespace.
        
        Args:
            text: Raw OCR text
            
        Returns:
            List of text sections
        """
        # Split on double newlines (paragraph breaks)
        sections = re.split(r'\n\s*\n', text)
        
        # Filter out very short sections (likely noise)
        sections = [s.strip() for s in sections if len(s.strip()) > 20]
        
        return sections
    
    def _clean_section(self, section: str) -> str:
        """
        Clean an OCR-extracted text section.
        
        Args:
            section: Raw text section
            
        Returns:
            Cleaned text section
        """
        # Replace multiple spaces with single space
        section = re.sub(r'\s+', ' ', section)
        
        # Replace multiple newlines with single newline
        section = re.sub(r'\n+', '\n', section)
        
        # Remove common OCR artifacts
        section = re.sub(r'[^\w\s\.,;:!?\-\'\"()%$#@&*+=/\\<>[\]{}]', '', section)
        
        return section.strip()
    
    async def score_sections_by_similarity(self, sections: List[str], query: str) -> List[Tuple[str, float]]:
        """
        Score text sections by similarity to query using memory manager.
        
        Args:
            sections: List of text sections
            query: Query to compare against
            
        Returns:
            List of (section, score) tuples, sorted by relevance
        """
        if not self.memory_manager or not hasattr(self.memory_manager, 'embedding_function'):
            # Can't score without memory manager
            return [(section, 0.5) for section in sections]
        
        try:
            # Generate query embedding
            query_embedding = self.memory_manager.embedding_function(query)
            
            # Generate section embeddings
            section_embeddings = []
            for section in sections:
                try:
                    embedding = self.memory_manager.embedding_function(section)
                    section_embeddings.append(embedding)
                except Exception:
                    # Skip sections that fail embedding
                    section_embeddings.append(None)
            
            # Score sections
            scored_sections = []
            for i, (section, embedding) in enumerate(zip(sections, section_embeddings)):
                if embedding is None:
                    # Skip sections without embeddings
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(query_embedding, embedding)
                
                # Apply enhancement
                enhanced_similarity = self._enhance_similarity(similarity)
                
                scored_sections.append((section, enhanced_similarity))
            
            # Sort by score
            scored_sections.sort(key=lambda x: x[1], reverse=True)
            
            return scored_sections
            
        except Exception as e:
            print(f"{self.get_time()} Error scoring sections: {str(e)}")
            # Return unscored sections as fallback
            return [(section, 0.5) for section in sections]
    
    def _calculate_similarity(self, embedding1: Any, embedding2: Any) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        import numpy as np
        
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
            
        embedding1_normalized = embedding1 / norm1
        embedding2_normalized = embedding2 / norm2
        
        # Calculate cosine similarity
        return float(np.dot(embedding1_normalized, embedding2_normalized))
    
    def _enhance_similarity(self, similarity: float) -> float:
        """
        Apply non-linear enhancement to similarity scores.
        
        Args:
            similarity: Raw similarity score (0.0-1.0)
            
        Returns:
            Enhanced similarity score
        """
        # Skip if no enhancement requested
        if self.similarity_enhancement_factor <= 0:
            return similarity
        
        # Apply non-linear enhancement
        if similarity > 0.6:
            # Boost high similarities (more confident matches)
            boost = (similarity - 0.6) * self.similarity_enhancement_factor * 2.0
            enhanced = min(1.0, similarity + boost)
        elif similarity < 0.4:
            # Reduce low similarities (less confident matches)
            reduction = (0.4 - similarity) * self.similarity_enhancement_factor * 2.0
            enhanced = max(0.0, similarity - reduction)
        else:
            # Middle range - moderate effect
            deviation = (similarity - 0.5) * self.similarity_enhancement_factor
            enhanced = 0.5 + deviation
        
        return enhanced
    
    async def add_to_memory(self, url: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Extract content and add top sections to memory.

        Args:
            url: URL to process
            query: Original query for relevance scoring
            top_k: Number of top sections to add

        Returns:
            List of added memory items
        """
        if not self.memory_manager:
            print(f"{self.get_time()} No memory manager available")
            return []

        # Extract content
        result = await self.extract_page_content(url)

        if not result['content_sections']:
            return []

        # Score sections
        scored_sections = await self.score_sections_by_similarity(
            result['content_sections'], query
        )

        # Add top sections to memory
        added_items = []
        for section, score in scored_sections[:top_k]:
            if score < 0.3:  # Skip very low relevance sections
                continue

            # Create metadata with better structure for retrieval
            metadata = {
                'source': 'web_ocr',
                'url': url,
                'title': result['title'],
                'domain': self._get_domain(url),
                'query': query,
                'relevance_score': score,
                'timestamp': time.time(),
                'content_type': 'extracted_text',
                'extraction_method': 'ocr'
            }

            # Format content to be more structured and usable in responses
            formatted_content = f"From {result['title']} ({self._get_domain(url)}): {section}"

            # Add to memory
            try:
                item_id = self.memory_manager.add(
                    content=formatted_content,
                    metadata=metadata
                )

                if item_id:
                    added_items.append({
                        'id': item_id,
                        'content': formatted_content,
                        'score': score,
                        'metadata': metadata
                    })
                    print(f"{self.get_time()} Added section to memory (score: {score:.2f})")
            except Exception as e:
                print(f"{self.get_time()} Error adding to memory: {e}")

        return added_items

    def _get_domain(self, url: str) -> str:
        """
        Extract domain name from URL.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Domain name
        """
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
                
            return domain
            
        except Exception:
            return "unknown"
            
    def format_for_context(self, result: Dict[str, Any], query: str, 
                          max_sections: int = 3) -> str:
        """
        Format OCR results for inclusion in context.
        
        Args:
            result: OCR result dictionary
            query: Original query
            max_sections: Maximum number of sections to include
            
        Returns:
            Formatted context string
        """
        if not result or not result.get('content_sections'):
            return ""
        
        # Build formatted output
        output = "WEB OCR RESULTS:\n\n"
        
        # Add title and URL
        output += f"Source: {result.get('title', 'Unknown Title')}\n"
        output += f"URL: {result.get('url', '')}\n\n"
        
        # Add content sections with default scores if we can't run the async function
        sections = result.get('content_sections', [])

        # For synchronous context, use a simpler scoring or fixed scores
        for i, section in enumerate(sections[:max_sections]):
            # Default score for synchronous format_for_context
            score = 0.5
            output += f"[Relevance: {score:.2f}] {section}\n\n"

        output += "Use the information above to help answer the query if relevant.\n"
        return output

    async def cleanup(self):
        """Clean up resources."""
        await self.close_browser()


# Fixed helper functions for synchronous code
def run_async(coroutine):
    """Run an async function from synchronous code, with proper event loop creation."""
    try:
        # Try to get the existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        # Create a new event loop if needed
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(coroutine)
    finally:
        # Don't close the loop as it may be reused
        pass

def extract_page_content_sync(extractor, url):
    """Synchronous wrapper for extract_page_content."""
    return run_async(extractor.extract_page_content(url))

def add_to_memory_sync(extractor, url, query, top_k=3):
    """Synchronous wrapper for add_to_memory."""
    return run_async(extractor.add_to_memory(url, query, top_k))

def cleanup_sync(extractor):
    """Synchronous wrapper for cleanup."""
    return run_async(extractor.cleanup())

def score_sections_sync(extractor, sections, query):
    """Synchronous wrapper for score_sections_by_similarity."""
    return run_async(extractor.score_sections_by_similarity(sections, query))

def format_for_context_with_scoring(extractor, result, query, max_sections=3):
    """
    Enhanced synchronous formatting with section scoring.
    This uses the synchronous scoring wrapper to get proper scored sections.
    """
    if not result or not result.get('content_sections'):
        return ""

    # Build formatted output
    output = "WEB OCR RESULTS:\n\n"

    # Add title and URL
    output += f"Source: {result.get('title', 'Unknown Title')}\n"
    output += f"URL: {result.get('url', '')}\n\n"

    # Get scored sections using the synchronous wrapper
    sections = result.get('content_sections', [])
    scored_sections = score_sections_sync(extractor, sections, query)

    # Add sections with scores
    for i, (section, score) in enumerate(scored_sections[:max_sections]):
        if score < 0.3:  # Skip very low relevance
            continue

        # Format section with score
        output += f"[Relevance: {score:.2f}] {section}\n\n"

    output += "Use the information above to help answer the query if relevant.\n"
    return output