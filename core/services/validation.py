"""
Citation Validation Service.

Implements Schema #3 from ARCHITECTURE.md:
    AI Citation → Fuzzy Match Check → Coordinate Extraction → Verification Status

This service verifies that AI-generated quotes actually exist in the source
PDF pages and retrieves their coordinates for frontend highlighting.

Verification Statuses:
    - VERIFIED: Exact match found, coordinates available (Green flag)
    - LIKELY: Fuzzy match confirmed but exact search failed (Yellow flag)  
    - UNVERIFIED: No reliable match found, possible hallucination (Red flag)
"""

import logging
from typing import List, Tuple

import fitz  # PyMuPDF
from rapidfuzz import fuzz

from core.models import Page


logger = logging.getLogger(__name__)

# Validation thresholds
FUZZY_MATCH_THRESHOLD = 85  # Minimum score to consider a match valid


class ValidationError(Exception):
    """Raised when validation process encounters an error."""
    pass


class ValidationService:
    """
    Service for verifying AI-generated citations against source documents.
    
    Uses a two-step verification process:
    1. Fuzzy matching to confirm text existence (catches minor variations)
    2. Exact search for coordinate extraction (for precise highlighting)
    """

    @staticmethod
    def repair_quote_anchor(page_text: str, dirty_anchor: str) -> str:
        """
        Repair an AI-generated anchor by snapping it to an exact substring from page text.

        Uses fuzzy substring alignment to locate the most likely matching region in
        the source page. If alignment score is strong, returns the exact fragment from
        page_text so downstream coordinate search works reliably.

        Args:
            page_text: Source page text to align against.
            dirty_anchor: Potentially noisy quote from the model.

        Returns:
            Repaired quote anchor (exact source substring) or original dirty_anchor.
        """
        if not page_text or not dirty_anchor:
            return dirty_anchor

        anchor = " ".join(dirty_anchor.split()).strip()
        source_text = page_text
        source_lower = source_text.lower()
        anchor_lower = anchor.lower()

        if not anchor_lower:
            return dirty_anchor

        try:
            alignment = fuzz.partial_ratio_alignment(anchor_lower, source_lower)
        except Exception as e:
            logger.debug(f"Anchor repair alignment failed: {e}")
            return dirty_anchor

        if not alignment or alignment.score < FUZZY_MATCH_THRESHOLD:
            return dirty_anchor

        start = max(0, alignment.dest_start)
        end = min(len(source_text), alignment.dest_end)

        if start >= end:
            return dirty_anchor

        # Expand to clean token boundaries to avoid clipped words.
        while start > 0 and source_text[start - 1].isalnum():
            start -= 1
        while end < len(source_text) and source_text[end].isalnum():
            end += 1

        repaired = source_text[start:end].strip()
        return repaired if repaired else dirty_anchor
    
    @staticmethod
    def verify_citation(page_id: int, quote_anchor: str) -> dict:
        """
        Verify if a quote exists on a specific PDF page and get its coordinates.
        
        This is the core "Hallucination Check" - it validates that AI-generated
        citations actually exist in the source material.
        
        Args:
            page_id: Primary key of the Page to search in.
            quote_anchor: The quote text to verify (from AI response).
            
        Returns:
            Dictionary with verification results:
            {
                "status": "VERIFIED" | "LIKELY" | "UNVERIFIED",
                "score": int,  # Fuzzy match score (0-100)
                "bboxes": [[x0, y0, x1, y1], ...]  # Bounding boxes for highlighting
            }
            
        Raises:
            Page.DoesNotExist: If no page exists with the given ID.
            ValidationError: If PDF cannot be opened.
        """
        result = {
            "status": "UNVERIFIED",
            "score": 0,
            "bboxes": []
        }
        
        # Handle empty or whitespace-only quotes
        if not quote_anchor or not quote_anchor.strip():
            logger.warning("Empty quote_anchor provided")
            return result
        
        # Step 1: Fetch the Page from database
        try:
            page = Page.objects.select_related('document').get(pk=page_id)
        except Page.DoesNotExist:
            logger.error(f"Page with ID {page_id} not found")
            raise
        
        # Normalize the quote for comparison
        quote_normalized = quote_anchor.lower().strip()
        
        # Step 2: Fuzzy Search - The Hallucination Filter
        # Compare against normalized page text for case-insensitive matching
        fuzzy_score = fuzz.partial_ratio(quote_normalized, page.text_norm)
        result["score"] = fuzzy_score
        
        logger.debug(
            f"Fuzzy match score for page {page_id}: {fuzzy_score} "
            f"(threshold: {FUZZY_MATCH_THRESHOLD})"
        )
        
        if fuzzy_score < FUZZY_MATCH_THRESHOLD:
            # Below threshold - likely a hallucination
            logger.info(
                f"Citation verification FAILED for page {page_id}: "
                f"score {fuzzy_score} < {FUZZY_MATCH_THRESHOLD}"
            )
            return result
        
        # Step 3: Coordinate Extraction - The Precision Step
        # Fuzzy match passed, now try to get exact coordinates from PDF
        bboxes = ValidationService._extract_coordinates(page, quote_anchor)
        
        if bboxes:
            # Exact match found - fully verified
            result["status"] = "VERIFIED"
            result["bboxes"] = bboxes
            logger.info(
                f"Citation VERIFIED for page {page_id}: "
                f"found {len(bboxes)} bounding box(es)"
            )
        else:
            # Fuzzy matched but exact search failed
            # This can happen due to minor OCR errors, hyphenation, etc.
            result["status"] = "LIKELY"
            logger.info(
                f"Citation LIKELY for page {page_id}: "
                f"fuzzy score {fuzzy_score} but no exact match"
            )
        
        return result
    
    @staticmethod
    def _extract_coordinates(page: Page, quote: str) -> List[List[float]]:
        """
        Extract bounding box coordinates for a quote from the PDF.
        
        Args:
            page: Page model instance.
            quote: The exact quote text to search for.
            
        Returns:
            List of bounding boxes as [x0, y0, x1, y1] coordinates.
            Empty list if no match found or PDF cannot be opened.
        """
        pdf_doc = None
        bboxes = []
        
        try:
            # Open the PDF file
            pdf_path = page.document.file.path
            pdf_doc = fitz.open(pdf_path)
            
            # Get the specific page (0-indexed in PyMuPDF, 1-indexed in our model)
            pdf_page = pdf_doc[page.page_number - 1]
            
            # Search for the quote text
            # search_for returns a list of Rect objects
            rects = pdf_page.search_for(quote, quads=False)
            
            if rects:
                # Convert Rect objects to list format [x0, y0, x1, y1]
                bboxes = [
                    [rect.x0, rect.y0, rect.x1, rect.y1]
                    for rect in rects
                ]
            else:
                # Try searching with normalized whitespace
                # Sometimes PDFs have different whitespace than expected
                normalized_quote = ' '.join(quote.split())
                if normalized_quote != quote:
                    rects = pdf_page.search_for(normalized_quote, quads=False)
                    bboxes = [
                        [rect.x0, rect.y0, rect.x1, rect.y1]
                        for rect in rects
                    ]
            
            logger.debug(f"Found {len(bboxes)} rectangles for quote on page {page.page_number}")
            
        except FileNotFoundError:
            logger.error(f"PDF file not found: {page.document.file.path}")
        except Exception as e:
            logger.error(f"Error extracting coordinates: {str(e)}")
        finally:
            if pdf_doc:
                pdf_doc.close()
        
        return bboxes
    
    @staticmethod
    def verify_multiple_citations(citations: List[Tuple[int, str]]) -> List[dict]:
        """
        Verify multiple citations in batch.
        
        Convenience method for verifying several citations at once.
        
        Args:
            citations: List of (page_id, quote_anchor) tuples.
            
        Returns:
            List of verification result dictionaries.
        """
        results = []
        for page_id, quote_anchor in citations:
            try:
                result = ValidationService.verify_citation(page_id, quote_anchor)
                result["page_id"] = page_id
                result["quote"] = quote_anchor
                results.append(result)
            except Page.DoesNotExist:
                results.append({
                    "page_id": page_id,
                    "quote": quote_anchor,
                    "status": "UNVERIFIED",
                    "score": 0,
                    "bboxes": [],
                    "error": "Page not found"
                })
        return results

