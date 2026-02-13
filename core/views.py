"""
Core API Views for UniSave.

This module provides the REST API endpoints that orchestrate the full
RAG (Retrieval-Augmented Generation) pipeline:
    1. Chunk Retrieval (keyword search)
    2. AI Generation (Gemini)
    3. Citation Validation (grounding check)
"""

import logging
import re
from typing import List, Tuple

from django.shortcuts import get_object_or_404

from rest_framework import status
from rest_framework.generics import ListAPIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import ChatMessage, Chunk, Document, Page
from core.serializers import (
    AnalyzeDocumentRequestSerializer,
    ChatMessageSerializer,
    DocumentSerializer,
)
from core.services import GeminiService, ValidationService
from core.services.ingestion import process_document, DocumentProcessingError


logger = logging.getLogger(__name__)

# Configuration
TOP_CHUNKS_COUNT = 5  # Number of top chunks to use for context
MIN_KEYWORD_LENGTH = 3  # Minimum length for a word to be considered a keyword


class DocumentUploadView(APIView):
    """
    API endpoint for uploading and listing PDF documents.
    
    POST /api/documents/ - Upload a new PDF document
    GET  /api/documents/ - List all documents (newest first)
    
    POST Request:
        Content-Type: multipart/form-data
        - file: PDF file (required)
        - title: Document title (optional, defaults to filename)
    
    POST Response (201 Created):
        {
            "id": 1,
            "title": "My Document",
            "file": "documents/my_document.pdf",
            "file_url": "http://localhost:8000/media/documents/my_document.pdf",
            "uploaded_at": "2024-01-15T10:30:00Z",
            "processing_status": "success" | "warning",
            "processing_message": "..."
        }
    
    GET Response (200 OK):
        [
            {
                "id": 1,
                "title": "My Document",
                "file": "documents/my_document.pdf",
                "file_url": "http://localhost:8000/media/documents/my_document.pdf",
                "uploaded_at": "2024-01-15T10:30:00Z"
            },
            ...
        ]
    """
    
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request):
        """Handle document file upload (PDF, DOCX, or PPTX)."""
        
        # Validate file presence
        if 'file' not in request.FILES:
            return Response(
                {"error": "No file provided. Please upload a document file."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        uploaded_file = request.FILES['file']
        file_name_lower = uploaded_file.name.lower()
        
        # Validate file type
        allowed_extensions = ['.pdf', '.docx', '.pptx']
        if not any(file_name_lower.endswith(ext) for ext in allowed_extensions):
            return Response(
                {"error": "Only PDF, DOCX, and PPTX files are allowed."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get or generate title
        title = request.data.get('title', '').strip()
        if not title:
            # Use filename without extension as title
            title = uploaded_file.name.rsplit('.', 1)[0]
        
        # Use serializer to create document (this triggers validate_file)
        serializer = DocumentSerializer(
            data={'title': title, 'file': uploaded_file},
            context={'request': request}
        )
        
        # Validate and save (validate_file will be called here)
        if not serializer.is_valid():
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )
        
        document = serializer.save()
        
        logger.info(f"Document uploaded: {document.title} (ID: {document.id})")
        
        # Process the PDF (extract text, create chunks)
        processing_status = "success"
        processing_message = "Document uploaded and processed successfully."
        
        try:
            result = process_document(document.id)
            processing_message = (
                f"Processed {result['pages_processed']} pages, "
                f"created {result['chunks_created']} chunks."
            )
            logger.info(f"PDF processing complete: {processing_message}")
            
        except DocumentProcessingError as e:
            # Processing failed, but document is saved
            processing_status = "warning"
            processing_message = f"Document saved but processing failed: {str(e)}"
            logger.warning(processing_message)
            
        except Exception as e:
            # Unexpected error
            processing_status = "warning"
            processing_message = f"Document saved but an unexpected error occurred: {str(e)}"
            logger.error(f"Unexpected error during PDF processing: {e}")
        
        # Serialize response
        response_data = serializer.data
        response_data['processing_status'] = processing_status
        response_data['processing_message'] = processing_message
        
        return Response(response_data, status=status.HTTP_201_CREATED)
    
    def get(self, request):
        """
        List all documents, ordered by newest first.
        
        Returns a list of all uploaded documents for building the sidebar history.
        """
        documents = Document.objects.all().order_by('-uploaded_at')
        serializer = DocumentSerializer(documents, many=True, context={'request': request})
        return Response(serializer.data)


class DocumentDetailView(APIView):
    """
    API endpoint for document detail operations.
    
    DELETE /api/documents/<pk>/
    
    Deletes a document and all associated data:
    - Document record
    - All Pages (cascade)
    - All Chunks (cascade)
    - All ChatMessages (cascade)
    - PDF file from storage
    
    Response (204 No Content): Success
    Response (404 Not Found): Document doesn't exist
    """
    
    def delete(self, request, pk):
        """
        Delete a document by ID.
        
        Args:
            request: HTTP request object.
            pk: Primary key (ID) of the document to delete.
            
        Returns:
            204 No Content on success.
            404 Not Found if document doesn't exist.
        """
        try:
            document = Document.objects.get(pk=pk)
        except Document.DoesNotExist:
            return Response(
                {"error": f"Document with ID {pk} not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Log before deletion
        document_title = document.title
        document_id = document.id
        
        # Delete the document (cascades to Pages, Chunks, ChatMessages)
        document.delete()
        
        logger.info(f"Document deleted: '{document_title}' (ID: {document_id})")
        
        return Response(status=status.HTTP_204_NO_CONTENT)


class ChunkRetriever:
    """
    Simple keyword-based chunk retrieval engine.
    
    Scores chunks based on keyword frequency matching.
    This is the "search engine" component of the RAG pipeline.
    """
    
    # Common stop words to exclude from keyword matching
    STOP_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
        'because', 'until', 'while', 'although', 'though', 'after', 'before',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
        'it', 'its', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she',
        'her', 'hers', 'herself', 'they', 'them', 'their', 'theirs', 'themselves',
    }
    
    @classmethod
    def extract_keywords(cls, text: str) -> List[str]:
        """
        Extract meaningful keywords from text.
        
        Removes stop words and short words, returns lowercase keywords.
        
        Args:
            text: Input text (e.g., user question).
            
        Returns:
            List of lowercase keyword strings.
        """
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = [
            word for word in words
            if word not in cls.STOP_WORDS and len(word) >= MIN_KEYWORD_LENGTH
        ]
        
        return keywords
    
    @classmethod
    def score_chunk(cls, chunk: Chunk, keywords: List[str]) -> int:
        """
        Score a chunk based on keyword frequency.
        
        Args:
            chunk: Chunk object to score.
            keywords: List of keywords to search for.
            
        Returns:
            Integer score (higher = more relevant).
        """
        # Use normalized text for case-insensitive matching
        chunk_text_lower = chunk.chunk_text.lower()
        
        score = 0
        for keyword in keywords:
            # Count occurrences of each keyword
            score += chunk_text_lower.count(keyword)
        
        return score
    
    @classmethod
    def retrieve_top_chunks(
        cls,
        doc_id: int = None,
        doc_ids: List[int] = None,
        question: str = "",
        top_n: int = TOP_CHUNKS_COUNT
    ) -> List[Chunk]:
        """
        Retrieve the most relevant chunks for a question.
        
        Supports both single document and multiple documents (Knowledge Library).
        
        Args:
            doc_id: Single document ID to search within (legacy support).
            doc_ids: List of document IDs to search across (preferred).
            question: User's question.
            top_n: Number of top chunks to return.
            
        Returns:
            List of Chunk objects, sorted by relevance (highest first).
        """
        # Determine which document IDs to use
        if doc_ids:
            document_ids = doc_ids
        elif doc_id:
            document_ids = [doc_id]
        else:
            logger.warning("No document ID(s) provided")
            return []
        
        # Get all chunks for the specified document(s)
        chunks = Chunk.objects.filter(
            page__document_id__in=document_ids
        ).select_related('page', 'page__document')
        
        if not chunks.exists():
            return []
        
        # Extract keywords from question
        keywords = cls.extract_keywords(question)
        
        if not keywords:
            # If no keywords extracted, return first N chunks
            logger.warning("No keywords extracted from question, returning first chunks")
            return list(chunks[:top_n])
        
        logger.debug(f"Searching with keywords: {keywords}")
        
        # Score each chunk
        scored_chunks: List[Tuple[int, Chunk]] = []
        for chunk in chunks:
            score = cls.score_chunk(chunk, keywords)
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Sort by score (descending) and return top N
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [chunk for score, chunk in scored_chunks[:top_n]]
        
        # If we didn't find enough matches, pad with sequential chunks
        if len(top_chunks) < top_n:
            existing_ids = {c.id for c in top_chunks}
            for chunk in chunks:
                if chunk.id not in existing_ids:
                    top_chunks.append(chunk)
                    if len(top_chunks) >= top_n:
                        break
        
        doc_ids_str = ', '.join(map(str, document_ids))
        logger.info(f"Retrieved {len(top_chunks)} chunks from document(s) [{doc_ids_str}]")
        return top_chunks


class AnalyzeDocumentView(APIView):
    """
    Main API endpoint for question-answering with citation validation.
    
    POST /api/analyze/
    
    This view orchestrates the complete RAG pipeline:
    1. Retrieval: Find relevant chunks using keyword search
    2. Generation: Get AI response with claims from Gemini
    3. Validation: Verify each citation against source PDF
    
    Request Body:
        {
            "document_id": int,     # Single document ID (legacy support)
            "document_ids": [int],  # List of document IDs (Knowledge Library mode)
            "question": str,        # User's question
            "mode": str             # Optional: "strict" (default) or "tutor"
        }
        
    Note: If both `document_id` and `document_ids` are provided, `document_ids` takes precedence.
    
    Response:
        {
            "answer": str,                  # AI-generated answer
            "claims": [                     # Verified claims with coordinates
                {
                    "claim": str,
                    "quote_anchor": str,
                    "page_hint": int,
                    "status": "VERIFIED" | "LIKELY" | "UNVERIFIED",
                    "score": int,
                    "bboxes": [[x0, y0, x1, y1], ...]
                }
            ],
            "chunks_used": int,             # Number of context chunks used
            "mode": str,                    # Mode used: "strict" or "tutor"
            "error": str (optional)         # Error message if any
        }
    """
    
    def post(self, request):
        """Handle POST request for document analysis."""
        
        # ─────────────────────────────────────────────────────────────
        # Step 1: VALIDATE REQUEST DATA using Serializer
        # ─────────────────────────────────────────────────────────────
        serializer = AnalyzeDocumentRequestSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get validated data
        validated_data = serializer.validated_data
        document_ids = validated_data.get('document_ids', [])
        document_id = validated_data.get('document_id')
        question = validated_data.get('question')
        mode = validated_data.get('mode', 'strict')
        
        # ─────────────────────────────────────────────────────────────
        # Step 2: DETERMINE DOCUMENT IDs for AI context
        # ─────────────────────────────────────────────────────────────
        # Determine which document IDs to use for AI context
        if document_ids:
            # Multiple documents mode (Knowledge Library)
            target_doc_ids = document_ids
        elif document_id:
            # Single document mode - wrap in list for consistency
            target_doc_ids = [document_id]
        else:
            # This shouldn't happen due to serializer validation, but safety check
            return Response(
                {"error": "Either document_id or document_ids is required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # ─────────────────────────────────────────────────────────────
        # Step 3: DETERMINE PRIMARY DOCUMENT (for ChatMessage FK)
        # ─────────────────────────────────────────────────────────────
        primary_doc_id = target_doc_ids[0]  # First document in list
        primary_doc = get_object_or_404(Document, pk=primary_doc_id)
        
        # Verify all other documents exist (if multiple)
        if len(target_doc_ids) > 1:
            documents = Document.objects.filter(pk__in=target_doc_ids)
            found_ids = set(documents.values_list('id', flat=True))
            missing_ids = set(target_doc_ids) - found_ids
            
            if missing_ids:
                return Response(
                    {"error": f"Document(s) with ID(s) {list(missing_ids)} not found"},
                    status=status.HTTP_404_NOT_FOUND
                )
            document_titles = ', '.join(doc.title for doc in documents)
        else:
            document_titles = primary_doc.title
        
        logger.info(
            f"Analyzing {len(target_doc_ids)} document(s) [{document_titles}] "
            f"with question: {question[:100]}... (mode={mode})"
        )
        
        # ─────────────────────────────────────────────────────────────
        # Step 3: SAVE USER MESSAGE - Chat History
        # ─────────────────────────────────────────────────────────────
        ChatMessage.objects.create(
            document=primary_doc,  # Associate with primary document
            sender='user',
            content=question,
            claims=[]
        )
        
        # ─────────────────────────────────────────────────────────────
        # Step 4: RETRIEVAL - The Search Engine (Multi-Document)
        # ─────────────────────────────────────────────────────────────
        top_chunks = ChunkRetriever.retrieve_top_chunks(
            doc_ids=target_doc_ids,  # Pass ALL document IDs for AI context
            question=question,
            top_n=TOP_CHUNKS_COUNT
        )
        
        if not top_chunks:
            doc_count = len(target_doc_ids)
            doc_text = "document" if doc_count == 1 else "documents"
            return Response({
                "answer": f"I couldn't find any relevant information in the selected {doc_text}. "
                         "Please make sure the document(s) have been processed.",
                "claims": [],
                "chunks_used": 0
            })
        
        # ─────────────────────────────────────────────────────────────
        # Step 5: GENERATION - The Brain (Gemini AI)
        # ─────────────────────────────────────────────────────────────
        ai_response = GeminiService.ask_with_context(
            question=question,
            context_chunks=top_chunks,  # Chunks from ALL documents
            mode=mode
        )
        
        # Check for AI errors
        if "error" in ai_response and not ai_response.get("answer"):
            return Response({
                "answer": "An error occurred while generating the response. Please try again.",
                "claims": [],
                "chunks_used": len(top_chunks),
                "error": ai_response.get("error")
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # ─────────────────────────────────────────────────────────────
        # Step 6: VALIDATION - The Lie Detector
        # ─────────────────────────────────────────────────────────────
        verified_claims = []
        
        for claim in ai_response.get("claims", []):
            page_hint = claim.get("page_hint", 1)
            quote_anchor = claim.get("quote_anchor", "")
            
            # Find the page by page_number within the target documents
            # Try each document until we find a match
            page = None
            found_doc_id = None
            
            for doc_id in target_doc_ids:
                try:
                    page = Page.objects.get(
                        document_id=doc_id,
                        page_number=page_hint
                    )
                    found_doc_id = doc_id
                    break
                except Page.DoesNotExist:
                    continue
            
            if page:
                # Repair minor OCR/spacing drift so quote exists in page text verbatim.
                repaired_quote_anchor = ValidationService.repair_quote_anchor(
                    page_text=page.text_raw or page.text_norm or "",
                    dirty_anchor=quote_anchor
                )

                # Verify the citation
                verification = ValidationService.verify_citation(
                    page_id=page.id,
                    quote_anchor=repaired_quote_anchor
                )
                
                # Merge claim with verification results
                verified_claim = {
                    "claim": claim.get("claim", ""),
                    "quote_anchor": repaired_quote_anchor,
                    "page_hint": page_hint,
                    "page_id": page.id,
                    "document_id": found_doc_id,  # Include document ID for multi-doc mode
                    "status": verification["status"],
                    "score": verification["score"],
                    "bboxes": verification["bboxes"]
                }
                
            else:
                # Page not found in any of the target documents - mark as unverified
                logger.warning(f"Page {page_hint} not found in any of documents {target_doc_ids}")
                verified_claim = {
                    "claim": claim.get("claim", ""),
                    "quote_anchor": quote_anchor,
                    "page_hint": page_hint,
                    "page_id": None,
                    "document_id": None,
                    "status": "UNVERIFIED",
                    "score": 0,
                    "bboxes": [],
                    "error": f"Page {page_hint} not found in any selected document"
                }
            
            verified_claims.append(verified_claim)
        
        # ─────────────────────────────────────────────────────────────
        # Step 7: SAVE AI MESSAGE - Chat History
        # ─────────────────────────────────────────────────────────────
        ChatMessage.objects.create(
            document=primary_doc,  # Associate with primary document
            sender='ai',
            content=ai_response.get("answer", ""),
            claims=verified_claims  # Save verified claims with status/bboxes
        )
        
        # ─────────────────────────────────────────────────────────────
        # Step 8: RESPONSE - Send to Frontend
        # ─────────────────────────────────────────────────────────────
        response_data = {
            "answer": ai_response.get("answer", ""),
            "claims": verified_claims,
            "chunks_used": len(top_chunks),
            "document_title": primary_doc.title,
            "document_ids": target_doc_ids,  # Include all document IDs used
            "mode": mode
        }
        
        # Log summary
        verified_count = sum(1 for c in verified_claims if c["status"] == "VERIFIED")
        likely_count = sum(1 for c in verified_claims if c["status"] == "LIKELY")
        unverified_count = sum(1 for c in verified_claims if c["status"] == "UNVERIFIED")
        
        logger.info(
            f"Analysis complete: {len(verified_claims)} claims "
            f"(✓{verified_count} ⚠{likely_count} ✗{unverified_count})"
        )
        
        return Response(response_data)


class DocumentHistoryView(ListAPIView):
    """
    API endpoint for retrieving chat history for a document.
    
    GET /api/documents/<doc_id>/history/
    
    Returns all chat messages (user questions and AI answers) for
    the specified document, ordered by creation time (oldest first).
    
    Response:
        [
            {
                "id": 1,
                "document": 5,
                "sender": "user",
                "content": "What is entropy?",
                "claims": [],
                "created_at": "2024-01-15T10:30:00Z"
            },
            {
                "id": 2,
                "document": 5,
                "sender": "ai",
                "content": "Entropy is a measure of disorder...",
                "claims": [{"claim": "...", "quote_anchor": "...", ...}],
                "created_at": "2024-01-15T10:30:05Z"
            }
        ]
    """
    
    serializer_class = ChatMessageSerializer
    
    def get_queryset(self):
        """Return messages for the specified document, ordered by creation time."""
        doc_id = self.kwargs.get('doc_id')
        return ChatMessage.objects.filter(document_id=doc_id).order_by('created_at')


class GenerateQuizView(APIView):
    """
    API endpoint for generating quiz questions from document(s).
    
    POST /api/quiz/
    
    Request Body:
        {
            "document_id": 1,  # Optional: single document ID (legacy)
            "document_ids": [1, 2]  # Optional: list of document IDs (preferred)
        }
        At least one of document_id or document_ids must be provided.
    
    Response:
        {
            "questions": [
                {
                    "question": "What is entropy?",
                    "options": ["Option A", "Option B", "Option C", "Option D", "Option E"],
                    "correct_index": 0,
                    "explanation": "Entropy is..."
                },
                ...
            ],
            "document_ids": [1, 2],
            "total_questions": 10
        }
    """
    
    def post(self, request):
        """Handle POST request for quiz generation."""
        
        # ─────────────────────────────────────────────────────────────
        # Step 1: VALIDATE REQUEST DATA
        # ─────────────────────────────────────────────────────────────
        document_ids = request.data.get('document_ids', [])
        document_id = request.data.get('document_id')
        
        # Determine which document IDs to use
        if document_ids:
            target_doc_ids = document_ids
        elif document_id:
            target_doc_ids = [document_id]
        else:
            return Response(
                {"error": "Either 'document_id' or 'document_ids' must be provided"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Verify documents exist
        documents = Document.objects.filter(id__in=target_doc_ids)
        if documents.count() != len(target_doc_ids):
            missing_ids = set(target_doc_ids) - set(documents.values_list('id', flat=True))
            return Response(
                {"error": f"Document(s) not found: {list(missing_ids)}"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # ─────────────────────────────────────────────────────────────
        # Step 2: RETRIEVE CONTEXT CHUNKS
        # ─────────────────────────────────────────────────────────────
        # Use a generic query to get broad context for quiz generation
        generic_query = "Important facts, definitions, and exam questions"
        
        # Retrieve chunks using the ChunkRetriever
        # We'll get more chunks for quiz generation (20 instead of 5)
        context_chunks = ChunkRetriever.retrieve_top_chunks(
            doc_ids=target_doc_ids,
            question=generic_query,
            top_n=20  # More chunks for better quiz coverage
        )
        
        if not context_chunks:
            return Response(
                {"error": "No content found in the selected document(s). Please ensure documents are processed."},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Build context text from chunks
        context_text = GeminiService._build_context(context_chunks)
        
        if not context_text or not context_text.strip():
            return Response(
                {"error": "No text content available in the selected document(s)."},
                status=status.HTTP_404_NOT_FOUND
            )
        
        logger.info(f"Generating quiz from {len(context_chunks)} chunks across document(s) {target_doc_ids}")
        
        # ─────────────────────────────────────────────────────────────
        # Step 3: GENERATE QUIZ
        # ─────────────────────────────────────────────────────────────
        quiz_questions = GeminiService.generate_quiz(context_text)
        
        if not quiz_questions:
            return Response(
                {"error": "Failed to generate quiz questions. Please try again."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # ─────────────────────────────────────────────────────────────
        # Step 4: RESPONSE
        # ─────────────────────────────────────────────────────────────
        response_data = {
            "questions": quiz_questions,
            "document_ids": target_doc_ids,
            "total_questions": len(quiz_questions)
        }
        
        logger.info(f"Successfully generated {len(quiz_questions)} quiz questions")
        
        return Response(response_data, status=status.HTTP_200_OK)
