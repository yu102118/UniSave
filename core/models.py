"""
Core models for the UniSave PDF processing pipeline.

This module implements the data models for storing uploaded PDF documents,
their extracted pages, and text chunks for retrieval-augmented generation (RAG).
"""

from django.db import models


class Document(models.Model):
    """
    Represents an uploaded document (PDF, DOCX, or PPTX).
    
    Stores the original file and metadata for tracking purposes.
    Acts as the root node for all related pages and chunks.
    """
    
    title = models.CharField(
        max_length=255,
        help_text="Human-readable title of the document"
    )
    file = models.FileField(
        upload_to='documents/',
        help_text="The original uploaded document file (PDF, DOCX, or PPTX)"
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return self.title


class Page(models.Model):
    """
    Represents a single page extracted from a PDF document.
    
    Stores both raw and normalized text for different use cases:
    - `text_raw`: Original text preserving formatting (for display/citations)
    - `text_norm`: Lowercased/normalized text (for case-insensitive search)
    
    Page dimensions (`width`, `height`) and `rotation` are stored to enable
    accurate coordinate mapping for frontend highlight rendering. When the AI
    returns a citation, we need to locate it on the page and draw a highlight
    box. Without knowing the original page dimensions and rotation, coordinate
    calculations would be incorrect, breaking the visual verification feature.
    """
    
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name='pages'
    )
    page_number = models.PositiveIntegerField(
        help_text="1-indexed page number within the document"
    )
    text_raw = models.TextField(
        blank=True,
        help_text="Original extracted text preserving case and formatting"
    )
    text_norm = models.TextField(
        blank=True,
        help_text="Normalized (lowercase) text for case-insensitive search"
    )
    
    # Page geometry for frontend rendering and highlight positioning
    width = models.FloatField(
        help_text="Page width in PDF points (1 point = 1/72 inch)"
    )
    height = models.FloatField(
        help_text="Page height in PDF points (1 point = 1/72 inch)"
    )
    rotation = models.IntegerField(
        default=0,
        help_text="Page rotation in degrees (0, 90, 180, or 270)"
    )
    
    class Meta:
        ordering = ['document', 'page_number']
        unique_together = ['document', 'page_number']
        indexes = [
            models.Index(fields=['document', 'page_number']),
        ]
    
    def __str__(self):
        return f"{self.document.title} — Page {self.page_number}"


class Chunk(models.Model):
    """
    Represents a text chunk from a page, optimized for retrieval.
    
    During ingestion, page text is split into overlapping chunks (~1000 chars)
    to enable granular search. Each chunk maintains its position via
    `chunk_index` so we can reconstruct the reading order.
    
    Note: Vector embeddings are intentionally omitted for MVP.
    We use keyword search (Django ORM + SQLite FTS) for simplicity.
    """
    
    page = models.ForeignKey(
        Page,
        on_delete=models.CASCADE,
        related_name='chunks'
    )
    chunk_index = models.PositiveIntegerField(
        help_text="0-indexed position of this chunk within the page"
    )
    chunk_text = models.TextField(
        help_text="The actual text content of this chunk"
    )
    
    class Meta:
        ordering = ['page', 'chunk_index']
        unique_together = ['page', 'chunk_index']
        indexes = [
            models.Index(fields=['page', 'chunk_index']),
        ]
    
    def __str__(self):
        preview = self.chunk_text[:50] + '...' if len(self.chunk_text) > 50 else self.chunk_text
        return f"Chunk {self.chunk_index} of Page {self.page.page_number}: {preview}"


class ChatMessage(models.Model):
    """
    Represents a message in the chat history for a document.
    
    Stores both user questions and AI responses, along with any
    citations/claims returned by the AI for later reference.
    """
    
    SENDER_CHOICES = [
        ('user', 'User'),
        ('ai', 'AI'),
    ]
    
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name='messages',
        help_text="The document this message belongs to"
    )
    sender = models.CharField(
        max_length=10,
        choices=SENDER_CHOICES,
        help_text="Who sent this message: 'user' or 'ai'"
    )
    content = models.TextField(
        help_text="The message content (question or answer text)"
    )
    claims = models.JSONField(
        default=list,
        blank=True,
        help_text="List of citations/claims from AI response (empty for user messages)"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']
        indexes = [
            models.Index(fields=['document', 'created_at']),
        ]
    
    def __str__(self):
        preview = self.content[:50] + '...' if len(self.content) > 50 else self.content
        return f"[{self.sender}] {preview}"