"""
Serializers for the core app.

Provides Django REST Framework serializers for API data transformation.
"""

import io
import logging

from rest_framework import serializers

from core.models import Document, ChatMessage


logger = logging.getLogger(__name__)


class DocumentSerializer(serializers.ModelSerializer):
    """
    Serializer for Document model.
    
    Includes a computed `file_url` field for frontend access to uploaded files.
    Also exposes `created_at` as an alias for `uploaded_at` for API consistency.
    """
    
    file_url = serializers.SerializerMethodField()
    created_at = serializers.DateTimeField(source='uploaded_at', read_only=True)
    
    class Meta:
        model = Document
        fields = ['id', 'title', 'file', 'file_url', 'uploaded_at', 'created_at']
        read_only_fields = ['id', 'uploaded_at', 'created_at']
    
    def validate_file(self, value):
        """
        Validate uploaded file page count.
        
        Enforces a maximum of 100 pages/slides per document to prevent system overload.
        
        Args:
            value: The uploaded file object.
            
        Returns:
            The file object if validation passes.
            
        Raises:
            serializers.ValidationError: If file exceeds 100 pages/slides.
        """
        MAX_PAGES = 100
        file_name_lower = value.name.lower()
        
        # 1. Check PDF
        if file_name_lower.endswith('.pdf'):
            try:
                from pypdf import PdfReader
                
                # Save current position
                current_pos = value.tell()
                # Reset to beginning
                value.seek(0)
                
                reader = PdfReader(value)
                page_count = len(reader.pages)
                
                # Reset file pointer to original position (or beginning)
                value.seek(0)
                
                if page_count > MAX_PAGES:
                    raise serializers.ValidationError(
                        f"PDF too large. Max {MAX_PAGES} pages allowed. (Got {page_count})"
                    )
                    
            except serializers.ValidationError:
                # Re-raise validation errors
                raise
            except Exception as e:
                # If pypdf fails to read, log warning but don't block upload
                # Processing will catch it later
                logger.warning(f"PDF Validation Warning: Could not count pages - {e}")
                # Reset file pointer in case of error
                value.seek(0)
        
        # 2. Check PPTX
        elif file_name_lower.endswith('.pptx'):
            try:
                from pptx import Presentation
                
                # Save current position
                current_pos = value.tell()
                # Reset to beginning
                value.seek(0)
                
                # Read file content into memory for Presentation
                # python-pptx needs a file-like object, so we use BytesIO
                file_content = value.read()
                value.seek(0)  # Reset for saving
                
                prs = Presentation(io.BytesIO(file_content))
                slide_count = len(prs.slides)
                
                if slide_count > MAX_PAGES:
                    raise serializers.ValidationError(
                        f"Presentation too large. Max {MAX_PAGES} slides allowed. (Got {slide_count})"
                    )
                    
            except serializers.ValidationError:
                # Re-raise validation errors
                raise
            except Exception as e:
                # If pptx fails to read, log warning but don't block upload
                # Processing will catch it later
                logger.warning(f"PPTX Validation Warning: Could not count slides - {e}")
                # Reset file pointer in case of error
                value.seek(0)
        
        # Ensure file pointer is at beginning before returning
        value.seek(0)
        return value
    
    def get_file_url(self, obj) -> str:
        """
        Return the absolute URL for the uploaded file.
        
        Args:
            obj: Document instance.
            
        Returns:
            Absolute URL string for the file, or empty string if no file.
        """
        if obj.file:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.file.url)
            return obj.file.url
        return ""


class ChatMessageSerializer(serializers.ModelSerializer):
    """
    Serializer for ChatMessage model.
    
    Serializes chat history entries including AI claims/citations.
    """
    
    class Meta:
        model = ChatMessage
        fields = ['id', 'document', 'sender', 'content', 'claims', 'created_at']
        read_only_fields = ['id', 'created_at']


class AnalyzeDocumentRequestSerializer(serializers.Serializer):
    """
    Serializer for document analysis request validation.
    
    Supports both single document and multiple documents (Knowledge Library mode).
    """
    
    document_id = serializers.IntegerField(required=False, help_text="Single document ID (legacy support)")
    document_ids = serializers.ListField(
        child=serializers.IntegerField(),
        required=False,
        allow_empty=False,
        help_text="List of document IDs for Knowledge Library mode"
    )
    question = serializers.CharField(
        required=True,
        allow_blank=False,
        help_text="The question to ask about the document(s)"
    )
    mode = serializers.ChoiceField(
        choices=['strict', 'tutor'],
        default='strict',
        required=False,
        help_text="Analysis mode: 'strict' (text-only) or 'tutor' (with general knowledge)"
    )
    
    def validate(self, attrs):
        """
        Ensure at least one of document_id or document_ids is provided.
        """
        document_id = attrs.get('document_id')
        document_ids = attrs.get('document_ids', [])
        
        if not document_id and not document_ids:
            raise serializers.ValidationError(
                "Either 'document_id' or 'document_ids' must be provided."
            )
        
        # If both are provided, document_ids takes precedence
        if document_ids and document_id:
            # Remove document_id from attrs to avoid confusion
            attrs.pop('document_id', None)
        
        return attrs

