# Core services package
from core.services.ingestion import (
    process_document,
    process_pdf_document,  # Backward compatibility alias
    DocumentProcessingError,
    PDFProcessingError,  # Backward compatibility alias
)
from core.services.validation import ValidationService, ValidationError
from core.services.ai import GeminiService, AIServiceError

__all__ = [
    'process_document',
    'process_pdf_document',  # Backward compatibility
    'DocumentProcessingError',
    'PDFProcessingError',  # Backward compatibility
    'ValidationService',
    'ValidationError',
    'GeminiService',
    'AIServiceError',
]

