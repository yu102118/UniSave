"""
URL configuration for the core app.

API Endpoints:
    POST   /api/documents/                    - Upload a PDF document
    GET    /api/documents/                    - List all documents
    DELETE /api/documents/<pk>/               - Delete a document
    GET    /api/documents/<id>/history/       - Get chat history for document
    POST   /api/analyze/                      - Analyze document with question (RAG pipeline)
    POST   /api/quiz/generate/                 - Generate quiz questions from document(s)
"""

from django.urls import path

from core.views import (
    AnalyzeDocumentView,
    DocumentDetailView,
    DocumentHistoryView,
    DocumentUploadView,
    GenerateQuizView,
)


urlpatterns = [
    path('documents/', DocumentUploadView.as_view(), name='document-upload'),
    path('documents/<int:pk>/', DocumentDetailView.as_view(), name='document-detail'),
    path('documents/<int:doc_id>/history/', DocumentHistoryView.as_view(), name='document-history'),
    path('analyze/', AnalyzeDocumentView.as_view(), name='analyze-document'),
    path('quiz/generate/', GenerateQuizView.as_view(), name='generate-quiz'),
]

