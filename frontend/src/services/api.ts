/**
 * API Service for UniSave backend communication.
 */

const API_BASE_URL = '/api';

export interface UploadResponse {
  id: number;
  title: string;
  file: string;
  file_url: string;
  uploaded_at: string;
  processing_status: 'success' | 'warning';
  processing_message: string;
}

export interface Claim {
  claim: string;
  quote_anchor: string;
  page_hint: number;
  page_id: number | null;
  status: 'VERIFIED' | 'LIKELY' | 'UNVERIFIED';
  score: number;
  bboxes: number[][];
}

export interface AnalyzeResponse {
  answer: string;
  claims: Claim[];
  chunks_used: number;
  document_title: string;
  error?: string;
}

export interface Document {
  id: number;
  title: string;
  file: string;
  file_url: string;
  uploaded_at: string;
}

/**
 * Upload a PDF document to the backend.
 */
export async function uploadDocument(file: File, title?: string): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  if (title) {
    formData.append('title', title);
  }

  const response = await fetch(`${API_BASE_URL}/documents/`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Upload failed');
  }

  return response.json();
}

/**
 * Get list of all documents.
 */
export async function getDocuments(): Promise<Document[]> {
  const response = await fetch(`${API_BASE_URL}/documents/`);
  
  if (!response.ok) {
    throw new Error('Failed to fetch documents');
  }

  return response.json();
}

/**
 * Analyze a document with a question.
 */
export async function analyzeDocument(docId: number, question: string): Promise<AnalyzeResponse> {
  const response = await fetch(`${API_BASE_URL}/analyze/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      doc_id: docId,
      question: question,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Analysis failed');
  }

  return response.json();
}

