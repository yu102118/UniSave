import { useState, useRef } from 'react';
import { uploadDocument, analyzeDocument, UploadResponse, AnalyzeResponse, Claim } from './services/api';

// Backend URL for media files
const BACKEND_URL = 'http://127.0.0.1:8000';

function App() {
  // Document state
  const [activeDocId, setActiveDocId] = useState<number | null>(null);
  const [currentFileUrl, setCurrentFileUrl] = useState<string | null>(null);
  const [documentTitle, setDocumentTitle] = useState<string>('');
  
  // UI state
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [highlights, setHighlights] = useState<Claim[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  /**
   * Handle file upload
   */
  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate PDF
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Please select a PDF file');
      return;
    }

    setIsUploading(true);
    setError(null);
    setUploadSuccess(false);
    setUploadMessage(null);

    try {
      const response: UploadResponse = await uploadDocument(file);
      
      // Set document state
      setActiveDocId(response.id);
      setCurrentFileUrl(BACKEND_URL + response.file_url);
      setDocumentTitle(response.title);
      
      // Clear previous analysis state for fresh UI
      setResult(null);
      setHighlights([]);
      setQuestion('');
      
      // Show success feedback
      setUploadSuccess(true);
      setUploadMessage(response.processing_message);
      
      // Auto-hide success message after 5 seconds
      setTimeout(() => {
        setUploadSuccess(false);
        setUploadMessage(null);
      }, 5000);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsUploading(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  /**
   * Handle document analysis
   */
  const handleAnalyze = async () => {
    if (!activeDocId) {
      setError('Please upload a document first');
      return;
    }
    
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResult(null);
    setHighlights([]);

    try {
      const response = await analyzeDocument(activeDocId, question);
      setResult(response);
      setHighlights(response.claims);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  };

  /**
   * Get status badge color
   */
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'VERIFIED': return 'bg-green-100 text-green-800 border-green-300';
      case 'LIKELY': return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'UNVERIFIED': return 'bg-red-100 text-red-800 border-red-300';
      default: return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  /**
   * Get status icon
   */
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'VERIFIED': return '✓';
      case 'LIKELY': return '⚠';
      case 'UNVERIFIED': return '✗';
      default: return '?';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="bg-slate-800/50 backdrop-blur-sm border-b border-slate-700 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
            📚 UniSave
          </h1>
          <span className="text-slate-400 text-sm">AI Study Assistant</span>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Left Panel - Upload & Question */}
          <div className="space-y-6">
            
            {/* Upload Section */}
            <section className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700 p-6">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <span>📄</span> Upload Document
              </h2>
              
              <div className="space-y-4">
                <label className={`
                  flex flex-col items-center justify-center w-full h-32 
                  border-2 border-dashed rounded-xl cursor-pointer
                  transition-all duration-200
                  ${uploadSuccess 
                    ? 'border-green-500 bg-green-500/10' 
                    : 'border-slate-600 hover:border-blue-500 hover:bg-slate-700/50'
                  }
                `}>
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    {isUploading ? (
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
                    ) : uploadSuccess ? (
                      <>
                        <span className="text-3xl mb-2">✅</span>
                        <p className="text-sm text-green-400">Upload successful!</p>
                      </>
                    ) : (
                      <>
                        <span className="text-3xl mb-2">📁</span>
                        <p className="text-sm text-slate-400">
                          <span className="font-semibold text-blue-400">Click to upload</span> or drag and drop
                        </p>
                        <p className="text-xs text-slate-500 mt-1">PDF files only</p>
                      </>
                    )}
                  </div>
                  <input 
                    ref={fileInputRef}
                    type="file" 
                    className="hidden" 
                    accept=".pdf"
                    onChange={handleUpload}
                    disabled={isUploading}
                  />
                </label>

                {/* Upload Success Message */}
                {uploadMessage && (
                  <div className={`p-3 rounded-lg text-sm ${
                    uploadSuccess 
                      ? 'bg-green-500/20 text-green-300 border border-green-500/30' 
                      : 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30'
                  }`}>
                    {uploadMessage}
                  </div>
                )}

                {/* Current Document Info */}
                {activeDocId && (
                  <div className="p-3 rounded-lg bg-slate-700/50 border border-slate-600">
                    <p className="text-sm text-slate-300">
                      <span className="text-slate-500">Active:</span>{' '}
                      <span className="font-medium text-white">{documentTitle}</span>
                    </p>
                    <p className="text-xs text-slate-500 mt-1">ID: {activeDocId}</p>
                  </div>
                )}
              </div>
            </section>

            {/* Question Section */}
            <section className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700 p-6">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <span>💬</span> Ask a Question
              </h2>
              
              <div className="space-y-4">
                <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="What would you like to know about this document?"
                  className="w-full h-24 px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-xl 
                           text-white placeholder-slate-400 resize-none
                           focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                           transition-all duration-200"
                  disabled={!activeDocId}
                />
                
                <button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing || !activeDocId || !question.trim()}
                  className={`
                    w-full py-3 px-6 rounded-xl font-semibold transition-all duration-200
                    flex items-center justify-center gap-2
                    ${isAnalyzing || !activeDocId || !question.trim()
                      ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                      : 'bg-gradient-to-r from-blue-500 to-emerald-500 text-white hover:shadow-lg hover:shadow-blue-500/25 hover:-translate-y-0.5'
                    }
                  `}
                >
                  {isAnalyzing ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <span>🔍</span>
                      Analyze
                    </>
                  )}
                </button>

                {!activeDocId && (
                  <p className="text-sm text-slate-500 text-center">
                    Upload a document first to ask questions
                  </p>
                )}
              </div>
            </section>

            {/* Error Display */}
            {error && (
              <div className="p-4 rounded-xl bg-red-500/20 border border-red-500/30 text-red-300">
                <p className="font-medium">⚠️ Error</p>
                <p className="text-sm mt-1">{error}</p>
              </div>
            )}
          </div>

          {/* Right Panel - PDF Viewer & Results */}
          <div className="space-y-6">
            
            {/* PDF Viewer */}
            <section className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700 p-6">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <span>📖</span> Document Preview
              </h2>
              
              {currentFileUrl ? (
                <div className="aspect-[3/4] bg-slate-900 rounded-xl overflow-hidden">
                  <iframe
                    src={currentFileUrl}
                    className="w-full h-full"
                    title="PDF Preview"
                  />
                </div>
              ) : (
                <div className="aspect-[3/4] bg-slate-900/50 rounded-xl flex items-center justify-center border-2 border-dashed border-slate-700">
                  <div className="text-center text-slate-500">
                    <span className="text-4xl block mb-2">📄</span>
                    <p>No document loaded</p>
                    <p className="text-sm mt-1">Upload a PDF to preview</p>
                  </div>
                </div>
              )}
            </section>

            {/* Analysis Results */}
            {result && (
              <section className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700 p-6">
                <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <span>🤖</span> AI Response
                </h2>
                
                {/* Answer */}
                <div className="prose prose-invert max-w-none mb-6">
                  <p className="text-slate-200 leading-relaxed">{result.answer}</p>
                </div>

                {/* Claims/Citations */}
                {highlights.length > 0 && (
                  <div className="space-y-3">
                    <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
                      Citations ({highlights.length})
                    </h3>
                    
                    {highlights.map((claim, index) => (
                      <div 
                        key={index}
                        className={`p-4 rounded-xl border ${getStatusColor(claim.status)}`}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex-1">
                            <p className="font-medium text-sm">{claim.claim}</p>
                            <p className="text-xs mt-2 opacity-75">
                              "{claim.quote_anchor}"
                            </p>
                            <p className="text-xs mt-1 opacity-60">
                              Page {claim.page_hint} • Match score: {claim.score}%
                            </p>
                          </div>
                          <span className="text-2xl" title={claim.status}>
                            {getStatusIcon(claim.status)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Metadata */}
                <div className="mt-4 pt-4 border-t border-slate-700 text-xs text-slate-500">
                  Used {result.chunks_used} text chunks for analysis
                </div>
              </section>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
