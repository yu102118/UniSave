"""
Gemini AI Service for UniSave.

Implements the Generative AI Layer that:
1. Constructs grounded prompts from context chunks
2. Calls Google Gemini 1.5 Flash API
3. Parses structured JSON responses with claims and citations

Supports two modes:
- STRICT: Only answers based on provided context (default)
- TUTOR: Uses general knowledge to explain exam questions found in context
"""

import json
import logging
import os
import re
from typing import List

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from core.models import Chunk


logger = logging.getLogger(__name__)


class AIServiceError(Exception):
    """Raised when AI service encounters an error."""
    pass


class GeminiService:
    """
    Service for interacting with Google Gemini API.
    
    Provides grounded question-answering using context from PDF chunks.
    Supports two modes: 'strict' (facts from text only) and 'tutor' (expert explanations).
    """
    
    MODEL_NAME = "gemini-flash-latest"
    
    # ─────────────────────────────────────────────────────────────────
    # STRICT MODE: Only answers based on provided context
    # ─────────────────────────────────────────────────────────────────
    STRICT_INSTRUCTION = """SYSTEM ROLE: You are an advanced AI Tutor called "UniSave", designed specifically for Azerbaijani students.

LANGUAGE RULES:
1. **Input Handling:** Users often type in "Translit" (Anglicized Azerbaijani). You MUST intelligently interpret 'e' as 'ə', 'ch'/'c' as 'ç', 'sh'/'s' as 'ş', etc., based on context. Never complain about their spelling.
2. **Output Formatting:** ALWAYS reply in **Standard, Grammatically Correct Azerbaijani** (using correct characters: ə, ö, ü, ğ, ı, ş, ç), unless the user specifically asks in Russian or English.
3. **Tone:** Academic but friendly, encouraging, and concise.

CONTEXT HANDLING:
Use the provided document context to answer. If the answer is not in the context, state that clearly in Azerbaijani.

You are UniSave, a strict academic assistant designed to help students understand their textbooks.

CRITICAL RULES:
1. Answer ONLY based on the provided Context below. Do not use external knowledge.
2. If the answer is NOT in the context, respond with: {"answer": "I cannot find information about this in the provided text.", "claims": []}
3. Be accurate and student-friendly in your explanations.
4. Every factual statement MUST be backed by a quote from the source.
5. IMPORTANT: Return strictly valid JSON using standard double quotes for keys and string values.

OUTPUT FORMAT:
You MUST return a valid JSON object. Do NOT use Markdown formatting (no ```json code blocks).
Return ONLY the raw JSON object, nothing else.

JSON SCHEMA:
{
    "answer": "A clear, student-friendly explanation that directly answers the question...",
    "claims": [
        {
            "claim": "A specific statement of fact derived from the text",
            "quote_anchor": "A unique, verbatim phrase (8-15 words) from the Context that proves this claim. MUST exist exactly in the source text.",
            "page_hint": <integer, the page number where this quote was found>
        }
    ]
}

QUOTE ANCHOR RULES:
- Must be 8-15 words, copied EXACTLY from the Context
- Must be unique enough to locate in the source
- Do NOT paraphrase or modify the quote
- Include the page number from the [Page X] markers

If you cannot find a direct quote to support a claim, do NOT include that claim."""

    # ─────────────────────────────────────────────────────────────────
    # TUTOR MODE: Expert explanations using general knowledge
    # ─────────────────────────────────────────────────────────────────
    TUTOR_INSTRUCTION = """SYSTEM ROLE: You are an advanced AI Tutor called "UniSave", designed specifically for Azerbaijani students.

LANGUAGE RULES:
1. **Input Handling:** Users often type in "Translit" (Anglicized Azerbaijani). You MUST intelligently interpret 'e' as 'ə', 'ch'/'c' as 'ç', 'sh'/'s' as 'ş', etc., based on context. Never complain about their spelling.
2. **Output Formatting:** ALWAYS reply in **Standard, Grammatically Correct Azerbaijani** (using correct characters: ə, ö, ü, ğ, ı, ş, ç), unless the user specifically asks in Russian or English.
3. **Tone:** Academic but friendly, encouraging, and concise.

CONTEXT HANDLING:
Use the provided document context to answer. If the answer is not in the context, state that clearly in Azerbaijani.

You are UniSave, an Expert Academic Tutor.
Your goal is to help a student answer questions found in their exam papers/textbooks.

LANGUAGE & CONTENT PROTOCOL:
1. DETECT User Language (e.g., Russian).
2. DETECT Context: Is the subject a foreign language (English, French, etc.) or Programming?
3. RESPONSE STRATEGY:
   - **Explanations/Logic:** Must be in the User's Language.
   - **Key Terms/Phrases/Code:** Must remain in the Target Language/Original Format.
   - **Format:** When listing terms, use the format: "**Term in Target Language** - (Translation/Explanation in User Language)".

PROTOCOL:
1. LOCATE THE QUESTION: 
   - Scan the provided Context to find the specific question/topic.
   - If NOT in Context, stop and say you cannot find it.

2. ANALYZE SOURCE:
   - Check if the direct ANSWER is present in the text.
   - Scenario A (Answer found): Use the text as primary source.
   - Scenario B (Answer missing): Use internal academic knowledge.

3. INTERNAL DOUBLE-CHECK (Scenario B only):
   - Retrieve standard academic answer.
   - Verify against alternative definitions.
   - Ensure consensus.

4. CITATION LOGIC:
   - **If Answer is in PDF:**
     - claim: The fact from text.
     - quote_anchor: Verbatim quote.
     - page_hint: Real page number.
   
   - **If Answer is from External Knowledge (Scenario B):**
     - You MUST add a citation indicating the source methodology.
     - claim: The specific concept, law, or standard applied.
     - quote_anchor: "External Academic Consensus" (Fixed string).
     - page_hint: 0 (Zero indicates external source).

OUTPUT FORMAT (JSON Only):
Return strictly valid JSON using standard double quotes for keys and string values.
{
    "answer": "Your clear explanation in User's Language, with Terms in Target Language...",
    "claims": [
        {
            "claim": "Concept or Fact used",
            "quote_anchor": "Verbatim text OR 'External Academic Consensus'",
            "page_hint": 1 OR 0
        }
    ]
}"""

    _configured = False
    
    @classmethod
    def _ensure_configured(cls) -> None:
        """Configure the Gemini API with the API key from environment."""
        if cls._configured:
            return
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not api_key:
            # Try loading from .env file if python-dotenv is available
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.environ.get("GOOGLE_API_KEY")
            except ImportError:
                pass
        
        if not api_key:
            raise AIServiceError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )
        
        genai.configure(api_key=api_key)
        cls._configured = True
        logger.info("Gemini API configured successfully")
    
    @classmethod
    def _build_context(cls, chunks: List[Chunk]) -> str:
        """
        Build a formatted context string from chunks.
        
        Each chunk is labeled with its page number for citation tracking.
        
        Args:
            chunks: List of Chunk objects to include in context.
            
        Returns:
            Formatted context string with page markers.
        """
        if not chunks:
            return ""
        
        context_parts = []
        current_page = None
        
        for chunk in chunks:
            page_num = chunk.page.page_number
            
            # Add page marker when page changes
            if page_num != current_page:
                context_parts.append(f"\n[Page {page_num}]")
                current_page = page_num
            
            context_parts.append(chunk.chunk_text)
        
        return "\n".join(context_parts)
    
    @classmethod
    def _clean_response(cls, response_text: str) -> str:
        """
        Clean AI response by removing markdown code blocks if present.
        
        Despite instructions, the model sometimes wraps JSON in code blocks.
        This method handles various markdown formats robustly.
        
        Args:
            response_text: Raw response from the AI.
            
        Returns:
            Cleaned text ready for JSON parsing.
        """
        if not response_text:
            return ""
        
        text = response_text.strip()
        
        # Remove markdown code block markers
        # Handle "```json" at the start
        if text.startswith("```json"):
            text = text[7:]  # Remove "```json"
        elif text.startswith("```"):
            text = text[3:]  # Remove "```"
        
        # Remove trailing "```"
        if text.endswith("```"):
            text = text[:-3]
        
        # Strip whitespace again after removing markers
        text = text.strip()
        
        # Try to extract JSON object using regex (fallback for complex cases)
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            text = json_match.group()
        
        return text.strip()
    
    @classmethod
    def _parse_response(cls, response_text: str) -> dict:
        """
        Parse AI response into structured dictionary.
        
        Robustly handles JSON parsing with fallback to show raw text
        if parsing fails, preventing frontend crashes.
        
        Args:
            response_text: Raw response text from the AI.
            
        Returns:
            Parsed dictionary with 'answer' and 'claims'.
            On parse error, returns fallback dict with raw text as answer.
        """
        # Clean markdown code blocks
        cleaned = cls._clean_response(response_text)
        
        if not cleaned:
            logger.warning("Empty response after cleaning")
            return {
                "answer": "The AI returned an empty response. Please try again.",
                "claims": []
            }
        
        try:
            # Attempt to parse as JSON
            result = json.loads(cleaned)

            if not isinstance(result, dict):
                logger.error(f"JSON root is not an object: {type(result)}")
                return {
                    "answer": cleaned,
                    "claims": [],
                    "parse_warning": "Response JSON root must be an object"
                }
            
            # Validate expected structure
            if "answer" not in result:
                result["answer"] = "Error: Response missing 'answer' field"
            if "claims" not in result:
                result["claims"] = []

            # Normalize base field types
            if not isinstance(result.get("answer"), str):
                result["answer"] = str(result.get("answer", ""))
            if not isinstance(result.get("claims"), list):
                result["claims"] = []
            
            # Ensure claims have required fields
            validated_claims = []
            for claim in result.get("claims", []):
                if isinstance(claim, dict) and "claim" in claim and "quote_anchor" in claim:
                    page_hint = claim.get("page_hint", 1)
                    try:
                        page_hint = int(page_hint)
                    except (TypeError, ValueError):
                        page_hint = 1
                    validated_claims.append({
                        "claim": claim.get("claim", ""),
                        "quote_anchor": claim.get("quote_anchor", ""),
                        "page_hint": page_hint
                    })
            result["claims"] = validated_claims
            
            return result
            
        except json.JSONDecodeError as e:
            # Log the error for debugging
            logger.error(f"JSON Parse Error: {e}")
            logger.error(f"Cleaned text (first 500 chars): {cleaned[:500]}")
            logger.debug(f"Full raw response: {response_text[:1000]}")
            
            # Fallback: Return the cleaned text as the answer
            # This prevents frontend crashes and still shows the user the AI's response
            return {
                "answer": cleaned,  # Show the raw text so user still sees the answer
                "claims": [],
                "parse_error": str(e),
                "parse_warning": "Response was not valid JSON, showing raw text"
            }
    
    @classmethod
    def ask_with_context(
        cls,
        question: str,
        context_chunks: List[Chunk],
        mode: str = "strict"
    ) -> dict:
        """
        Ask a question using provided context chunks.
        
        Constructs a grounded prompt, calls Gemini API, and returns
        a structured response with verifiable claims.
        
        Args:
            question: The user's question.
            context_chunks: List of Chunk objects providing context.
            mode: Either "strict" (facts from text only) or "tutor" (expert explanations).
                  Default is "strict".
            
        Returns:
            Dictionary with structure:
            {
                "answer": str,
                "claims": [
                    {
                        "claim": str,
                        "quote_anchor": str,
                        "page_hint": int
                    },
                    ...
                ],
                "mode": str,
                "error": str (optional)
            }
            
        Raises:
            AIServiceError: If API key is missing or API call fails critically.
        """
        # Ensure API is configured
        cls._ensure_configured()
        
        # Build context from chunks
        context = cls._build_context(context_chunks)
        
        if not context:
            return {
                "answer": "No context was provided. Please upload a document first.",
                "claims": [],
                "mode": mode
            }
        
        # Select instruction and temperature based on mode
        if mode == "tutor":
            system_instruction = cls.TUTOR_INSTRUCTION
            temperature = 0.2  # Low temperature for stable structured output
            logger.info("Using TUTOR mode (general knowledge enabled)")
        else:
            system_instruction = cls.STRICT_INSTRUCTION
            temperature = 0.2  # Low temperature to reduce quote drift/hallucinations
            logger.info("Using STRICT mode (text-only facts)")
        
        # Construct the user prompt
        user_prompt = f"""Context:
{context}

Question: {question}

Remember: Return ONLY a valid JSON object with "answer" and "claims" fields. No markdown, no code blocks."""

        logger.info(f"Sending question to Gemini: {question[:100]}...")
        logger.debug(f"Context length: {len(context)} chars from {len(context_chunks)} chunks")
        
        try:
            # Safety settings - disable all blocking for academic content
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            # Create model instance with selected system instruction
            model = genai.GenerativeModel(
                model_name=cls.MODEL_NAME,
                system_instruction=system_instruction,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    top_p=0.95,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                ),
                safety_settings=safety_settings,
            )
            
            # Generate response
            response = model.generate_content(user_prompt)
            
            # Check for blocked content
            if not response.text:
                logger.warning("Empty response from Gemini API")
                return {
                    "answer": "The AI was unable to generate a response. Please rephrase your question.",
                    "claims": [],
                    "mode": mode,
                    "error": "Empty response"
                }
            
            # Parse and return
            result = cls._parse_response(response.text)
            result["mode"] = mode  # Include mode in response
            logger.info(f"Successfully got response with {len(result.get('claims', []))} claims")
            return result
            
        except Exception as e:
            error_msg = f"Gemini API error: {str(e)}"
            logger.error(error_msg)
            return {
                "answer": "An error occurred while processing your question. Please try again.",
                "claims": [],
                "mode": mode,
                "error": error_msg
            }
    
    @classmethod
    def generate_quiz(cls, context_text: str) -> List[dict]:
        """
        Generate a multiple-choice quiz from the provided context text.
        
        Creates 5-option multiple choice questions (A, B, C, D, E) based strictly
        on the provided text. Adapts quantity based on text density (up to 10 questions).
        
        Args:
            context_text: The text content to generate quiz questions from.
            
        Returns:
            List of quiz question dictionaries, each with structure:
            {
                "question": str,
                "options": List[str] (5 options: A, B, C, D, E),
                "correct_index": int (0=A, 1=B, 2=C, 3=D, 4=E),
                "explanation": str
            }
            
        Raises:
            AIServiceError: If API key is missing or API call fails critically.
        """
        # Ensure API is configured
        cls._ensure_configured()
        
        if not context_text or not context_text.strip():
            logger.warning("Empty context provided for quiz generation")
            return []
        
        prompt = f"""
        Act as a Strict Academic Examiner.
        Based on the following text, generate a Multiple Choice Quiz.
        
        RULES:
        1. If the text is dense, generate up to 10 high-quality questions.
        2. If the text is a list of existing questions, extract 10 random ones.
        3. Options must be labelled A, B, C, D, E.
        4. RETURN ONLY A JSON ARRAY. Do not write introductory text.
        5. IMPORTANT: The output must be a list enclosed in square brackets [ ... ].

        OUTPUT FORMAT:
        [
            {{
                "question": "Question text...",
                "options": ["Option A", "Option B", "Option C", "Option D", "Option E"],
                "correct_index": 0,
                "explanation": "Why A is correct..."
            }}
        ]

        TEXT:
        {context_text[:15000]}
        """
        
        try:
            # Safety settings - disable all blocking for academic content
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            # Create model instance
            model = genai.GenerativeModel(
                model_name=cls.MODEL_NAME,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    top_p=0.95,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                ),
                safety_settings=safety_settings,
            )
            
            response = model.generate_content(prompt)
            raw_text = response.text.strip()

            # Clean Markdown
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            
            raw_text = raw_text.strip()

            # Robust Parsing
            try:
                # Attempt 1: Standard parse
                quiz_data = json.loads(raw_text)
                
                # Validate it's a list
                if not isinstance(quiz_data, list):
                    logger.error(f"Quiz response is not a list: {type(quiz_data)}")
                    return []
                
                return quiz_data
            except json.JSONDecodeError:
                # Attempt 2: Auto-repair (Add missing brackets)
                try:
                    quiz_data = json.loads(f"[{raw_text}]")
                    logger.info("Successfully repaired JSON by adding brackets")
                    
                    # Validate it's a list
                    if not isinstance(quiz_data, list):
                        logger.error(f"Quiz response is not a list after repair: {type(quiz_data)}")
                        return []
                    
                    return quiz_data
                except json.JSONDecodeError as e2:
                    logger.error(f"Failed to parse Quiz JSON. Error: {e2}")
                    logger.error(f"Raw text (first 100 chars): {raw_text[:100]}...")
                    return []
            
        except Exception as e:
            logger.error(f"Error generating quiz: {e}")
            return []
    
    @classmethod
    def health_check(cls) -> dict:
        """
        Check if the Gemini service is properly configured and accessible.
        
        Returns:
            Dictionary with status and any error messages.
        """
        try:
            cls._ensure_configured()
            
            # Try a minimal API call
            model = genai.GenerativeModel(cls.MODEL_NAME)
            response = model.generate_content("Respond with only: OK")
            
            return {
                "status": "healthy",
                "model": cls.MODEL_NAME,
                "response_test": "OK" in response.text
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
