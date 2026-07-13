import hashlib
import os
import json
from typing import Literal, List, Optional
from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai import types

class GeminiSource(BaseModel):
    title: str
    url: str
    snippet: str

class GeminiVerificationResult(BaseModel):
    verdict: Literal["Likely True", "Likely False", "Mixed", "Unverifiable"]
    fake_likelihood: float      # 0-100, model's own estimate
    confidence: float           # 0-100, HOW sure it is (separate from the likelihood itself)
    claims_checked: list[str] = Field(default_factory=list)
    summary: str                # 1-3 plain-language sentences
    sources: list[GeminiSource] = Field(default_factory=list)

class GeminiVerificationService:
    def __init__(self):
        self.cache = {}
        # We instantiate the client dynamically when needed to catch missing keys correctly.

    def _get_client(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        return genai.Client(api_key=api_key)

    def _get_cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
        
    def _create_fallback_response(self, error_reason: str) -> dict:
        print(f"GEMINI FALLBACK TRIGGERED: {error_reason}")
        return {
            "verdict": "Unverifiable",
            "fake_likelihood": 50.0,
            "confidence": 0.0,
            "claims_checked": [],
            "summary": f"Web verification unavailable: {error_reason}",
            "sources": [],
            "search_queries": [],
            "search_suggestions_html": None
        }

    async def verify_article(self, text: str) -> dict:
        client = self._get_client()
        if not client:
            return self._create_fallback_response("GEMINI_API_KEY is missing or invalid.")
        
        # Check if enabled
        if str(os.getenv("ENABLE_GEMINI_VERIFICATION", "true")).lower() != "true":
            return self._create_fallback_response("Gemini verification is disabled via configuration.")

        # Cap text to ~500 words
        short_text = " ".join(text.split()[:500])
        cache_key = self._get_cache_key(short_text)
        if cache_key in self.cache:
            return self.cache[cache_key]

        verification_prompt = f"""You are a fact-verification assistant for a news-credibility tool. Given the article text, identify the 2-5 most important, independently checkable factual claims (not opinions or style). Research each using Google Search. Based on current, reliable sources, judge whether the article is accurate, misleading, or false. Return only the structured result — no commentary outside the schema.

Article Text:
{short_text}
"""
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "none")
        if thinking_level.lower() == "none":
            thinking_config = None
        else:
            # currently only gemini-2.0-pro-exp supports thinking, gemini-2.5-flash does not
            # if thinking is none, we don't pass it.
            thinking_config = types.ThinkingConfig(thinking_budget=1024)

        config_args = {
            "tools": [types.Tool(google_search=types.GoogleSearch())],
            "temperature": 0.2,
        }
        
        if thinking_config is not None:
            config_args["thinking_config"] = thinking_config
        
        # Add instructions to the prompt to force JSON format, since we can't use response_mime_type with tools
        json_instruction = """
IMPORTANT: You MUST return your response as a valid JSON object EXACTLY matching the following structure. Do not wrap it in markdown block quotes (```json), just return the raw JSON object. You may include Google Search citation markers like [1] or [2] inside the string fields.

EXTREMELY IMPORTANT - AVOID RECITATION FILTERS: 
You MUST NOT quote any text directly from the article or from the search results.
You MUST paraphrase everything in your own unique words. 
For the 'snippet' field in sources, write a completely original 1-sentence summary rather than copying text from the search results.
{
  "verdict": "Likely True" | "Likely False" | "Mixed" | "Unverifiable",
  "fake_likelihood": float (0-100),
  "confidence": float (0-100),
  "claims_checked": ["string"],
  "summary": "string",
  "sources": [{"title": "string", "url": "string", "snippet": "string"}]
}
"""
        full_prompt = verification_prompt + json_instruction

        try:
            # We add a timeout in the aio context if possible. 
            # The python SDK doesn't natively take timeout in GenerateContentConfig in all versions, we might need asyncio.wait_for at the caller level.
            import asyncio
            timeout_seconds = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "15"))
            
            async def _call_gemini():
                return await client.aio.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(**config_args)
                )
                
            response = await asyncio.wait_for(_call_gemini(), timeout=timeout_seconds)

            if not response.text:
                reason = "Unknown"
                if response.candidates:
                    reason = str(response.candidates[0].finish_reason)
                return self._create_fallback_response(f"Model returned empty response. Finish reason: {reason}")

            raw_text = response.text.strip()
            # Clean up potential markdown formatting
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()
            
            # Parse the JSON string manually
            try:
                parsed_json = json.loads(raw_text)
                # Create the pydantic object
                validated = GeminiVerificationResult(**parsed_json)
            except json.JSONDecodeError as e:
                return self._create_fallback_response(f"Gemini returned invalid JSON: {e}. Raw: {raw_text[:100]}")
            except ValidationError as e:
                return self._create_fallback_response(f"Gemini returned missing/invalid fields: {e}")
            
            # Extract grounding metadata
            search_queries = []
            search_suggestions_html = None
            
            try:
                candidate = response.candidates[0]
                if candidate.grounding_metadata:
                    if candidate.grounding_metadata.web_search_queries:
                        search_queries = list(candidate.grounding_metadata.web_search_queries)
                    if candidate.grounding_metadata.search_entry_point:
                        search_suggestions_html = candidate.grounding_metadata.search_entry_point.rendered_content
            except Exception as metadata_e:
                # If metadata fails, we still have the verification result
                print(f"Warning: Failed to parse grounding metadata: {metadata_e}")
            
            result_dict = validated.model_dump()
            result_dict["search_queries"] = search_queries
            result_dict["search_suggestions_html"] = search_suggestions_html
            
            # Cache it
            self.cache[cache_key] = result_dict
            return result_dict
            
        except asyncio.TimeoutError:
            return self._create_fallback_response("Gemini API call timed out.")
        except ValidationError as e:
            return self._create_fallback_response(f"Gemini API returned invalid schema: {e}")
        except Exception as e:
            # Catches 429 quota limits, API errors, JSON parse errors, etc.
            return self._create_fallback_response(f"Gemini API error: {str(e)}")
