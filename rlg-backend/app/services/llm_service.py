"""
LLM Service - Local LLM via Ollama
100% offline after initial model download
"""
from typing import Optional, AsyncGenerator, Dict, Any
import httpx
import json
from dataclasses import dataclass

from app.core.config import settings


@dataclass
class LLMResponse:
    """Response from the LLM"""
    text: str
    tokens_used: int
    model: str
    finish_reason: str


class LLMService:
    """
    Local LLM service using Ollama.
    Runs completely offline with models like:
    - mistral (7B, good balance)
    - llama3 (8B, high quality)
    - phi3 (3.8B, fast)
    - gemma2 (9B, good reasoning)
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: int = 120
    ):
        self.model = model or settings.OLLAMA_MODEL
        self.host = host or settings.OLLAMA_HOST
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,  # Low temp for grounded responses
        max_tokens: int = 1024,
        stop_sequences: Optional[list] = None
    ) -> LLMResponse:
        """
        Generate a response from the local LLM.
        Uses low temperature for factual, grounded responses.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "stop": stop_sequences or []
            }
        }
        
        try:
            response = self.client.post(
                f"{self.host}/api/generate",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            return LLMResponse(
                text=data.get("response", ""),
                tokens_used=data.get("eval_count", 0),
                model=self.model,
                finish_reason=data.get("done_reason", "unknown")
            )
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}")
    
    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ) -> AsyncGenerator[str, None]:
        """
        Stream response tokens for real-time output.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.host}/api/generate",
                json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done"):
                            break
    
    def get_available_models(self) -> list:
        """List available Ollama models"""
        try:
            response = self.client.get(f"{self.host}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model if not available"""
        try:
            response = self.client.post(
                f"{self.host}/api/pull",
                json={"name": model_name},
                timeout=600  # Models can take a while
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = self.client.get(f"{self.host}/api/tags")
            return response.status_code == 200
        except Exception:
            return False


class ExtractiveLLMService(LLMService):
    """
    Specialized LLM service for extractive QA.
    Forces the model to quote directly from sources.
    """
    
    def extract_answer(
        self,
        context: str,
        question: str,
        context_chunks: list
    ) -> Dict[str, Any]:
        """
        Generate answer with forced extraction.
        Returns answer with source mappings.
        """
        prompt = self._build_extractive_prompt(context, question, context_chunks)
        response = self.generate(prompt, temperature=0.0)  # Zero temp for extraction
        
        # Parse the structured response
        return self._parse_extractive_response(response.text, context_chunks)
    
    def _build_extractive_prompt(
        self,
        context: str,
        question: str,
        context_chunks: list
    ) -> str:
        """Build prompt that forces extractive behavior"""
        return f"""You are an EXTRACTIVE question answering system. You MUST follow these rules:

CRITICAL RULES:
1. Your answer MUST use EXACT QUOTES from the sources
2. Place quotes inside "quotation marks"
3. Add citation markers [1], [2] after each quote
4. If you cannot find the answer in the sources, respond: "NOT_FOUND"
5. Do NOT paraphrase - use the exact words from sources

{context}

QUESTION: {question}

EXTRACTIVE ANSWER (quotes with citations only):"""
    
    def _parse_extractive_response(
        self,
        response: str,
        context_chunks: list
    ) -> Dict[str, Any]:
        """Parse the extractive response to verify quotes"""
        import re
        
        # Check for NOT_FOUND
        if "NOT_FOUND" in response.upper():
            return {
                "answer": None,
                "found": False,
                "quotes": []
            }
        
        # Extract quoted passages and citations
        quote_pattern = r'"([^"]+)"\s*\[(\d+)\]'
        quotes = re.findall(quote_pattern, response)
        
        verified_quotes = []
        for quote, citation_num in quotes:
            idx = int(citation_num) - 1
            if 0 <= idx < len(context_chunks):
                chunk = context_chunks[idx]
                # Verify quote exists in source
                if quote.lower() in chunk.content.lower():
                    verified_quotes.append({
                        "quote": quote,
                        "citation": f"[{citation_num}]",
                        "source": chunk.citation,
                        "verified": True
                    })
                else:
                    verified_quotes.append({
                        "quote": quote,
                        "citation": f"[{citation_num}]",
                        "source": chunk.citation,
                        "verified": False  # Quote not found in source!
                    })
        
        return {
            "answer": response,
            "found": True,
            "quotes": verified_quotes,
            "all_verified": all(q["verified"] for q in verified_quotes) if verified_quotes else False
        }


# Singleton instances
llm_service = LLMService()
extractive_llm = ExtractiveLLMService()
