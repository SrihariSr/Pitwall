"""
Unified LLM client for Pitwall agents and orchestrator.

Wraps Gemini behind a typed, async interface. Every Gemini call goes through
this layer so swapping to Claude (or any other provider) is a one-file change.
"""
import asyncio
import json
import os
from typing import TypeVar
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError

_RATE_LIMIT_INTERVAL_SECONDS = 4.5
_rate_limit_lock = asyncio.Lock()
_last_call_time: float = 0.0

load_dotenv()

T = TypeVar("T", bound=BaseModel)

class LLMError(Exception):
    """
    Parent class for all LLM client failures.
    """

class LLMSchemaError(LLMError):
    """
    Error if the model returns content that doesn't match the expected schema.
    """

class LLMNetworkError(LLMError):
    """
    Error at the API or network level.
    """

class LLMClient:
    """
    The connection between Pitwall and the LLM.

    Currently utilises Google Gemini 2.5 flash, can be swapped out for any LLM at any point.
    """

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not found! Set it up in .env or pass an API key.")
        
        self._client = genai.Client(api_key=key)
    
    async def _call_with_retry(
            self,
            model: str,
            user_prompt: str,
            config: genai_types.GenerateContentConfig,
        ) -> str:
            """Make the API call with one retry on network failure, respecting rate limit."""
            global _last_call_time
    
            last_error: Exception | None = None
            for attempt in range(2):
                # Enforce inter-call spacing to stay under free-tier limits.
                async with _rate_limit_lock:
                    elapsed = asyncio.get_event_loop().time() - _last_call_time
                    if elapsed < _RATE_LIMIT_INTERVAL_SECONDS:
                        await asyncio.sleep(_RATE_LIMIT_INTERVAL_SECONDS - elapsed)
                    _last_call_time = asyncio.get_event_loop().time()
    
                try:
                    response = await asyncio.to_thread(
                        self._client.models.generate_content,
                        model=model,
                        contents=user_prompt,
                        config=config,
                    )
                    if not response.text:
                        raise LLMSchemaError("Model returned empty response")
                    return response.text
                except Exception as e:
                    last_error = e
                    if attempt == 0:
                        await asyncio.sleep(1.0)
                        continue
                    raise LLMNetworkError(f"LLM call failed after retry: {e}") from e
    
            raise LLMNetworkError(f"Unexpected retry exit: {last_error}")

    async def generate_structured(
    self,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: type[T],
    temperature: float = 0.2
    ) -> T:
        """
        Call the LLM, expect structured JSON output, return a parsed Pydantic object.
        
        The prompt structure (system first, user last) is deliberately ordered for
        Gemini's implicit prompt caching: stable content at the start gets cached,
        variable content at the end is the cheap differential.

        Parameters:
            model: Gemini model name, e.g. "gemini-2.5-flash"
            system_prompt: The agent's role and reasoning instructions
            user_prompt: The specific situation to reason about
            response_schema: A Pydantic model class describing the expected output
            temperature: Sampling temperature. Low (0.1-0.3) for analytical work,
            higher only for creative tasks like Radio Narrator.

        Returns:
            An instance of response_schema, validated.
        
        Raises:
            LLMNetworkError: API or network failure (after one retry).
            LLMSchemaError: Response didn't match the schema after parsing.
        """

        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_schema
        )

        text = await self._call_with_retry(model, user_prompt, config)

        # Gemini's structured output mode returns valid JSON
        # but I am validating manually as a defensive measure.
        try:
            data = json.loads(text)
            return response_schema.model_validate(data)
        except json.JSONDecodeError as e:
            raise LLMSchemaError(f"Model returned non-JSON: {e}. \nRaw text: {text[:300]}") from e
        except ValidationError as e:
            raise LLMSchemaError(f"Model's output failed schema validation: {e}")
    
