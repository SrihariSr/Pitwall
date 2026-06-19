"""
Unified LLM client for Pitwall agents and orchestrator.

Supports two backends:
- gemini: cloud, free-tier-constrained, fast
- ollama: local, unlimited, slower (via native /api/chat)

Switch via LLM_BACKEND env var (default: gemini).
"""
import asyncio
import json
import os
from typing import TypeVar

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError

load_dotenv()

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base class for all LLM client failures."""


class LLMSchemaError(LLMError):
    """The model returned content that didn't match the expected schema."""


class LLMNetworkError(LLMError):
    """A network or API-level failure occurred."""


_RATE_LIMIT_INTERVAL_SECONDS = 4.5
_rate_limit_lock = asyncio.Lock()
_last_call_time: float = 0.0

_OLLAMA_URL = "http://localhost:11434/api/chat"


def _strip_code_fences(text: str) -> str:
    """Defensive: strip ```json ... ``` fences if a model adds them."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]  # drop closing fence
        text = "\n".join(lines).strip()
    return text


class LLMClient:
    """The single seam between Pitwall and any LLM provider."""

    def __init__(self, backend: str | None = None, api_key: str | None = None) -> None:
        self.backend = (backend or os.getenv("LLM_BACKEND", "gemini")).lower()

        if self.backend == "gemini":
            key = api_key or os.getenv("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY not found. Set it in .env.")
            self._gemini_client = genai.Client(api_key=key)
        elif self.backend == "ollama":
            # No persistent client needed; httpx.AsyncClient is created per-call.
            self._gemini_client = None
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}. Use 'gemini' or 'ollama'.")

    async def generate_structured(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[T],
        temperature: float = 0.2,
    ) -> T:
        if self.backend == "gemini":
            text = await self._call_gemini(
                model, system_prompt, user_prompt, response_schema, temperature,
            )
        else:
            text = await self._call_ollama(
                model, system_prompt, user_prompt, response_schema, temperature,
            )

        text = _strip_code_fences(text)

        try:
            data = json.loads(text)
            return response_schema.model_validate(data)
        except json.JSONDecodeError as e:
            raise LLMSchemaError(
                f"Model returned non-JSON: {e}. Raw text: {text[:300]}"
            ) from e
        except ValidationError as e:
            raise LLMSchemaError(f"Model output failed schema validation: {e}") from e

    async def _call_gemini(
        self, model: str, system_prompt: str, user_prompt: str,
        response_schema: type[T], temperature: float,
    ) -> str:
        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        global _last_call_time
        for attempt in range(2):
            async with _rate_limit_lock:
                elapsed = asyncio.get_event_loop().time() - _last_call_time
                if elapsed < _RATE_LIMIT_INTERVAL_SECONDS:
                    await asyncio.sleep(_RATE_LIMIT_INTERVAL_SECONDS - elapsed)
                _last_call_time = asyncio.get_event_loop().time()

            try:
                response = await asyncio.to_thread(
                    self._gemini_client.models.generate_content,
                    model=model,
                    contents=user_prompt,
                    config=config,
                )
                if not response.text:
                    raise LLMSchemaError("Gemini returned empty response")
                return response.text
            except Exception as e:
                if attempt == 0:
                    await asyncio.sleep(1.0)
                    continue
                raise LLMNetworkError(f"Gemini call failed: {e}") from e

        raise LLMNetworkError("Gemini call exhausted retries")

    async def _call_ollama(
        self, model: str, system_prompt: str, user_prompt: str,
        response_schema: type[T], temperature: float,
    ) -> str:
        """Native Ollama /api/chat call with structured-output constraint.

        Uses Ollama's native API directly (not the OpenAI-compatible layer)
        because the compatibility layer silently drops the `format` parameter,
        leaving the model unconstrained.
        """
        schema = response_schema.model_json_schema()

        augmented_system = (
            f"{system_prompt}\n\n"
            f"Respond with a single JSON object whose top-level keys are the "
            f"schema fields. Do not wrap the object in an outer key. Do not "
            f"include markdown fences or commentary."
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": augmented_system},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": schema,
            "think": False,
            "options": {
                "temperature": temperature,
            },
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            for attempt in range(2):
                try:
                    resp = await client.post(_OLLAMA_URL, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    text = data.get("message", {}).get("content", "")
                    if not text:
                        raise LLMSchemaError("Ollama returned empty content")
                    return text
                except Exception as e:
                    if attempt == 0:
                        await asyncio.sleep(1.0)
                        continue
                    raise LLMNetworkError(f"Ollama call failed: {e}") from e

        raise LLMNetworkError("Ollama call exhausted retries")
