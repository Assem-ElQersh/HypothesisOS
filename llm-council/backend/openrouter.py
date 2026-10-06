import os
import sys
import json
import asyncio
import httpx

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import config

async def query_model(model_name: str, messages: list, timeout: float = 120.0) -> dict:
    """
    Query OpenRouter API for LLM completion.
    If OPENROUTER_API_KEY is missing or API call fails, returns error status
    to prevent fabricating fake scientific metrics.
    """
    api_key = config.OPENROUTER_API_KEY

    if not api_key:
        print(f"[OpenRouter Guard] OPENROUTER_API_KEY is not set in environment.")
        return {"error": "OPENROUTER_API_KEY missing", "status": "API_UNAVAILABLE", "content": None}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **config.PROMPT_CACHE_HEADERS
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2048
    }

    print(f"[OpenRouter] Querying {model_name}...")
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(
                f"{config.OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return {"content": content, "status": "SUCCESS", "raw": data}
        except Exception as e:
            print(f"[OpenRouter Error] {e}")
            return {"error": str(e), "status": "API_UNAVAILABLE", "content": None}

async def query_models_parallel(models: list, messages: list, timeout: float = 120.0) -> list:
    """ Query multiple LLMs in parallel """
    tasks = [query_model(m, messages, timeout=timeout) for m in models]
    return await asyncio.gather(*tasks, return_exceptions=True)
