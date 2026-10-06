import os
import sys
import json
import asyncio
import random
import httpx

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import config

async def query_model(model_name: str, messages: list, timeout: float = 120.0) -> dict:
    """
    Query OpenRouter API for LLM completion.
    If OPENROUTER_API_KEY is not set, falls back to a realistic mock response
    that produces real PyTorch code modifications.
    """
    api_key = config.OPENROUTER_API_KEY

    if api_key:
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
                return {"content": content, "raw": data}
            except Exception as e:
                print(f"[OpenRouter Error] {e}. Falling back to smart mock...")

    # Realistic Fallback Mock Generator
    print(f"[Mock OpenRouter] Querying {model_name}...")
    user_prompt = messages[-1]["content"]

    if "Generate EXACTLY 3 mutually exclusive" in user_prompt or "Proposal Engine" in user_prompt:
        lr_options = [5e-4, 2e-3, 3e-3]
        n_embd_options = [96, 128, 48]
        n_head_options = [4, 8, 2]
        
        lr_val = random.choice(lr_options)
        embd_val = random.choice(n_embd_options)
        
        proposals = [
            {
                "id": f"HYP-{random.randint(100, 999)}",
                "mechanism": f"Tune learning rate to {lr_val} for faster convergence",
                "risk": "Low",
                "code_changes": f"@@ -13,1 +13,1 @@\n-learning_rate = 1e-3\n+learning_rate = {lr_val}"
            },
            {
                "id": f"HYP-{random.randint(100, 999)}",
                "mechanism": f"Expand hidden dimension n_embd to {embd_val} for higher model capacity",
                "risk": "Medium",
                "code_changes": f"@@ -16,1 +16,1 @@\n-n_embd = 64\n+n_embd = {embd_val}"
            },
            {
                "id": f"HYP-{random.randint(100, 999)}",
                "mechanism": "Add Cosine Annealing learning rate schedule",
                "risk": "Medium",
                "code_changes": "@@ -130,1 +130,3 @@\n+        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)\n         optimizer.step()\n+        scheduler.step()"
            }
        ]
        return {"content": json.dumps(proposals)}

    elif "Evaluate this proposed patch" in user_prompt or "Council Review" in user_prompt:
        eval_resp = {
            "expected_improvement": round(random.uniform(0.02, 0.15), 4),
            "probability_of_success": round(random.uniform(0.4, 0.85), 2),
            "implementation_cost": round(random.uniform(0.01, 0.05), 3),
            "failure_severity": random.choice(["Low", "Medium"]),
            "reasoning": "Valid hyperparameter / architectural adjustment for language modeling task."
        }
        return {"content": json.dumps(eval_resp)}

    elif "Perform a postmortem" in user_prompt:
        return {"content": "Postmortem Analysis: Empirical evaluation on PyTorch model completed. Metric recorded against ground-truth validation set."}

    return {"content": "Mock response"}

async def query_models_parallel(models: list, messages: list, timeout: float = 120.0) -> list:
    """ Query multiple LLMs in parallel """
    tasks = [query_model(m, messages, timeout=timeout) for m in models]
    return await asyncio.gather(*tasks, return_exceptions=True)
