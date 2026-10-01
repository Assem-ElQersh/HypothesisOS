import json
import random
import re

async def query_model(model_name, messages, timeout=120.0):
    print(f"[Mock LLM] Querying {model_name}...")
    user_prompt = messages[-1]["content"]
    
    if "Generate EXACTLY 3 mutually exclusive" in user_prompt:
        # Mock Proposal Generation
        proposals = [
            {
                "id": f"HYP-{random.randint(100,999)}",
                "mechanism": "Reduce learning rate by 10% to prevent divergence",
                "risk": "Low",
                "code_changes": "@@ -6,3 +6,3 @@\n     time.sleep(1)\n-    val_bpb = 3.42 + random.uniform(-0.05, 0.05)\n+    val_bpb = 3.20 + random.uniform(-0.05, 0.05)\n     peak_vram_mb = 18000"
            },
            {
                "id": f"HYP-{random.randint(100,999)}",
                "mechanism": "Increase batch size",
                "risk": "Medium",
                "code_changes": "@@ -6,3 +6,3 @@\n     time.sleep(1)\n-    val_bpb = 3.42 + random.uniform(-0.05, 0.05)\n+    val_bpb = 3.42 + random.uniform(-0.05, 0.05)\n     peak_vram_mb = 36000"
            }
        ]
        return {"content": json.dumps(proposals)}
        
    elif "Evaluate this proposed patch" in user_prompt:
        # Mock Council Evaluation
        response = {
            "expected_improvement": 0.1,
            "probability_of_success": 0.6,
            "implementation_cost": 0.1,
            "failure_severity": "low",
            "reasoning": "This seems safe."
        }
        return {"content": json.dumps(response)}
        
    elif "Perform a postmortem" in user_prompt:
        # Mock Postmortem
        return {"content": "Postmortem Analysis: The patch was successfully applied and the baseline was reduced."}
        
    return {"content": ""}

async def query_models_parallel(models, messages, timeout=120.0):
    pass
