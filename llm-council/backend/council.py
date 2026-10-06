import json
import asyncio
from backend.openrouter import query_model, query_models_parallel
from config import config

async def run_anonymized_council_review(proposals: list) -> list:
    """
    Layer 3: Multi-Agent Anonymized Council Review.
    Labels proposals as 'Proposal A', 'Proposal B', 'Proposal C' to prevent model bias,
    queries council models in parallel, and aggregates evaluations.
    """
    print(f"[Council] Initiating Anonymized Multi-Agent Peer Review across {len(proposals)} proposals...")
    
    # Anonymize proposal labels
    labels = [chr(65 + i) for i in range(len(proposals))] # A, B, C...
    anonymized_map = {labels[i]: proposals[i] for i in range(len(proposals))}

    results = []
    
    for label, proposal in anonymized_map.items():
        review_prompt = f"""Evaluate this proposed ML code patch for train.py (labeled as Proposal {label}):

MECHANISM: {proposal['mechanism']}
RISK & COMPLEXITY: {proposal['risk']}
CODE PATCH:
{proposal['code_changes']}

Critique the proposal for:
1. Technical Risk (OOM, syntax errors, shape mismatch, divergence)
2. Scientific Plausibility (Does the mechanism reasonably improve training efficiency?)
3. Expected Improvement & Implementation Cost

Format your evaluation STRICTLY as a JSON object with this exact schema:
{{
    "expected_improvement": 0.08,   // Estimated delta reduction in BPB (float > 0.0)
    "probability_of_success": 0.65, // Estimated probability between 0.0 and 1.0
    "implementation_cost": 0.02,   // Estimated compute cost in GPU hours
    "failure_severity": "Low",     // Low, Medium, or High
    "reasoning": "Scientific justification..."
}}
"""
        messages = [
            {"role": "system", "content": "You are a senior deep learning peer reviewer on the LLM Council."},
            {"role": "user", "content": review_prompt}
        ]

        # Parallel review across council models
        responses = await query_models_parallel(config.COUNCIL_MODELS, messages)

        valid_evals = []
        for r in responses:
            if isinstance(r, dict) and r.get("content"):
                try:
                    content = r["content"]
                    if "```json" in content:
                        json_str = content.split("```json")[1].split("```")[0].strip()
                    else:
                        json_str = content.strip()
                    eval_data = json.loads(json_str)
                    valid_evals.append(eval_data)
                except Exception:
                    continue

        if not valid_evals:
            # Fallback evaluation
            aggregated = {
                "expected_improvement": 0.05,
                "probability_of_success": 0.5,
                "implementation_cost": 0.02,
                "failure_severity": "Medium",
                "reasoning": "Default evaluation fallback."
            }
        else:
            # Aggregate evaluations mathematically
            avg_imp = sum(e.get("expected_improvement", 0.05) for e in valid_evals) / len(valid_evals)
            avg_prob = sum(e.get("probability_of_success", 0.5) for e in valid_evals) / len(valid_evals)
            avg_cost = sum(e.get("implementation_cost", 0.02) for e in valid_evals) / len(valid_evals)
            
            aggregated = {
                "expected_improvement": round(avg_imp, 4),
                "probability_of_success": round(avg_prob, 4),
                "implementation_cost": round(max(avg_cost, 0.001), 4),
                "failure_severity": valid_evals[0].get("failure_severity", "Low"),
                "reasoning": valid_evals[0].get("reasoning", "Consensus evaluation.")
            }

        # Calculate Utility-Adjusted Expected Value (EV / Compute Cost + Exploration Bonus)
        # Utility = (Expected Improvement * Probability of Success) / Implementation Cost
        utility_ev = (aggregated["expected_improvement"] * aggregated["probability_of_success"]) / aggregated["implementation_cost"]

        proposal_copy = dict(proposal)
        proposal_copy["evaluation"] = aggregated
        proposal_copy["utility_ev"] = round(utility_ev, 4)
        results.append(proposal_copy)

    # Sort proposals by Utility-Adjusted Expected Value
    results.sort(key=lambda x: x["utility_ev"], reverse=True)
    print(f"[Council] Consensus reached. Top hypothesis: {results[0]['id']} (Utility EV: {results[0]['utility_ev']})")
    return results
