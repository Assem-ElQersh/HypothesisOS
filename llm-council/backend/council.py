import json
import asyncio
import random
import numpy as np
from backend.openrouter import query_model, query_models_parallel
from config import config

async def run_anonymized_council_review(proposals: list) -> list:
    """
    Layer 3: Multi-Agent Anonymized Council Review & Ranking.
    Presents ALL candidate proposals together to each reviewer model with randomized labels,
    requires rank ordering (1st, 2nd, 3rd...), and calculates inter-reviewer model disagreement.
    """
    print(f"[Council] Initiating Multi-Proposal Anonymized Peer Review across {len(proposals)} candidate hypotheses...")
    
    # Randomize proposal labels for this review session
    labels = [f"Proposal {chr(65 + i)}" for i in range(len(proposals))] # Proposal A, Proposal B, Proposal C...
    shuffled_indices = list(range(len(proposals)))
    random.shuffle(shuffled_indices)

    label_to_prop = {}
    proposal_text_block = ""
    for idx, orig_idx in enumerate(shuffled_indices):
        lbl = labels[idx]
        prop = proposals[orig_idx]
        label_to_prop[lbl] = prop
        proposal_text_block += f"\n--- {lbl} (ID: {prop['id']}) ---\n"
        proposal_text_block += f"MECHANISM: {prop['mechanism']}\n"
        proposal_text_block += f"RISK & COMPLEXITY: {prop.get('risk', 'Medium')}\n"
        proposal_text_block += f"CODE PATCH:\n{prop['code_changes']}\n"

    review_prompt = f"""You are a senior deep learning peer reviewer on the LLM Council.
Evaluate and rank all of the following candidate ML code patches for train.py:

{proposal_text_block}

CRITERIA:
1. Technical Risk (syntax errors, shape mismatch, divergence, GPU OOM)
2. Scientific Plausibility (does the change reasonably improve convergence/loss?)
3. Utility (Expected Improvement / Implementation Cost)

Provide your evaluation STRICTLY as a JSON dictionary matching this exact schema:
{{
    "rankings": ["Proposal A", "Proposal B", "Proposal C"], // Ranked from best to worst
    "evaluations": {{
        "Proposal A": {{
            "expected_improvement": 0.08,   // Delta reduction in BPB (> 0.0)
            "probability_of_success": 0.70, // Float 0.0 to 1.0
            "implementation_cost": 0.01,   // Estimated GPU hours
            "technical_risk": "Low",       // Low, Medium, High
            "reasoning": "Detailed justification..."
        }}
    }}
}}
"""
    messages = [
        {"role": "system", "content": "You are an expert ML peer reviewer on the LLM Council."},
        {"role": "user", "content": review_prompt}
    ]

    # Query council reviewer models in parallel
    responses = await query_models_parallel(config.COUNCIL_MODELS, messages)

    reviewer_rankings = []
    proposal_scores = {lbl: [] for lbl in label_to_prop.keys()}

    for r in responses:
        if isinstance(r, dict) and r.get("status") == "SUCCESS" and r.get("content"):
            try:
                content = r["content"]
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                else:
                    json_str = content.strip()
                data = json.loads(json_str)

                if "rankings" in data:
                    reviewer_rankings.append(data["rankings"])
                
                evals = data.get("evaluations", {})
                for lbl, ev_info in evals.items():
                    if lbl in proposal_scores:
                        imp = float(ev_info.get("expected_improvement", 0.05))
                        prob = float(ev_info.get("probability_of_success", 0.5))
                        cost = float(ev_info.get("implementation_cost", 0.01))
                        cost = max(cost, 0.001)
                        utility = (imp * prob) / cost
                        proposal_scores[lbl].append({
                            "imp": imp,
                            "prob": prob,
                            "cost": cost,
                            "utility": utility,
                            "reasoning": ev_info.get("reasoning", "")
                        })
            except Exception:
                continue

    # If OpenRouter API calls failed or were unconfigured
    if not reviewer_rankings or sum(len(v) for v in proposal_scores.values()) == 0:
        print(f"[Council Notice] OpenRouter API unavailable or returned unparseable reviews. Cannot rank proposals without live reviewer consensus.")
        return []

    # Calculate Consensus & Inter-Reviewer Disagreement
    scored_results = []
    for lbl, prop in label_to_prop.items():
        scores = proposal_scores[lbl]
        if not scores:
            continue
        
        utilities = [s["utility"] for s in scores]
        mean_utility = float(np.mean(utilities))
        disagreement_std = float(np.std(utilities)) if len(utilities) > 1 else 0.0

        avg_imp = float(np.mean([s["imp"] for s in scores]))
        avg_prob = float(np.mean([s["prob"] for s in scores]))
        avg_cost = float(np.mean([s["cost"] for s in scores]))

        prop_copy = dict(prop)
        prop_copy["label"] = lbl
        prop_copy["evaluation"] = {
            "expected_improvement": round(avg_imp, 4),
            "probability_of_success": round(avg_prob, 4),
            "implementation_cost": round(avg_cost, 4),
            "reviewer_count": len(scores),
            "disagreement_std": round(disagreement_std, 4),
            "reasoning": scores[0]["reasoning"]
        }
        prop_copy["utility_ev"] = round(mean_utility, 4)
        scored_results.append(prop_copy)

    # Sort by Consensus Utility-Adjusted Expected Value
    scored_results.sort(key=lambda x: x["utility_ev"], reverse=True)
    print(f"[Council Consensus] Selected Top Proposal: {scored_results[0]['id']} (Utility EV: {scored_results[0]['utility_ev']}, Disagreement STD: {scored_results[0]['evaluation']['disagreement_std']})")
    return scored_results
