import json
import asyncio
import random
import numpy as np
from backend.openrouter import query_model, query_models_parallel
from config import config

def compute_kendalls_w(rank_matrices: list, num_proposals: int) -> float:
    """
    Computes Kendall's W Coefficient of Concordance across reviewer ranking vectors.
    rank_matrices: list of lists, where each list is rank assigned to proposal 0..K-1.
    Returns W in [0.0, 1.0]. 1.0 = perfect consensus, 0.0 = complete disagreement.
    """
    m = len(rank_matrices) # Number of reviewers
    k = num_proposals # Number of proposals
    
    if m < 2 or k < 2:
        return 1.0

    # Sum of ranks for each proposal
    rank_sums = np.sum(rank_matrices, axis=0)
    mean_rank_sum = np.mean(rank_sums)
    
    s = np.sum((rank_sums - mean_rank_sum) ** 2)
    w = (12.0 * s) / ((m ** 2) * (k ** 3 - k))
    return float(np.clip(w, 0.0, 1.0))

async def run_anonymized_council_review(proposals: list) -> list:
    """
    Layer 3: Multi-Agent Anonymized Council Review & Ranking.
    Presents ALL candidate proposals together to reviewer models with randomized labels,
    aggregates ordinal rank orders via Borda count, calculates Kendall's W concordance & rank disagreement,
    and ranks by Utility-Adjusted Expected Value incorporating Information Gain & Novelty.
    """
    print(f"[Council] Initiating Multi-Proposal Anonymized Peer Review across {len(proposals)} candidate hypotheses...")
    
    if not proposals:
        return []

    labels = [f"Proposal {chr(65 + i)}" for i in range(len(proposals))]
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
3. Information Gain (learning value regardless of success/failure)
4. Architectural Novelty (non-redundancy vs standard modifications)
5. Utility (Expected Value / Implementation Cost)

Provide your evaluation STRICTLY as a JSON dictionary matching this exact schema:
{{
    "rankings": ["Proposal A", "Proposal B", "Proposal C"], // Explicit order from 1st best to last
    "evaluations": {{
        "Proposal A": {{
            "expected_improvement": 0.08,   // Delta reduction in BPB (> 0.0)
            "probability_of_success": 0.70, // Float 0.0 to 1.0
            "information_gain": 0.60,       // Float 0.0 to 1.0
            "novelty": 0.50,                // Float 0.0 to 1.0
            "implementation_cost": 0.01,   // Estimated compute cost
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

    responses = await query_models_parallel(config.COUNCIL_MODELS, messages)

    reviewer_rank_lists = []
    proposal_evals = {lbl: [] for lbl in label_to_prop.keys()}

    for r in responses:
        if isinstance(r, dict) and r.get("status") == "SUCCESS" and r.get("content"):
            try:
                content = r["content"]
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                else:
                    json_str = content.strip()
                data = json.loads(json_str)

                if "rankings" in data and isinstance(data["rankings"], list):
                    reviewer_rank_lists.append(data["rankings"])
                
                evals = data.get("evaluations", {})
                for lbl, ev_info in evals.items():
                    if lbl in proposal_evals:
                        imp = float(ev_info.get("expected_improvement", 0.05))
                        prob = float(ev_info.get("probability_of_success", 0.5))
                        info = float(ev_info.get("information_gain", 0.3))
                        nov = float(ev_info.get("novelty", 0.3))
                        cost = max(float(ev_info.get("implementation_cost", 0.01)), 0.001)
                        
                        ev = (prob * imp + config.LAMBDA_INFO * info + config.BETA_NOVELTY * nov) / cost

                        proposal_evals[lbl].append({
                            "imp": imp,
                            "prob": prob,
                            "info": info,
                            "nov": nov,
                            "cost": cost,
                            "ev": ev,
                            "reasoning": ev_info.get("reasoning", "")
                        })
            except Exception:
                continue

    if not reviewer_rank_lists or sum(len(v) for v in proposal_evals.values()) == 0:
        print(f"[Council Notice] OpenRouter API unavailable or returned unparseable reviews.")
        return []

    # Calculate Mean-Rank Matrix & Kendall's W Concordance
    lbl_list = list(label_to_prop.keys())
    num_props = len(lbl_list)
    rank_matrices = []

    for r_list in reviewer_rank_lists:
        r_vector = []
        for lbl in lbl_list:
            if lbl in r_list:
                r_vector.append(r_list.index(lbl) + 1)
            else:
                r_vector.append(num_props)
        rank_matrices.append(r_vector)

    kendalls_w = compute_kendalls_w(rank_matrices, num_props)
    rank_disagreement = round(1.0 - kendalls_w, 4)

    avg_ranks = np.mean(rank_matrices, axis=0)

    scored_results = []
    for idx, lbl in enumerate(lbl_list):
        prop = label_to_prop[lbl]
        evals = proposal_evals[lbl]
        if not evals:
            continue
        
        avg_imp = float(np.mean([e["imp"] for e in evals]))
        avg_prob = float(np.mean([e["prob"] for e in evals]))
        avg_info = float(np.mean([e["info"] for e in evals]))
        avg_nov = float(np.mean([e["nov"] for e in evals]))
        avg_cost = float(np.mean([e["cost"] for e in evals]))
        avg_ev = float(np.mean([e["ev"] for e in evals]))

        mean_rank = float(avg_ranks[idx])
        rank_bonus = 0.5 + 0.5 * ((num_props - mean_rank + 1) / num_props)
        final_score = avg_ev * rank_bonus

        prop_copy = dict(prop)
        prop_copy["label"] = lbl
        prop_copy["evaluation"] = {
            "expected_improvement": round(avg_imp, 4),
            "probability_of_success": round(avg_prob, 4),
            "information_gain": round(avg_info, 4),
            "novelty": round(avg_nov, 4),
            "implementation_cost": round(avg_cost, 4),
            "reviewer_count": len(evals),
            "kendalls_w": round(kendalls_w, 4),
            "rank_disagreement": rank_disagreement,
            "mean_rank": round(mean_rank, 2),
            "reasoning": evals[0]["reasoning"]
        }
        prop_copy["utility_ev"] = round(final_score, 4)
        scored_results.append(prop_copy)

    scored_results.sort(key=lambda x: x["utility_ev"], reverse=True)
    print(f"[Council Consensus] Top Proposal: {scored_results[0]['id']} (Score: {scored_results[0]['utility_ev']}, Kendall's W: {kendalls_w:.4f}, Disagreement: {rank_disagreement})")
    return scored_results
