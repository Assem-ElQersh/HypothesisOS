import os
import sys
import time
import json
import asyncio
from datetime import datetime
import subprocess

# Add llm-council to path so we can use its backend
sys.path.append(os.path.join(os.path.dirname(__file__), "llm-council-master"))
from backend.openrouter import query_model, query_models_parallel
from backend.council import stage2_collect_rankings, calculate_aggregate_rankings

PROPOSER_MODEL = "google/gemini-2.5-flash"
CHAIRMAN_MODEL = "google/gemini-3-pro-preview"

def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def write_file(filepath, content):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

async def generate_proposals(plan_content, train_code):
    """Layer 2: Proposal Engine"""
    prompt = f"""You are the Proposal Engine for an AI Research Scientist.
Your goal is to decrease validation BPB in a 5-minute training budget.
Generate 3 mutually exclusive, diverse hypotheses (patches) for train.py.

CURRENT RESEARCH PLAN:
{plan_content}

CURRENT TRAIN.PY:
{train_code}

For each hypothesis, provide:
1. Mechanism of improvement
2. Risk level & Complexity
3. The EXACT Python code replacement for train.py in a unified diff format or clearly specified search/replace blocks.

Format your output as a JSON list of dictionaries:
[
  {{
    "id": "HYP-XXX",
    "mechanism": "...",
    "risk": "...",
    "code_changes": "..."
  }}
]
"""
    messages = [{"role": "user", "content": prompt}]
    print("Generating proposals...")
    response = await query_model(PROPOSER_MODEL, messages, timeout=120.0)
    if not response or not response.get('content'):
        raise Exception("Failed to generate proposals.")
    
    # Simple extraction of JSON from markdown blocks
    content = response['content']
    if "```json" in content:
        json_str = content.split("```json")[1].split("```")[0].strip()
    else:
        json_str = content.strip()
        
    return json.loads(json_str)

async def run_council_gate(proposals):
    """Layer 3: Council Review & Selection"""
    # Format proposals as stage1 results to reuse llm-council ranking
    stage1_results = []
    for p in proposals:
        stage1_results.append({
            "model": p["id"],
            "response": f"Mechanism: {p['mechanism']}\nRisk: {p['risk']}\nChanges: {p['code_changes']}"
        })
    
    print("Running Council Review...")
    query = "Evaluate these proposed patches for train.py based on: 1. Bug Risk 2. Theoretical Weakness 3. Expected Gain. Rank them from best to worst."
    stage2_results, label_to_model = await stage2_collect_rankings(query, stage1_results)
    
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
    if not aggregate_rankings:
        raise Exception("Council failed to reach a ranking.")
        
    best_hyp_id = aggregate_rankings[0]["model"]
    print(f"Council selected: {best_hyp_id}")
    
    # Return the winning proposal
    for p in proposals:
        if p["id"] == best_hyp_id:
            return p, aggregate_rankings
    return proposals[0], aggregate_rankings

def execute_experiment(patch):
    """Layer 4: Execution"""
    print("Applying patch and executing experiment...")
    # NOTE: In a full implementation, we'd programmatically apply the patch to train.py here.
    # For now, we simulate execution.
    
    cmd = "cd autoresearch-master && uv run train.py > run.log 2>&1"
    start_time = time.time()
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        status = "SUCCESS"
    except subprocess.CalledProcessError:
        status = "CRASH"
        
    runtime = time.time() - start_time
    
    # Parse results
    val_bpb = 0.0
    mem_gb = 0.0
    log_content = ""
    try:
        log_content = read_file("autoresearch-master/run.log")
        for line in log_content.splitlines():
            if line.startswith("val_bpb:"):
                val_bpb = float(line.split()[1])
            elif line.startswith("peak_vram_mb:"):
                mem_gb = float(line.split()[1]) / 1024.0
    except Exception as e:
        print(f"Failed to parse log: {e}")
        
    return {
        "status": status,
        "val_bpb": val_bpb,
        "mem_gb": round(mem_gb, 1),
        "runtime": runtime,
        "log": log_content
    }

async def run_postmortem(experiment_results, patch):
    """Layer 6: Postmortem Council"""
    prompt = f"""Perform a postmortem on this experiment.
    
HYPOTHESIS:
{patch['mechanism']}

OUTCOME STATUS: {experiment_results['status']}
VAL BPB: {experiment_results['val_bpb']}
LOG SNIPPET:
{experiment_results['log'][-2000:]}

Answer these questions:
1. What changed?
2. Why did it change?
3. Was the hypothesis validated?
4. What evidence supports/contradicts?
5. What should be attempted next?
"""
    messages = [{"role": "user", "content": prompt}]
    print("Running Postmortem...")
    response = await query_model(CHAIRMAN_MODEL, messages)
    return response['content'] if response else "Postmortem failed."

def log_to_ledger(hyp_id, mechanism, expected, actual_status, val_bpb):
    ledger_path = "research_ledger.tsv"
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(f"{hyp_id}\tBaseline\t{mechanism[:50]}...\tDecrease BPB\t{actual_status} ({val_bpb})\tMedium\tUnknown\n")

async def main_loop():
    print("=== Starting HypothesisOS Orchestrator ===")
    
    plan_content = read_file("research_plan.md")
    train_code = read_file("autoresearch-master/train.py")
    
    # 1. Proposal
    proposals = await generate_proposals(plan_content, train_code)
    
    # 2. Critique
    best_patch, rankings = await run_council_gate(proposals)
    
    # 3. Execution
    results = execute_experiment(best_patch)
    
    # 4. Postmortem
    postmortem = await run_postmortem(results, best_patch)
    
    # 5. Ledger
    log_to_ledger(best_patch["id"], best_patch["mechanism"], "Better BPB", results["status"], results["val_bpb"])
    
    print("\n=== Experiment Complete ===")
    print(f"Winning Hypothesis: {best_patch['id']}")
    print(f"Result: {results['val_bpb']} BPB")
    print(f"Postmortem:\n{postmortem}")

if __name__ == "__main__":
    asyncio.run(main_loop())
