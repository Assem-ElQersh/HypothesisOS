import os
import sys
import time
import json
import asyncio
import subprocess
import shutil
from datetime import datetime

# Add llm-council to path so we can use its backend
sys.path.append(os.path.join(os.path.dirname(__file__), "llm-council"))
from backend.openrouter import query_model

PROPOSER_MODEL = "google/gemini-2.5-flash"
CHAIRMAN_MODEL = "google/gemini-3-pro-preview"

def read_file(filepath):
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def write_file(filepath, content):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

# Load the stable system prompt once
SYSTEM_PROMPT = read_file(".agents/system_prompt.txt")

async def generate_proposals(plan_content, train_code):
    """Layer 2: Proposal Engine"""
    user_prompt = f"""Your goal is to decrease validation BPB in a 5-minute training budget.
Generate EXACTLY 3 mutually exclusive, diverse hypotheses (patches) for train.py.

CURRENT RESEARCH PLAN:
{plan_content}

CURRENT TRAIN.PY:
{train_code}

For each hypothesis, provide:
1. Mechanism of improvement
2. Risk level & Complexity
3. The EXACT Python code replacement for train.py. Provide it as a list of search/replace blocks.

Format your output STRICTLY as a JSON list of dictionaries with this exact schema:
[
  {{
    "id": "HYP-001",
    "mechanism": "Use adamw to improve convergence",
    "risk": "Low",
    "code_changes": "@@ -6,3 +6,3 @@\\n     time.sleep(1)\\n-    val_bpb = 3.42 + random.uniform(-0.05, 0.05)\\n+    val_bpb = 3.20 + random.uniform(-0.05, 0.05)\\n     peak_vram_mb = 18000"
  }}
]
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    print("Generating proposals...")
    response = await query_model(PROPOSER_MODEL, messages, timeout=120.0)
    if not response or not response.get('content'):
        raise Exception("Failed to generate proposals.")
    
    content = response['content']
    if "```json" in content:
        json_str = content.split("```json")[1].split("```")[0].strip()
    else:
        json_str = content.strip()
        
    proposals = json.loads(json_str)
    # Basic validation
    assert len(proposals) > 0, "Must generate at least 1 proposal"
    return proposals

async def run_council_gate(proposals):
    """Layer 3: Council Review & Selection"""
    print("Running Council Review...")
    
    scored_proposals = []
    
    for p in proposals:
        query = f"""Evaluate this proposed patch for train.py:
Mechanism: {p['mechanism']}
Risk: {p['risk']}
Changes: {p['code_changes']}

Provide your evaluation STRICTLY as a JSON object with this schema:
{{
    "expected_improvement": 0.1,  // float (estimated delta in BPB)
    "probability_of_success": 0.5, // float between 0 and 1
    "implementation_cost": 0.1,   // float in GPU hours
    "failure_severity": "low",    // string (low/medium/high)
    "reasoning": "..."            // string
}}
"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
        
        response = await query_model(CHAIRMAN_MODEL, messages)
        content = response['content']
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        else:
            json_str = content.strip()
            
        evaluation = json.loads(json_str)
        expected_improvement = evaluation.get("expected_improvement", 0.0)
        prob_success = evaluation.get("probability_of_success", 0.0)
        
        # Calculate EV programmatically
        ev = expected_improvement * prob_success
        p["evaluation"] = evaluation
        p["expected_value"] = ev
        scored_proposals.append(p)
        
    # Select the highest expected-value experiment
    scored_proposals.sort(key=lambda x: x["expected_value"], reverse=True)
    best_hyp = scored_proposals[0]
    
    print(f"Council selected: {best_hyp['id']} with EV: {best_hyp['expected_value']:.4f}")
    return best_hyp

def apply_patch(train_path, patch):
    """Apply the unified diff string to the target file."""
    # Since writing a full patch applicator is complex in a mock script,
    # we'll use a very simplified approach: we just let the patch fail if it's 
    # not a valid unified diff that standard 'patch' can handle, or we 
    # extract the new code manually if it's just a simple string replacement mock.
    
    # In our mock environment, code_changes might be simple diffs.
    # Let's save the diff to a file and use the `patch` CLI tool.
    with open("temp.patch", "w") as f:
        f.write(patch["code_changes"] + "\n")
        
    try:
        subprocess.run(["patch", train_path, "temp.patch"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Patch failed: {e.stderr}")
        return False
    finally:
        if os.path.exists("temp.patch"):
            os.remove("temp.patch")

def execute_experiment(patch):
    """Layer 4: Execution"""
    train_path = "autoresearch/train.py"
    backup_path = "autoresearch/train.py.bak"
    
    # 1. Backup baseline
    if os.path.exists(train_path):
        shutil.copy2(train_path, backup_path)
    
    # 2. Apply patch
    print(f"Applying patch {patch['id']} to {train_path}...")
    success = apply_patch(train_path, patch)
    if not success:
        # Revert
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, train_path)
        return {
            "status": "INVALID",
            "val_bpb": None,
            "mem_gb": None,
            "runtime": 0,
            "log": "Failed to apply patch."
        }
    
    print("Executing experiment via modal_runner.py...")
    cmd = "python modal_runner.py"
    start_time = time.time()
    
    status = "FAILED"
    try:
        # In a real environment, this spins up the Modal runner
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        pass
        
    runtime = time.time() - start_time
    
    # 3. Parse results from log
    val_bpb = None
    mem_gb = None
    log_content = ""
    try:
        log_content = read_file("autoresearch/run.log")
        # Check if the runner itself reported a specific error
        if "TIMEOUT EXPIRED" in log_content:
            status = "TIMEOUT"
        elif "OutOfMemoryError" in log_content:
            status = "OOM"
        else:
            status = "VALID" # Default assumed valid if log exists, unless metrics are missing
            
        for line in log_content.splitlines():
            if line.startswith("val_bpb:"):
                val_bpb = float(line.split()[1])
            elif line.startswith("peak_vram_mb:"):
                mem_gb = float(line.split()[1]) / 1024.0
    except Exception as e:
        print(f"Failed to parse log: {e}")
        status = "FAILED"
        
    # Priority 1: Make invalid evidence impossible to interpret as success
    if val_bpb is None and status == "VALID":
        status = "METRIC_MISSING"

    # Restore baseline for the next experiment (or keep it if we accept it, but typically we want isolated trials)
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, train_path)
        
    return {
        "status": status,
        "val_bpb": val_bpb,
        "mem_gb": round(mem_gb, 1) if mem_gb else None,
        "runtime": runtime,
        "log": log_content
    }

async def run_postmortem(experiment_results, patch, baseline_bpb):
    """Layer 6: Postmortem Council"""
    user_prompt = f"""Perform a postmortem on this experiment.
    
HYPOTHESIS: {patch['id']} - {patch['mechanism']}
APPLIED DIFF:
{patch['code_changes']}

BASELINE BPB: {baseline_bpb}
TREATMENT BPB: {experiment_results['val_bpb']}
OUTCOME STATUS: {experiment_results['status']}
LOG SNIPPET:
{experiment_results['log'][-2000:]}

Answer these questions:
1. What changed mathematically and computationally?
2. Was the hypothesis validated?
3. What evidence supports/contradicts?
4. What should be attempted next?
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    print("Running Postmortem...")
    response = await query_model(CHAIRMAN_MODEL, messages)
    return response['content'] if response else "Postmortem failed."

def update_research_state(patch, results, postmortem, baseline_bpb):
    state_path = "memory/research_state.json"
    state = {}
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
            
    val_bpb = results["val_bpb"]
    delta = (val_bpb - baseline_bpb) if val_bpb is not None else None
    
    # Log hypothesis outcome
    state.setdefault("hypotheses", {})[patch["id"]] = {
        "status": results["status"],
        "confidence": patch["evaluation"]["probability_of_success"], # Dummy before confidence
        "delta_bpb": delta,
        "postmortem": postmortem
    }
    
    if results["status"] == "VALID" and delta is not None and delta < 0:
        state["proven_wins"].append(patch["id"])
        state["current_best"]["val_bpb"] = val_bpb
        state["current_best"]["experiment_id"] = patch["id"]
    elif results["status"] in ["VALID", "FAILED", "OOM", "TIMEOUT"]:
        state["proven_failures"].append(patch["id"])
        
    state.setdefault("trajectory", []).append({
        "timestamp": datetime.now().isoformat(),
        "hypothesis_id": patch["id"],
        "delta": delta,
        "status": results["status"]
    })
    
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

def log_to_ledger(patch, expected_improvement, results, baseline_bpb, conf_before):
    ledger_path = "research_ledger.tsv"
    
    # Priority 6: Ledger format
    # Hypothesis ID | Parent Hypothesis | Patch Applied | Expected Outcome | Actual Outcome | Confidence Before | Confidence After
    
    if not os.path.exists(ledger_path):
        with open(ledger_path, "w") as f:
            f.write("Hypothesis ID\tParent Hypothesis\tPatch Applied\tExpected Outcome\tActual Outcome\tConfidence Before\tConfidence After\n")
            
    val_bpb = results["val_bpb"]
    delta = (val_bpb - baseline_bpb) if val_bpb is not None else "N/A"
    actual_outcome = f"{results['status']} (Delta: {delta})"
    conf_after = "High" if results["status"] == "VALID" and delta != "N/A" and delta < 0 else "Low"
    
    # Basic inline diff preview
    patch_preview = patch['code_changes'].replace("\n", "\\n")[:50]
    
    row = f"{patch['id']}\tBaseline\t{patch_preview}...\tDecrease BPB by {expected_improvement}\t{actual_outcome}\t{conf_before}\t{conf_after}\n"
    
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(row)

async def main_loop():
    print("=== Starting HypothesisOS Orchestrator ===")
    
    # 0. Load State
    state_path = "memory/research_state.json"
    if not os.path.exists(state_path):
        print("Error: Research state missing.")
        return
        
    with open(state_path, "r") as f:
        state = json.load(f)
    
    baseline_bpb = state.get("current_best", {}).get("val_bpb", 0.0)
    
    plan_content = read_file("research_plan.md")
    train_code = read_file("autoresearch/train.py")
    
    # 1. Proposal
    proposals = await generate_proposals(plan_content, train_code)
    
    # 2. Critique
    best_patch = await run_council_gate(proposals)
    
    # 3. Execution
    results = execute_experiment(best_patch)
    
    # 4. Postmortem
    postmortem = await run_postmortem(results, best_patch, baseline_bpb)
    
    # 5. Ledger & State Update
    update_research_state(best_patch, results, postmortem, baseline_bpb)
    log_to_ledger(
        patch=best_patch,
        expected_improvement=best_patch["evaluation"]["expected_improvement"],
        results=results,
        baseline_bpb=baseline_bpb,
        conf_before=best_patch["evaluation"]["probability_of_success"]
    )
    
    print("\n=== Experiment Complete ===")
    print(f"Winning Hypothesis: {best_patch['id']}")
    print(f"Result: {results['val_bpb']} BPB (Status: {results['status']})")
    print(f"Postmortem:\n{postmortem}")

if __name__ == "__main__":
    asyncio.run(main_loop())
