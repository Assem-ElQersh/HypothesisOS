import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import sys
import time
import json
import ast
import py_compile
import asyncio
import subprocess
import shutil
import argparse
import hashlib
import re
from datetime import datetime

# Path setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm-council"))

from config import config
from backend.openrouter import query_model
from backend.council import run_anonymized_council_review
from memory.research_tree import ResearchTree

def read_file(filepath):
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def write_file(filepath, content):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

def clean_llm_text(text: str) -> str:
    """ Strips LLM artifact tokens like <|tool_call_start|> or markdown system wrappers """
    if not text:
        return ""
    text = re.sub(r"<\|.*?\|>", "", text)
    return text.strip()

def get_file_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

INITIAL_EVALUATOR_HASH = get_file_sha256(config.EVALUATE_SCRIPT)

def lock_evaluator_file(filepath: str):
    """ Locks evaluate.py with read-only permissions (chmod 444) """
    if os.path.exists(filepath):
        try:
            os.chmod(filepath, 0o444)
        except Exception:
            pass

def verify_evaluator_integrity(filepath: str) -> bool:
    current_hash = get_file_sha256(filepath)
    if INITIAL_EVALUATOR_HASH and current_hash != INITIAL_EVALUATOR_HASH:
        print(f"[SECURITY ALERT] Evaluator integrity check FAILED! File {filepath} was modified!")
        return False
    return True

SYSTEM_PROMPT = read_file(".agents/system_prompt.txt")
if not SYSTEM_PROMPT:
    SYSTEM_PROMPT = (
        "You are an autonomous AI ML Research Scientist. "
        "Your objective is to propose scientific hypotheses to decrease validation BPB in a PyTorch language model. "
        "Generate realistic code modifications for train.py only."
    )

async def generate_proposals(plan_content: str, train_code: str, tree_state: dict) -> list:
    """ Layer 2: Proposal Engine with Memory Insights """
    insights = tree_state.get("research_insights", {})
    insights_str = (
        f"PROVEN WINS ({len(insights.get('proven_wins', []))}):\n" +
        ("\n".join(insights.get("proven_wins", [])) if insights.get("proven_wins") else "None yet.") + "\n\n" +
        f"PROVEN FAILURES ({len(insights.get('proven_failures', []))}):\n" +
        ("\n".join(insights.get("proven_failures", [])) if insights.get("proven_failures") else "None yet.")
    )

    user_prompt = f"""You are the Proposal Engine for an autonomous ML research scientist.
Goal: Decrease validation BPB (bits per byte) on a PyTorch Transformer language model.

CURRENT RESEARCH STATE:
{plan_content}

SYNTHESIZED RESEARCH INSIGHTS (DO NOT REPEAT KNOWN FAILURES):
{insights_str}

CURRENT TRAIN.PY CODE:
```python
{train_code}
```

REQUIREMENTS:
1. Generate EXACTLY 3 mutually exclusive, diverse hypotheses (code patches for train.py).
2. DO NOT alter evaluation code or override metric computations. Only modify hyperparameters, neural network layers, optimizer, or training schedule in train.py.
3. Provide code changes as clean, unified diff strings or replace blocks.

Format your output STRICTLY as a JSON list of dictionaries with this exact schema:
[
  {{
    "id": "HYP-101",
    "mechanism": "Increase transformer layers from 2 to 4 to improve representation capacity",
    "risk": "Medium",
    "code_changes": "@@ -17,1 +17,1 me\\n-n_layer = 2\\n+n_layer = 4"
  }}
]
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    print("[Proposal Engine] Generating 3 candidate hypotheses...")
    response = await query_model(config.PROPOSER_MODEL, messages, timeout=120.0)
    
    if not response or response.get("status") != "SUCCESS" or not response.get("content"):
        err = response.get("error", "API Call Failed")
        print(f"[Proposal Error] LLM Query Failed: {err}")
        return []

    content = response['content']
    if "```json" in content:
        json_str = content.split("```json")[1].split("```")[0].strip()
    else:
        json_str = content.strip()

    try:
        proposals = json.loads(json_str)
        assert isinstance(proposals, list) and len(proposals) > 0, "Must generate a non-empty proposal list"
        return proposals
    except Exception as e:
        print(f"[Proposal Error] JSON parsing failed: {e}")
        return []

def validate_patch_security(patch: dict) -> bool:
    """
    IMMUTABLE EVALUATION BOUNDARY GUARD:
    Ensures proposals ONLY modify train.py and do NOT tamper with evaluation,
    ground-truth val_bpb calculation, or import shortcuts.
    """
    code_changes = patch.get("code_changes", "")
    
    forbidden_targets = ["evaluate.py", "prepare.py"]
    for forbidden in forbidden_targets:
        if forbidden in code_changes:
            print(f"[Security Guard] REJECTED {patch['id']}: Attempted to modify immutable file '{forbidden}'.")
            return False

    if "val_bpb =" in code_changes or "val_bpb:" in code_changes:
        print(f"[Security Guard] REJECTED {patch['id']}: Attempted to hardcode ground-truth val_bpb metric.")
        return False

    return True

def validate_patch_ast_syntax(filepath: str) -> bool:
    """
    PRE-EXECUTION AST SYNTAX & COMPILATION VALIDATION:
    Ensures proposed Python code parses cleanly before wasting compute time.
    """
    try:
        source = read_file(filepath)
        ast.parse(source)
        py_compile.compile(filepath, doraise=True)
        return True
    except (SyntaxError, py_compile.PyCompileError) as e:
        print(f"[AST Syntax Check Failed] {e}")
        return False

def apply_patch_to_code(train_code: str, patch: dict) -> tuple[bool, str]:
    """ Apply diff or replacement block to train.py """
    patch_str = patch.get("code_changes", "")
    lines = train_code.splitlines(keepends=True)

    if "@@" in patch_str:
        temp_patch = "temp_proposal.patch"
        write_file(temp_patch, patch_str + "\n")
        
        try:
            res = subprocess.run(["patch", config.TRAIN_SCRIPT, temp_patch], capture_output=True, text=True)
            if res.returncode == 0:
                patched_code = read_file(config.TRAIN_SCRIPT)
                return True, patched_code
            else:
                print(f"[Patch Error] Unified diff failed: {res.stderr}")
        except Exception as e:
            print(f"[Patch Error] {e}")
        finally:
            if os.path.exists(temp_patch):
                os.remove(temp_patch)

    try:
        new_code = train_code
        for line in patch_str.splitlines():
            if line.startswith("-") and not line.startswith("---"):
                old_line = line[1:].strip()
            elif line.startswith("+") and not line.startswith("+++"):
                new_line = line[1:].strip()
                if 'old_line' in locals() and old_line in new_code:
                    new_code = new_code.replace(old_line, new_line, 1)

        if new_code != train_code:
            write_file(config.TRAIN_SCRIPT, new_code)
            return True, new_code
    except Exception as e:
        print(f"[Patch Error] Fallback replacement failed: {e}")

    return False, train_code

def execute_experiment(patch: dict, backend: str = config.EXECUTION_BACKEND) -> dict:
    """
    Layer 5: Isolated Execution Layer with Cryptographic Integrity Verification.
    Dispatches to local Python process or Modal container.
    """
    # 0. Evaluator Cryptographic Integrity Verification
    if not verify_evaluator_integrity(config.EVALUATE_SCRIPT):
        return {"status": "SECURITY_VIOLATION", "val_bpb": None, "val_loss": None, "runtime": 0, "log": "Evaluator file tampering detected!"}

    train_path = config.TRAIN_SCRIPT
    backup_path = train_path + ".bak"

    shutil.copy2(train_path, backup_path)

    # 1. Security Guard
    if not validate_patch_security(patch):
        return {"status": "SECURITY_VIOLATION", "val_bpb": None, "val_loss": None, "runtime": 0, "log": "Security violation"}

    # 2. Apply Patch
    print(f"[Execution] Applying patch {patch['id']}...")
    success, patched_code = apply_patch_to_code(read_file(backup_path), patch)
    if not success:
        shutil.copy2(backup_path, train_path)
        return {"status": "INVALID_PATCH", "val_bpb": None, "val_loss": None, "runtime": 0, "log": "Patch application failed"}

    # 3. Pre-Execution AST Syntax Validation
    if not validate_patch_ast_syntax(train_path):
        print(f"[Execution] Rejected patch {patch['id']} due to syntax error before launch.")
        shutil.copy2(backup_path, train_path)
        return {"status": "INVALID_SYNTAX", "val_bpb": None, "val_loss": None, "runtime": 0, "log": "Syntax validation error"}

    start_time = time.time()

    # 4. Dispatch based on Backend (local vs modal)
    if backend == "modal":
        print(f"[Execution] Dispatching PyTorch run to Modal Container (modal_runner.py)...")
        try:
            modal_res = subprocess.run(
                ["modal", "run", "modal_runner.py"],
                capture_output=True,
                text=True,
                timeout=config.EXECUTION_TIMEOUT_SEC + 60
            )
            log_output = modal_res.stdout + "\n" + modal_res.stderr
            if modal_res.returncode != 0:
                shutil.copy2(backup_path, train_path)
                return {"status": "MODAL_ERROR", "val_bpb": None, "val_loss": None, "runtime": time.time() - start_time, "log": log_output}
            
            # Parse structured JSON output from Modal runner
            if "---MODAL_OUTPUT_JSON_START---" in modal_res.stdout:
                json_part = modal_res.stdout.split("---MODAL_OUTPUT_JSON_START---")[1].split("---MODAL_OUTPUT_JSON_END---")[0].strip()
                eval_data = json.loads(json_part)
                runtime = time.time() - start_time
                return {
                    "status": eval_data.get("status", "VALID"),
                    "val_bpb": eval_data.get("val_bpb"),
                    "val_loss": eval_data.get("val_loss"),
                    "peak_vram_mb": eval_data.get("peak_vram_mb", 0.0),
                    "runtime": round(runtime, 2),
                    "log": eval_data.get("full_log", log_output),
                    "backup_path": backup_path
                }
            else:
                shutil.copy2(backup_path, train_path)
                return {"status": "MODAL_ERROR", "val_bpb": None, "val_loss": None, "runtime": time.time() - start_time, "log": log_output}

        except Exception as e:
            shutil.copy2(backup_path, train_path)
            return {"status": "MODAL_ERROR", "val_bpb": None, "val_loss": None, "runtime": time.time() - start_time, "log": str(e)}

    else:
        print(f"[Execution] Running local PyTorch training script ({config.TRAIN_SCRIPT})...")
        try:
            train_res = subprocess.run(
                [sys.executable, config.TRAIN_SCRIPT],
                capture_output=True,
                text=True,
                timeout=config.EXECUTION_TIMEOUT_SEC
            )
            train_log = train_res.stdout + "\n" + train_res.stderr
            
            if train_res.returncode != 0:
                print(f"[Execution Error] PyTorch training crashed:\n{train_log[-500:]}")
                shutil.copy2(backup_path, train_path)
                return {"status": "CRASH", "val_bpb": None, "val_loss": None, "runtime": time.time() - start_time, "log": train_log}
        except subprocess.TimeoutExpired:
            print("[Execution Error] PyTorch training timed out.")
            shutil.copy2(backup_path, train_path)
            return {"status": "TIMEOUT", "val_bpb": None, "val_loss": None, "runtime": time.time() - start_time, "log": "Timeout expired"}

    # 5. Run Immutable Ground-Truth Evaluator (Local mode)
    print(f"[Execution] Running Immutable Ground-Truth Evaluator ({config.EVALUATE_SCRIPT})...")
    try:
        eval_res = subprocess.run(
            [sys.executable, config.EVALUATE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=30
        )
        eval_log = eval_res.stdout
        eval_data = json.loads(eval_log.strip().splitlines()[-1])
        runtime = time.time() - start_time

        return {
            "status": eval_data.get("status", "VALID"),
            "val_bpb": eval_data.get("val_bpb"),
            "val_loss": eval_data.get("val_loss"),
            "peak_vram_mb": eval_data.get("peak_vram_mb"),
            "runtime": round(runtime, 2),
            "log": eval_log,
            "backup_path": backup_path
        }

    except Exception as e:
        print(f"[Execution Error] Evaluation failed: {e}")
        shutil.copy2(backup_path, train_path)
        return {"status": "METRIC_MISSING", "val_bpb": None, "val_loss": None, "runtime": time.time() - start_time, "log": str(e)}

async def run_postmortem(results: dict, patch: dict, baseline_bpb: float) -> str:
    """ Layer 6: Postmortem Council """
    status = results.get("status")
    val_bpb = results.get("val_bpb")

    if status != "VALID" or val_bpb is None:
        return (
            f"POSTMORTEM ANALYSIS for {patch['id']}:\n"
            f"Execution Status: {status}.\n"
            f"Failure Reason: Experiment failed during execution or evaluation. "
            f"No empirical metric improvements recorded."
        )

    user_prompt = f"""Perform a rigorous postmortem on this ML experiment.

HYPOTHESIS: {patch['id']} - {patch['mechanism']}
APPLIED CODE DIFF:
{patch['code_changes']}

BASELINE BPB: {baseline_bpb}
TREATMENT BPB: {val_bpb}
DELTA BPB: {val_bpb - baseline_bpb:.4f}
OUTCOME STATUS: {status}

Answer:
1. What changed computationally and in terms of model performance?
2. Was the hypothesis empirically validated?
3. What evidence supports or contradicts the mechanism?
4. What next hypothesis should be explored?
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    response = await query_model(config.CHAIRMAN_MODEL, messages)
    if not response or response.get("status") != "SUCCESS":
        return f"Postmortem complete for {patch['id']} (Status: {status})."
    
    raw_content = response.get('content', 'Postmortem complete.')
    return clean_llm_text(raw_content)

def log_to_ledger(patch: dict, results: dict, baseline_bpb: float):
    """ Record experiment into research_ledger.tsv """
    ledger_path = config.RESEARCH_LEDGER_PATH
    if not os.path.exists(ledger_path):
        write_file(
            ledger_path,
            "Hypothesis ID\tParent Hypothesis\tPatch Applied\tExpected Outcome\tActual Outcome\tConfidence Before\tConfidence After\n"
        )

    val_bpb = results.get("val_bpb")
    delta = (val_bpb - baseline_bpb) if (val_bpb is not None and baseline_bpb is not None) else "N/A"
    actual_outcome = f"{results['status']} (Val BPB: {val_bpb}, Delta: {delta})"
    
    conf_before = patch.get("evaluation", {}).get("probability_of_success", 0.5)
    conf_after = "High" if (results["status"] == "VALID" and val_bpb is not None and baseline_bpb is not None and val_bpb < baseline_bpb) else "Low"
    
    patch_snippet = patch.get("code_changes", "").replace("\n", "\\n")[:60]

    row = f"{patch['id']}\t{patch.get('parent_id', 'root')}\t{patch_snippet}\tDecrease BPB\t{actual_outcome}\t{conf_before}\t{conf_after}\n"
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(row)

async def main_autonomous_loop(max_experiments: int = config.DEFAULT_MAX_EXPERIMENTS, backend: str = config.EXECUTION_BACKEND):
    print("=========================================================")
    print("=== HypothesisOS Autonomous ML Research Loop ===")
    print("=========================================================")
    print(f"Backend Execution Mode: {backend}")
    
    lock_evaluator_file(config.EVALUATE_SCRIPT)
    tree = ResearchTree()
    
    # Run initial baseline training & evaluation if needed
    if tree.data["current_best"]["val_bpb"] == float("inf"):
        print("[Setup] Preparing dataset and running baseline training...")
        subprocess.run([sys.executable, config.PREPARE_SCRIPT], check=False)
        subprocess.run([sys.executable, config.TRAIN_SCRIPT], check=False)
        
        print("[Setup] Running initial baseline evaluation...")
        eval_res = subprocess.run([sys.executable, config.EVALUATE_SCRIPT], capture_output=True, text=True)
        try:
            data = json.loads(eval_res.stdout.strip().splitlines()[-1])
            tree.set_baseline(data["val_bpb"], data["val_loss"])
            print(f"[Setup] Ground-truth baseline established: {data['val_bpb']} BPB (Val Loss: {data['val_loss']})")
        except Exception as e:
            print(f"[Setup Error] Initial baseline evaluation failed: {e}\nOutput was: {eval_res.stdout}")
            return

    exp_counter = 0

    while exp_counter < max_experiments:
        exp_counter += 1
        print(f"\n--- [AUTONOMOUS ITERATION {exp_counter}/{max_experiments}] ---")

        # Tree Search: Select parent expansion node using UCB1
        active_node_id = tree.select_expansion_node()
        active_node = tree.data["nodes"].get(active_node_id, {})
        baseline_bpb = active_node.get("val_bpb")
        if baseline_bpb is None:
            baseline_bpb = tree.data["current_best"]["val_bpb"]

        print(f"Selected Expansion Node: '{active_node_id}' (Baseline: {baseline_bpb} BPB)")

        plan_content = read_file(config.RESEARCH_PLAN_PATH)
        train_code = read_file(config.TRAIN_SCRIPT)

        tree.data["research_insights"] = tree.get_research_insights()

        # 1. Proposal Generation (Layer 2)
        proposals = await generate_proposals(plan_content, train_code, tree.data)
        if not proposals:
            print("[Loop Aborted] Proposal generation failed or OpenRouter API unavailable.")
            break

        # 2. Multi-Proposal Council Review & Ranking (Layer 3)
        evaluated_proposals = await run_anonymized_council_review(proposals)
        if not evaluated_proposals:
            print("[Loop Aborted] Council review unavailable. OpenRouter API key missing or models unresponsive.")
            break

        best_hypothesis = evaluated_proposals[0]
        best_hypothesis["parent_id"] = active_node_id

        print(f"[Experiment Selection] Selected Hypothesis: {best_hypothesis['id']}")
        print(f"  Mechanism: {best_hypothesis['mechanism']}")
        print(f"  Utility EV Score: {best_hypothesis['utility_ev']}")

        node_id = tree.add_experiment_node(best_hypothesis, parent_id=active_node_id)

        # 3. Execution (Layer 5)
        results = execute_experiment(best_hypothesis, backend=backend)

        # 4. Postmortem (Layer 6)
        postmortem = await run_postmortem(results, best_hypothesis, baseline_bpb)

        # 5. KEEP-OR-REVERT SEMANTICS
        val_bpb = results.get("val_bpb")
        is_win = (results["status"] == "VALID" and val_bpb is not None and val_bpb < baseline_bpb)
        backup_path = results.get("backup_path", config.TRAIN_SCRIPT + ".bak")

        if is_win:
            print(f"🎉 [WINNER!] {best_hypothesis['id']} improved BPB from {baseline_bpb} to {val_bpb} (Delta: {val_bpb - baseline_bpb:.4f})")
            print(f"[Keep-or-Revert] KEPT patch in {config.TRAIN_SCRIPT}. New baseline established!")
            if os.path.exists(backup_path):
                os.remove(backup_path)
        else:
            print(f"❌ [REJECTED] {best_hypothesis['id']} failed to beat baseline (Status: {results['status']}, Val BPB: {val_bpb}).")
            print(f"[Keep-or-Revert] REVERTED {config.TRAIN_SCRIPT} to previous baseline.")
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, config.TRAIN_SCRIPT)
                os.remove(backup_path)

        # 6. Update Research Tree & Ledger
        tree.update_experiment_result(node_id, results, postmortem, baseline_bpb)
        log_to_ledger(best_hypothesis, results, baseline_bpb)

        print(f"[Iteration {exp_counter} Complete] Research Tree state saved.")

    print("\n=========================================================")
    print("=== Autonomous Research Search Session Complete ===")
    print(f"Final Best Metric: {tree.data['current_best']['val_bpb']} BPB")
    print(f"Total Proven Wins: {len(tree.data['proven_wins'])}")
    print(f"Total Proven Failures: {len(tree.data['proven_failures'])}")
    print("=========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HypothesisOS Autonomous Researcher")
    parser.add_argument("--max-experiments", type=int, default=config.DEFAULT_MAX_EXPERIMENTS, help="Maximum number of experiments")
    parser.add_argument("--execution-backend", type=str, default=config.EXECUTION_BACKEND, choices=["local", "modal"], help="Execution backend (local or modal)")
    args = parser.parse_args()

    asyncio.run(main_autonomous_loop(max_experiments=args.max_experiments, backend=args.execution_backend))
