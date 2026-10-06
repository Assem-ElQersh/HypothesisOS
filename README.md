# HypothesisOS

HypothesisOS is a hierarchical AI research scientist framework designed to maximize empirical ML research progress per unit of compute.

It marries two core paradigms:
1. **AutoResearch (Execution):** A PyTorch empirical training loop bounded by a **100% self-contained immutable ground-truth evaluator** (`evaluate.py`), utilizing true wall-clock compute budgets and **Keep-or-Revert** branch semantics.
2. **LLM Council (Judgment):** A multi-agent consensus system running **multi-proposal anonymized peer review** across parallel LLM models to rank candidate proposals by Utility-Adjusted Expected Value (EV) and inter-reviewer model disagreement.

---

## Key Scientific Architecture & Rigorous Guards

### 1. Hard Evaluation Boundary (`evaluate.py` vs `train.py`)
- **Mutable Space (`autoresearch/train.py`)**: The LLM Proposal Engine generates code patches for model architecture, hyperparameters, optimizers, and learning rate schedules in `train.py`.
- **Immutable Space (`autoresearch/evaluate.py`)**: `evaluate.py` is **100% self-contained** and imports zero objects from `train.py`. It constructs its own evaluation model architecture directly from `model.pt` metadata and evaluates ground-truth loss on `data/val.pt`.
- **Pre-Execution AST Syntax Guard**: Patched Python source is validated via `ast.parse()` and `py_compile.compile()` before execution. Invalid syntax is rejected immediately as `INVALID_SYNTAX`.
- **Security Guard**: Static code analysis rejects any patch attempting to modify `evaluate.py` or hardcode ground-truth metrics.

### 2. Multi-Proposal Council Ranking & Disagreement
- All candidate proposals (`Proposal A`, `Proposal B`, `Proposal C`) are presented simultaneously to each reviewer model with randomized labels.
- Evaluated in parallel across peer-review models (`google/gemini-2.5-flash`, `anthropic/claude-3.5-sonnet`, `meta-llama/llama-3.3-70b-instruct`).
- **Utility-Adjusted Expected Value Selection**:
  $$\text{Utility EV} = \frac{\text{Expected Improvement} \times \text{Probability of Success}}{\text{Implementation Cost (GPU hours)}}$$
- Measures and records inter-reviewer model disagreement standard deviation.

### 3. Wall-Clock Compute Budget & Keep-or-Revert Semantics
- **Wall-Clock Time Budget Loop**: Every model iteration executes inside a wall-clock deadline loop (`while time.time() < deadline:`), guaranteeing every candidate architecture gets an identical compute opportunity.
- **Keep-or-Revert Branch Management**:
  - **WIN** (`val_bpb < baseline_bpb`): The code patch remains **KEPT** in `train.py` on disk as the new baseline for subsequent iterations.
  - **LOSS / CRASH / FAILED**: The code is immediately **REVERTED** to the baseline backup.

### 4. Non-Leaky Benchmark & Reproducibility
- **Non-Leaky Corpus**: Uses a diverse, non-repeating character-level text dataset with strictly disjoint train and validation token splits.
- **Determinism**: Enforces fixed random seeds (`torch.manual_seed(42)`) across training data ordering, model initialization, and validation evaluation.

### 5. Consolidated Research Tree Memory
- **Research Tree (`memory/research_tree.json`)**: Manages a single authoritative directed experiment graph tracking hypothesis node parentage, git diffs, metrics, and postmortems.
- **Strict API Error Handling**: If OpenRouter API keys are unconfigured or calls fail, the system aborts selection rather than fabricating fake mock scores or hallucinated postmortems.

---

## Project Structure

```text
HypothesisOS/
├── config.py                 # System configuration, execution backend, & model choices
├── orchestrator.py           # Rebuilt continuous loop with AST syntax check & keep-or-revert
├── modal_runner.py           # Offloads PyTorch execution to Modal GPU container
├── research_plan.md          # Dynamically updated research trajectory & plan
├── research_ledger.tsv       # Hard record of Hypothesis -> Expected -> Actual BPB
├── autoresearch/
│   ├── prepare.py            # Non-leaky dataset preparation script
│   ├── train.py              # Mutable PyTorch model & wall-clock training loop
│   └── evaluate.py           # 100% self-contained immutable evaluator (0 imports from train.py)
├── llm-council/
│   └── backend/
│       ├── openrouter.py     # Async OpenRouter API client with prompt caching & error guards
│       └── council.py        # Multi-proposal anonymized peer review & disagreement scoring
├── memory/
│   ├── research_tree.py      # Directed graph memory data structure
│   └── research_tree.json    # Single authoritative persistent research tree state
└── .agents/                  # Master agent policy & operational rules
```

---

## Quickstart

### Prerequisites
- Python 3.10+
- PyTorch & `httpx` (`pip install torch httpx numpy`)
- Set `OPENROUTER_API_KEY` environment variable for live multi-model LLM calls:
  ```bash
  export OPENROUTER_API_KEY="your-openrouter-key"
  ```

### Running an Autonomous Research Session

Run a 3-iteration autonomous research search session:

```bash
# Local Execution
python orchestrator.py --max-experiments 3 --execution-backend local

# Modal Container GPU Execution
python orchestrator.py --max-experiments 3 --execution-backend modal
```

The orchestrator will:
1. Establish ground-truth baseline evaluation via self-contained `evaluate.py`.
2. Generate candidate code patches for `train.py`.
3. Pre-validate syntax via AST parsing.
4. Conduct multi-proposal anonymized peer review across LLM Council models.
5. Execute PyTorch model training under fixed wall-clock time budget.
6. Apply **Keep-or-Revert**: keep winning code in `train.py` or revert losing code.
7. Update `memory/research_tree.json` and `research_plan.md`.
