# HypothesisOS

HypothesisOS is a hierarchical AI research scientist framework designed to maximize empirical ML research progress per unit of compute.

It marries two core paradigms:
1. **AutoResearch (Execution):** A PyTorch empirical training loop bounded by an **immutable ground-truth evaluator** (`evaluate.py`), utilizing fixed-budget trials with **Keep-or-Revert** branch semantics.
2. **LLM Council (Judgment):** A multi-agent consensus system running **anonymized peer review** across parallel LLM models to score proposals by Utility-Adjusted Expected Value (EV).

---

## Mission: Build a Production-Grade AI Research Scientist

The objective is not to produce a chatbot or a prompt demo, but an autonomous research system. The core scientific principle is that **generation, evaluation, and execution must remain separate.** Empirical results from the immutable ground-truth evaluator always override theoretical arguments—the GPU is the source of truth.

---

## Key Scientific Architecture & Guards

### 1. Immutable Evaluation Boundary (`evaluate.py` vs `train.py`)
- **Mutable Space (`autoresearch/train.py`)**: The LLM Proposal Engine generates code patches for model architecture, hyperparameters, optimizers, and learning rate schedules in `train.py`.
- **Immutable Space (`autoresearch/evaluate.py`)**: Dataset loading, validation evaluation loops, cross-entropy calculation, and ground-truth `val_bpb` computation are strictly isolated in `evaluate.py`.
- **Security Boundary Guard**: The orchestrator inspects proposals via static checks. Any patch attempting to modify `evaluate.py` or hardcode `val_bpb` in `train.py` is flagged as a `SECURITY_VIOLATION` and rejected.

### 2. Multi-Agent Anonymized LLM Council & OpenRouter
- Proposals are anonymized as `Proposal A`, `Proposal B`, `Proposal C` to eliminate model bias.
- Evaluated in parallel across multiple peer-review models (e.g. Gemini 2.5 Pro, Claude 3.5 Sonnet, Llama 3.3 70B via OpenRouter).
- **Utility-Adjusted Expected Value Selection**:
  $$\text{Utility EV} = \frac{\text{Expected Improvement} \times \text{Probability of Success}}{\text{Implementation Cost (GPU hours)}}$$

### 3. Continuous Loop & Keep-or-Revert Semantics
- **Continuous Execution Loop**: Runs autonomous search sessions continuously (`python orchestrator.py --max-experiments N`).
- **Keep-or-Revert Branch Management**:
  - **WIN** (`val_bpb < baseline_bpb`): The code patch remains **KEPT** in `train.py` on disk as the new baseline for subsequent iterations.
  - **LOSS / CRASH / FAILED**: The code is immediately **REVERTED** to the baseline code backup. Code on disk always stays synchronized with `current_best`.

### 4. Directed Research Tree Memory
- **Research Tree (`memory/research_tree.json`)**: Manages a directed experiment graph tracking hypothesis node parentage, git diffs, ground-truth metrics, and postmortem evidence.
- **Postmortem Falsification Guard**: Forces postmortems to explicitly declare execution failure if metrics are missing or code crashes, eliminating false success narratives.

---

## Project Structure

```text
HypothesisOS/
├── config.py                 # System configuration, model choices, & OpenRouter headers
├── orchestrator.py           # Rebuilt continuous loop with keep-or-revert & EV selection
├── modal_runner.py           # Offloads PyTorch execution to serverless GPU containers
├── research_plan.md          # Dynamically updated research trajectory & plan
├── research_ledger.tsv       # Hard record of Hypothesis -> Expected -> Actual BPB
├── autoresearch/
│   ├── prepare.py            # Dataset preparation & tokenization script
│   ├── train.py              # Mutable PyTorch NanoGPT model & training loop
│   └── evaluate.py           # Immutable ground-truth validation evaluator
├── llm-council/
│   └── backend/
│       ├── openrouter.py     # Async OpenRouter API client with prompt caching
│       └── council.py        # Multi-agent anonymized peer review & Utility EV
├── memory/
│   ├── research_tree.py      # Directed graph memory data structure
│   └── research_tree.json    # Persistent research state & node history
└── .agents/                  # System instructions & master policies
```

---

## Quickstart

### Prerequisites
- Python 3.10+
- PyTorch & `httpx` (`pip install torch httpx`)
- (Optional) `OPENROUTER_API_KEY` environment variable for live multi-model LLM calls.

### Running an Autonomous Research Session

Run a 5-iteration continuous autonomous research search session:

```bash
python orchestrator.py --max-experiments 5
```

The orchestrator will:
1. Run initial ground-truth baseline evaluation (`evaluate.py`).
2. Generate 3 candidate code patches for `train.py`.
3. Conduct anonymized peer review across LLM Council models and pick the top Utility EV hypothesis.
4. Execute PyTorch model training and measure ground-truth validation BPB.
5. Apply **Keep-or-Revert**: keep winning code in `train.py` or revert losing code.
6. Record nodes in `memory/research_tree.json` and update `research_plan.md`.

---

## Prompt Caching & Efficiency

To maximize LLM cost-efficiency and inference speed, HypothesisOS leverages **OpenRouter Prompt Caching**:
- The master agent policy (`.agents/system_prompt.txt`) is isolated in the `{"role": "system"}` context.
- Dynamic data (current code, logs) are injected via the `{"role": "user"}` context.
- Prompt cache headers (`HTTP-Referer`, `X-Title`) are transmitted on every API request.
