# HypothesisOS

HypothesisOS is a hierarchical AI research scientist framework designed to maximize empirical progress per unit of compute.

It marries two paradigms:
1. **AutoResearch (Execution):** A fixed-budget empirical training loop to validate modifications to a neural network architecture.
2. **LLM Council (Judgment):** A multi-agent consensus system used to filter proposals, evaluate bug risk, and synthesize postmortem results.

## Mission: Build a Hierarchical AI Research Scientist
The objective is not to produce a chatbot, but an autonomous research system. The core principle is that **generation, evaluation, and execution must remain separate.** Empirical results always override theoretical arguments, and the GPU is the source of truth.

## The Operating Loop Paradigm

1. **Layer 1: Research Planner:** Maintains the current objective, known constraints, active hypotheses, and termination criteria. Never modify code without explicitly linking the modification to a hypothesis in the plan.
2. **Layer 2: Proposal Engine:** Generates 3-5 mutually exclusive candidate hypotheses based on the plan. Evaluates risk, complexity, and expected mechanism of improvement.
3. **Layer 3: Council Review:** A multi-agent council evaluates each proposal independently for technical risk (OOM, divergence) and scientific plausibility. It exists to reject bad experiments before they consume compute.
4. **Layer 4: Experiment Selection:** Selects only the highest expected-value experiment. Do not run multiple expensive experiments simultaneously unless explicitly instructed.
5. **Layer 5: Execution:** Execution must be isolated from reasoning. The selected patch is applied and run against a fixed time budget. Only empirical evidence is collected (validation BPB, runtime, memory).
6. **Layer 6: Postmortem Council:** After execution, the council analyzes the evidence to answer why the outcome occurred and whether the hypothesis was validated.

## Project Structure
```text
HypothesisOS/
├── autoresearch/        # Pure execution layer (fixed budget GPU training)
├── llm-council/         # API backend for the multi-agent judgement layers
├── .agents/             # System instructions and operational rules
├── council/             # Storage for peer-review deliberation state
├── experiments/         # Output artifacts from executed patches
├── memory/              # Persistent trajectory and knowledge base
├── reports/             # Generated postmortems and synthetic findings
├── orchestrator.py      # The main Python loop binding the hierarchy together
├── modal_runner.py      # Offloads execution to serverless H100s
├── research_plan.md     # Current active objectives and constraints
└── research_ledger.tsv  # Hard record of Hypothesis -> Expected -> Actual BPB
```

## Prompt Caching & System Prompt
To maximize LLM cost-efficiency and inference speed, HypothesisOS heavily leverages **Prompt Caching**. 
- The master agent policy is defined in `.agents/system_prompt.txt`.
- The system prompt is kept completely **stable** and isolated in the `{"role": "system"}` context.
- All task-specific and dynamic data (like current code and logs) are strictly injected via the `{"role": "user"}` message.
This ensures the prefix >= 1024 tokens remains an exact match across every execution loop, cutting input-token costs by up to 90%.

## Anti-Failure Rules
- Reject circular reasoning, self-grading, and popularity arguments.
- Never claim success without empirical support.
- The objective is not to produce code or explanations, but to maximize validated research progress per unit of compute.
