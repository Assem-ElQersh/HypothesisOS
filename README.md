# HypothesisOS

HypothesisOS is a hierarchical AI research scientist framework designed to maximize empirical progress per unit of compute. 

It marries two paradigms:
1. **AutoResearch (Execution):** A fixed-budget empirical training loop to validate modifications to a neural network architecture.
2. **LLM Council (Judgment):** A multi-agent consensus system used to filter proposals, evaluate bug risk, and synthesize postmortem results.

## Architecture
- **Proposal Layer:** A low-cost model generates diverse, mutually exclusive patches.
- **Critique Layer:** The council scores and filters patches based on risk and expected gain.
- **Execution Layer:** Only the winning patch is run against the GPU for empirical truth.
- **Learning Layer:** The council writes a postmortem to the research ledger to inform future hypotheses.
