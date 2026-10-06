---
name: debater
description: Multi-agent Generator-Critic debate framework supervised by Antigravity model. Triggered by "call the debater", "debater", or requests for Generator-Critic research debate loops under Antigravity supervision.
---

# Supervised Generator-Critic Debate Skill

This skill configures Antigravity to act as the Supervisor (Third Wheel / Overseer) monitoring a two-agent debate loop between Agent A (Generator) and Agent B (Critic).

## Supervisory Role & Control Invariants

1. **Agent A (Generator)**: Google AI Studio (`gemini-2.5-pro` or `gemini-2.5-flash`). Synthesizes technical drafts and revisions.
2. **Agent B (Critic)**: OpenRouter Free Tier (`meta-llama/llama-3.3-70b-instruct:free`). Evaluates drafts for logic gaps and vulnerabilities.
3. **Agent C (Antigravity - Supervisor)**: Monitors every turn, validates critique accuracy, prevents premature `[APPROVED]` consensus, and injects corrective policy instructions if agents drift.

## Execution Workflow

1. Initialize conversation history with user task prompt.
2. Execute Generator step -> obtain Draft N.
3. Execute Critic step -> obtain Critique N.
4. **Antigravity Intervention Gate**:
   - Inspect Draft N and Critique N.
   - Verify if Critique N is mathematically/logically sound.
   - If Critic output `[APPROVED]` prematurely while unaddressed flaws exist, Antigravity overrides approval, injects required corrections, and forces another revision round.
   - If Critic drifted into invalid critique, Antigravity rectifies critique before passing to Generator.
5. Terminate only when Antigravity Supervisor verifies strict compliance and zero remaining vulnerabilities.

## How to Run in Any Project

Place this skill definition in the project's workspace directory at:
`.agents/skills/debater/SKILL.md`

Place the supervisory orchestration notebook at:
`debate_loop.ipynb`
