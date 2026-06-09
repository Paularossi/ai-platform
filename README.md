# Multi-Agent Deliberation Platform

A Streamlit-based platform for running structured multi-agent deliberation experiments with LLMs. Implements the OMAS (Orchestrated Multi-Agent System) framework from the paper *Orchestration of Multi-Agent Systems as a Mechanism Design Problem*.

---

## Overview

Multiple LLM agents with distinct roles deliberate on a topic over several rounds. After each round, agents score each other's contributions (peer review), receive ECU (experimental currency unit) payouts, and an Orchestrator adjusts scoring weights to maximise social welfare. The experiment ends when a coalition forms or the maximum number of rounds is reached.

---

## Project structure

```
ai-platform/
├── app/
│   ├── Main.py                  # Streamlit entry point
│   ├── components/utils.py      # restore_draft() helper
│   └── pages/
│       ├── 1_Welcome.py         # Step 1: experiment name, topic description
│       ├── 2_Agent Setup.py     # Step 2: agents, protocol, ECU settings
│       ├── 3_Instructions.py    # Step 3: base instructions, per-agent overrides
│       ├── 4_Review.py          # Step 4: review config, save/load draft, launch
│       └── 5_Run.py             # Run page: live feed, downloads, nav buttons
├── core/
│   ├── state.py                 # AgentOutput, PeerReviewOutput dataclasses
│   ├── hub.py                   # CommunicationHub: routing, context, logging
│   ├── agent.py                 # Agent: prompt construction, LLM call, parsing
│   ├── ecu.py                   # PeerReviewRound, CoalitionTracker, EcuLedger
│   ├── orchestrator.py          # Orchestrator: Algorithm 1 weight updating
│   └── protocols/
│       ├── __init__.py          # RunEvent dataclass
│       ├── crowd.py             # Simultaneous protocol (blind Phase 1)
│       └── gossip.py            # Sequential protocol (origination/anchoring)
├── tests/
│   └── test_pipeline.py         # 81-test dry-run pipeline test suite
├── experiments/
├── literature/
├── chess_test.json
├── instructions.txt
├── requirements.txt
└── README.md
```

---

## Setup
### Prerequisites

- **Python 3.11** with packages listed in `requirements.txt`
- **OpenAI API access** (for GPT-4o)

Set your OpenAI API key as `OPENAI_API_KEY` in your environment, or enter it on the Run page.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Paularossi/ai-platform.git
   cd ai-platform
   ```

2. **Set up a virtual environment (venv)**

   Create the virtual environment with:
   ```bash
   python -m venv venv
   ```

   And then activate it with:
   ```bash
   venv\Scripts\activate.bat
   ```

3. **Install Python dependencies (change your CUDA version):**
   ```bash
   pip install -r requirements.txt
   ```

4. Run app

   ```bash
   streamlit run app/Main.py
   ```

---

## Experiment flow (4 steps)

### Step 1 - Overview
Name, author, version. Brief topic description for your own reference (the actual deliberation question is set in Step 3).

### Step 2 - Agent Setup

**Protocol:**
- **Simultaneous** - all agents contribute blindly in Phase 1, then peer-review in Phase 2. Use for studying pure deliberation quality.
- **Sequential** - agents contribute one by one; later agents see earlier agents' same-round contributions. Use for origination bias and anchoring effects.

**Information flow (two separate axes):**

| Setting | What it controls |
|---|---|
| **φ₁ Phase 1 visibility** | What each agent sees from the previous round before writing their contribution. `Blind` / `Previous round` / `Full history` |
| **φ₂ Phase 2 review depth** | What a reviewer sees about the agent they score. `current_only` / `previous_round` / `full_history` |

**Stopping rule:** `Max cycles`, `Convergence` (coalition), or `Either`.

**ECU / peer review - T/S/O information condition:**

| Condition | What agents know |
|---|---|
| **T (Transparent)** | Full weight vector + all agents' cumulative balances |
| **S (Semi-transparent)** | Noisy weight estimates + own balance only |
| **O (Opaque)** | No ECU or weight information |

**Coalition threshold τ:** minimum consensus score required (both directions) to form a coalition.

**Orchestrator:** when enabled, adjusts weights every K rounds (Algorithm 1). See below.

### Step 3 - Instructions & topic
Write the deliberation question and base instructions for all agents. Optionally add per-agent prompt overrides. The question is injected into each agent's user message at runtime - do not hardcode it in the instructions.

### Step 4 - Review
Check configuration, save/load JSON draft, launch.

---

## Round structure

**Phase 1 - Contributions**

Agents write a position statement. Their context contains:
- The deliberation question (always)
- Previous round contributions from other agents (if φ₁ ≠ Blind)
- Their own previous contribution labelled "(your previous contribution)"
- Peer review scores they received last round
- ECU balances / weights (T or S condition only)

Round 1 is always blind regardless of φ₁ (no prior contributions exist).

**Phase 2 - Peer review**

Each agent scores every other agent on five quality dimensions (0–1) and writes a one-sentence justification. What the reviewer sees depends on φ₂:
- `current_only` - only the current round's contribution
- `previous_round` - current + previous round side-by-side (enables scoring whether position changed)
- `full_history` - full contribution trajectory

---

## ECU mechanism

```
ecu_i = Σ_q  w_q × mean_peer_score_q(agent_i)
```

**Five quality dimensions:**
- `depth_breadth` - analytical rigour with sufficient breadth
- `depth` - depth and evidence of reasoning
- `clarity` - clarity and structure
- `completeness` - how fully the contribution addresses the question
- `consensus` - whether the position changed meaningfully from the previous round (requires φ₂ = previous_round or full_history), or how constructively the contribution advances agreement

**Coalition:** formed when all agent pairs have mutual consensus scores ≥ τ. Coalition scores are derived directly from the consensus dimension - no separate question is asked.

---

## Orchestrator (Algorithm 1)

When enabled, runs every K rounds after peer review completes.

Two separate mechanisms:

**Quality weight optimisation (coordinate-descent):**
For `depth_breadth`, `depth`, `clarity`, `completeness`: tries w_q ± ε (normalised within this subset so their sum stays constant). Updates if SW_quality improves.

**Consensus incentive rule:**
- Mean consensus < 0.5 → raise consensus weight by ε (agents disagree; create stronger incentive to converge)
- Mean consensus > 0.7 → lower consensus weight by ε (agents converging; reduce pressure)
- Mean consensus in [0.5, 0.7] → no change

The two mechanisms are independent. This ensures the Orchestrator drives convergence rather than simply down-weighting low-scoring dimensions.

---

## Output

After a run, two downloads are available:

**Results CSV** - one row per item:
- `coalition_final`, `coalition_size`, `coalition_reached`
- `ecu_{AgentName}` - final cumulative balance
- `pr_{AgentName}_{dimension}` - mean peer score from last round

**Full log JSON** - complete record including:
- `log` - contributions, ECU scores, ECU earned per agent per round
- `peer_review_log` - dimension scores and justifications per reviewer per round
- `coalition_history` - coalition members and size after each round
- `orchestrator` - weight update history (dimension, old/new weight, SW values)
- `prompt_log` - every prompt and response, labelled by phase (`contribution` or `peer_review`). Use to verify φ₁/φ₂/T/S/O settings are applied correctly.

---

## Experimental conditions

The platform supports a 3-axis experimental design:

| Axis | Options | Controls |
|---|---|---|
| φ₁ | Blind / Previous round / Full history | Cross-round contribution visibility |
| φ₂ | current_only / previous_round / full_history | Reviewer's history depth |
| T/S/O | Transparent / Semi-transparent / Opaque | ECU mechanism visibility |

The canonical baseline condition is **φ₁=Previous round × φ₂=previous_round × O**.

---

## Running tests

```bash
python tests/test_pipeline.py
```

81 tests, no API key required (dry_run=True). Covers: event sequence, Phase 1 blindness, peer review structure, ECU calculation, self-assessment, coalition tracking, prompt construction, serialisation.

---

## Key design decisions

**No hardcoded prompts.** System prompts contain only user-defined content (role description, base instructions, guidelines). The deliberation item is always injected at runtime from the dataset - never baked into the instructions.

**Coalition from consensus.** The coalition score is the consensus dimension score. No separate agreement question is asked. This keeps the prompt domain-agnostic and avoids redundancy.

**Orchestrator separates quality from convergence.** Quality dimensions are optimised by coordinate-descent (normalised). Consensus weight is controlled by a direct rule based on observed consensus level. This prevents the common failure mode where the Orchestrator down-weights a dimension simply because agents score badly on it.

**Prompt log for debugging.** Every LLM call is logged with the full prompt and response in the main JSON log. Inspect `prompt_log` to verify visibility settings, check what agents actually see, and diagnose unexpected behaviour.
