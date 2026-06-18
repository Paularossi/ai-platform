# Multi-Agent Deliberation Platform

A Streamlit-based platform for running structured multi-agent deliberation experiments with LLMs. Implements the OMAS (Orchestrated Multi-Agent System) framework from the paper *Orchestration of Multi-Agent Systems as a Mechanism Design Problem*.

---

## Overview

Multiple LLM agents with distinct roles deliberate on a topic over several rounds. Each round has two phases:
 
- **Phase 1 (Contributions):** agents write a position statement, optionally seeing previous rounds.
- **Phase 2 (Peer review):** agents score each other's contributions on four quality dimensions, receiving ECU (experimental currency unit) payouts. An Orchestrator uses the agents' own importance votes to update the ECU weight vector each round.
The experiment ends when a coalition forms, the maximum number of rounds is reached, or (optionally) contributions converge.

Note: check [this](https://github.com/Mathews-Tom/Agentic-Design-Patterns) book to understand better agentic design.

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
│   ├── providers.py             # LLM provider wrappers (OpenAI, Anthropic)
│   ├── ecu.py                   # PeerReviewRound, CoalitionTracker, EcuLedger
│   ├── orchestrator.py          # Orchestrator: importance-vote gradient (Algorithm 1)
│   ├── orchestrator_sandbox.py  # Archived: coordinate-search version (not used)
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
- **OpenAI API key** for GPT-4o agents (`OPENAI_API_KEY`)
- **Anthropic API key** for Claude agents (`ANTHROPIC_API_KEY`)

Set your keys in your environment, or enter them on the Run page.

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

### Step 1 — Overview
Name, author, and a brief description of the experiment for your own reference. The actual deliberation question is set in Step 3.

### Step 2 - Agent Setup

**Agents:** configure name, provider (OpenAI or Anthropic), model, and role description. The role description is injected into the system prompt and defines the agent's perspective or mandate.

**Protocol:**
| Setting | Description | Use for |
|---|---|---|
| **Simultaneous** (crowd) | All agents contribute blindly in Phase 1, then peer-review in Phase 2 | Pure deliberation quality, no anchoring |
| **Sequential** (gossip) | Agents contribute one by one; later agents see earlier same-round contributions | Origination bias, anchoring, information cascade |

**Information flow (two separate axes):**

| Axis | Setting | What it controls |
|---|---|---|
| **φ₁** | `Blind` / `Previous round` / `Full history` | What each agent sees from prior rounds before writing their contribution |
| **φ₂** | `current_only` / `previous_round` / `full_history` | What a reviewer sees about the agent they score during peer review |
 
Round 1 is always blind for φ₁ regardless of setting — no prior contributions exist.
 
**Stopping rule:** `Max cycles`, `Convergence` (stable coalition), or `Either`.

**ECU / peer review - T/S/O information condition:**

| Condition | What agents know during Phase 1 |
|---|---|
| **T (Transparent)** | Full ECU weight vector + all agents' cumulative balances |
| **S (Semi-transparent)** | Noisy weight estimates (±20%) + own balance only |
| **O (Opaque)** | No ECU or weight information |

**Coalition threshold τ:** minimum consensus score required from both agents (mutual) to count as a coalition pair. A coalition requires at least two agents.
 
**Orchestrator:** when enabled, updates ECU weights every K rounds using agents' importance votes (see below).

### Step 3 - Instructions & topic
Write the deliberation question and base instructions for all agents. Optionally add per-agent prompt overrides. The question is injected into each agent's user message at runtime.

### Step 4 - Review
Inspect the full configuration, save or load a JSON draft, then launch the experiment.

---

## Round structure

**Phase 1 - Contributions**

Each agent writes a position statement. Their context contains:
- The deliberation question (always)
- Previous round contributions from other agents (if φ₁ ≠ Blind)
- Their own previous contribution labelled "(your previous contribution)"
- Peer review scores they received last round
- ECU balances and weights (T or S condition only)

Round 1 is always blind regardless of φ₁ (no prior contributions exist).

**Phase 2 - Peer review**

Each agent scores every other agent on four quality dimensions (0–1) and writes a one-sentence justification. If the Orchestrator is enabled, agents also distribute 100 importance points across the four dimensions, answering: *"Given the topic of the debate and your role, which dimensions are most important to you?"*

What the reviewer sees depends on φ₂:
- `current_only` - only the current round's contribution
- `previous_round` - current + previous round side-by-side (enables scoring whether position changed)
- `full_history` - full contribution trajectory

---

## ECU mechanism

```
ecu_i = Σ_q  w_q × mean_peer_score_q(agent_i)
```

**Payout formula:**
 
$$\text{ecu}_i^{(t)} = \sum_{q \in \mathcal{Q}} w_q^{ECU} \cdot \frac{1}{n-1} \sum_{j \neq i} s_{ji}^{(t)}(q)$$
 
where $s_{ji}^{(t)}(q)$ is the score agent $j$ gives agent $i$ on dimension $q$ in round $t$.
 
**Social welfare formula** (fixed weights $w^{SW}$, set once at experiment start):
 
$$SW^{(t)} = \sum_{q \in \mathcal{Q}} w_q^{SW} \cdot \frac{1}{n} \sum_{i \in \mathcal{N}} \frac{1}{n-1} \sum_{j \neq i} s_{ji}^{(t)}(q)$$


**Four quality dimensions:**
 
| Dimension | Description |
|---|---|
| `depth_breadth` | Analytical rigour with sufficient breadth across relevant angles |
| `depth` | Depth of reasoning and quality of evidence |
| `clarity` | Clarity, precision, and structure of the contribution |
| `consensus` | Whether the position changed meaningfully from the previous round in response to others; on round 1, how constructively the contribution opens dialogue |
 
**Coalition:** formed when all agent pairs have mutual consensus scores ≥ τ. A coalition requires at least two agents. Coalition scores are derived directly from the consensus dimension — no separate agreement question is asked.
 
**Reputation:** an agent's reputation in a given round is their ECU earned in that round (not cumulative), so it rises and falls with recent performance.

---

## Orchestrator (Algorithm 1)

When enabled, runs every K rounds after peer review completes. Uses an **importance-vote gradient** approach (no extra API calls).

**Update rule:**
 
1. During Phase 2, each agent distributes 100 importance points across the four dimensions.
2. Average votes across agents: $\bar{v}_q = \frac{1}{n} \sum_i v_{iq}$ (sums to 100).
3. Normalise: $\hat{v}_q = \bar{v}_q / 100$ (sums to 1).
4. Gradient step with $1/t$ learning rate: $w_q^{ECU}(t+1) = w_q^{ECU}(t) + \frac{1}{t} \cdot \hat{v}_q$
5. Renormalise to initial sum to keep the ECU budget stable.
The $1/t$ schedule satisfies the Robbins-Monro conditions: larger updates early, diminishing over time, converging in ratio as rounds accumulate.
 
The archived coordinate-search version (sandbox re-runs) is in `orchestrator_sandbox.py` and is not used by the platform.

---

## Providers
 
Two LLM providers are supported. Each is a thin wrapper around its SDK (no LiteLLM dependency).
 
| Provider | SDK | Models |
|---|---|---|
| **OpenAI** | `openai` | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `o1`, `o1-mini` |
| **Anthropic** | `anthropic` | `claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5` |
 
The Run page automatically detects which providers are used by the configured agents and shows an API key input for each. Agents from different providers can participate in the same experiment.
 
Adding a new provider: subclass `LLMProvider` in `core/providers.py`, add it to `PROVIDERS`, `PROVIDER_MODELS`, and `API_KEY_ENV_VARS`.
 
---

## Output

After a run, two downloads are available:

**Results CSV** - one row per item:
- `coalition_final`, `coalition_size`, `coalition_reached`
- `ecu_{AgentName}` - final cumulative ECU balance
- `pr_{AgentName}_{dimension}` - mean peer score from last round

**Full log JSON** — complete record including:
- `log` — contributions, ECU scores, ECU earned per agent per round
- `peer_review_log` — dimension scores, justifications, and importance votes per reviewer per round
- `coalition_history` — coalition members and size after each round
- `orchestrator` — full importance-vote update history per round (raw votes, mean votes, normalised votes, learning rate, old and new weights)
- `social_welfare_history` — SW value and weight snapshot after each round
- `prompt_log` — every prompt and response, labelled by phase (`contribution` or `peer_review`). Use to verify φ₁/φ₂/T/S/O settings are applied correctly.

---

## Running tests

```bash
python tests/test_pipeline.py
```

81 tests, no API key required (uses `dry_run=True`). Covers: event sequence, Phase 1 blindness in round 1, peer review structure, ECU calculation, self-assessment, coalition tracking (minimum size 2, mutual threshold), prompt construction, importance vote parsing, and JSON serialisation.

---

## Key design decisions

**No hardcoded prompts.** System prompts contain only user-defined content (role description, base instructions, guidelines). The deliberation item is always injected at runtime from the dataset — never baked into the instructions.
 
**Coalition from consensus.** The coalition score is derived directly from the consensus dimension score. No separate agreement question is asked. This keeps the prompt domain-agnostic and avoids redundancy.
 
**Two separate weight vectors.** $w^{SW}$ (social welfare weights) are fixed by the experimenter and define what good deliberation looks like from the social planner's perspective. $w^{ECU}$ (incentive weights) are the Orchestrator's decision variable and are updated by Algorithm 1. The connection between them is indirect: changing $w^{ECU}$ shifts agent incentives, which changes contributions, which changes peer scores, which changes $SW^{(t)}$ even though $w^{SW}$ never moves.
 
**Participatory weight updating.** The Orchestrator uses agents' own importance votes rather than sandbox re-runs, making weight updating free (no extra API calls) and participatory — agents themselves determine the gradient direction.
 
**One client per provider.** Each provider class holds a class-level singleton client. All agents using the same provider share one connection object, avoiding repeated client instantiation across contributions and peer reviews.
 
**Prompt log for debugging.** Every LLM call is logged with the full prompt and response. Inspect `prompt_log` in the JSON download to verify visibility settings, check what agents actually see, and diagnose unexpected behaviour.