# Multi-Agent Deliberation Platform

A Streamlit-based platform for running structured multi-agent deliberation experiments with LLMs. Implements the OMAS (Orchestrated Multi-Agent System) framework from the paper *Orchestration of Multi-Agent Systems as a Mechanism Design Problem*.

---

## Overview

Multiple LLM agents with distinct roles deliberate on a single topic over one or more rounds. Each round has two phases:

- **Phase 1 (Contributions):** agents write a position statement, optionally seeing previous rounds. In the UI, responses stream in live as they're generated.
- **Phase 2 (Peer review - if enabled):** agents score each other's contributions on configurable quality dimensions, receiving ECU (experimental currency unit) payouts. An Orchestrator uses the agents' own importance votes to update the ECU weight vector each round.

A run can be **automatic** (plays straight through and stops on the configured rule) or **manual** (pauses before every agent turn so you can review, edit, or write the prompt yourself, and end the debate whenever you decide it's over). The run ends when an outcome is selected, and the transcript and log become available to download.

Note: check [this](https://github.com/Mathews-Tom/Agentic-Design-Patterns) book to understand better agentic design.

---

## Project structure

```
ai-platform/
├── app/
│   ├── Main.py                  # Streamlit entry point
│   ├── components/utils.py      # restore_draft() helper
│   └── pages/
│       ├── 1_Welcome.py         # Step 1: experiment name, author
│       ├── 2_Agent Setup.py     # Step 2: agents, protocol, run mode, ECU settings
│       ├── 3_Instructions.py    # Step 3: deliberation topic, base instructions, per-agent overrides
│       ├── 4_Review.py          # Step 4: review config, save/load draft, launch
│       └── 5_Run.py             # Step 5: live feed (streamed or manual step-through), outcome, downloads
├── core/
│   ├── state.py                 # AgentOutput, PeerReviewOutput dataclasses
│   ├── hub.py                   # CommunicationHub: routing, context, logging
│   ├── agent.py                 # Agent: prompt construction, LLM call/stream, response parsing
│   ├── providers.py             # LLM provider wrappers (OpenAI, Anthropic, Google) — complete() and stream()
│   ├── ecu.py                   # PeerReviewRound, CoalitionTracker, EcuLedger
│   ├── orchestrator.py          # Orchestrator: importance-vote gradient (Algorithm 1)
│   ├── runner.py                # Shared construction helpers (used by UI and batch runner)
│   └── protocols/
│       ├── __init__.py          # RunEvent dataclass, build_ecu_info_str helper
│       ├── crowd.py             # Simultaneous protocol (blind Phase 1)
│       └── gossip.py            # Sequential protocol (origination/anchoring)
├── experiments/
│   ├── run_experiment.py        # Generic batch runner (no UI required)
│   └── test_thinking.json       # Example UI draft (single scenario)
├── tests/
│   └── test_pipeline.py         # 81-test dry-run pipeline test suite
├── literature/
├── requirements.txt
└── README.md
```

---

## Setup
### Prerequisites

- **Python 3.11** with packages listed in `requirements.txt`
- API keys for the providers you use set up in an .env file (see below)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Paularossi/ai-platform.git
   cd ai-platform
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate.bat
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys** - create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GOOGLE_API_KEY=...
   ```
   Only keys for providers you actually use are required. The `.env` file should be gitignored.

5. **Run the UI:**
   ```bash
   streamlit run app/Main.py
   ```

6. **Or run an experiment without the UI:**
   ```bash
   python experiments/run_experiment.py experiments/chess_test.json
   ```
---

## Experiment flow (UI - 5 steps)

### Step 1 — Welcome
Name and author for the experiment, for your own reference.

### Step 2 — Agent Setup

**Agents:** configure name, provider (OpenAI, Anthropic, or Google), model, and role description. The role description is injected into the system prompt and defines the agent's perspective or mandate.

**Protocol:**
| Setting | Description | Use for |
|---|---|---|
| **Simultaneous** (crowd) | All agents contribute blindly in Phase 1, then peer-review in Phase 2 | Pure deliberation quality, no anchoring |
| **Sequential** (gossip) | Agents contribute one by one; later agents see earlier same-round contributions | Origination bias, anchoring, information cascade |

**Run mode** (decided upfront, part of the protocol definition):
| Mode | Behaviour |
|---|---|
| **Automatic** | Plays straight through using the stopping rule below |
| **Manual (step-through)** | Pauses before every agent turn so you can review and edit the exact prompt, then approve it to send. You decide when the debate ends — there's no round limit or stopping rule to configure |

**Sequential order** (Sequential setting only): `Fixed`, `Randomized each cycle`, or `Custom order` (rank agents explicitly).

**Information flow (two separate axes):**

| Axis | Setting | What it controls |
|---|---|---|
| **φ₁** | `Blind` / `Previous Round` / `Full History` | What each agent sees from prior rounds before writing their contribution |
| **φ₂** | `Current Only` / `Previous Round` / `Full History` | What a reviewer sees about the agent they score during peer review |

Round 1 is always blind for φ₁ regardless of setting since no prior contributions exist.

**Stopping rule** (Automatic mode only): `Max cycles`, `Convergence` (stable coalition), or `Either`.

**ECU / peer review — T/S/O information condition:**

| Condition | What agents see during Phase 1 and Phase 2 |
|---|---|
| **T (Transparent)** | Full ECU weight vector + all agents' cumulative balances |
| **S (Semi-transparent)** | Noisy weight estimates (±20% per dimension) + own balance only |
| **O (Opaque)** | No ECU or weight information |

Under T and S, the system prompt informs agents that their contributions are peer-reviewed on named quality dimensions and that they earn ECU based on those scores, but gives no instruction on how to respond to this information.

**Coalition threshold τ:** minimum consensus score required from both agents (mutual) to count as a coalition pair. A coalition requires at least two agents.

**Orchestrator:** when enabled, updates ECU weights every K rounds using agents' importance votes (see below).

### Step 3 — Instructions & topic
Write the single deliberation topic and base instructions for all agents. Optionally add per-agent prompt overrides and guideline notes. The topic is injected into each agent's user message at runtime.

The prompt preview updates live based on the current φ₁ and ECU information condition settings, and omits any peer-review-specific text when ECU scoring is turned off.

### Step 4 — Review & Launch
Inspect the full configuration, save or load a JSON draft, then launch.

### Step 5 — Run
Watch the deliberation unfold as a live chat feed, with each contribution streaming in as it's generated. In manual mode, you approve (and can edit) every prompt before it's sent, and end the run yourself with "Stop here." Once the run ends, select an outcome — how the debate settled — with optional notes; saving it closes the debate and unlocks the transcript and log downloads.

---

## Round structure

**Phase 1 — Contributions**

Each agent writes a position statement. Their context contains:
- The deliberation topic (always)
- Previous round contributions from other agents (if φ₁ ≠ Blind)
- Their own previous contribution, labelled "(your contribution)"
- Peer review scores received last round (if ECU enabled)
- ECU balances and weights (T or S condition only)

Round 1 is always blind regardless of φ₁ since no prior contributions exist. In an automatic or manual UI run, the response streams into the live feed token by token as the model generates it.

**Phase 2 — Peer review**

Each agent scores every other agent on the configured quality dimensions (0-1) and writes a one-sentence justification. If the Orchestrator is enabled, agents also distribute 100 importance points across the dimensions.

What the reviewer sees depends on φ₂:
- `Current Only` - only the current round's contribution
- `Previous Round` - current + previous round side-by-side
- `Full History` - full contribution trajectory across all rounds

---

## ECU mechanism

**Payout formula:**

$$\text{ecu}_i^{(t)} = \sum_{q \in \mathcal{Q}} w_q^{ECU} \cdot \frac{1}{n-1} \sum_{j \neq i} s_{ji}^{(t)}(q)$$

where $s_{ji}^{(t)}(q)$ is the score agent $j$ gives agent $i$ on dimension $q$ in round $t$.

**Social welfare formula** (fixed weights $w^{SW}$, set once at experiment start):

$$SW^{(t)} = \sum_{q \in \mathcal{Q}} w_q^{SW} \cdot \frac{1}{n} \sum_{i \in \mathcal{N}} \frac{1}{n-1} \sum_{j \neq i} s_{ji}^{(t)}(q)$$

**Quality dimensions** are fully configurable per experiment (name, label, rubric, ECU weight, SW weight). The default set is:

| Dimension | Description |
|---|---|
| `depth_breadth` | Analytical rigour with sufficient breadth across relevant angles |
| `depth` | Depth of reasoning and quality of evidence |
| `clarity` | Clarity, precision, and structure of the contribution |
| `consensus` | Whether the contribution constructively advances group agreement or productively engages with opposing views |

Rubrics are defined in the experiment config and injected into the peer review prompt verbatim and no interpretation is hardcoded.

**Coalition:** formed when all agent pairs have mutual consensus scores ≥ τ. A coalition requires at least two agents. Coalition scores are derived directly from the consensus dimension and no separate agreement question is asked.

**Reputation:** an agent's reputation in a given round is their ECU earned in that round (not cumulative), so it rises and falls with recent performance.

---

## Orchestrator (Algorithm 1)

When enabled, runs every K rounds after peer review completes. Uses an **importance-vote gradient** approach (no extra API calls).

**Update rule:**

1. During Phase 2, each agent distributes 100 importance points across the dimensions.
2. Average votes across agents: $\bar{v}_q = \frac{1}{n} \sum_i v_{iq}$ (sums to 100).
3. Normalise: $\hat{v}_q = \bar{v}_q / 100$ (sums to 1).
4. Gradient step with $1/t$ learning rate: $w_q^{ECU}(t+1) = w_q^{ECU}(t) + \frac{1}{t} \cdot \hat{v}_q$
5. Renormalise to initial sum to keep the ECU budget stable.

The $1/t$ schedule satisfies the Robbins-Monro conditions: larger updates early, diminishing over time, converging in ratio as rounds accumulate.

---

## Outcome types

At the end of a run — automatic or manual — you classify how the deliberation settled and log it alongside the transcript. The platform never infers this automatically; it's a judgment call made from reading the log:

| Outcome | Description |
|---|---|
| **Full consensus** | Agents converge to a single stable position |
| **Static equilibrium** | A stable distribution of positions (e.g. 20% vs 80%) |
| **Dynamic equilibrium** | A partially stable distribution with periodic shifts between positions |
| **Chaotic state** | Unstable, non-converging, highly variable outputs |

Consensus is one valid outcome among several, not the default to aim for. Saving an outcome (with optional notes) closes the debate and unlocks the downloads below.

---

## Batch experiments (no UI)

Experiments can be run programmatically from the command line using `experiments/run_experiment.py`. This is useful for running multiple scenario combinations without touching the UI.

```bash
python experiments/run_experiment.py experiments/test_thinking.json
```

**Experiment JSON format** - two variants are accepted:

1. **UI draft** (flat): the JSON saved by the UI's "Save draft" button can be passed directly. The embedded `protocol` block becomes a single scenario.

2. **Batch format**: defines a shared `base` config and a list of `scenarios`, each overriding the `protocol` block:
   ```json
   {
     "meta": { "name": "...", "author": "..." },
     "base": {
       "agents": [...],
       "task": { "description": "..." },
       "instructions": { "base_instructions": "...", "guideline_notes": "..." },
       "agent_prompt_overrides": { "AgentName": "..." },
       "ecu": { "enabled": true, ... }
     },
     "scenarios": [
       { "name": "sim_blind", "protocol": { "setting": "Simultaneous", "visibility_mode": "Blind", ... } },
       { "name": "seq_full",  "protocol": { "setting": "Sequential",   "visibility_mode": "Full History", ... } }
     ]
   }
   ```

**Output** — written to `experiments/results/`:
- One JSON file per scenario with the full result (contributions, peer reviews, ECU ledger, coalition history, orchestrator updates, prompt log)
- A summary CSV across all scenarios

---

## Providers

Three LLM providers are supported. Each is a thin wrapper around its SDK, exposing both a blocking `complete()` call and a `stream()` generator for live token-by-token output.

| Provider | SDK | Models |
|---|---|---|
| **OpenAI** | `openai` | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `o1`, `o1-mini` |
| **Anthropic** | `anthropic` | `claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5` |
| **Google** | `google-genai` | `gemini-3.5-flash`, `gemini-3.1-flash-lite`, `gemini-2.5-pro`, `gemini-2.5-flash` |

Agents from different providers can participate in the same experiment.

Adding a new provider: subclass `LLMProvider` in `core/providers.py` (implement both `complete()` and `stream()`), add it to `PROVIDERS`, `PROVIDER_MODELS`, and `API_KEY_ENV_VARS`.

---

## Output

**From a UI run**, once an outcome has been saved:
- **Transcript (.md)** — a readable, round-by-round record of the deliberation, ending with the outcome and any notes. Meant to read or share directly.
- **Full log (.json)** — the complete machine-readable record: contributions, peer reviews, ECU ledger, coalition history, orchestrator updates, the outcome, and every prompt sent.

**From a batch run**, per scenario:
- One JSON file with the same full record as above.
- A summary CSV across all scenarios in the run — `coalition_final`, `coalition_size`, `coalition_reached`, `ecu_{AgentName}` (final cumulative balance), `pr_{AgentName}_{dimension}` (mean peer score from the last round).

In both cases, the JSON's `prompt_log` records every prompt and response, labelled by phase (`contribution` or `peer_review`) — use it to verify φ₁/φ₂/T/S/O settings were applied correctly.

---

## Running tests

```bash
python tests/test_pipeline.py
```

81 tests, no API key required (uses `dry_run=True`). Covers: event sequence, Phase 1 blindness in round 1, peer review structure, ECU calculation, self-assessment, coalition tracking (minimum size 2, mutual threshold), prompt construction, importance vote parsing, and JSON serialisation.

---

## Key design decisions

**No hardcoded prompts.** System prompts contain only user-defined content (role description, base instructions, guidelines, dimension rubrics). The deliberation topic is always injected at runtime and never baked into the instructions. Dimension rubrics are defined entirely in the experiment config and rendered verbatim into the peer review prompt.

**ECU feedback without direction.** Under T or S conditions, agents are told what they can observe (scores, balances, weights) but are given no instruction on how to respond to this information. The goal is to observe whether agents adapt their strategy organically, not to coach them toward higher scores.

**Coalition from consensus.** The coalition score is derived directly from the consensus dimension score. No separate agreement question is asked. This keeps the prompt domain-agnostic and avoids redundancy.

**Two separate weight vectors.** $w^{SW}$ (social welfare weights) are fixed by the experimenter and define what good deliberation looks like from the social planner's perspective. $w^{ECU}$ (incentive weights) are the Orchestrator's decision variable and are updated by Algorithm 1. The connection between them is indirect: changing $w^{ECU}$ shifts agent incentives, which changes contributions, which changes peer scores, which changes $SW^{(t)}$ even though $w^{SW}$ never moves.

**Participatory weight updating.** The Orchestrator uses agents' own importance votes rather than sandbox re-runs, making weight updating free (no extra API calls) and participatory - agents themselves determine the gradient direction.

**Outcome classification stays human.** The four outcome types (full consensus, static equilibrium, dynamic equilibrium, chaotic state) are never auto-detected from the log — whoever runs the experiment reads the transcript and picks one, with consensus deliberately not privileged as the "correct" result.

**Shared construction layer.** `core/runner.py` contains all experiment construction logic (building agents, protocols, hubs, ECU components). Both the Streamlit UI and the batch runner import from it, ensuring they run identical pipelines. The UI owns streaming and rendering; `runner.py` owns everything else.

**One client per provider.** Each provider class holds a class-level singleton client. All agents using the same provider share one connection object, avoiding repeated client instantiation across contributions and peer reviews.

**Prompt log for debugging.** Every LLM call is logged with the full prompt and response. Inspect `prompt_log` in the JSON output to verify visibility settings, check what agents actually see, and diagnose unexpected behaviour.
