# Multi-Agent Deliberation Platform

A Streamlit-based platform for running structured multi-agent deliberation experiments with LLMs. Implements the OMAS (Orchestrated Multi-Agent System) framework from the paper *Orchestration of Multi-Agent Systems as a Mechanism Design Problem*.

---

## Overview

Multiple LLM agents with distinct roles deliberate on a topic over several rounds. Each round has two phases:
 
- **Phase 1 (Contributions):** agents write a position statement, optionally seeing previous rounds.
- **Phase 2 (Peer review):** agents score each other's contributions on configurable quality dimensions, receiving ECU (experimental currency unit) payouts. An Orchestrator uses the agents' own importance votes to update the ECU weight vector each round.

The experiment ends when a coalition forms, the maximum number of rounds is reached, or (optionally) contributions converge.

Note: check [this](https://github.com/Mathews-Tom/Agentic-Design-Patterns) book to understand better agentic design.

---

## Project structure

```
ai-platform/
├── app/
│   ├── Main.py                  # Streamlit entry point - nav: Setup / Debate / History
│   └── pages/
│       ├── 1_Welcome.py         # Setup: Agent 0 topic entry, moderator model, hard limits
│       ├── 5_Run.py             # Debate: live feed, downloads (md/pdf/json), nav buttons
│       └── 6_History.py         # History: browse past debates from the shared Supabase log
├── core/
│   ├── state.py                 # AgentOutput, PeerReviewOutput dataclasses
│   ├── hub.py                   # CommunicationHub: routing, context, logging
│   ├── agent.py                 # Agent: prompt construction, LLM call, parsing
│   ├── agent_zero.py            # AgentZero: Agent 0 mode's super-agent (init/decide/brief)
│   ├── providers.py             # LLM provider wrappers (OpenAI, Anthropic, Google)
│   ├── ecu.py                   # PeerReviewRound, CoalitionTracker, EcuLedger
│   ├── orchestrator.py          # Orchestrator: importance-vote update (Algorithm 1)
│   ├── runner.py                # Shared construction helpers (used by UI and batch runner)
│   ├── pdf_export.py            # Renders an Agent 0 final brief to PDF
│   ├── db.py                    # Supabase-backed persistent debate log (History page)
│   └── protocols/
│       ├── __init__.py          # RunEvent, shared peer-review + round-finalisation helpers
│       ├── crowd.py             # Simultaneous protocol (blind Phase 1)
│       ├── gossip.py            # Sequential protocol (origination/anchoring)
│       └── agent_zero_loop.py   # Agent 0 mode's dynamic outer loop
├── experiments/
│   ├── run_experiment.py        # Generic batch runner (no UI required)
│   └── chess_test.json          # Example manual-mode config (single scenario)
├── tests/
│   └── test_pipeline.py         # Offline pipeline regression tests
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
   The live Agent 0 app requires all three provider keys because Agent 0 can select agents from any provider. The `.env` file should be gitignored.

5. **Run the UI:**
   ```bash
   streamlit run app/Main.py
   ```

6. **Or run an experiment without the UI:**
   ```bash
   python experiments/run_experiment.py experiments/chess_test.json
   ```
---

## Agent 0 mode (the live app)

The Streamlit app's navigation ([app/Main.py](app/Main.py)) wires up three pages - **Setup → Debate → History** - and Agent 0 mode is the only thing they run. Manual mode (fixed roster, human-configured protocol) has no UI; it is supported as a config shape via [core/runner.py](core/runner.py) and the batch runner.

**Setup** ([1_Welcome.py](app/pages/1_Welcome.py)) - two text fields. **Topic** is the only required one: the deliberation question, shown to Agent 0 and to every debating agent every round. **Final brief instructions** (optional) is where you steer *how* Agent 0 writes its final brief - section order, length, language, anything about the brief's shape (nothing about it is prescribed in code; see "No hardcoded prompts" below). The two are kept structurally separate on purpose: brief instructions reach only Agent 0's own initialization/decision calls (`cfg["task"]["brief_instructions"]`), never `hub.item_data`, so a debating agent can never mistake "how the eventual brief should look" for something to argue about in its own contribution. Setup also picks which model runs Agent 0 itself, and sets the hard bounds (`max_rounds`, `max_agents`, `max_total_spawns`) that cap runaway cost regardless of what Agent 0 decides.

**Debate** ([5_Run.py](app/pages/5_Run.py)) - a live per-round feed (contributions, peer-review scores, ECU balances, Agent 0's reasoning and any roster/instruction changes), then the final brief with four downloads: the brief as Markdown or PDF, a plain transcript JSON (just what each agent said, round by round), and the full log JSON (every prompt/response, scores, ECU ledger, coalition history).

**History** ([6_History.py](app/pages/6_History.py)) - every debate run from the live app is logged to a shared Supabase (Postgres) database via [core/db.py](core/db.py), independent of any one user's session. This page lists past debates (topic, model, round count, how it ended) and can show any past debate's stored final brief on demand. API calls are saved incrementally in `llm_calls`, including Agent 0 initialization, contributions, reviews, decisions and final briefs. History lets you inspect prompts, responses, failures, effective settings and token usage.

---

## Manual mode config (batch runner) - OUTDATED (see other branch)

The steps below describe the config shape manual mode uses - agents, protocol, ECU settings, instructions - written by hand as JSON. This shape is what `core/runner.py` and `experiments/run_experiment.py` (the batch runner) consume; it's the way to run fixed-roster, human-configured deliberations rather than Agent 0-designed ones.

### Agents and protocol

**Agents:** configure name, provider (OpenAI, Anthropic, or Google), model, and role description. The role description is injected into the system prompt and defines the agent's perspective or mandate.

**Protocol:**
| Setting | Description | Use for |
|---|---|---|
| **Simultaneous** (crowd) | All agents contribute blindly in Phase 1, then peer-review in Phase 2 | Pure deliberation quality, no anchoring |
| **Sequential** (gossip) | Agents contribute one by one; later agents see earlier same-round contributions | Origination bias, anchoring, information cascade |

**Information flow (two separate axes):**

| Axis | Setting | What it controls |
|---|---|---|
| **φ₁** | `Blind` / `Previous Round` / `Full History` | What each agent sees from prior rounds before writing their contribution |
| **φ₂** | `Current Only` / `Previous Round` / `Full History` | What a reviewer sees about the agent they score during peer review |

Round 1 is always blind for φ₁ regardless of setting since no prior contributions exist.

**Stopping rule:** `Max cycles`, `Convergence` (stable coalition), or `Either`.

**ECU / peer review — T/S/O information condition:**

| Condition | What agents see during Phase 1 and Phase 2 |
|---|---|
| **T (Transparent)** | Full ECU weight vector + all agents' cumulative balances |
| **S (Semi-transparent)** | Noisy weight estimates (±20% per dimension) + own balance only |
| **O (Opaque)** | No ECU or weight information |

Under T and S, the system prompt informs agents that their contributions are peer-reviewed on named quality dimensions and that they earn ECU based on those scores, but gives no instruction on how to respond to this information.

**Coalition threshold τ:** minimum consensus score required from both agents (mutual) to count as a coalition pair. A coalition requires at least two agents.

**Orchestrator:** when enabled, updates ECU weights every K rounds using agents' importance votes (see below).

### Instructions & topic
Write the deliberation question (`task.description`) and base instructions for all agents. Optionally add per-agent prompt overrides and guideline notes. The question is injected into each agent's user message at runtime.

---

## Round structure

**Phase 1 — Contributions**

Each agent writes a position statement. Their context contains:
- The deliberation question (always)
- Previous round contributions from other agents (if φ₁ ≠ Blind)
- Their own previous contribution, labelled "(your contribution)"
- Peer review scores received last round (if ECU enabled)
- ECU balances and weights (T or S condition only)

Round 1 is always blind regardless of φ₁ since no prior contributions exist.

**Phase 2 — Peer review**

Each agent scores every other agent on the configured quality dimensions (0-1) and writes a one-sentence justification. If the Orchestrator is enabled, agents also distribute 100 importance points across the dimensions.

What the reviewer sees depends on φ₂:
- `Current Only` - only the current round's contribution
- `Previous Round` - current + previous round side-by-side
- `Full History` - full contribution trajectory across all rounds

---

## Hallucination and risk peer assessments

Agent 0 runs optionally collect separate assessments during peer review (enabled by default in Setup). Reviewers flag specific suspected factual issues, can report that a claim cannot be verified, and assess the quality of risk reasoning on a 0-1 scale or report it as unassessable. These are peer judgments, not independently verified findings. They do not change ECU payments, importance votes or social-welfare weights. Agent 0 receives them when deciding and writing the final brief.

The Debate page shows reviewer-level findings. Full JSON exports preserve the parsed assessments, while persistent API call records retain their original prompts and responses.

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
| `consensus` | In superagent mode, compatibility with the reviewer's current substantive position, using a fixed rubric |

Rubrics are defined in the experiment config and injected into the peer review prompt verbatim and no interpretation is hardcoded.

**Coalition:** formed when all agent pairs have mutual consensus scores ≥ τ. A coalition requires at least two agents. Coalition scores are derived directly from the consensus dimension and no separate agreement question is asked.

**Reputation:** an agent's reputation in a given round is their ECU earned in that round (not cumulative), so it rises and falls with recent performance.

---

## Orchestrator (Algorithm 1)

When enabled, runs every K rounds after peer review completes. Uses an **importance-vote update** approach (no extra API calls).

**Update rule:**

1. During Phase 2, each agent distributes 100 importance points across the dimensions.
2. Average votes across agents: $\bar{v}_q = \frac{1}{n} \sum_i v_{iq}$ (sums to 100).
3. Normalise: $\hat{v}_q = \bar{v}_q / 100$ (sums to 1).
4. Vote step with $1/t$ learning rate: $w_q^{ECU}(t+1) = w_q^{ECU}(t) + \frac{1}{t} \cdot \hat{v}_q$
5. Renormalise to initial sum to keep the ECU budget stable.

The $1/t$ schedule gives diminishing updates. Votes are not an objective gradient, and no optimality or convergence guarantee is claimed.

---

## Batch experiments (no UI)

Experiments can be run programmatically from the command line using `experiments/run_experiment.py`. This is useful for running multiple scenario combinations without touching the UI.

```bash
python experiments/run_experiment.py experiments/chess_test.json
python experiments/run_experiment.py experiments/congestion_pricing.json --dry-run
```

**Experiment JSON format** - two variants are accepted:

1. **Flat config**: a single experiment config with an embedded `protocol` block, which becomes a single scenario.

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

Three LLM providers are supported. Each is a thin wrapper around its SDK.

| Provider | SDK | Models |
|---|---|---|
| **OpenAI** | `openai` | `gpt-6-luna`, `gpt-4.1-mini`, `gpt-6.1-sol`, `gpt-4o` |
| **Anthropic** | `anthropic` | `claude-sonnet-5-5`, `claude-haiku-4-5` |
| **Google** | `google-genai` | `gemini-3.8-flash`, `gemini-3.5-flash-lite` |

Agents from different providers can participate in the same experiment. Empty, truncated and malformed responses are recorded as errors. Call records retain provider-reported input, output, cache and reasoning token counts. The full Agent 0 log also aggregates usage by model.

Adding a new provider: subclass `LLMProvider` in `core/providers.py`, add it to `PROVIDERS`, `PROVIDER_MODELS`, and `API_KEY_ENV_VARS`.

---

## Output

This expands on the two file kinds the batch runner writes to `experiments/results/` (see "Batch experiments" above). The live app's Agent 0 mode instead offers downloads directly in the browser (brief as Markdown/PDF, transcript JSON, full log JSON) - see "Agent 0 mode (the live app)" above.

**Results CSV** - one row per scenario/item:
- `coalition_final`, `coalition_size`, `coalition_reached`
- `ecu_{AgentName}` - final cumulative ECU balance
- `pr_{AgentName}_{dimension}` - mean peer score from last round

**Full log JSON** - complete record including:
- `log` - contributions, ECU scores, ECU earned per agent per round
- `peer_review_log` - dimension scores, justifications, and importance votes per reviewer per round
- `coalition_history` - coalition members and size after each round
- `orchestrator` - full importance-vote update history per round (raw votes, mean votes, normalised votes, learning rate, old and new weights)
- `ecu_ledger` - final weights, balances, and social welfare history
- `prompt_log` - every prompt and response, labelled by phase (`contribution` or `peer_review`). Use to verify φ₁/φ₂/T/S/O settings are applied correctly.

---

## Running tests

```bash
python -m unittest discover -s tests
python tests/check_models.py --all                 # preview only, no paid calls
python tests/check_models.py --all --live          # paid checks, using the app's peer-review path
```

Offline tests cover the pipeline, all provider adapters, sequential score visibility, failed reviews, roster selection, token accounting and persistence. Live checks exercise contributions, structured Agent 0 decisions and repeated peer reviews. They print request counts and effective output budgets before running. Set the relevant API keys in the environment before running the script.

---

## Key design decisions

**No hardcoded prompts.** System prompts contain only user-defined content (role description, base instructions, guidelines, dimension rubrics). The deliberation item is always injected at runtime and never baked into the instructions. Dimension rubrics are defined entirely in the experiment config and rendered verbatim into the peer review prompt. The same principle applies to Agent 0's final brief: its length, structure, and language aren't prescribed anywhere in code (see [core/agent_zero.py](core/agent_zero.py) - the system prompt only requires it be substantive, not any particular shape). Format instructions belong in the topic pre-prompt, same as everything else Agent 0 decides.

**Brief export doesn't assume English or Latin script.** [core/pdf_export.py](core/pdf_export.py) titles the PDF from the brief's own leading heading - in whatever language Agent 0 wrote it - falling back to the topic text rather than a fixed label, and renders with a bundled/system Unicode font instead of ReportLab's base14 Helvetica so Cyrillic, Greek, Vietnamese, and similar scripts display correctly instead of dropping to blanks. This doesn't extend to right-to-left scripts (Arabic, Hebrew) or CJK/Indic scripts, which need bidi reordering and complex text shaping that ReportLab's Paragraph flowable doesn't do - a real gap, not a solved one.

**ECU feedback without direction.** Under T or S conditions, agents are told what they can observe (scores, balances, weights) but are given no instruction on how to respond to this information. The goal is to observe whether agents adapt their strategy organically, not to coach them toward higher scores.

**Coalition from consensus.** The coalition score is derived directly from the consensus dimension score. No separate agreement question is asked. This keeps the prompt domain-agnostic and avoids redundancy.

**Two separate weight vectors.** $w^{SW}$ (social welfare weights) are fixed by the experimenter and define what good deliberation looks like from the social planner's perspective. $w^{ECU}$ (incentive weights) are the Orchestrator's decision variable and are updated by Algorithm 1. The connection between them is indirect: changing $w^{ECU}$ shifts agent incentives, which changes contributions, which changes peer scores, which changes $SW^{(t)}$ even though $w^{SW}$ never moves.

**Participatory weight updating.** The Orchestrator uses agents' own importance votes rather than sandbox re-runs, making weight updating free (no extra API calls) and participatory - agents determine the relative importance signal.

**Shared construction layer.** `core/runner.py` contains all experiment construction logic (building agents, protocols, hubs, ECU components). Both the Streamlit UI and the batch runner import from it, ensuring they run identical pipelines. The UI owns streaming and rendering; `runner.py` owns everything else.

**One client per provider.** Each cached provider instance reuses its SDK client; request diagnostics and usage are isolated by thread. All agents using the same provider share one connection object, avoiding repeated client instantiation across contributions and peer reviews.

**Prompt log for debugging.** Every LLM call is logged with the full prompt and response. Inspect `prompt_log` in the JSON output to verify visibility settings, check what agents actually see, and diagnose unexpected behaviour.

Superagent consistency: consensus uses a fixed agreement rubric with configurable weights; missing agreement scores cannot form coalitions. Terminal rounds execute no roster or instruction changes. Round exports distinguish `ecu_weights` used for allocations from `next_ecu_weights`. Missing current-round SW is explicitly marked unavailable.

Peer reviews use provider-native structured responses, with local validation of score ranges and exact flagged excerpts. Gemini 3 calls allow at least 8,192 output tokens (including thinking), using low thinking for Flash and minimal for 3.5 Flash Lite.

Peer-review responses use a reusable array-entry schema to limit grammar complexity as panels grow; target coverage is validated locally. Flagged quotes may differ in whitespace only and are stored as their original source excerpts.

The Debate page exports a standalone HTML transcript containing the panel, rules, all rounds, peer assessments, final brief, balances and expandable API diagnostics.

Web search can be enabled on Setup for panel contributions (off by default), including agents added later. It uses OpenAI Responses web search, Anthropic web search, and Gemini Google Search grounding. Moderator and peer-review calls do not search. Citations appear in contributions; provider search metadata and token usage are saved in API diagnostics, the database, and the HTML/JSON logs. OpenAI/Anthropic allow up to two tool uses per contribution; Gemini query counts are logged but not hard-capped. Search and token fees apply; enabling search does not force its use. Counts across providers are different billing units, not a shared cost estimate.

To include a paid native-search check per model: `python tests/check_models.py --all --web-search --live`. Omit `--live` to preview.

A separate setup toggle lets Agent 0 research a cited shared briefing before designing the panel (one extra model call). Every panel agent receives that briefing, including later additions; later moderator calls do not search. Run-page and HTML summaries show search counts and expandable queries per agent in each round, with Agent 0 setup search shown under Panel and rules. Raw metadata remains in the JSON log and database diagnostics.
