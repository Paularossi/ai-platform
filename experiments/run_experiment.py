"""
experiments/run_experiment.py

Generic batch runner — loads an experiment JSON file and runs every scenario.

Usage
-----
    python experiments/run_experiment.py experiments/chess_test.json
    python experiments/run_experiment.py experiments/chess_test.json --dry-run

Results land in experiments/results/<scenario_name>_<timestamp>.json
A summary CSV is written to experiments/results/summary_<timestamp>.csv

Experiment JSON format
----------------------
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
    { "name": "my_scenario", "protocol": { "setting": "Simultaneous", ... } },
    ...
  ]
}

The "base" section is identical to the draft JSON saved by the UI (Step 4 → Save draft),
so you can design an experiment in the UI, save the draft, and use it here directly.

API keys
--------
Set your API keys in the .env file before running.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from core.runner import (
    build_agents,
    build_ecu_components,
    build_hub,
    build_peer_reviewer,
    build_protocol,
    collect_result,
    results_to_df,
)

_AMS = ZoneInfo("Europe/Amsterdam")
RESULTS_DIR = Path(__file__).parent / "results"


def load_experiment(path: Path) -> tuple[dict, list[dict], dict]:
    """
    Load and validate an experiment JSON. Returns (base_cfg, scenarios, meta).

    Accepts two formats:
    - Batch format: { "meta": {...}, "base": {...}, "scenarios": [...] }
    - UI draft format: flat dict saved by the UI (no "base" wrapper); the
      embedded "protocol" block becomes a single scenario named "default".
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    if "base" in raw:
        # Batch format
        base = raw["base"]
        scenarios = raw.get("scenarios", [])
    else:
        # UI draft format — treat the whole file as the base config and
        # promote the embedded protocol into a single scenario.
        protocol = raw.pop("protocol", {})
        base = raw
        scenario_name = raw.get("overview", {}).get("name", "default").replace(" ", "_").lower()
        scenarios = [{"name": scenario_name, "protocol": protocol}]

    if not scenarios:
        raise ValueError(f"'{path.name}' has no scenarios and no embedded protocol.")
    if not base.get("agents"):
        raise ValueError("'agents' is empty — add at least one agent.")
    if not base.get("task", {}).get("description", "").strip():
        raise ValueError("'task.description' is empty — set the deliberation topic.")

    meta = base.get("meta") or raw.get("meta") or {}
    return base, scenarios, meta


def build_cfg(base: dict, scenario: dict) -> dict:
    """Merge base config with a scenario's protocol override."""
    cfg = copy.deepcopy(base)
    cfg["protocol"] = scenario["protocol"]
    return cfg


def run_scenario(base: dict, scenario: dict, dry_run: bool = False) -> dict:
    """Run one scenario and return its result dict."""
    cfg = build_cfg(base, scenario)
    review_depth = cfg["protocol"].get("review_depth", "Previous Round")

    agents = build_agents(cfg)
    agent_names = [a.name for a in agents]
    peer_reviewer = build_peer_reviewer(cfg, review_depth)

    topic = cfg.get("task", {}).get("description", "")
    item_id = "item_0"
    item_data = {"topic": topic}

    ledger, coalition, orchestrator = build_ecu_components(cfg, agent_names, review_depth)
    protocol = build_protocol(
        agents, cfg,
        peer_reviewer=peer_reviewer,
        coalition_tracker=coalition,
        orchestrator=orchestrator,
        dry_run=dry_run,
    )
    hub = build_hub(item_id, item_data, cfg, agent_names, ledger)

    max_cycles = cfg["protocol"].get("max_cycles", "?")
    current_cycle = -1
    for event in protocol.run_iter(hub):
        if event.cycle != current_cycle:
            current_cycle = event.cycle
            print(f"  round {current_cycle + 1}/{max_cycles}", flush=True)
        if event.kind == "dispatch":
            print(f"    {event.agent_name} ...", end="", flush=True)
        elif event.kind == "submission":
            print(" done", flush=True)
        elif event.kind == "peer_review":
            print(f"    peer review ...", flush=True)

    result = collect_result(hub, ledger, coalition, orchestrator)
    result["scenario"] = scenario["name"]
    result["protocol_config"] = cfg["protocol"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument("experiment", help="Path to the experiment JSON file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls (placeholder outputs). Tests the pipeline.")
    args = parser.parse_args()

    exp_path = Path(args.experiment)
    if not exp_path.exists():
        print(f"Error: '{exp_path}' not found.")
        sys.exit(1)

    base, scenarios, meta = load_experiment(exp_path)

    print(f"Experiment : {meta.get('name', exp_path.stem)}")
    print(f"Author     : {meta.get('author', '—')}")
    print(f"Agents     : {', '.join(a['name'] for a in base['agents'])}")
    print(f"Scenarios  : {len(scenarios)}")
    print(f"Dry run    : {args.dry_run}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(_AMS).strftime("%Y%m%d_%H%M%S")
    all_results: list[dict] = []

    for scenario in scenarios:
        name = scenario["name"]
        proto = scenario["protocol"]
        print(f"\n{'='*60}")
        print(f"Scenario  : {name}")
        print(f"  setting : {proto.get('setting')}")
        print(f"  φ₁      : {proto.get('visibility_mode')}")
        print(f"  φ₂      : {proto.get('review_depth')}")
        print(f"  cycles  : {proto.get('max_cycles')}")
        print(f" ECU visibility : {proto.get('ecu', {}).get('info_condition', False)}")

        result = run_scenario(base, scenario, dry_run=args.dry_run)
        all_results.append(result)

        out_path = RESULTS_DIR / f"{name}_{timestamp}.json"
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"  → saved {out_path.name}")

        coalition_hist = result.get("coalition_history", {}).get("history", [])
        coalition_str = "none"
        if coalition_hist:
            members = coalition_hist[-1].get("coalition", [])
            coalition_str = ", ".join(members) if len(members) >= 2 else "none"
        print(f"  turns={result['num_turns']}  coalition={coalition_str}  "
              f"ecu={result.get('ecu_balances', {})}")

    summary_df = results_to_df(all_results)
    summary_df.insert(0, "scenario", [r["scenario"] for r in all_results])
    csv_path = RESULTS_DIR / f"summary_{timestamp}.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV → {csv_path.name}")
    print("Done.")


if __name__ == "__main__":
    main()
