"""
tests/test_pipeline.py

Test suite for the multi-agent deliberation platform.

Run from the project root:
    python tests/test_pipeline.py

Covers:
    1. Event sequence and count
    2. Phase 1 blindness (no cross-visibility within a round)
    3. Peer review scores structure and values
    4. ECU calculation (weighted peer-averaged scores)
    5. Self-assessment mode
    6. Coalition tracking at different thresholds
    7. Prompt construction (system prompt vs user message separation)
    8. Serialisation to dict / JSON
"""

import sys
import json

sys.path.insert(0, ".")

from core.state import AgentOutput
from core.hub import CommunicationHub, ContextPacket
from core.agent import Agent, _build_system_prompt, _build_user_message
from core.ecu import PeerReviewRound, CoalitionTracker, EcuLedger, DEFAULT_DIMENSIONS
from core.protocols.crowd import CrowdProtocol


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

AGENTS_CFG = [
    {
        "name": "Economist",
        "provider": "OpenAI",
        "model": "gpt-4o",
        "role": "Custom",
        "custom_role": "Argue from welfare economics. Cite empirical evidence.",
    },
    {
        "name": "Urban Planner",
        "provider": "OpenAI",
        "model": "gpt-4o",
        "role": "Custom",
        "custom_role": "Balance accessibility and mobility across the city.",
    },
    {
        "name": "Citizens Advocate",
        "provider": "OpenAI",
        "model": "gpt-4o",
        "role": "Custom",
        "custom_role": "Represent distributional concerns and equity.",
    },
]

EXPERIMENT_CFG = {
    "protocol": {
        "setting": "Crowd (parallel)",
        "max_cycles": 2,
        "stopping_rule": "Max cycles",
        "visibility_mode": "Summary only",
    },
    "instructions": {
        "base_instructions": (
            "You are a policy advisor. Give a concise position statement "
            "(2-3 sentences) from your perspective."
        ),
        "guideline_notes": "",
    },
    "agent_prompt_overrides": {},
}

ITEM = {"question": "Should the city introduce congestion charges on its inner ring road?"}


def build_run(max_cycles=2, include_self=False, threshold=0.6, info_cond="opaque"):
    """Build and run a full dry-run experiment, returning all components."""
    agents = [Agent(cfg, EXPERIMENT_CFG) for cfg in AGENTS_CFG]
    agent_names = [a.name for a in agents]
    weights = {d["name"]: 1.0 for d in DEFAULT_DIMENSIONS}

    peer_reviewer = PeerReviewRound(dry_run=True, include_self_assessment=include_self)
    ledger = EcuLedger(agent_names, weights=weights, include_self_assessment=include_self)
    coalition = CoalitionTracker(threshold=threshold)

    cfg = dict(EXPERIMENT_CFG)
    cfg["protocol"] = dict(cfg["protocol"])
    cfg["protocol"]["max_cycles"] = max_cycles

    hub = CommunicationHub(
        item_id="q1",
        item_data=ITEM,
        visibility_mode="Summary only",
        agent_names=agent_names,
        ledger=ledger,
        ecu_info_condition=info_cond,
    )
    protocol = CrowdProtocol(
        agents, cfg,
        peer_reviewer=peer_reviewer,
        coalition_tracker=coalition,
        dry_run=True,
    )
    events = list(protocol.run_iter(hub))
    return hub, events, ledger, coalition, agents


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

PASS, FAIL = 0, 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        print(f"  ✓ {name}")
        PASS += 1
    else:
        print(f"  ✗ {name}" + (f": {detail}" if detail else ""))
        FAIL += 1


# ---------------------------------------------------------------------------
# Test 1: Event sequence
# ---------------------------------------------------------------------------

def test_event_sequence() -> None:
    print("\n=== TEST 1: Event sequence ===")
    hub, events, *_ = build_run(max_cycles=2)
    kinds = [e.kind for e in events]

    check(
        "20 total events (3 dispatch + 3 submission + 3 peer_review + 1 ecu_update) × 2 cycles",
        len(events) == 20,
        f"got {len(events)}",
    )
    check(
        "Correct event-kind counts",
        kinds.count("dispatch") == 6
        and kinds.count("submission") == 6
        and kinds.count("peer_review") == 6
        and kinds.count("ecu_update") == 2,
        str({k: kinds.count(k) for k in set(kinds)}),
    )
    check(
        "Phase 1 (submission) before Phase 2 (peer_review) in each cycle",
        kinds.index("peer_review") > kinds.index("submission"),
    )


# ---------------------------------------------------------------------------
# Test 2: Phase 1 blindness
# ---------------------------------------------------------------------------

def test_phase1_blindness() -> None:
    print("\n=== TEST 2: Phase 1 blindness ===")
    hub, events, *_ = build_run(max_cycles=2)

    cycle0_dispatches = [e for e in events if e.kind == "dispatch" and e.cycle == 0]
    check(
        "Round-0 packets have empty visible history",
        all(len(e.packet.visible_history) == 0 for e in cycle0_dispatches),
    )

    cycle1_dispatches = [e for e in events if e.kind == "dispatch" and e.cycle == 1]
    check(
        "Round-1 packets have non-empty visible history",
        all(len(e.packet.visible_history) > 0 for e in cycle1_dispatches),
    )

    check(
        "Round-1 visible history contains all 3 agents' contributions",
        all(len(e.packet.visible_history) == 3 for e in cycle1_dispatches),
        f"got {[len(e.packet.visible_history) for e in cycle1_dispatches]}",
    )


# ---------------------------------------------------------------------------
# Test 3: Peer review structure
# ---------------------------------------------------------------------------

def test_peer_review_structure() -> None:
    print("\n=== TEST 3: Peer review scores ===")
    hub, *_ = build_run(max_cycles=2)

    check(
        "Hub has 6 peer reviews (3 agents × 2 cycles)",
        len(hub.peer_review_log) == 6,
        f"got {len(hub.peer_review_log)}",
    )

    for review in hub.peer_review_log:
        check(
            f"{review.reviewer_name} (cycle {review.cycle}): scored 2 peers",
            len(review.scores) == 2,
            f"scored {len(review.scores)}",
        )
        for reviewed, scores in review.scores.items():
            check(
                f"  {review.reviewer_name}→{reviewed}: {len(scores)} dimensions: {list(scores.keys())}", len(scores) == len(DEFAULT_DIMENSIONS),
                str(list(scores.keys())),
            )
            check(
                f"  {review.reviewer_name}→{reviewed}: all 0.5 (dry-run)",
                all(abs(v - 0.5) < 0.001 for v in scores.values()),
            )
        check(
            f"{review.reviewer_name}: coalition scores for 2 peers",
            len(review.coalition_scores) == 2,
        )


# ---------------------------------------------------------------------------
# Test 4: ECU calculation
# ---------------------------------------------------------------------------

def test_ecu_calculation() -> None:
    print("\n=== TEST 4: ECU calculation ===")
    hub, _, ledger, *_ = build_run(max_cycles=2)

    check(
        "Ledger has 6 records (3 agents × 2 cycles)",
        len(ledger.history) == 6,
        f"got {len(ledger.history)}",
    )

    expected_per_turn = 4 * 1.0 * 0.5  # 4 dims × w=1.0 × score=0.5
    for rec in ledger.history:
        check(
            f"{rec.agent_name} cycle {rec.cycle}: ecu = {rec.ecu_earned:.3f}",
            abs(rec.ecu_earned - expected_per_turn) < 0.001,
            f"expected {expected_per_turn}",
        )

    expected_total = expected_per_turn * 2
    for name, bal in hub.ecu_balances.items():
        check(
            f"{name} total balance = {bal:.3f}",
            abs(bal - expected_total) < 0.001,
            f"expected {expected_total}",
        )


# ---------------------------------------------------------------------------
# Test 5: Self-assessment
# ---------------------------------------------------------------------------

def test_self_assessment() -> None:
    print("\n=== TEST 5: Self-assessment ===")
    hub, _, ledger, *_ = build_run(max_cycles=1, include_self=True)

    for review in hub.peer_review_log:
        check(
            f"{review.reviewer_name} has self_scores",
            review.self_scores is not None,
        )

    # With λ=0.5: ecu = sum_q w_q * (λ*self + peer_mean) / (1 + λ)
    # dry-run: self=0.5, peer_mean=0.5 → ecu = sum_q 1.0 * 0.5 = 2.0 (4 dims)
    expected_sa = 4 * 1.0 * (0.5 * 0.5 + 0.5) / 1.5
    for rec in ledger.history:
        check(
            f"{rec.agent_name} SA ecu = {rec.ecu_earned:.4f}",
            abs(rec.ecu_earned - expected_sa) < 0.001,
            f"expected {expected_sa:.4f}",
        )


# ---------------------------------------------------------------------------
# Test 6: Coalition tracking
# ---------------------------------------------------------------------------

def test_coalition() -> None:
    print("\n=== TEST 6: Coalition tracking ===")
    _, _, _, coalition, _ = build_run(max_cycles=2, threshold=0.6)

    check("Coalition history has 2 entries", len(coalition.history) == 2)
    for entry in coalition.history:
        check(
            "No coalition at τ=0.6 (dry-run scores 0.5 < 0.6)",
            entry["size"] <= 1,
            f"got size {entry['size']}",
        )

    _, _, _, coalition2, _ = build_run(max_cycles=1, threshold=0.4)
    check(
        "Full coalition at τ=0.4 (0.5 ≥ 0.4)",
        coalition2.history[-1]["size"] == 3,
        f"got {coalition2.history[-1]['size']}",
    )


# ---------------------------------------------------------------------------
# Test 7: Prompt construction — no item data in system prompt
# ---------------------------------------------------------------------------

def test_prompt_construction() -> None:
    print("\n=== TEST 7: Prompt construction ===")

    system = _build_system_prompt(
        "Economist",
        "Argue from welfare economics.",
        "You are a policy advisor.",
        "",
        {},
    )
    check("System prompt contains agent name", "Economist" in system)
    check("System prompt contains custom role description", "welfare economics" in system)
    check("System prompt has NO item data", "congestion" not in system.lower())

    packet0 = ContextPacket(
        item_id="q1",
        item_data=ITEM,
        current_contribution=None,
        visible_history=[],
        cycle=0,
        agent_name="Economist",
    )
    msg0 = _build_user_message(packet0)
    check("Round-0 user message contains item data", "congestion charges" in msg0)
    check("Round-0 user message has no history", "Previous round" not in msg0)

    prev = AgentOutput(
        agent_name="Urban Planner",
        cycle=0,
        contribution="Spatial displacement matters.",
        ecu_scores={"depth_breadth": 0.7, "depth": 0.6, "clarity": 0.8,
                    "consensus": 0.5},
        ecu_earned=3.3,
    )
    packet1 = ContextPacket(
        item_id="q1",
        item_data=ITEM,
        current_contribution="Spatial displacement matters.",
        visible_history=[prev],
        cycle=1,
        agent_name="Economist",
    )
    msg1 = _build_user_message(packet1)
    check("Round-1 message has previous round section", "Previous round" in msg1)
    check("Round-1 message shows peer contributions", "Urban Planner" in msg1)
    check("Round-1 message shows peer review scores", "Peer review scores" in msg1)
    check("Round-1 message shows ECU earned", "3.30 ecus" in msg1)
    check("Round-1 message ends with Your turn marker", "Your turn" in msg1)


# ---------------------------------------------------------------------------
# Test 8: Serialisation
# ---------------------------------------------------------------------------

def test_serialisation() -> None:
    print("\n=== TEST 8: Serialisation ===")
    hub, *_ = build_run(max_cycles=2)

    d = hub.to_dict()
    check("to_dict has log with 6 entries", len(d.get("log", [])) == 6)
    check("to_dict has peer_review_log with 6 entries",
          len(d.get("peer_review_log", [])) == 6)
    check("to_dict has ecu section", "ecu" in d)
    check("Each peer review entry has scores field",
          all("scores" in p for p in d["peer_review_log"]))
    check("Each peer review entry has coalition_scores field",
          all("coalition_scores" in p for p in d["peer_review_log"]))
    check("Each log entry has ecu_scores field",
          all("ecu_scores" in e for e in d["log"]))

    json_str = json.dumps(d)
    check("Full dict serialises to JSON without error", len(json_str) > 200)

    reparsed = json.loads(json_str)
    check("JSON round-trips correctly",
          reparsed["item_id"] == hub.item_id
          and len(reparsed["log"]) == 6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_event_sequence()
    test_phase1_blindness()
    test_peer_review_structure()
    test_ecu_calculation()
    test_self_assessment()
    test_coalition()
    test_prompt_construction()
    test_serialisation()

    print(f"\n{'=' * 50}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("All tests pass ✓")
        sys.exit(0)
    else:
        print("Some tests FAILED ✗")
        sys.exit(1)