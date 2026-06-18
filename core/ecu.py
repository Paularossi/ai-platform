"""
core/ecu.py

ECU mechanism — peer review based.

Components
----------

  PeerReviewRound
  ---------------
  Prompts each agent to score all other agents on the five quality
  dimensions and report a coalition agreement score (0-1) per peer.
  Returns a list of PeerReviewOutput objects.

  ECU formula:
    ecu_i = sum_q  w_q^ECU * (1/(n-1)) * sum_{j≠i} s_{ji}(q)

  Social welfare formula:
    SW = sum_q w_q^SW * (1/n) * sum_i (1/(n-1)) * sum_{j≠i} s_{ji}(q)

  The SW weights are fixed social-planner valuations. The ECU weights are
  variable Orchestrator incentives and are the only weights updated by the
  Orchestrator.

  If self-assessment is enabled, self-scores are included with weight λ:
    ecu_i = sum_q  w_q * [ λ*s_{ii}(q) + (1/(n-1))*sum_{j≠i} s_{ji}(q) ] / (1 + λ)

  CoalitionTracker
  ----------------
  After each round's peer review, finds the largest subset of agents
  with mutual coalition agreement ≥ τ (in both directions).

  EcuLedger
  ---------
  Unchanged — accumulates per-agent ecu balances and records history.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from core.state import PeerReviewOutput


# ---------------------------------------------------------------------------
# Quality dimensions
# ---------------------------------------------------------------------------

DEFAULT_DIMENSIONS: list[dict] = [
    {
        "name": "depth_breadth",
        "label": "Depth-Breadth",
        "rubric": (
            "Does the contribution explore the topic with both analytical rigour "
            "and sufficient breadth? "
            "0 = too narrow or too diffuse; 1 = well-balanced depth and breadth."
        ),
    },
    {
        "name": "depth",
        "label": "Depth",
        "rubric": (
            "Is the reasoning deep and well-evidenced? "
            "0 = superficial assertion; 1 = rigorous argument with supporting logic or evidence."
        ),
    },
    {
        "name": "clarity",
        "label": "Clarity",
        "rubric": (
            "Is the contribution clearly written and easy to understand? "
            "0 = ambiguous or poorly structured; 1 = precise and well-structured."
        ),
    },
    {
        "name": "consensus",
        "label": "Consensus",
        "rubric": (
            "Does the contribution constructively advance group agreement or "
            "productively engage with opposing views? "
            "0 = purely divisive; 1 = constructively bridges perspectives."
        ),
    },
]


# ---------------------------------------------------------------------------
# PeerReviewRound
# ---------------------------------------------------------------------------

class PeerReviewRound:
    """
    Builds the Phase 2 peer review prompt and parses the response.

    The actual LLM call is made by the Agent via the protocol — not here.
    This class handles prompt construction and response parsing only.

    Parameters
    ----------
    dimensions : list[dict]
    include_self_assessment : bool
    review_depth : str
        φ₂ — how much contribution history the reviewer sees.
        "current_only"   → only the current round's contribution
        "previous_round" → current + previous round side-by-side
        "full_history"   → full trajectory across all rounds
    dry_run : bool
    """

    def __init__(
        self,
        dimensions: list[dict] | None = None,
        include_self_assessment: bool = False,
        review_depth: str = "previous_round",
        dry_run: bool = False,
        reviewer_provider: str = "OpenAI",
        reviewer_model: str = "gpt-4o",
    ):
        self.dimensions = dimensions or DEFAULT_DIMENSIONS
        self.include_self_assessment = include_self_assessment
        self.review_depth = review_depth
        self.dry_run = dry_run
        self.reviewer_provider = reviewer_provider
        self.reviewer_model = reviewer_model

    def build_prompt(
        self,
        reviewer_name: str,
        reviewer_contribution: Any,
        all_contributions: dict[str, Any],
        cycle: int,
        item_context: str = "",
        ecu_info: str = "",
        reviewer_role: str = "",
        agent_histories: dict[str, list[str]] | None = None,
        collect_importance_votes: bool = False,
    ) -> str:
        """
        Build the peer review prompt.

        Parameters
        ----------
        agent_histories : dict[str, list[str]] | None
            Per-agent contribution history from hub.build_review_history().
            {agent_name: [contribution_round_0, contribution_round_1, ...]}
            Used when review_depth != "current_only".
        """
        review_targets = list(all_contributions.keys()) if self.include_self_assessment \
            else [n for n in all_contributions if n != reviewer_name]

        dim_names = [d["name"] for d in self.dimensions]
        # On cycle 0 there is no previous round — treat as current_only
        # regardless of review_depth to avoid confusing agents.
        is_first_round = (cycle == 0)
        has_history = (
            agent_histories
            and self.review_depth != "current_only"
            and not is_first_round
        )
        lines: list[str] = []

        lines.append(
            f"You are {reviewer_name}."
            + (f" {reviewer_role}." if reviewer_role else "")
            + " The contribution round has just ended. Complete the peer review below."
        )
        lines.append(f"Topic: {item_context}")
        lines.append("")

        if ecu_info:
            lines.append(ecu_info)
            lines.append("")

        if is_first_round:
            lines.append(
                "NOTE: This is Round 1. No previous contributions exist. "
                "Agents are contributing for the first time."
            )
            lines.append("")

        lines.append("Your contribution this round:")
        contrib_text = (
            reviewer_contribution if isinstance(reviewer_contribution, str)
            else json.dumps(reviewer_contribution, ensure_ascii=False)
        )
        lines.append(f"  {contrib_text}")
        lines.append("")

        # Show peer contributions — with or without history
        if has_history:
            depth_label = {
                "previous_round": "current round + previous round",
                "full_history": "all rounds",
            }.get(self.review_depth, "")
            lines.append(
                f"Contributions per agent ({depth_label}) — "
                "most recent is labelled [current]:"
            )
            for name, contrib in all_contributions.items():
                if name == reviewer_name and not self.include_self_assessment:
                    continue
                lines.append(f"  [{name}]")
                history = agent_histories.get(name, [])
                # Show history entries oldest-first, mark the last as [current]
                if len(history) > 1:
                    for i, h in enumerate(history[:-1]):
                        lines.append(f"    Round {cycle - (len(history)-1-i)}: {h}")
                if history:
                    lines.append(f"    [current]: {history[-1]}")
                else:
                    current_text = contrib if isinstance(contrib, str) \
                        else json.dumps(contrib, ensure_ascii=False)
                    lines.append(f"    [current]: {current_text}")
        else:
            lines.append("Contributions this round:")
            for name, contrib in all_contributions.items():
                if name == reviewer_name and not self.include_self_assessment:
                    continue
                text = contrib if isinstance(contrib, str) \
                    else json.dumps(contrib, ensure_ascii=False)
                lines.append(f"  [{name}]: {text}")
        lines.append("")

        lines.append("For each agent, provide:")
        lines.append(f"  1. Quality scores on {len(self.dimensions)} dimensions (0.00 to 1.00 each).")
        lines.append("  2. A one-sentence justification summarising your overall assessment.")
        if has_history:
            lines.append(
                "For the consensus dimension specifically: "
                "score whether this agent's position CHANGED meaningfully from their "
                "previous round in response to others' arguments. "
                "0 = no change at all, 1 = substantially updated position."
            )
        elif is_first_round:
            lines.append(
                "For the consensus dimension specifically: "
                "this is Round 1 — there is no previous contribution to compare against. "
                "Score instead how constructively this contribution opens dialogue "
                "and invites agreement. "
                "0 = purely adversarial or closed; 1 = constructively invites convergence."
            )
        lines.append("")
        lines.append("Quality dimension rubrics:")
        for d in self.dimensions:
            lines.append(f"  {d['label']} ({d['name']}): {d['rubric']}")
        lines.append("")

        if collect_importance_votes:
            lines.append(
                "After scoring all agents, also provide your IMPORTANCE VOTE: "
                "distribute exactly 100 points across the quality dimensions to reflect "
                "which dimensions matter most to you, given the topic of the debate and your role. "
                "You may assign 0 to a dimension if you consider it unimportant. "
                "The points must sum to exactly 100."
            )
            lines.append("")

        lines.append("Respond ONLY with a JSON object in this exact format:")
        lines.append("{")
        for name in review_targets:
            lines.append(f'  "{name}": {{')
            for dim in dim_names:
                lines.append(f'    "{dim}": <score 0.00-1.00>,')
            lines.append('    "justification": "<one sentence>"')
            lines.append("  },")
        if collect_importance_votes:
            lines.append('  "importance_votes": {')
            for i, dim in enumerate(dim_names):
                comma = "," if i < len(dim_names) - 1 else ""
                lines.append(f'    "{dim}": <integer points>{comma}')
            lines.append("  }")
        lines.append("}")
        lines.append("No other text. No markdown fences.")
        return "\n".join(lines)

    def parse(
        self,
        reviewer_name: str,
        cycle: int,
        raw: str,
        all_contributions: dict[str, Any],
    ) -> PeerReviewOutput:
        """
        Parse the agent's raw response into a PeerReviewOutput.

        Coalition scores are derived from the consensus dimension score —
        no separate coalition_agreement field is asked or parsed.
        """
        review_targets = list(all_contributions.keys()) if self.include_self_assessment \
            else [n for n in all_contributions if n != reviewer_name]
        dim_names = [d["name"] for d in self.dimensions]

        if self.dry_run:
            scores = {
                name: {d: 0.5 for d in dim_names}
                for name in review_targets if name != reviewer_name
            }
            self_scores = (
                {d: 0.5 for d in dim_names}
                if self.include_self_assessment else None
            )
            return PeerReviewOutput(
                reviewer_name=reviewer_name,
                cycle=cycle,
                scores=scores,
                self_scores=self_scores,
                coalition_scores={n: 0.5 for n in all_contributions if n != reviewer_name},
                coalition_justifications={},
                importance_votes={},
                raw_response="[dry-run]",
            )

        scores: dict[str, dict[str, float]] = {}
        self_scores: dict[str, float] | None = None
        justifications: dict[str, str] = {}
        importance_votes: dict[str, float] = {}

        try:
            clean = re.sub(r"```[a-z]*\n?", "", raw).strip()
            clean = re.sub(r",(\s*[}\]])", r"\1", clean)
            parsed = json.loads(clean)

            for name in review_targets:
                if name not in parsed:
                    continue
                entry = parsed[name]
                dim_scores: dict[str, float] = {}
                for dim in dim_names:
                    try:
                        dim_scores[dim] = max(0.0, min(1.0, float(entry.get(dim, 0.5))))
                    except (ValueError, TypeError):
                        dim_scores[dim] = 0.5

                if name == reviewer_name:
                    self_scores = dim_scores
                else:
                    scores[name] = dim_scores
                    justifications[name] = str(entry.get("justification", ""))

            # Parse importance votes if present
            if "importance_votes" in parsed and isinstance(parsed["importance_votes"], dict):
                raw_votes = parsed["importance_votes"]
                importance_votes = {
                    d: max(0.0, float(raw_votes.get(d, 0.0)))
                    for d in dim_names
                }
                total = sum(importance_votes.values())
                if total > 0:
                    # Normalise to sum exactly 100
                    importance_votes = {d: v * 100.0 / total for d, v in importance_votes.items()}

        except Exception as exc:
            print(f"[PeerReviewRound] Parse error for {reviewer_name}: {exc}")
            for name in review_targets:
                if name != reviewer_name:
                    scores[name] = {d: 0.5 for d in dim_names}
                    justifications[name] = ""

        # Derive coalition scores from the consensus dimension
        coalition: dict[str, float] = {
            name: scores[name].get("consensus", 0.5)
            for name in scores
        }

        return PeerReviewOutput(
            reviewer_name=reviewer_name,
            cycle=cycle,
            scores=scores,
            self_scores=self_scores,
            coalition_scores=coalition,
            coalition_justifications=justifications,
            importance_votes=importance_votes,
            raw_response=raw,
        )

    # Keep run() as a convenience wrapper that makes the LLM call directly
    # — used when no Agent object is available (e.g. standalone testing).
    # In the main protocol flow, the protocol calls agent.call() instead.
    def run(
        self,
        reviewer_name: str,
        reviewer_contribution: Any,
        all_contributions: dict[str, Any],
        cycle: int,
        item_context: str = "",
        ecu_info: str = "",
    ) -> PeerReviewOutput:

        prompt = self.build_prompt(
            reviewer_name, reviewer_contribution, all_contributions,
            cycle, item_context, ecu_info,
        )
        try:
            from core.providers import get_provider
            raw = get_provider(self.reviewer_provider).complete(
                model=self.reviewer_model,
                system_prompt="You are a structured peer reviewer. Follow the instructions exactly.",
                user_message=prompt,
                max_tokens=800,
            )
        except Exception as exc:
            raw = f"[ERROR: {exc}]"

        return self.parse(reviewer_name, cycle, raw, all_contributions)


# ---------------------------------------------------------------------------
# CoalitionTracker
# ---------------------------------------------------------------------------

class CoalitionTracker:
    """
    Finds the largest coalition given a round's peer review outputs.

    A coalition is a subset S ⊆ N such that for all i,j ∈ S (i≠j):
        coalition_scores[i][j] ≥ τ  AND  coalition_scores[j][i] ≥ τ

    Parameters
    ----------
    threshold : float
        Minimum mutual agreement score τ (default 0.6).
    """

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self._history: list[dict] = []

    def find_coalition(
        self, reviews: list[PeerReviewOutput]
    ) -> list[str]:
        """
        Return the largest coalition from this round's peer review outputs.
        If multiple coalitions of the same size exist, returns the first found.
        """
        # Build agreement matrix: agree[i][j] = score i gives j
        agree: dict[str, dict[str, float]] = {}
        for r in reviews:
            agree[r.reviewer_name] = r.coalition_scores

        agents = list(agree.keys())
        n = len(agents)

        best: list[str] = []

        # Enumerate all subsets of size ≥ 2 (a coalition requires mutual agreement
        # between at least two agents; single-agent subsets are meaningless)
        for size in range(n, 1, -1):
            if size <= len(best):
                break
            for subset in _subsets(agents, size):
                if self._is_coalition(subset, agree):
                    if len(subset) > len(best):
                        best = list(subset)
            if best:
                break

        record = {
            "coalition": best,
            "size": len(best),
            "threshold": self.threshold,
            "agreement_matrix": {
                i: {j: agree.get(i, {}).get(j, 0.0) for j in agents}
                for i in agents
            },
        }
        self._history.append(record)
        return best

    def _is_coalition(
        self, subset: list[str], agree: dict[str, dict[str, float]]
    ) -> bool:
        for i in subset:
            for j in subset:
                if i == j:
                    continue
                if agree.get(i, {}).get(j, 0.0) < self.threshold:
                    return False
                if agree.get(j, {}).get(i, 0.0) < self.threshold:
                    return False
        return True

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "history": self._history,
        }


def _subsets(items: list, size: int):
    """Yield all subsets of `items` of given size."""
    from itertools import combinations
    yield from combinations(items, size)


# ---------------------------------------------------------------------------
# EcuLedger (unchanged interface, updated formula)
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    """ECU record for one agent in one round."""
    agent_name: str
    cycle: int
    item_id: str
    peer_scores: dict[str, dict[str, float]]  # {reviewer: {dim: score}}
    self_scores: dict[str, float] | None
    aggregated_scores: dict[str, float]        # mean per dimension
    weights: dict[str, float]                  # legacy alias: ECU weights
    ecu_weights: dict[str, float]
    sw_weights: dict[str, float]
    social_welfare: float | None
    ecu_earned: float
    contribution_preview: str

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "cycle": self.cycle,
            "item_id": self.item_id,
            "peer_scores": self.peer_scores,
            "self_scores": self.self_scores,
            "aggregated_scores": {k: round(v, 4) for k, v in self.aggregated_scores.items()},
            "weights": self.weights,
            "ecu_weights": self.ecu_weights,
            "sw_weights": self.sw_weights,
            "social_welfare": None if self.social_welfare is None else round(self.social_welfare, 4),
            "ecu_earned": round(self.ecu_earned, 4),
            "contribution_preview": self.contribution_preview,
        }


class EcuLedger:
    """
    Accumulates ecu balances from peer review rounds.

    Parameters
    ----------
    agent_names : list[str]
    dimensions : list[dict]
    weights : dict[str, float]
        Backward-compatible alias for initial ECU weight vector w^ECU.
    ecu_weights : dict[str, float]
        Variable Orchestrator incentive weights w^ECU used for ECU payouts.
    sw_weights : dict[str, float]
        Fixed social-planner valuation weights w^SW used for social welfare.
    include_self_assessment : bool
        If True, self-scores contribute with weight lambda_self.
    lambda_self : float
        Weight of self-score relative to peer scores (default 0.5).
    """

    def __init__(
        self,
        agent_names: list[str],
        dimensions: list[dict] | None = None,
        weights: dict[str, float] | None = None,
        ecu_weights: dict[str, float] | None = None,
        sw_weights: dict[str, float] | None = None,
        include_self_assessment: bool = False,
        lambda_self: float = 0.5,
    ):
        self.dimensions = dimensions or DEFAULT_DIMENSIONS
        self.dim_names = [d["name"] for d in self.dimensions]
        self.include_self_assessment = include_self_assessment
        self.lambda_self = lambda_self

        default_weights = {d["name"]: 1.0 for d in self.dimensions}

        # w^ECU: variable incentive weights. `weights` is kept as a legacy alias.
        self.ecu_weights: dict[str, float] = dict(default_weights)
        if weights:
            self.ecu_weights.update(weights)
        if ecu_weights:
            self.ecu_weights.update(ecu_weights)

        # w^SW: fixed social-planner valuation weights. These are not updated
        # by the Orchestrator.
        self.sw_weights: dict[str, float] = dict(default_weights)
        if sw_weights:
            self.sw_weights.update(sw_weights)

        self._balances: dict[str, float] = {name: 0.0 for name in agent_names}
        self._history: list[TurnRecord] = []
        self._social_welfare_history: list[dict[str, Any]] = []

    def record_from_reviews(
        self,
        agent_name: str,
        cycle: int,
        item_id: str,
        contribution: Any,
        reviews: list[PeerReviewOutput],
    ) -> float:
        """
        Compute and record ecu for agent_name from a round's peer review outputs.

        Implements the ECU formula:
            ecu_i = sum_q w_q^ECU * mean_peer_score_q(i)

        With optional self-assessment (eq. weighted):
            ecu_i = sum_q w_q * [λ*s_ii(q) + mean_peer_q(i)] / (1 + λ)
        """
        # Collect peer scores for this agent
        peer_scores: dict[str, dict[str, float]] = {}
        self_scores: dict[str, float] | None = None

        for review in reviews:
            if review.reviewer_name == agent_name:
                # Self-assessment
                if review.self_scores:
                    self_scores = review.self_scores
            elif agent_name in review.scores:
                peer_scores[review.reviewer_name] = review.scores[agent_name]

        # Aggregate: mean per dimension across peer reviewers
        agg: dict[str, float] = {}
        n_peers = len(peer_scores)

        for dim in self.dim_names:
            if n_peers > 0:
                peer_mean = sum(
                    peer_scores[r].get(dim, 0.5) for r in peer_scores
                ) / n_peers
            else:
                peer_mean = 0.5

            if self.include_self_assessment and self_scores:
                self_val = self_scores.get(dim, 0.5)
                agg[dim] = (self.lambda_self * self_val + peer_mean) / (1 + self.lambda_self)
            else:
                agg[dim] = peer_mean

        # ECU_i^(t) = Σ_q w_q^ECU · (1/(n-1))Σ_{j≠i}s_ji^(t)(q)
        ecu = sum(agg.get(d, 0.0) * self.ecu_weights.get(d, 1.0) for d in self.dim_names)

        sw = self.compute_social_welfare(reviews)

        self._balances[agent_name] = self._balances.get(agent_name, 0.0) + ecu

        contrib_str = (
            contribution if isinstance(contribution, str)
            else json.dumps(contribution, ensure_ascii=False)
        )

        self._history.append(TurnRecord(
            agent_name=agent_name,
            cycle=cycle,
            item_id=item_id,
            peer_scores=peer_scores,
            self_scores=self_scores,
            aggregated_scores=agg,
            weights=dict(self.ecu_weights),
            ecu_weights=dict(self.ecu_weights),
            sw_weights=dict(self.sw_weights),
            social_welfare=sw,
            ecu_earned=ecu,
            contribution_preview=contrib_str[:120],
        ))

        return ecu

    @property
    def weights(self) -> dict[str, float]:
        """Backward-compatible alias for variable ECU weights w^ECU."""
        return self.ecu_weights

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Update variable ECU incentive weights w^ECU."""
        self.ecu_weights.update(new_weights)

    def update_ecu_weights(self, new_weights: dict[str, float]) -> None:
        """Explicit alias for updating w^ECU."""
        self.update_weights(new_weights)

    def update_sw_weights(self, new_weights: dict[str, float]) -> None:
        """Update fixed SW valuation weights manually; the Orchestrator should not call this."""
        self.sw_weights.update(new_weights)

    def mean_peer_scores(self, reviews: list[PeerReviewOutput]) -> dict[str, float]:
        """
        Compute s̄_q^(t) = (1/n)Σ_i (1/(n-1))Σ_{j≠i}s_ji^(t)(q).

        This is the mean peer score for each dimension across all reviewed
        agents and reviewers in a round.
        """
        by_dim: dict[str, list[float]] = {d: [] for d in self.dim_names}
        for review in reviews:
            for dim_scores in review.scores.values():
                for dim in self.dim_names:
                    if dim in dim_scores:
                        by_dim[dim].append(float(dim_scores[dim]))
        return {dim: (sum(vals) / len(vals) if vals else 0.0) for dim, vals in by_dim.items()}

    def compute_social_welfare(self, reviews: list[PeerReviewOutput]) -> float:
        """Compute SW^(t) = Σ_q w_q^SW · s̄_q^(t)."""
        means = self.mean_peer_scores(reviews)
        return sum(self.sw_weights.get(dim, 1.0) * means.get(dim, 0.0) for dim in self.dim_names)

    def record_social_welfare(self, cycle: int, reviews: list[PeerReviewOutput]) -> float:
        """Compute and store social welfare for one round."""
        means = self.mean_peer_scores(reviews)
        sw = sum(self.sw_weights.get(dim, 1.0) * means.get(dim, 0.0) for dim in self.dim_names)
        self._social_welfare_history.append({
            "cycle": cycle,
            "mean_peer_scores": {k: round(v, 4) for k, v in means.items()},
            "sw_weights": dict(self.sw_weights),
            "ecu_weights": dict(self.ecu_weights),
            "social_welfare": round(sw, 4),
        })
        return sw

    @property
    def balances(self) -> dict[str, float]:
        return dict(self._balances)

    @property
    def history(self) -> list[TurnRecord]:
        return list(self._history)

    def balance_for(self, agent_name: str) -> float:
        """Return the cumulative ECU balance for one agent."""
        return self._balances.get(agent_name, 0.0)

    def last_round_ecu_for(self, agent_name: str) -> float:
        """
        Return the ECU earned by an agent in the most recent round.

        Used as a reputation signal: reflects current performance rather than
        cumulative history, so reputation can rise and fall across rounds.
        Returns 0.0 if the agent has no recorded history.
        """
        agent_records = [r for r in reversed(self._history) if r.agent_name == agent_name]
        if not agent_records:
            return 0.0
        latest_cycle = agent_records[0].cycle
        cycle_records = [r for r in agent_records if r.cycle == latest_cycle]
        return sum(r.ecu_earned for r in cycle_records)

    def reputation_scores(self) -> dict[str, float]:
        """
        Return {agent_name: last_round_ecu} for all agents.

        Convenience method for passing reputation context into prompts or
        weighting peer review scores.
        """
        return {name: self.last_round_ecu_for(name) for name in self._balances}

    @property
    def social_welfare_history(self) -> list[dict[str, Any]]:
        return list(self._social_welfare_history)

    def scores_by_dimension(self) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {d: [] for d in self.dim_names}
        for rec in self._history:
            for dim, score in rec.aggregated_scores.items():
                if dim in result:
                    result[dim].append(score)
        return result

    def to_dict(self) -> dict:
        return {
            "weights": self.ecu_weights,
            "ecu_weights": self.ecu_weights,
            "sw_weights": self.sw_weights,
            "balances": {k: round(v, 4) for k, v in self._balances.items()},
            "social_welfare_history": self.social_welfare_history,
            "history": [r.to_dict() for r in self._history],
        }