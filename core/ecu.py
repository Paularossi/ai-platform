"""
core/ecu.py

ECU mechanism — peer review based (OMAS Section 4, meeting update).

Components
----------

  PeerReviewRound
  ---------------
  Prompts each agent to score all other agents on the five quality
  dimensions and report a coalition agreement score (0-1) per peer.
  Returns a list of PeerReviewOutput objects.

  Ecu formula (eq. 1 in simulation section):
    ecu_i = sum_q  w_q * (1/(n-1)) * sum_{j≠i} s_{ji}(q)

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
        "name": "completeness",
        "label": "Completeness",
        "rubric": (
            "Does the contribution adequately address the question? "
            "0 = incomplete, misses key aspects; 1 = thorough and fully responsive."
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

    The actual LLM call is made by the Agent (via agent.call() with a
    peer-review ContextPacket) — not here. This class only handles
    prompt construction and response parsing.

    Parameters
    ----------
    dimensions : list[dict]
    include_self_assessment : bool
    dry_run : bool
        If True, build_prompt() still works but parse() returns uniform 0.5 scores.
    """

    def __init__(
        self,
        dimensions: list[dict] | None = None,
        include_self_assessment: bool = False,
        dry_run: bool = False,
    ):
        self.dimensions = dimensions or DEFAULT_DIMENSIONS
        self.include_self_assessment = include_self_assessment
        self.dry_run = dry_run

    def build_prompt(
        self,
        reviewer_name: str,
        reviewer_contribution: Any,
        all_contributions: dict[str, Any],
        cycle: int,
        item_context: str = "",
        ecu_info: str = "",
    ) -> str:
        """Build the peer review prompt to be sent to the reviewing agent."""
        review_targets = list(all_contributions.keys()) if self.include_self_assessment \
            else [n for n in all_contributions if n != reviewer_name]

        dim_names = [d["name"] for d in self.dimensions]
        lines: list[str] = []

        lines.append(
            f"You are {reviewer_name}. The contribution round has just ended. "
            "Now complete the peer review table below."
        )
        lines.append(f"Topic: {item_context}")
        lines.append("")

        if ecu_info:
            lines.append(ecu_info)
            lines.append("")

        lines.append("Your contribution this round:")
        contrib_text = (
            reviewer_contribution if isinstance(reviewer_contribution, str)
            else json.dumps(reviewer_contribution, ensure_ascii=False)
        )
        lines.append(f"  {contrib_text}")
        lines.append("")

        lines.append("All contributions this round:")
        for name, contrib in all_contributions.items():
            if name == reviewer_name:
                continue
            text = contrib if isinstance(contrib, str) \
                else json.dumps(contrib, ensure_ascii=False)
            lines.append(f"  [{name}]: {text}")
        lines.append("")

        lines.append(
            "Score each agent listed below on the five quality dimensions "
            "(0.00 to 1.00 each). Also report a coalition agreement score "
            "(0.00 = completely disagree with their position, "
            "1.00 = fully agree) and a one-sentence justification."
        )
        lines.append("")
        lines.append("Scoring rubrics:")
        for d in self.dimensions:
            lines.append(f"  {d['label']} ({d['name']}): {d['rubric']}")
        lines.append("")
        lines.append("Respond ONLY with a JSON object in this exact format:")
        lines.append("{")
        for name in review_targets:
            lines.append(f'  "{name}": {{')
            for dim in dim_names:
                lines.append(f'    "{dim}": <score 0.00-1.00>,')
            lines.append('    "coalition_agreement": <score 0.00-1.00>,')
            lines.append('    "coalition_justification": "<one sentence>"')
            lines.append("  },")
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
        """Parse the agent's raw response into a PeerReviewOutput."""
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
                coalition_justifications={n: "[dry-run]" for n in all_contributions if n != reviewer_name},
                raw_response="[dry-run]",
            )

        scores: dict[str, dict[str, float]] = {}
        self_scores: dict[str, float] | None = None
        coalition: dict[str, float] = {}
        justifications: dict[str, str] = {}

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
                    try:
                        coalition[name] = max(0.0, min(1.0, float(entry.get("coalition_agreement", 0.5))))
                    except (ValueError, TypeError):
                        coalition[name] = 0.5
                    justifications[name] = str(entry.get("coalition_justification", ""))

        except Exception as exc:
            print(f"[PeerReviewRound] Parse error for {reviewer_name}: {exc}")
            for name in review_targets:
                if name != reviewer_name:
                    scores[name] = {d: 0.5 for d in dim_names}
                    coalition[name] = 0.5
                    justifications[name] = ""

        return PeerReviewOutput(
            reviewer_name=reviewer_name,
            cycle=cycle,
            scores=scores,
            self_scores=self_scores,
            coalition_scores=coalition,
            coalition_justifications=justifications,
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
        if self.dry_run:
            return self.parse(reviewer_name, cycle, "[dry-run]", all_contributions)

        prompt = self.build_prompt(
            reviewer_name, reviewer_contribution, all_contributions,
            cycle, item_context, ecu_info,
        )
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=800,
            )
            raw = response.choices[0].message.content or "{}"
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

        # Enumerate all subsets (n is small, typically 3-4)
        for size in range(n, 0, -1):
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
    weights: dict[str, float]
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
        Initial weight vector w.
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
        include_self_assessment: bool = False,
        lambda_self: float = 0.5,
    ):
        self.dimensions = dimensions or DEFAULT_DIMENSIONS
        self.dim_names = [d["name"] for d in self.dimensions]
        self.include_self_assessment = include_self_assessment
        self.lambda_self = lambda_self

        self.weights: dict[str, float] = {d["name"]: 1.0 for d in self.dimensions}
        if weights:
            self.weights.update(weights)

        self._balances: dict[str, float] = {name: 0.0 for name in agent_names}
        self._history: list[TurnRecord] = []

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

        Implements eq. (1) from the simulation section:
            ecu_i = sum_q w_q * mean_peer_score_q(i)

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

        # ECU = weighted sum of aggregated scores
        ecu = sum(agg.get(d, 0.0) * self.weights.get(d, 1.0) for d in self.dim_names)

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
            weights=dict(self.weights),
            ecu_earned=ecu,
            contribution_preview=contrib_str[:120],
        ))

        return ecu

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Update weight vector w. Called by Orchestrator local search."""
        self.weights.update(new_weights)

    @property
    def balances(self) -> dict[str, float]:
        return dict(self._balances)

    @property
    def history(self) -> list[TurnRecord]:
        return list(self._history)

    def balance_for(self, agent_name: str) -> float:
        return self._balances.get(agent_name, 0.0)

    def scores_by_dimension(self) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {d: [] for d in self.dim_names}
        for rec in self._history:
            for dim, score in rec.aggregated_scores.items():
                if dim in result:
                    result[dim].append(score)
        return result

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "balances": {k: round(v, 4) for k, v in self._balances.items()},
            "history": [r.to_dict() for r in self._history],
        }