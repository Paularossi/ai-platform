"""
core/db.py

Persistent debate log for Agent 0 mode, backed by Supabase (Postgres).

Debates, rounds, dimensions and roster rows describe each experiment.
llm_calls stores prompts, answer text, failures, effective settings and token
usage for every request, including Agent 0 initialization and final briefs.
Thinking text and the entire result JSON are not duplicated in the database.

This module is only used by the live Streamlit app (see app/pages/5_Run.py
and app/pages/6_History.py). The batch runner (experiments/run_experiment.py)
is untouched - it keeps writing local JSON files as before.
"""

from __future__ import annotations

from core.llm.providers import uses_default_temperature

import os
from contextlib import contextmanager
from typing import Any

import psycopg2
import psycopg2.extras

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS debates (
    id                  BIGSERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    topic               TEXT NOT NULL,
    az_provider         TEXT,
    az_model            TEXT,
    az_temperature      DOUBLE PRECISION,
    max_rounds          INTEGER,
    max_agents          INTEGER,
    max_total_spawns    INTEGER,
    num_rounds          INTEGER,
    num_turns           INTEGER,
    ended_reason        TEXT,
    final_brief         TEXT,
    status              TEXT NOT NULL DEFAULT 'completed'
);

CREATE TABLE IF NOT EXISTS rounds (
    id                    BIGSERIAL PRIMARY KEY,
    debate_id             BIGINT NOT NULL REFERENCES debates(id) ON DELETE CASCADE,
    cycle                 INTEGER NOT NULL,
    social_welfare        DOUBLE PRECISION,
    end_debate            BOOLEAN,
    agent_zero_reasoning  TEXT
);

CREATE TABLE IF NOT EXISTS ecu_dimensions (
    id                 BIGSERIAL PRIMARY KEY,
    debate_id          BIGINT NOT NULL REFERENCES debates(id) ON DELETE CASCADE,
    dimension_name     TEXT,
    label              TEXT,
    rubric             TEXT,
    initial_ecu_weight DOUBLE PRECISION,  -- set by Agent 0 at initialization
    final_ecu_weight   DOUBLE PRECISION,  -- after the Orchestrator's vote-based updates
    sw_weight          DOUBLE PRECISION   -- fixed for the whole debate (never optimised)
);

CREATE TABLE IF NOT EXISTS agents (
    id            BIGSERIAL PRIMARY KEY,
    debate_id     BIGINT NOT NULL REFERENCES debates(id) ON DELETE CASCADE,
    agent_name    TEXT,
    role          TEXT,
    provider      TEXT,
    model         TEXT,
    added_cycle   INTEGER,          -- null = present in the starting panel
    removed_cycle INTEGER,          -- null = never removed
    final_balance DOUBLE PRECISION  -- final cumulative ECU balance
);

CREATE INDEX IF NOT EXISTS idx_rounds_debate_id ON rounds(debate_id);
CREATE INDEX IF NOT EXISTS idx_ecu_dimensions_debate_id ON ecu_dimensions(debate_id);
CREATE INDEX IF NOT EXISTS idx_agents_debate_id ON agents(debate_id);
"""

# Additive schema changes for existing databases.
# Using IF NOT EXISTS / IF EXISTS everywhere (rather than assuming a clean
# database) means this file stays the single source of truth whether a
# project is brand new or already has an older version of these tables.
_ALTER_DEBATES = """
ALTER TABLE debates
    ADD COLUMN IF NOT EXISTS initial_roster_size  INTEGER,
    ADD COLUMN IF NOT EXISTS final_roster_size     INTEGER,
    ADD COLUMN IF NOT EXISTS visibility_mode        TEXT,
    ADD COLUMN IF NOT EXISTS review_depth           TEXT,
    ADD COLUMN IF NOT EXISTS info_condition         TEXT,
    ADD COLUMN IF NOT EXISTS coalition_threshold    DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS final_social_welfare   DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS final_brief            TEXT,
    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'completed';
"""

_ALTER_ROUNDS = """
ALTER TABLE rounds
    ADD COLUMN IF NOT EXISTS roster_size              INTEGER,
    ADD COLUMN IF NOT EXISTS coalition                 TEXT[],
    ADD COLUMN IF NOT EXISTS agents_added              TEXT[],
    ADD COLUMN IF NOT EXISTS agents_removed             TEXT[],
    ADD COLUMN IF NOT EXISTS instructions_issued_count INTEGER;
"""

_ALTER_AGENTS = """
ALTER TABLE agents
    ADD COLUMN IF NOT EXISTS final_balance DOUBLE PRECISION;
"""

_CREATE_CALLS = """
CREATE TABLE IF NOT EXISTS llm_calls (
    call_id UUID PRIMARY KEY,
    debate_id BIGINT NOT NULL REFERENCES debates(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL,
    cycle INTEGER,
    agent_name TEXT NOT NULL,
    phase TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    error TEXT,
    stop_reason TEXT,
    duration_ms INTEGER,
    parameters JSONB NOT NULL DEFAULT '{}',
    usage JSONB NOT NULL DEFAULT '{}',
    diagnostics JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_debate ON llm_calls(debate_id, created_at);
"""


def _get_db_url() -> str | None:
    """Read the connection string from Streamlit secrets, falling back to the environment."""
    try:
        import streamlit as st
        if "SUPABASE_DB_URL" in st.secrets:
            return st.secrets["SUPABASE_DB_URL"]
    except Exception:
        pass
    return os.environ.get("SUPABASE_DB_URL")


@contextmanager
def get_connection():
    """Open a new connection to the Supabase Postgres database. Raises if no URL is configured."""
    db_url = _get_db_url()
    if not db_url:
        raise RuntimeError(
            "SUPABASE_DB_URL is not set. Add it to .streamlit/secrets.toml locally, "
            "or to the app's Secrets panel on Streamlit Cloud."
        )
    conn = psycopg2.connect(db_url, connect_timeout=10)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def ensure_schema() -> None:
    """Create the tables if needed and add any columns an existing database is missing."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLES)
            cur.execute(_ALTER_DEBATES)
            cur.execute(_ALTER_ROUNDS)
            cur.execute(_ALTER_AGENTS)
            cur.execute(_CREATE_CALLS)
        conn.commit()


def _build_agent_registry(design: dict[str, Any], rounds: list[dict[str, Any]]) -> dict[str, dict]:
    """
    Reconstruct every agent that ever appeared in the debate, with when they
    joined and left, from the starting roster plus each round's roster
    snapshot and agents_added/agents_removed lists. No new data is captured
    here - this just structures what's already in the result.
    """
    registry: dict[str, dict] = {}

    for a in design.get("agents", []):
        registry[a["name"]] = {
            "role": a.get("role"), "provider": a.get("provider"), "model": a.get("model"),
            "added_cycle": None, "removed_cycle": None,
        }

    for r in rounds:
        cycle = r["cycle"]
        roster_by_name = {a["name"]: a for a in r.get("roster", [])}
        added_specs = {a["name"]: a for a in r.get("agents_added_specs", [])}
        for name in r.get("agents_added", []):
            spec = added_specs.get(name) or roster_by_name.get(name, {})
            registry[name] = {
                "role": spec.get("role"), "provider": spec.get("provider"), "model": spec.get("model"),
                "added_cycle": cycle, "removed_cycle": None,
            }
        for name in r.get("agents_removed", []):
            if name in registry:
                registry[name]["removed_cycle"] = cycle

    return registry


def save_experiment(result: dict[str, Any], cfg: dict[str, Any], debate_id: int | None = None) -> int:
    """
    Insert one Agent 0 debate result: the debate row (including its final
    brief), its panel (with final ECU balance), its ECU dimensions, and its
    per-round rows. Returns the new debate id.
    """
    az_cfg = cfg.get("agent_zero", {})
    topic = cfg.get("task", {}).get("description", "")
    design = result.get("agent_zero_design", {})
    rounds = result.get("rounds", [])

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO debates
                    (id, topic, az_provider, az_model, az_temperature,
                     max_rounds, max_agents, max_total_spawns,
                     num_rounds, num_turns, ended_reason, final_brief,
                     initial_roster_size, final_roster_size, final_social_welfare,
                     visibility_mode, review_depth, info_condition, coalition_threshold)
                VALUES (COALESCE(%s, nextval(pg_get_serial_sequence('debates', 'id'))),
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    topic = EXCLUDED.topic,
                    az_provider = EXCLUDED.az_provider,
                    az_model = EXCLUDED.az_model,
                    az_temperature = EXCLUDED.az_temperature,
                    max_rounds = EXCLUDED.max_rounds,
                    max_agents = EXCLUDED.max_agents,
                    max_total_spawns = EXCLUDED.max_total_spawns,
                    num_rounds = EXCLUDED.num_rounds,
                    num_turns = EXCLUDED.num_turns,
                    ended_reason = EXCLUDED.ended_reason,
                    final_brief = EXCLUDED.final_brief,
                    initial_roster_size = EXCLUDED.initial_roster_size,
                    final_roster_size = EXCLUDED.final_roster_size,
                    final_social_welfare = EXCLUDED.final_social_welfare,
                    visibility_mode = EXCLUDED.visibility_mode,
                    review_depth = EXCLUDED.review_depth,
                    info_condition = EXCLUDED.info_condition,
                    coalition_threshold = EXCLUDED.coalition_threshold,
                    status = 'completed'
                RETURNING id
                """,
                (
                    debate_id, topic,
                    az_cfg.get("provider"),
                    az_cfg.get("model"),
                    (None if uses_default_temperature(az_cfg.get("provider", ""), az_cfg.get("model", ""))
                     else az_cfg.get("temperature")),
                    az_cfg.get("max_rounds"),
                    az_cfg.get("max_agents"),
                    az_cfg.get("max_total_spawns"),
                    len(rounds),
                    result.get("num_turns"),
                    result.get("ended_reason"),
                    result.get("final_brief"),
                    len(result.get("initial_roster", [])),
                    len(result["final_roster"]) if "final_roster" in result else (len(rounds[-1]["roster"]) if rounds else None),
                    rounds[-1]["social_welfare"] if rounds else None,
                    design.get("visibility_mode"),
                    design.get("review_depth"),
                    design.get("info_condition"),
                    design.get("coalition_threshold"),
                ),
            )
            debate_id = cur.fetchone()[0]
            for table in ("ecu_dimensions", "agents", "rounds"):
                cur.execute(f"DELETE FROM {table} WHERE debate_id = %s", (debate_id,))

            final_weights = result.get("final_ecu_weights", {})
            dim_rows = [
                (
                    debate_id,
                    d.get("name"),
                    d.get("label"),
                    d.get("rubric"),
                    d.get("weight"),
                    final_weights.get(d.get("name")),
                    d.get("sw_weight"),
                )
                for d in design.get("ecu_dimensions", [])
            ]
            if dim_rows:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO ecu_dimensions
                        (debate_id, dimension_name, label, rubric,
                         initial_ecu_weight, final_ecu_weight, sw_weight)
                    VALUES %s
                    """,
                    dim_rows,
                )

            agent_registry = _build_agent_registry(design, rounds)
            final_balances = result.get("final_ecu_balances", {})
            agent_rows = [
                (debate_id, name, info["role"], info["provider"], info["model"],
                 info["added_cycle"], info["removed_cycle"], final_balances.get(name))
                for name, info in agent_registry.items()
            ]
            if agent_rows:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO agents
                        (debate_id, agent_name, role, provider, model,
                         added_cycle, removed_cycle, final_balance)
                    VALUES %s
                    """,
                    agent_rows,
                )

            round_rows = [
                (
                    debate_id,
                    r["cycle"],
                    r.get("social_welfare"),
                    r.get("end_debate", False),
                    r.get("agent_zero_reasoning", ""),
                    len(r.get("roster", [])),
                    r.get("coalition", []),
                    r.get("agents_added", []),
                    r.get("agents_removed", []),
                    len(r.get("agent_instructions_issued", {})),
                )
                for r in rounds
            ]
            if round_rows:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO rounds
                        (debate_id, cycle, social_welfare, end_debate, agent_zero_reasoning,
                         roster_size, coalition, agents_added, agents_removed,
                         instructions_issued_count)
                    VALUES %s
                    """,
                    round_rows,
                )
            for call in result.get("debug", {}).get("prompt_log", []):
                if call.get("call_id"):
                    _insert_call(cur, debate_id, call)
        conn.commit()

    return debate_id


def list_debates(limit: int = 50) -> list[dict[str, Any]]:
    """
    Return the most recent debates, newest first, without the (potentially
    long) final_brief text - use get_debate_brief(id) to fetch one on demand.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, created_at, topic, az_provider, az_model, az_temperature, status,
                       max_rounds, max_agents, num_rounds, num_turns, ended_reason,
                       initial_roster_size, final_roster_size, final_social_welfare,
                       visibility_mode, review_depth, info_condition, coalition_threshold
                FROM debates
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return list(cur.fetchall())


def get_debate_brief(debate_id: int) -> str | None:
    """Return the stored final brief for one debate, or None if not found."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT final_brief FROM debates WHERE id = %s", (debate_id,))
            row = cur.fetchone()
            return row[0] if row else None


def begin_experiment(cfg):
    """Create the parent row before any API calls, retaining interrupted runs."""
    az = cfg.get("agent_zero", {})
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO debates
                (topic, az_provider, az_model, max_rounds, max_agents, max_total_spawns, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'running') RETURNING id""",
                (cfg.get("task", {}).get("description", ""), az.get("provider"), az.get("model"),
                 az.get("max_rounds"), az.get("max_agents"), az.get("max_total_spawns")))
            return cur.fetchone()[0]


def _insert_call(cur, debate_id, call):
    diagnostics = {k: call[k] for k in ("request_id", "response_id", "block_types", "web_search") if call.get(k) is not None}
    cur.execute("""INSERT INTO llm_calls
        (call_id, debate_id, created_at, cycle, agent_name, phase, provider, model,
         system_prompt, prompt, response, status, error, stop_reason, duration_ms, parameters, usage, diagnostics)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (call_id) DO UPDATE SET status=EXCLUDED.status, error=EXCLUDED.error""",
        (call["call_id"], debate_id, call["created_at"], call.get("cycle"), call.get("agent", "Unknown"),
         call.get("phase", "unknown"), call["provider"], call["model"], call.get("system_prompt", ""),
         call.get("prompt", ""), call.get("response", ""), call["status"], call.get("error"),
         call.get("stop_reason"), call.get("duration_ms"), psycopg2.extras.Json(call.get("parameters") or {}),
         psycopg2.extras.Json(call.get("usage") or {}), psycopg2.extras.Json(diagnostics)))


def save_call(debate_id, call):
    with get_connection() as conn:
        with conn.cursor() as cur:
            _insert_call(cur, debate_id, call)


def mark_experiment_failed(debate_id):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE debates SET status = 'failed' WHERE id = %s", (debate_id,))


def get_debate_calls(debate_id):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM llm_calls WHERE debate_id = %s ORDER BY created_at", (debate_id,))
            return [dict(row) for row in cur.fetchall()]
