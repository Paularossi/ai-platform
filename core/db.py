"""
core/db.py

Persistent debate log for the multi-agent lab, backed by Supabase (Postgres).

Six tables: who the student is, the debate itself (config + outcome +
reflection + token total), who was on the panel, every contribution, every
peer review, and every prompt sent (including edits a student made to it in
Manual mode). Unlike the superagent_testing branch, full contribution text and 
peer-review content are stored here, not just summaries.

Auth is a bare student ID. There is no roster yet: any non-empty ID
is accepted and gets its own row in `students` the first time it's seen.
Validating against an actual roster later is a small addition on top of this.

This module is only used by the live Streamlit app (app/pages/5_Run.py,
saving on outcome-save - see _render_outcome_and_downloads).
"""

from __future__ import annotations

import os
from typing import Any

import psycopg2
import psycopg2.extras

# Placeholder for testing - change later.
# Warn-only: nothing in the app blocks a student from running over this.
COURSE_TOKEN_QUOTA = 100_000

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS students (
    student_id  TEXT PRIMARY KEY,
    first_seen  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS debates (
    id                 BIGSERIAL PRIMARY KEY,
    student_id         TEXT NOT NULL REFERENCES students(student_id),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    experiment_name    TEXT,
    author             TEXT,
    topic              TEXT,
    protocol_setting   TEXT,
    run_mode           TEXT,
    visibility_mode    TEXT,
    review_depth       TEXT,
    order_type         TEXT,
    max_cycles         INTEGER,
    stopping_rule      TEXT,
    num_agents         INTEGER,
    has_human_agent    BOOLEAN,
    ecu_enabled        BOOLEAN,
    num_turns          INTEGER,
    converged          BOOLEAN,
    outcome_label      TEXT,
    outcome_notes      TEXT,
    reflection         TEXT,
    total_tokens       INTEGER
);

CREATE TABLE IF NOT EXISTS agents (
    id                 BIGSERIAL PRIMARY KEY,
    debate_id          BIGINT NOT NULL REFERENCES debates(id) ON DELETE CASCADE,
    agent_name         TEXT,
    role               TEXT,
    provider           TEXT,
    model              TEXT,
    temperature        DOUBLE PRECISION,
    is_human           BOOLEAN,
    final_ecu_balance  DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS contributions (
    id            BIGSERIAL PRIMARY KEY,
    debate_id     BIGINT NOT NULL REFERENCES debates(id) ON DELETE CASCADE,
    agent_name    TEXT,
    cycle         INTEGER,
    contribution  TEXT,
    raw_response  TEXT,
    total_tokens  INTEGER,
    ecu_earned    DOUBLE PRECISION,
    timestamp     TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS peer_reviews (
    id                BIGSERIAL PRIMARY KEY,
    debate_id         BIGINT NOT NULL REFERENCES debates(id) ON DELETE CASCADE,
    reviewer_name     TEXT,
    cycle             INTEGER,
    scores            JSONB,
    self_scores       JSONB,
    justifications    JSONB,
    importance_votes  JSONB,
    raw_response      TEXT,
    timestamp         TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS prompts (
    id               BIGSERIAL PRIMARY KEY,
    debate_id        BIGINT NOT NULL REFERENCES debates(id) ON DELETE CASCADE,
    cycle            INTEGER,
    agent_name       TEXT,
    phase            TEXT,
    prompt           TEXT,
    original_prompt  TEXT,
    response         TEXT
);

CREATE INDEX IF NOT EXISTS idx_debates_student_id ON debates(student_id);
CREATE INDEX IF NOT EXISTS idx_debates_created_at ON debates(created_at);
CREATE INDEX IF NOT EXISTS idx_agents_debate_id ON agents(debate_id);
CREATE INDEX IF NOT EXISTS idx_contributions_debate_id ON contributions(debate_id);
CREATE INDEX IF NOT EXISTS idx_peer_reviews_debate_id ON peer_reviews(debate_id);
CREATE INDEX IF NOT EXISTS idx_prompts_debate_id ON prompts(debate_id);
"""


def _get_db_url() -> str | None:
    """Read the connection string from Streamlit secrets, falling back to the environment."""
    try:
        import streamlit as st
        if "SUPABASE_DB_URL_COURSE" in st.secrets:
            return st.secrets["SUPABASE_DB_URL_COURSE"]
    except Exception:
        pass
    return os.environ.get("SUPABASE_DB_URL_COURSE")


def get_connection():
    """Open a new connection to the Supabase Postgres database. Raises if no URL is configured."""
    db_url = _get_db_url()
    if not db_url:
        raise RuntimeError(
            "SUPABASE_DB_URL_COURSE is not set. Add it to .streamlit/secrets.toml locally, "
            "or to the app's Secrets panel on Streamlit Cloud."
        )
    return psycopg2.connect(db_url)


def ensure_schema() -> None:
    """Create every table (and index) if it doesn't already exist. Safe to call on every app start."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLES)
        conn.commit()


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------

def upsert_student(student_id: str) -> None:
    """Record a student's first/last-seen time. Called once per session, on the Welcome page."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO students (student_id) VALUES (%s)
                ON CONFLICT (student_id) DO UPDATE SET last_seen = now()
                """,
                (student_id,),
            )
        conn.commit()


def get_total_token_usage(student_id: str) -> int:
    """
    All-time sum of total_tokens across this student's debates - what the
    course-wide quota (COURSE_TOKEN_QUOTA) is checked against. 0 for a
    student with no debates yet (including one not in `students` at all).
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(SUM(total_tokens), 0) FROM debates WHERE student_id = %s",
                (student_id,),
            )
            return cur.fetchone()[0]


def update_debate_fields(debate_id: int, **fields: Any) -> None:
    """
    Update one or more columns on an already-saved debate row - used when a
    student re-saves the outcome, or saves/edits the reflection after the
    debate row already exists, so those don't insert a duplicate row.
    Pass column=value kwargs, e.g. update_debate_fields(id, reflection="...").
    """
    if not fields:
        return
    set_clause = ", ".join(f"{col} = %s" for col in fields)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE debates SET {set_clause} WHERE id = %s",
                (*fields.values(), debate_id),
            )
        conn.commit()


# ---------------------------------------------------------------------------
# Saving a completed debate
# ---------------------------------------------------------------------------

def save_debate(result: dict[str, Any], cfg: dict[str, Any], student_id: str) -> int:
    """
    Insert one completed debate: the debate row (config + outcome +
    reflection + token total), its panel, every contribution, every peer
    review, and every prompt sent. Returns the new debate id.

    `result` is the dict from core.runner.collect_result(), with "outcome"
    and "reflection" added by app/pages/5_Run.py once the student saves
    them (see _render_outcome_and_downloads - saving the outcome is what
    triggers this call).
    """
    upsert_student(student_id)

    overview = cfg.get("overview", {})
    protocol = cfg.get("protocol", {})
    ecu_cfg = cfg.get("ecu", {})
    agent_cfgs = cfg.get("agents", [])
    outcome = result.get("outcome") or {}

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO debates
                    (student_id, experiment_name, author, topic,
                     protocol_setting, run_mode, visibility_mode, review_depth,
                     order_type, max_cycles, stopping_rule, num_agents,
                     has_human_agent, ecu_enabled, num_turns, converged,
                     outcome_label, outcome_notes, reflection, total_tokens)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    student_id,
                    overview.get("name"),
                    overview.get("author"),
                    cfg.get("task", {}).get("description"),
                    protocol.get("setting"),
                    protocol.get("run_mode"),
                    protocol.get("visibility_mode"),
                    protocol.get("review_depth"),
                    protocol.get("order_type"),
                    protocol.get("max_cycles"),
                    protocol.get("stopping_rule"),
                    len(agent_cfgs),
                    any(a.get("provider") == "Human" for a in agent_cfgs),
                    ecu_cfg.get("enabled", False),
                    result.get("num_turns"),
                    result.get("converged"),
                    outcome.get("label"),
                    outcome.get("notes"),
                    result.get("reflection"),
                    result.get("total_tokens"),
                ),
            )
            debate_id = cur.fetchone()[0]

            final_balances = result.get("ecu_balances", {})
            agent_rows = [
                (
                    debate_id, a.get("name"), a.get("role"), a.get("provider"), a.get("model"),
                    a.get("temperature"), a.get("provider") == "Human",
                    final_balances.get(a.get("name")),
                )
                for a in agent_cfgs
            ]
            if agent_rows:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO agents
                        (debate_id, agent_name, role, provider, model,
                         temperature, is_human, final_ecu_balance)
                    VALUES %s
                    """,
                    agent_rows,
                )

            contribution_rows = [
                (
                    debate_id, o.get("agent_name"), o.get("cycle"), o.get("contribution"),
                    o.get("raw_response"), o.get("total_tokens"), o.get("ecu_earned"),
                    o.get("timestamp"),
                )
                for o in result.get("log", [])
            ]
            if contribution_rows:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO contributions
                        (debate_id, agent_name, cycle, contribution,
                         raw_response, total_tokens, ecu_earned, timestamp)
                    VALUES %s
                    """,
                    contribution_rows,
                )

            review_rows = [
                (
                    debate_id, r.get("reviewer_name"), r.get("cycle"),
                    psycopg2.extras.Json(r.get("scores", {})),
                    psycopg2.extras.Json(r.get("self_scores")) if r.get("self_scores") is not None else None,
                    psycopg2.extras.Json(r.get("justifications", {})),
                    psycopg2.extras.Json(r.get("importance_votes", {})),
                    r.get("raw_response"), r.get("timestamp"),
                )
                for r in result.get("peer_review_log", [])
            ]
            if review_rows:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO peer_reviews
                        (debate_id, reviewer_name, cycle, scores, self_scores,
                         justifications, importance_votes, raw_response, timestamp)
                    VALUES %s
                    """,
                    review_rows,
                )

            prompt_rows = [
                (
                    debate_id, p.get("cycle"), p.get("agent"), p.get("phase"),
                    p.get("prompt"), p.get("original_prompt"), p.get("response"),
                )
                for p in result.get("prompt_log", [])
            ]
            if prompt_rows:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO prompts
                        (debate_id, cycle, agent_name, phase, prompt, original_prompt, response)
                    VALUES %s
                    """,
                    prompt_rows,
                )
        conn.commit()

    return debate_id


# ---------------------------------------------------------------------------
# Reading back (tutor view / History page)
# ---------------------------------------------------------------------------

def list_debates(student_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """
    Return the most recent debates, newest first. Pass student_id to scope
    to one student (the History page); omit it for a tutor-wide overview.
    """
    where = "WHERE student_id = %s" if student_id else ""
    params: tuple = (student_id, limit) if student_id else (limit,)
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT id, student_id, created_at, experiment_name, author, topic,
                       protocol_setting, run_mode, num_agents, has_human_agent,
                       num_turns, converged, outcome_label, total_tokens
                FROM debates
                {where}
                ORDER BY created_at DESC
                LIMIT %s
                """,
                params,
            )
            return list(cur.fetchall())


def get_debate(debate_id: int) -> dict[str, Any] | None:
    """Return one debate's full row (config, outcome, reflection), or None if not found."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM debates WHERE id = %s", (debate_id,))
            row = cur.fetchone()
            return dict(row) if row else None
