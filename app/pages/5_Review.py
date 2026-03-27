"""Review Page - full experiment summary + save draft"""

import json
from datetime import datetime

import streamlit as st
from components.utils import restore_draft

st.set_page_config(page_title="Review Experiment", page_icon="🧠", layout="wide")

# ---------- sidebar ----------
st.sidebar.title("Experiment Builder")
st.sidebar.caption("Step 5 of 5")
st.sidebar.progress(5 / 5)
st.sidebar.markdown(
    """
**Steps**
1. Task
2. Agents
3. Instructions
4. Dataset
5. Review
"""
)

st.title("Review & Save")
st.caption("Check your full experiment setup before saving or launching.")


# ---------- helpers ----------

def val(v, fallback="-"):
    """Return v if truthy, else fallback."""
    if v is None:
        return fallback
    if isinstance(v, list):
        return ", ".join(str(x) for x in v) if v else fallback
    if isinstance(v, str):
        return v.strip() or fallback
    return v


def check(label: str, ok: bool):
    """Render a green tick or red cross next to a readiness check."""
    icon = "✅" if ok else "❌"
    st.markdown(f"{icon} {label}")


def build_draft() -> dict:
    """
    Assemble the full experiment config from session state.
    Merges everything written by all 5 pages into one clean dict.
    The dataset binary (images) is intentionally excluded - only metadata is saved.
    """
    cfg = st.session_state.get("experiment_config", {})

    draft = {
        "meta": {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "app_version": "0.1",
        },
        "overview": cfg.get("overview", {
            "name": st.session_state.get("exp_name", ""),
            "author": st.session_state.get("author", ""),
            "protocol_id": st.session_state.get("protocol_id", ""),
        }),
        "task": cfg.get("task", {
            "category": st.session_state.get("task_category", ""),
            "modalities": st.session_state.get("modalities", []),
            "description": st.session_state.get("task_description", ""),
        }),
        "schemas": cfg.get("schemas", {
            "input_fields": st.session_state.get("input_fields", []),
            "output_fields": st.session_state.get("output_fields", []),
        }),
        "questions": cfg.get("questions", st.session_state.get("question_sets", [])),
        "instructions": cfg.get("instructions", {
            "base_instructions": st.session_state.get("base_instructions", ""),
            "guideline_notes": st.session_state.get("guideline_notes", ""),
        }),
        "agent_prompt_overrides": cfg.get(
            "agent_prompt_overrides",
            st.session_state.get("agent_prompt_overrides", {})
        ),
        "agents": cfg.get("agents", st.session_state.get("agents", [])),
        "protocol": cfg.get("protocol", {}),
        "dataset": {
            k: v for k, v in cfg.get("dataset", {}).items()
            # exclude binary data - only keep metadata
            if k not in ("image_bytes", "raw_bytes")
        },
    }
    return draft


def readiness_checks(draft: dict) -> tuple[list[str], list[str]]:
    """Return (warnings, errors) based on the draft content."""
    warnings, errors = [], []

    if not draft["overview"].get("name"):
        warnings.append("Experiment name is not set.")
    if not draft["task"].get("description"):
        warnings.append("Task description is empty.")
    if not draft["schemas"].get("input_fields"):
        errors.append("No input fields defined (Step 1).")
    if not draft["schemas"].get("output_fields"):
        errors.append("No output fields defined (Step 1).")
    if not draft["agents"]:
        errors.append("No agents configured (Step 2).")
    if not draft["questions"]:
        warnings.append("No questions loaded (Step 3). Agents will have no structured output to fill in.")
    if not draft["instructions"].get("base_instructions"):
        warnings.append("Base instructions are empty (Step 3).")
    if not draft["dataset"].get("filename"):
        warnings.append("No dataset uploaded yet (Step 4).")

    return warnings, errors


# ---------- load state ----------
cfg = st.session_state.get("experiment_config", {})
draft = build_draft()
warnings, errors = readiness_checks(draft)

# ---------- layout ----------
main_col, action_col = st.columns([2.2, 1], gap="large")

with main_col:

    # ── Readiness banner ──────────────────────────────────────────────────────
    if errors:
        st.error(
            f"**{len(errors)} issue(s) require attention before launching:**\n"
            + "\n".join(f"- {e}" for e in errors)
        )
    elif warnings:
        st.warning(
            f"**{len(warnings)} warning(s):** some fields are incomplete.\n"
            + "\n".join(f"- {w}" for w in warnings)
        )
    else:
        st.success("✅ All required fields are filled in. Ready to launch.")

    # ── 1. Overview ───────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("1. Overview")
        ov = draft["overview"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Experiment name", val(ov.get("name")))
        c2.metric("Author", val(ov.get("author")))
        c3.metric("Protocol version", val(ov.get("protocol_id")))

    # ── 2. Task ───────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("2. Task")
        task = draft["task"]
        c1, c2 = st.columns(2)
        c1.markdown(f"**Category**  \n{val(task.get('category'))}")
        c2.markdown(f"**Modalities**  \n{val(task.get('modalities'))}")
        st.markdown(f"**Description**  \n{val(task.get('description'))}")

        schemas = draft["schemas"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Input fields**")
            for f in schemas.get("input_fields", []):
                st.markdown(f"- `{f['name']}` ({f['type']})")
            if not schemas.get("input_fields"):
                st.markdown("_None defined_")
        with c2:
            st.markdown("**Output fields**")
            for f in schemas.get("output_fields", []):
                st.markdown(f"- `{f['name']}` ({f['type']})")
            if not schemas.get("output_fields"):
                st.markdown("_None defined_")

    # ── 3. Protocol & agents ──────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("3. Protocol & agents")
        proto = draft["protocol"]

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Setting**  \n{val(proto.get('setting'))}")
        c2.markdown(f"**Platform mode**  \n{val(proto.get('platform_mode'))}")
        c3.markdown(f"**Supervision**  \n{val(proto.get('supervision_mode'))}")

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Visibility**  \n{val(proto.get('visibility_mode'))}")
        c2.markdown(f"**Order**  \n{val(proto.get('order_type'))}")
        c3.markdown(f"**Stopping rule**  \n{val(proto.get('stopping_rule'))} (max {val(proto.get('max_cycles'))} cycles)")

        if proto.get("initializer_agent"):
            st.markdown(f"**Initializer agent**  \n{proto['initializer_agent']}")
        if proto.get("judge_agent") and proto.get("setting") == "Court (judge-based)":
            st.markdown(f"**Judge agent**  \n{proto['judge_agent']}")

        agents = draft["agents"]
        if agents:
            st.divider()
            st.markdown(f"**{len(agents)} agent(s)**")
            cols = st.columns(min(len(agents), 3))
            for i, agent in enumerate(agents):
                with cols[i % 3]:
                    with st.container(border=True):
                        st.markdown(f"**{agent.get('name', f'Agent {i+1}')}**")
                        st.caption(agent.get("role", "-"))
                        st.markdown(f"`{agent.get('provider', '?')}` / `{agent.get('model', '?')}`")
                        override = draft["agent_prompt_overrides"].get(agent.get("name", ""), "")
                        if override.strip():
                            st.caption("✏️ Has prompt override")
        else:
            st.markdown("_No agents configured._")

    # ── 4. Instructions & questions ───────────────────────────────────────────
    with st.container(border=True):
        st.subheader("4. Instructions & questions")

        instructions = draft["instructions"]
        base = instructions.get("base_instructions", "").strip()
        notes = instructions.get("guideline_notes", "").strip()
        questions = draft["questions"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Base instructions", "✅ Set" if base else "-")
        c2.metric("Guideline notes", "✅ Set" if notes else "-")
        c3.metric("Questions loaded", len(questions))

        if questions:
            st.divider()
            for q in questions:
                field_type = q.get("field_type", "")
                badge = {
                    "single_label": "🔘",
                    "multi_label": "☑️",
                    "boolean": "✅",
                    "score": "📊",
                    "text": "📝",
                    "ranking": "🔢",
                }.get(field_type, "•")
                n_opts = len(q.get("options", []))
                st.markdown(
                    f"{badge} `{q.get('field_name', '?')}` - "
                    f"{field_type.replace('_', '-')} - "
                    f"{n_opts} option(s)"
                )

    # ── 5. Dataset ────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("5. Dataset")
        ds = draft["dataset"]

        if not ds.get("filename"):
            st.markdown("_No dataset uploaded._")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("File", ds.get("filename", "-"))
            if ds.get("num_images"):
                c2.metric("Images", ds["num_images"])
            if ds.get("num_rows"):
                c3.metric("Metadata rows", ds["num_rows"])

            mapping = ds.get("column_mapping", {})
            active_mapping = {k: v for k, v in mapping.items() if v != "- not mapped -"}
            if active_mapping:
                st.divider()
                st.markdown("**Column mapping**")
                for field, col in active_mapping.items():
                    st.markdown(f"- `{field}` ← `{col}`")

    # ── Navigation ─────────────────────────────────────────────────────────────
    st.divider()
    nav1, _, nav3 = st.columns([2, 2, 2])
    with nav1:
        if st.button("← Back"):
            st.switch_page("pages/4_Dataset.py")
    with nav3:
        launch_disabled = bool(errors)
        if st.button(
            "🚀 Launch experiment",
            type="primary",
            disabled=launch_disabled,
            help="Fix the errors above before launching." if launch_disabled else "",
        ):
            #st.info("Experiment runner coming soon!", icon="🔧")
            st.switch_page("pages/6_Run.py")


# ---------- action column ─────────────────────────────────────────────────────
with action_col:

    # ── Readiness checklist ───────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Readiness")
        check("Experiment name set", bool(draft["overview"].get("name")))
        check("Input fields defined", bool(draft["schemas"].get("input_fields")))
        check("Output fields defined", bool(draft["schemas"].get("output_fields")))
        check("Agents configured", bool(draft["agents"]))
        check("Protocol configured", bool(draft["protocol"]))
        check("Instructions set", bool(draft["instructions"].get("base_instructions", "").strip()))
        check("Questions loaded", bool(draft["questions"]))
        check("Dataset uploaded", bool(draft["dataset"].get("filename")))

    # ── Save draft ────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Save draft")
        st.caption(
            "Downloads the full experiment configuration as a JSON file. "
            "Dataset images are not included - only metadata."
        )

        exp_name = draft["overview"].get("name", "experiment").strip() or "experiment"
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
        default_filename = f"{exp_name.lower().replace(' ', '_')}_{timestamp}.json"

        filename_input = st.text_input(
            "Filename",
            value=default_filename,
            key="draft_filename",
        )
        if not filename_input.endswith(".json"):
            filename_input += ".json"

        draft_json = json.dumps(draft, indent=2, ensure_ascii=False)

        st.download_button(
            label="⬇ Download draft JSON",
            data=draft_json,
            file_name=filename_input,
            mime="application/json",
            type="primary",
            width='stretch',
        )

        st.caption(f"~{len(draft_json) // 1024 + 1} KB")

    # ── Load draft ────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Load draft")
        st.caption("Upload a previously saved draft JSON to restore the experiment configuration.")

        uploaded_draft = st.file_uploader(
            "Upload draft JSON",
            type=["json"],
            key="draft_uploader",
        )

        if uploaded_draft is not None:
            if st.button("Restore from draft", type="secondary"):
                try:
                    loaded = json.load(uploaded_draft)
                    restore_draft(loaded)
                    st.success("Draft restored. You can now navigate to any step to review or edit.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load draft: {e}")
