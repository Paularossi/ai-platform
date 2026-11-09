"""Setup page for Agent 0 mode.

Agent 0 designs the debate: the panel of agents, the instructions they get,
the criteria they are scored on, and the visibility rules. The person using
this page only supplies the topic, picks which model runs Agent 0, and sets
hard limits on how large or long the debate can run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from core.llm.providers import API_KEY_ENV_VARS, PROVIDER_MODELS, PROVIDERS, uses_default_temperature, reset_provider


def init_state() -> None:
    defaults = {
        "az_topic": "",
        "az_brief_instructions": "",
        "az_provider": "Anthropic",
        "az_model": "claude-sonnet-5-5",
        "az_temperature": 0.0,
        "az_max_rounds": 10,
        "az_review_checks": True,
        "az_web_search": False,
        "az_setup_search": False,
        "az_max_agents": 6,
        "az_max_total_spawns": 8,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

st.title("New Debate")
st.caption(
    "Enter a topic. Agent 0 assembles the panel, sets the ground rules, and runs the "
    "debate to a policy brief."
)

st.divider()

with st.container(border=True):
    st.subheader("Topic")
    st.session_state.az_topic = st.text_area(
        "Topic",
        value=st.session_state.az_topic,
        placeholder="Should the city introduce congestion charges on its inner ring road?",
        height=100,
        label_visibility="collapsed",
        key="widget_topic",
    )

with st.container(border=True):
    st.subheader("Final brief instructions (optional)")
    st.caption(
        "How Agent 0 should structure, format, or shape the brief it writes once the debate ends "
        " - e.g. section order, length, language (only Agent 0 sees this). "
    )
    st.session_state.az_brief_instructions = st.text_area(
        "Final brief instructions",
        value=st.session_state.az_brief_instructions,
        placeholder="e.g. Open with a two-sentence introduction, then a dedicated section on X, then a conclusion.",
        height=80,
        label_visibility="collapsed",
        key="widget_brief_instructions",
    )

@st.cache_data(show_spinner=False)
def _extract_upload(name, data):
    from core.documents import extract_documents
    return extract_documents(name, data)


shared_documents = []
documents_ready = True
with st.container(border=True):
    st.subheader("Shared documents (optional)")
    st.caption("Selected text is supplied to Agent 0, every panel member and peer reviewer on each call. "
               "It is included in downloaded logs and transcripts and sent to the panel's providers. "
               "PDFs use extracted text only, without images or OCR. ZIP files are read locally, not uploaded to model APIs.")
    uploads = st.file_uploader("Upload source documents", type=["md", "txt", "tex", "bib", "py", "json", "csv", "yaml", "yml", "pdf", "zip"],
                               accept_multiple_files=True, key="shared_document_uploads")
    from core.documents import document_context, MAX_BYTES
    candidates = []
    if sum(upload.size for upload in uploads) > MAX_BYTES:
        st.error("Uploads exceed 20 MB in total. Remove some files.")
        documents_ready = False
    else:
        for upload in uploads:
            try:
                records, notices = _extract_upload(upload.name, upload.getvalue())
                candidates.extend(records)
                if notices:
                    with st.expander(f"Extraction notices: {upload.name}"):
                        st.text("\n".join(notices))
            except (ValueError, ImportError) as exc:
                st.error(str(exc))
                documents_ready = False
    if candidates:
        selected = st.multiselect("Sources to share", range(len(candidates)), default=list(range(len(candidates))),
                                  format_func=lambda i: candidates[i]["name"])
        shared_documents = [candidates[i] for i in selected]
        try:
            source_context = document_context(shared_documents)
            st.caption(f"{len(shared_documents)} sources, {len(source_context):,} characters. "
                       "Text is sent again with each relevant call and adds input-token costs. "
                       "The 100,000-character package limit does not guarantee room for a long debate.")
        except ValueError as exc:
            st.error(str(exc))
            documents_ready = False
        with st.expander("Preview selected extracted text"):
            for document in shared_documents:
                st.text(document["name"])
                st.code(document["text"], language=None)


with st.container(border=True):
    st.subheader("Moderator")
    st.caption(
        "Agent 0 runs the debate: it picks who takes part, decides what they are scored on, "
        "and calls the debate when it's done. Panel members can use OpenAI, Anthropic, and Google models. "
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.az_provider = st.selectbox(
            "Provider", PROVIDERS,
            index=PROVIDERS.index(st.session_state.az_provider),
            key="widget_provider",
        )
    with c2:
        model_options = PROVIDER_MODELS.get(st.session_state.az_provider, [])
        current_model = st.session_state.az_model if st.session_state.az_model in model_options else model_options[0]
        st.session_state.az_model = st.selectbox(
            "Model", model_options,
            index=model_options.index(current_model),
            key="widget_model",
        )
    with c3:
        st.session_state.az_temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0,
            value=st.session_state.get("az_temperature", 0.0), step=0.01,
            help="0 keeps Agent 0's decisions consistent. Higher values add variety. "
                 "A few models don't support this and will run at their own default.",
            key="widget_temperature",
            disabled=uses_default_temperature(st.session_state.az_provider, st.session_state.az_model),
        )

        if uses_default_temperature(st.session_state.az_provider, st.session_state.az_model):
            st.caption("This model uses its default temperature.")

st.session_state.az_review_checks = st.checkbox(
    "Include hallucination and risk peer assessments", value=st.session_state.az_review_checks,
    help="Collect suspected factual issues and risk-reasoning quality separately from ECU scores. These are peer judgments, not verified facts.",
    key="widget_review_checks",
)

st.session_state.az_web_search = st.toggle(
    "Enable web search for panel contributions", value=st.session_state.az_web_search,
    help="All panel agents may search, including agents added later. "
         "Search adds provider fees and token costs. ",
    key="widget_web_search",
)

st.session_state.az_setup_search = st.toggle(
    "Enable superagent web search during setup", value=st.session_state.az_setup_search,
    help="One additional research call before panel design. Its cited briefing is shared with all agents. "
         "Search and token fees apply. Later moderator decisions and final synthesis do not search.",
    key="widget_setup_search",
)

with st.expander("Limits"):
    st.caption(
        "These cap how long and how large the debate can grow, regardless of what "
        "Agent 0 decides."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.az_max_rounds = st.number_input(
            "Max rounds", min_value=1, max_value=50, value=st.session_state.az_max_rounds, step=1,
            key="widget_max_rounds",
        )
    with c2:
        st.session_state.az_max_agents = st.number_input(
            "Max panel size", min_value=1, max_value=10, value=st.session_state.az_max_agents, step=1,
            key="widget_max_agents",
        )
    with c3:
        st.session_state.az_max_total_spawns = st.number_input(
            "Max agents created", min_value=1, max_value=20, value=st.session_state.az_max_total_spawns, step=1,
            help="Counts the starting panel plus every agent added later.",
            key="widget_max_total_spawns",
        )

with st.container(border=True):
    st.subheader("API keys")
    st.caption("One key per provider covers all its models. Set all three so Agent 0 can choose across providers.")
    api_keys: dict[str, str] = {}
    for provider in PROVIDERS:
        env_var = API_KEY_ENV_VARS.get(provider, f"{provider.upper()}_API_KEY")
        if os.environ.get(env_var):
            st.success(f"{provider} key set", icon=":material/check_circle:")
            api_keys[provider] = ""
        else:
            api_keys[provider] = st.text_input(
                f"{provider} API key", type="password", key=f"az_api_key_{provider}",
            )

st.divider()

topic_ready = bool(st.session_state.az_topic.strip())
if not topic_ready:
    st.warning("Enter a topic to start.")

keys_ready = all(os.environ.get(API_KEY_ENV_VARS[p]) or api_keys.get(p) for p in PROVIDERS)
if not keys_ready:
    st.info("Set a key for each provider before starting.")

if st.button("Start debate", type="primary", disabled=not (topic_ready and keys_ready and documents_ready), use_container_width=True):
    for provider, key in api_keys.items():
        if key:
            reset_provider(provider)
            os.environ[API_KEY_ENV_VARS.get(provider, f"{provider.upper()}_API_KEY")] = key

    st.session_state.experiment_config = {
        "mode": "agent_zero",
        "shared_documents": shared_documents,
        "web_search_enabled": bool(st.session_state.az_web_search),
        "task": {
            "description": st.session_state.az_topic.strip(),
            "brief_instructions": st.session_state.az_brief_instructions.strip(),
        },
        # "instructions" and "ecu" are left unset here on purpose: Agent 0
        # designs them in its initialization call (see core/agent_zero.py).
        "agent_zero": {
            "web_search_enabled": bool(st.session_state.az_setup_search),
            "provider": st.session_state.az_provider,
            "model": st.session_state.az_model,
            "temperature": float(st.session_state.get("az_temperature", 0.0)),
            "review_checks_enabled": bool(st.session_state.az_review_checks),
            "max_rounds": int(st.session_state.az_max_rounds),
            "max_agents": int(st.session_state.az_max_agents),
            "max_total_spawns": int(st.session_state.az_max_total_spawns),
        },
    }
    st.session_state.pop("az_result", None)
    st.session_state.pop("az_pending_calls", None)
    st.session_state.pop("az_save_error", None)
    st.switch_page("pages/5_Run.py")
