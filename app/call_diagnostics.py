"""Shared inspector for current and persisted model requests."""
import json
import streamlit as st


def render_call_diagnostics(calls, key):
    with st.expander(f"API call diagnostics ({len(calls)} calls)"):
        if not calls:
            st.caption("No call records available. Older debates predate request logging.")
            return
        failures = [c for c in calls if c.get("status") == "error"]
        if failures:
            st.warning(f"{len(failures)} call(s) failed. Select one to inspect the exact error and response.")
        labels = []
        for i, c in enumerate(calls):
            turn = f"Round {c['cycle'] + 1}" if c.get("cycle") is not None else "Setup"
            labels.append(f"{i + 1}. {turn} · {c.get('agent', c.get('agent_name'))} · {c.get('phase')} · {c.get('model')} · {c.get('status')}")
        i = st.selectbox("Call", range(len(calls)), format_func=lambda i: labels[i], key=f"{key}_call")
        c = calls[i]
        if c.get("error"):
            st.error(c["error"])
        st.caption(f"Stop reason: {c.get('stop_reason') or 'not reported'} · Duration: {c.get('duration_ms', '?')} ms")
        for label, field in (("System prompt", "system_prompt"), ("User prompt", "prompt"), ("Response text", "response")):
            st.markdown(f"**{label}**")
            st.code(c.get(field) or "(empty)", language=None)
        st.json({"parameters": c.get("parameters"), "usage": c.get("usage"),
                 "diagnostics": c.get("diagnostics", {k: c.get(k) for k in ("request_id", "response_id", "block_types", "web_search")})}, expanded=False)
        st.download_button("Download API calls (.json)", json.dumps(calls, indent=2, ensure_ascii=False, default=str),
                           file_name="api_calls.json", mime="application/json", key=f"{key}_download")
