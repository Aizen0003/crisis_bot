"""
Chat interface component for the Crisis Intelligence Command Center.
Handles message display, user input, and the full RAG interaction loop.

Render model (single code path): `handle_chat_input` only captures input, runs
the workflow, stores the turn in session_state, and triggers a rerun. ALL
messages — including the one just produced — are rendered exactly once by
`render_message_history` from session_state. This avoids double-rendering the
latest message/evidence panel and keeps `st.chat_input` anchored below a
height-bounded, scrollable history container instead of being pushed down the
viewport as the conversation grows.
"""

import os

import streamlit as st

from src.graph import run_crisis_graph


def init_chat_state():
    """Initialize session state for chat."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_message_history():
    """Render the full conversation (messages + images + evidence) from state.

    Everything lives inside a single height-bounded, scrollable container so a
    growing conversation scrolls *inside* the box rather than dragging the chat
    input down the page each turn. This is the ONLY place messages and their
    evidence panels are rendered.
    """
    history = st.container(height=500)
    with history:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Only assistant turns carry evidence/images/errors.
                if message["role"] != "assistant":
                    continue

                if message.get("error"):
                    st.error(message["error"])

                if message.get("show_images", True):
                    _render_images(message.get("images") or [])
                else:
                    st.info("📷 Image display suppressed by user request.")

                if message.get("sources"):
                    with st.expander("🔍 Evidence & Reasoning Trace", expanded=False):
                        _render_evidence_panel(message)


def _render_images(images: list[dict]):
    """Render retrieved images inline, flagging any that are missing on disk."""
    for img in images:
        filename = img.get("filename", "")
        if filename and os.path.exists(filename):
            st.image(
                filename,
                caption=f"📸 {img['description']} (Score: {img['score']:.2f})",
                use_container_width=True,
            )
        else:
            # Don't silently skip — say the file is missing so the panel isn't
            # mysteriously empty when an expected image can't be loaded.
            st.caption(f"🖼️ Image unavailable: {os.path.basename(filename) or 'unknown'}")


def _render_evidence_panel(message: dict):
    """Render the expandable evidence panel for an assistant message."""
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("**📄 Text Sources Retrieved:**")
        if message.get("sources"):
            for i, src in enumerate(message["sources"], 1):
                severity = src.get("metadata", {}).get("severity", "")
                icon = {
                    "CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢",
                }.get(severity, "")
                st.caption(
                    f"{icon} **[Source {i}]** (Score: {src['score']:.3f})\n"
                    f"{src['text'][:150]}..."
                )
        else:
            st.caption("No relevant text sources found.")

    with col2:
        st.markdown("**🖼️ Visual Evidence:**")
        images = message.get("images") or []
        if images:
            for img_info in images:
                filename = img_info.get("filename", "")
                if filename and os.path.exists(filename):
                    st.image(filename, width=200)
                    st.caption(f"Match: {img_info['score']:.2f} — {img_info['description']}")
                else:
                    st.caption(
                        f"🖼️ unavailable — {img_info['description']} "
                        f"(Match: {img_info['score']:.2f})"
                    )
        else:
            st.caption("No matching images found.")

    # Reasoning trace
    if message.get("reasoning_trace"):
        st.markdown("**🧠 Reasoning Trace:**")
        st.markdown(
            f'<div class="reasoning-trace">{message["reasoning_trace"]}</div>',
            unsafe_allow_html=True,
        )


def handle_chat_input(disaster_filter: str = None, region_filter: str = None):
    """
    Capture the user's input, run the RAG workflow, store the turn, and rerun.

    Rendering is intentionally NOT done here — see the module docstring. We only
    mutate session_state and trigger a rerun so `render_message_history` shows
    the new turn through the single render path.
    """
    prompt = st.chat_input("Ask about a disaster scenario...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})

    # ── Run the LangGraph RAG workflow (retrieval + synthesis + memory) ──
    with st.spinner("🧠 Running crisis intelligence workflow..."):
        results = run_crisis_graph(
            query=prompt,
            filters={
                "disaster": disaster_filter,
                "region": region_filter,
            },
        )

    # Store the assistant turn. Episodic memory in Qdrant is persisted inside the
    # workflow's persist_memory node, so we do not write it again here.
    st.session_state.messages.append({
        "role": "assistant",
        "content": results["response"],
        "images": results["image_results"],
        "sources": results["text_results"],
        "reasoning_trace": results["reasoning_trace"],
        "show_images": results["show_images"],
        "error": results.get("error"),
    })

    # Re-run so the freshly stored turn renders through render_message_history().
    st.rerun()
