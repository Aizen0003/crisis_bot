"""
Main dashboard layout component.
Assembles the sidebar, header, tabs, and all sub-components.
"""

import streamlit as st
from src.ui.styles import get_custom_css
from src.ui.chat import init_chat_state, render_message_history, handle_chat_input
from src.ui.map_view import render_map_view
from src.ui.analytics import render_analytics
from src.qdrant_manager import ensure_collections
from src.memory import reset_scenario
from src.config import (
    APP_TITLE, APP_ICON, APP_LAYOUT,
    TEXT_RELEVANCE_THRESHOLD, IMAGE_RELEVANCE_THRESHOLD,
)


def setup_page():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout=APP_LAYOUT,
        initial_sidebar_state="expanded",
    )
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Ensure Qdrant collections exist
    ensure_collections()
    
    # Initialize chat state
    init_chat_state()


def render_header():
    """Render the main header bar: title + a single, honest status line."""
    st.markdown(
        f"""
        <div class="main-header">
            <div>
                <h1>{APP_ICON} {APP_TITLE}</h1>
                <p class="subtitle">Multimodal RAG for National Disaster Response
                &nbsp;·&nbsp; LangGraph orchestration &nbsp;·&nbsp; Qdrant vector search</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> dict:
    """
    Render the sidebar: search filter, scenario controls, and compact system info.

    Returns the active filter settings. Only the filters the graph actually
    applies from the UI are exposed here — no dead controls. (Database stats live
    in the Analytics tab, so they are not duplicated here.)
    """
    filters = {}

    with st.sidebar:
        st.markdown("## 🧠 Command Console")

        # ── Search Filter ────────────────────────────────────────────
        # Only disaster type is exposed: it's the one filter the graph applies
        # from the UI (triage supplies an automatic fallback when unset). Severity
        # is deliberately NOT a filter — keyword-triaged severity is noisy and
        # hard-filtering on it hurts recall.
        st.markdown("### 🔎 Search Filter")

        disaster_filter = st.selectbox(
            "Disaster Type",
            ["All", "Flood", "Earthquake", "Cyclone", "Landslide", "Fire",
             "Tsunami", "Drought", "Industrial", "Infrastructure"],
            key="disaster_filter",
        )
        filters["disaster"] = disaster_filter

        st.markdown("---")

        # ── Scenario Controls ────────────────────────────────────────
        st.markdown("### ⚙️ Scenario Controls")

        if st.button("🔄 Start New Scenario", use_container_width=True, type="primary"):
            success = reset_scenario()
            if success:
                st.success("✅ Conversation memory wiped! Base data preserved.")
            else:
                st.error("❌ Failed to clear memory.")
            st.session_state.messages = []
            st.rerun()

        st.caption(
            "Clears conversation history while preserving "
            "all ingested disaster reports and images."
        )

        st.markdown("---")

        # ── System Info (compact) ────────────────────────────────────
        st.markdown("### ℹ️ System Info")
        st.caption("**Orchestration:** LangGraph + LangChain")
        st.caption("**Models:** MiniLM-L6-v2 · CLIP ViT-B-32 · Gemini 2.5 Flash")
        st.caption("**Vector DB:** Qdrant Cloud")
        st.caption(
            f"**Thresholds:** text > {TEXT_RELEVANCE_THRESHOLD} · "
            f"image > {IMAGE_RELEVANCE_THRESHOLD}"
        )

    return filters


def render_main_content(filters: dict):
    """Render the main tabbed content area."""
    tab_chat, tab_map, tab_analytics = st.tabs([
        "💬 Command Chat",
        "🗺️ Situation Map",
        "📊 Analytics",
    ])
    
    with tab_chat:
        # st.chat_input lives inside this tab (not at the top level) so it does
        # not bleed onto the Map/Analytics tabs. render_message_history() bounds
        # the conversation in a scrollable, height-fixed container so the input
        # stays anchored just below it instead of being dragged down. See chat.py.
        render_message_history()
        handle_chat_input(disaster_filter=filters.get("disaster"))

    with tab_map:
        render_map_view()
    
    with tab_analytics:
        render_analytics()
