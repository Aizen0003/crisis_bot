"""
Main dashboard layout component.
Assembles the sidebar, header, tabs, and all sub-components.
"""

import streamlit as st
from src.ui.styles import get_custom_css
from src.ui.chat import init_chat_state, render_message_history, handle_chat_input
from src.ui.map_view import render_map_view
from src.ui.analytics import render_analytics
from src.qdrant_manager import ensure_collections, get_collection_stats
from src.memory import reset_scenario
from src.config import APP_TITLE, APP_ICON, APP_LAYOUT


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
    """Render the main header bar."""
    st.markdown(
        f"""
        <div class="main-header">
            <div>
                <h1>{APP_ICON} {APP_TITLE}</h1>
                <p class="subtitle">Multimodal RAG System for National Disaster Response 
                &nbsp;|&nbsp; 
                <span class="live-indicator"><span class="pulse-dot"></span>LIVE</span>
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> dict:
    """
    Render the sidebar with system controls and filters.
    Returns the active filter settings.
    """
    filters = {}
    
    with st.sidebar:
        st.markdown("## 🧠 Command Console")
        
        # ── System Status ────────────────────────────────────────────
        stats = get_collection_stats()
        episodic = stats.get("user_episodic_memory", {})
        multimodal = stats.get("disaster_multimodal", {})
        
        st.markdown(
            f"""
            <div style="background: rgba(22,27,34,0.8); border: 1px solid #30363d; 
                        border-radius: 8px; padding: 0.8rem; margin-bottom: 1rem;">
                <p style="margin:0; font-size: 0.75rem; color: #8b949e; 
                          text-transform: uppercase; letter-spacing: 0.08em;">
                    Database Status
                </p>
                <p style="margin: 0.3rem 0 0 0; font-size: 0.85rem; color: #e6edf3;">
                    📄 {episodic.get('points_count', 0)} text vectors &nbsp;|&nbsp; 
                    🖼️ {multimodal.get('points_count', 0)} image vectors
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # ── Search Filters ───────────────────────────────────────────
        st.markdown("### 🔎 Search Filters")
        
        disaster_filter = st.selectbox(
            "Disaster Type",
            ["All", "Flood", "Earthquake", "Cyclone", "Landslide", "Fire",
             "Tsunami", "Drought", "Industrial", "Infrastructure"],
            key="disaster_filter",
        )
        
        severity_filter = st.selectbox(
            "Severity Level",
            ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
            key="severity_filter",
        )
        
        filters["disaster"] = disaster_filter
        filters["severity"] = severity_filter
        
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
        
        # ── System Info ──────────────────────────────────────────────
        st.markdown("### ℹ️ System Info")
        st.caption("**Models:** MiniLM-L6-v2 + CLIP ViT-B-32")
        st.caption("**LLM:** Google Gemini 2.5 Flash")
        st.caption("**Vector DB:** Qdrant Cloud")
        st.caption("**Text Threshold:** > 0.35")
        st.caption("**Image Threshold:** > 0.22")
    
    return filters


def render_main_content(filters: dict):
    """Render the main tabbed content area."""
    tab_chat, tab_map, tab_analytics = st.tabs([
        "💬 Command Chat",
        "🗺️ Situation Map",
        "📊 Analytics",
    ])
    
    with tab_chat:
        render_message_history()
        handle_chat_input(
            disaster_filter=filters.get("disaster"),
            severity_filter=filters.get("severity"),
        )
    
    with tab_map:
        render_map_view()
    
    with tab_analytics:
        render_analytics()
