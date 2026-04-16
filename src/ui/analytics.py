"""
Analytics dashboard component.
Displays statistics, charts, and collection health metrics.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.qdrant_manager import get_qdrant_client, get_collection_stats
from src.config import COLLECTION_EPISODIC, SEVERITY_LEVELS


def _fetch_all_reports() -> list[dict]:
    """Fetch all system reports for analytics."""
    client = get_qdrant_client()
    try:
        results, _ = client.scroll(
            collection_name=COLLECTION_EPISODIC,
            scroll_filter={
                "must": [
                    {"key": "role", "match": {"value": "system_report"}},
                ]
            },
            limit=500,
            with_vectors=False,
        )
        return [p.payload for p in results]
    except Exception:
        return []


def render_analytics():
    """Render the full analytics dashboard."""
    st.markdown("### 📊 Crisis Analytics Dashboard")
    
    # ── Collection Health ────────────────────────────────────────────
    stats = get_collection_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    episodic = stats.get("user_episodic_memory", {})
    multimodal = stats.get("disaster_multimodal", {})
    
    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <p class="metric-value">{episodic.get('points_count', 0)}</p>
                <p class="metric-label">Text Reports</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <p class="metric-value">{multimodal.get('points_count', 0)}</p>
                <p class="metric-label">Visual Evidence</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        total = episodic.get('points_count', 0) + multimodal.get('points_count', 0)
        st.markdown(
            f"""<div class="metric-card">
                <p class="metric-value">{total}</p>
                <p class="metric-label">Total Vectors</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        status_emoji = "🟢" if episodic.get("status") == "green" else "🟡"
        st.markdown(
            f"""<div class="metric-card">
                <p class="metric-value">{status_emoji}</p>
                <p class="metric-label">System Status</p>
            </div>""",
            unsafe_allow_html=True,
        )
    
    st.markdown("---")
    
    # ── Disaster Analytics ───────────────────────────────────────────
    reports = _fetch_all_reports()
    
    if not reports:
        st.info("📈 No data available for analytics. Run `python ingest_bulk.py` first.")
        return
    
    df = pd.DataFrame(reports)
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Disaster Type Distribution
        if "disaster_type" in df.columns:
            type_counts = df["disaster_type"].value_counts().reset_index()
            type_counts.columns = ["Disaster Type", "Count"]
            
            fig_type = px.pie(
                type_counts,
                values="Count",
                names="Disaster Type",
                title="Disaster Type Distribution",
                color_discrete_sequence=px.colors.qualitative.Dark2,
                hole=0.4,
            )
            fig_type.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e6edf3",
                title_font_size=14,
                showlegend=True,
                height=350,
            )
            st.plotly_chart(fig_type, use_container_width=True)
    
    with col_right:
        # Severity Distribution
        if "severity" in df.columns:
            sev_counts = df["severity"].value_counts().reset_index()
            sev_counts.columns = ["Severity", "Count"]
            
            color_map = {k: v["color"] for k, v in SEVERITY_LEVELS.items()}
            
            fig_sev = px.bar(
                sev_counts,
                x="Severity",
                y="Count",
                title="Severity Breakdown",
                color="Severity",
                color_discrete_map=color_map,
            )
            fig_sev.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e6edf3",
                title_font_size=14,
                showlegend=False,
                height=350,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(48,54,61,0.5)"),
            )
            st.plotly_chart(fig_sev, use_container_width=True)
    
    # Regional Distribution
    if "region" in df.columns:
        region_counts = df["region"].value_counts().head(15).reset_index()
        region_counts.columns = ["Region", "Count"]
        
        fig_region = px.bar(
            region_counts,
            x="Count",
            y="Region",
            orientation="h",
            title="Top 15 Affected Regions",
            color="Count",
            color_continuous_scale="YlOrRd",
        )
        fig_region.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3",
            title_font_size=14,
            height=400,
            yaxis=dict(autorange="reversed"),
            xaxis=dict(showgrid=True, gridcolor="rgba(48,54,61,0.5)"),
        )
        st.plotly_chart(fig_region, use_container_width=True)
    
    # ── Source Agency Distribution ────────────────────────────────────
    if "source_agency" in df.columns:
        st.markdown("#### 📡 Reporting Agencies")
        agency_counts = df["source_agency"].value_counts().head(10)
        for agency, count in agency_counts.items():
            st.markdown(f"• **{agency}**: {count} reports")
