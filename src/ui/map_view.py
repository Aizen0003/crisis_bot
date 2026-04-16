"""
GIS Map View component.
Renders an interactive Folium map showing disaster locations
extracted from the ingested data logs.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
from src.qdrant_manager import get_qdrant_client
from src.config import COLLECTION_EPISODIC, SEVERITY_LEVELS


# Map disaster types to marker colors
DISASTER_COLORS = {
    "flood": "blue",
    "earthquake": "red",
    "cyclone": "purple",
    "landslide": "orange",
    "fire": "red",
    "tsunami": "darkblue",
    "drought": "beige",
    "industrial": "gray",
    "infrastructure": "darkred",
    "other": "lightgray",
}

DISASTER_ICONS = {
    "flood": "tint",
    "earthquake": "exclamation-triangle",
    "cyclone": "cloud",
    "landslide": "mountain",
    "fire": "fire",
    "tsunami": "water",
    "drought": "sun",
    "industrial": "industry",
    "infrastructure": "road",
    "other": "info-sign",
}


def _fetch_geolocated_reports() -> list[dict]:
    """Fetch all system reports that have coordinates from Qdrant."""
    client = get_qdrant_client()
    
    try:
        # Scroll through all system_report entries
        results, _ = client.scroll(
            collection_name=COLLECTION_EPISODIC,
            scroll_filter={
                "must": [
                    {"key": "role", "match": {"value": "system_report"}},
                ]
            },
            limit=200,
            with_vectors=False,
        )
        
        reports = []
        for point in results:
            payload = point.payload
            if payload.get("lat") and payload.get("lon"):
                reports.append({
                    "text": payload.get("chat_text", ""),
                    "lat": payload["lat"],
                    "lon": payload["lon"],
                    "region": payload.get("region", "Unknown"),
                    "disaster_type": payload.get("disaster_type", "other"),
                    "severity": payload.get("severity", "LOW"),
                    "source_agency": payload.get("source_agency", "Unknown"),
                })
        
        return reports
    except Exception as e:
        st.error(f"Failed to fetch map data: {e}")
        return []


def render_map_view():
    """Render the interactive crisis map."""
    st.markdown("### 🗺️ Crisis Situation Map")
    st.caption("Real-time visualization of disaster locations extracted from ingested reports.")
    
    reports = _fetch_geolocated_reports()
    
    if not reports:
        st.info(
            "📍 No geolocated reports found. Run the data ingestion script "
            "(`python ingest_bulk.py`) to populate the map."
        )
        # Show empty map centered on India
        m = folium.Map(location=[22.5, 82.0], zoom_start=5, tiles="CartoDB dark_matter")
        st_folium(m, use_container_width=True, height=500)
        return
    
    # ── Build the map ────────────────────────────────────────────────
    m = folium.Map(
        location=[22.5, 82.0],  # Center of India
        zoom_start=5,
        tiles="CartoDB dark_matter",
    )
    
    # Track severity counts for the legend
    severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    type_counts = {}
    
    for report in reports:
        dtype = report["disaster_type"]
        severity = report["severity"]
        
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        type_counts[dtype] = type_counts.get(dtype, 0) + 1
        
        # Marker color based on disaster type
        color = DISASTER_COLORS.get(dtype, "lightgray")
        icon_name = DISASTER_ICONS.get(dtype, "info-sign")
        
        # Build popup content
        sev_config = SEVERITY_LEVELS.get(severity, SEVERITY_LEVELS["LOW"])
        popup_html = f"""
        <div style="font-family: Inter, sans-serif; min-width: 200px;">
            <h4 style="margin:0; color: #333;">{report['region']}</h4>
            <p style="margin: 4px 0;">
                <span style="background: {sev_config['color']}; color: white; padding: 2px 8px; 
                       border-radius: 4px; font-size: 11px; font-weight: 600;">
                    {severity}
                </span>
                <span style="background: #eee; padding: 2px 8px; border-radius: 4px; 
                       font-size: 11px; margin-left: 4px;">
                    {dtype.upper()}
                </span>
            </p>
            <p style="font-size: 12px; color: #555; margin-top: 8px;">
                {report['text'][:200]}...
            </p>
            <p style="font-size: 11px; color: #888; margin-top: 4px;">
                📡 {report['source_agency']}
            </p>
        </div>
        """
        
        folium.Marker(
            location=[report["lat"], report["lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{sev_config['icon']} {report['region']} — {dtype.title()}",
            icon=folium.Icon(color=color, icon=icon_name, prefix="fa"),
        ).add_to(m)
    
    # ── Render ───────────────────────────────────────────────────────
    st_folium(m, use_container_width=True, height=550)
    
    # ── Map Legend / Stats ───────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 Severity Distribution**")
        for sev, count in severity_counts.items():
            if count > 0:
                config = SEVERITY_LEVELS.get(sev, {})
                st.markdown(
                    f'{config.get("icon", "⚪")} **{sev}**: {count} reports'
                )
    with col2:
        st.markdown("**🏷️ Disaster Types**")
        for dtype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"• **{dtype.title()}**: {count}")
