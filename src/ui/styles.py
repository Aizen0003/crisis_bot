"""
Custom CSS styles for the Crisis Intelligence Command Center.
Dark theme with crisis-appropriate color system.
"""


def get_custom_css() -> str:
    """Return the full custom CSS for the application."""
    return """
    <style>
    /* ══════════════════════════════════════════════════════════════════
       CRISIS INTELLIGENCE — DARK COMMAND CENTER THEME
       ══════════════════════════════════════════════════════════════════ */

    /* ── Global ─────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header Bar ─────────────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1a0a0a 100%);
        border: 1px solid rgba(255, 23, 68, 0.3);
        border-radius: 12px;
        padding: 1.2rem 1.8rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: 0 4px 20px rgba(255, 23, 68, 0.1);
    }

    .main-header h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF1744, #FF6D00, #FFD600);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }

    .main-header .subtitle {
        color: #8b949e;
        font-size: 0.85rem;
        margin: 0;
        font-weight: 400;
    }

    /* ── Status Indicators ──────────────────────────────────────────── */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.05em;
    }

    .status-critical { background: rgba(255,23,68,0.15); color: #FF1744; border: 1px solid rgba(255,23,68,0.3); }
    .status-high { background: rgba(255,145,0,0.15); color: #FF9100; border: 1px solid rgba(255,145,0,0.3); }
    .status-medium { background: rgba(255,234,0,0.15); color: #FFEA00; border: 1px solid rgba(255,234,0,0.3); }
    .status-low { background: rgba(0,230,118,0.15); color: #00E676; border: 1px solid rgba(0,230,118,0.3); }

    /* ── Metric Cards ───────────────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(145deg, #161b22, #0d1117);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        transition: transform 0.2s, border-color 0.3s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255,23,68,0.4);
    }

    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #e6edf3;
        margin: 0;
    }

    .metric-card .metric-label {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 0.3rem 0 0 0;
    }

    /* ── Chat Messages ──────────────────────────────────────────────── */
    .stChatMessage {
        border-radius: 10px !important;
        border: 1px solid #30363d !important;
    }

    /* ── Evidence Panel ─────────────────────────────────────────────── */
    .evidence-panel {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.5rem;
    }

    .evidence-panel .evidence-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #58a6ff;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }

    /* ── Severity Badge ─────────────────────────────────────────────── */
    .severity-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.05em;
    }

    /* ── Sidebar ────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #30363d;
    }

    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 1rem;
        color: #e6edf3;
        border-bottom: 1px solid #30363d;
        padding-bottom: 0.5rem;
    }

    /* ── Map Container ──────────────────────────────────────────────── */
    .map-container {
        border: 1px solid #30363d;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Reasoning Trace ────────────────────────────────────────────── */
    .reasoning-trace {
        background: rgba(13, 17, 23, 0.9);
        border: 1px solid #1f6feb33;
        border-left: 3px solid #1f6feb;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #8b949e;
        line-height: 1.6;
        white-space: pre-wrap;
    }

    /* ── Tab Styling ────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #30363d;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: #8b949e;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
    }

    .stTabs [aria-selected="true"] {
        color: #FF1744 !important;
        border-bottom: 2px solid #FF1744 !important;
    }

    /* ── Animations ─────────────────────────────────────────────────── */
    @keyframes pulse-red {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background: #FF1744;
        border-radius: 50%;
        display: inline-block;
        animation: pulse-red 1.5s ease-in-out infinite;
        margin-right: 0.5rem;
    }

    .live-indicator {
        display: inline-flex;
        align-items: center;
        font-size: 0.75rem;
        color: #FF1744;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Scrollbar ──────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #484f58; }

    </style>
    """
