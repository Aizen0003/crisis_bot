"""
Centralized configuration for the Crisis Intelligence Command Center.
All thresholds, model names, collection configs, and environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# ── Model Configuration ──────────────────────────────────────────────
TEXT_ENCODER_MODEL = "all-MiniLM-L6-v2"
CLIP_MODEL = "clip-ViT-B-32"
GEMINI_MODEL = "gemini-2.5-flash"

# ── Vector Dimensions ────────────────────────────────────────────────
TEXT_VECTOR_DIM = 384
CLIP_VECTOR_DIM = 512

# ── Qdrant Collection Names ──────────────────────────────────────────
COLLECTION_EPISODIC = "user_episodic_memory"
COLLECTION_MULTIMODAL = "disaster_multimodal"

# ── Search Thresholds ────────────────────────────────────────────────
TEXT_RELEVANCE_THRESHOLD = 0.35
IMAGE_RELEVANCE_THRESHOLD = 0.30   # Raised from 0.22 to filter weak CLIP matches
TEXT_SEARCH_LIMIT = 5
IMAGE_SEARCH_LIMIT = 5             # Fetch more, then deduplicate

# ── Base Evidence Role ───────────────────────────────────────────────
# Base disaster reports are stored with this role. Conversation memory
# (user/assistant turns) is stored separately so it does not pollute
# grounded base-evidence retrieval.
SYSTEM_REPORT_ROLE = "system_report"

# ── Memory Decay ─────────────────────────────────────────────────────
MEMORY_DECAY_HOURS = 48          # Memories older than this get reduced weight
MEMORY_DECAY_FACTOR = 0.7        # Multiplier applied to decayed memories
MAX_REINFORCEMENT_BOOST = 1.5    # Max boost from repeated queries

# ── Severity Levels ──────────────────────────────────────────────────
SEVERITY_LEVELS = {
    "CRITICAL": {"color": "#FF1744", "icon": "🔴", "priority": 1},
    "HIGH":     {"color": "#FF9100", "icon": "🟠", "priority": 2},
    "MEDIUM":   {"color": "#FFEA00", "icon": "🟡", "priority": 3},
    "LOW":      {"color": "#00E676", "icon": "🟢", "priority": 4},
}

# ── Disaster Type Classification ─────────────────────────────────────
DISASTER_KEYWORDS = {
    "flood": ["flood", "water-logging", "waterlogging", "water-rescue", "inundation", "deluge",
             "submerged", "overflow", "embankment", "breach", "rising", "river", "water level",
             "sewer", "waterborne", "boats deployed", "stranded", "rescue raft"],
    "earthquake": ["earthquake", "seismic", "tremor", "aftershock", "richter", "epicenter"],
    "cyclone": ["cyclone", "hurricane", "typhoon", "storm surge", "gust", "wind damage",
               "depression", "cyclonic", "high wave", "high tide"],
    "landslide": ["landslide", "mudslide", "slope failure", "rockfall", "debris flow",
                 "hill-slope", "slope stability", "boulder", "slope anchor"],
    "fire": ["fire", "blaze", "conflagration", "wildfire", "forest fire",
             "stubble burning", "arson", "smoke plume", "foam unit"],
    "tsunami": ["tsunami", "tidal wave", "run-up", "wave warning", "shoreline"],
    "drought": ["drought", "water crisis", "dried", "water scarcity", "famine"],
    "industrial": ["industrial", "chemical", "hazmat", "explosion", "gas leak",
                   "mine pit", "warehouse", "toxic plume", "coal-pit"],
    "infrastructure": ["bridge failure", "dam overflow", "road collapse", "track washout",
                       "derailment", "building collapse", "structural collapse",
                       "collapse", "masonry", "structural failure"],
}

# ── Location Coordinates (Indian cities/regions for geocoding) ───────
LOCATION_COORDINATES = {
    "kerala": (10.8505, 76.2711), "idukki": (9.8494, 76.9720),
    "mumbai": (19.0760, 72.8777), "andheri": (19.1136, 72.8697),
    "odisha": (20.9517, 85.0985), "puri": (19.7983, 85.8315),
    "uttarakhand": (30.0668, 79.0193), "chamoli": (30.4025, 79.3250),
    "assam": (26.2006, 92.9376), "dhubri": (26.0220, 89.9797),
    "guwahati": (26.1445, 91.7362), "kaziranga": (26.5775, 93.1711),
    "karnataka": (15.3173, 75.7139), "yellapur": (14.9640, 74.7620),
    "mangalore": (12.9141, 74.8560),
    "gujarat": (22.2587, 71.1924), "bhuj": (23.2420, 69.6669), "kutch": (23.7337, 69.8597),
    "tamil nadu": (11.1271, 78.6569), "coimbatore": (11.0168, 76.9558), "chennai": (13.0827, 80.2707),
    "west bengal": (22.9868, 87.8550), "kolkata": (22.5726, 88.3639), "kalimpong": (27.0667, 88.4695),
    "maharashtra": (19.7515, 75.7139), "lonavala": (18.7546, 73.4062), "raigad": (18.5151, 73.1823),
    "andhra pradesh": (15.9129, 79.7400), "vishakhapatnam": (17.6868, 83.2185), "nellore": (14.4426, 79.9865),
    "telangana": (18.1124, 79.0193), "hyderabad": (17.3850, 78.4867), "ranga reddy": (17.2543, 78.3870),
    "bihar": (25.0961, 85.3131), "patna": (25.6093, 85.1376), "hajipur": (25.6876, 85.2122), "supaul": (26.1229, 86.6046),
    "punjab": (31.1471, 75.3412), "ludhiana": (30.9010, 75.8573), "amritsar": (31.6340, 74.8723),
    "himachal pradesh": (31.1048, 77.1734), "rohtang": (32.3722, 77.2474), "kullu": (31.9592, 77.1089),
    "jammu": (33.7782, 76.5762), "kashmir": (34.0837, 74.7973), "kishtwar": (33.3153, 75.7700),
    "goa": (15.2993, 74.1240), "anjuna": (15.5735, 73.7407),
    "andaman": (11.7401, 92.6586),
    "chhattisgarh": (21.2787, 81.8661), "korba": (22.3595, 82.7501),
    "madhya pradesh": (22.9734, 78.6569), "hoshangabad": (22.7470, 77.7217),
    "rajasthan": (27.0238, 74.2179), "jaipur": (26.9124, 75.7873), "jaisalmer": (26.9157, 70.9083),
    "jharkhand": (23.6102, 85.2799), "dhanbad": (23.7957, 86.4304),
    "meghalaya": (25.4670, 91.3662), "shillong": (25.5788, 91.8933),
    "sikkim": (27.5330, 88.5122), "pelling": (27.3003, 88.2367),
    "nagaland": (26.1584, 94.5624), "dimapur": (25.9065, 93.7273),
    "manipur": (24.6637, 93.9063),
    "tripura": (23.9408, 91.9882), "agartala": (23.8315, 91.2868),
    "arunachal pradesh": (28.2180, 94.7278), "bomdila": (27.2652, 92.4223),
    "mizoram": (23.1645, 92.9376), "aizawl": (23.7271, 92.7176),
    "haryana": (29.0588, 76.0856), "karnal": (29.6857, 76.9905),
    "uttar pradesh": (26.8467, 80.9462), "kanpur": (26.4499, 80.3319),
    "delhi": (28.7041, 77.1025), "new delhi": (28.6139, 77.2090),
    "konkan": (16.5000, 73.5000),
    "ernakulam": (9.9816, 76.2999), "kozhikode": (11.2588, 75.7804),
    "ganjam": (19.5860, 84.3333),
}

# ── Image Folder Path ────────────────────────────────────────────────
IMAGE_FOLDER = "data_images"
TEXT_FILE = "data_logs.txt"

# ── UI Configuration ─────────────────────────────────────────────────
APP_TITLE = "Crisis Intelligence Command Center"
APP_ICON = "🚨"
APP_LAYOUT = "wide"


# ── Configuration Validation ─────────────────────────────────────────
class ConfigError(RuntimeError):
    """Raised when required environment configuration is missing or invalid."""


_REQUIRED_ENV = {
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "QDRANT_URL": QDRANT_URL,
    "QDRANT_API_KEY": QDRANT_API_KEY,
}


def validate_config(require: tuple[str, ...] = ("GEMINI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")):
    """
    Validate that required environment variables are present.

    Raises ConfigError with a clear, human-readable message listing the
    missing keys and how to fix them. Never prints secret values.

    Args:
        require: which keys must be present for the current operation.
    """
    missing = [key for key in require if not _REQUIRED_ENV.get(key, "").strip()]
    if missing:
        raise ConfigError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ".\n\nFix: copy `.env.example` to `.env` and fill in your keys:\n"
            "    cp .env.example .env\n"
            "Required keys: GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY.\n"
            "(Values are read from the environment / .env and are never logged.)"
        )
