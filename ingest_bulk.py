"""
Bulk Data Ingestion Script
===========================
Ingests text logs and images into Qdrant vector collections
with automatic metadata enrichment:
  - Disaster type classification
  - Severity assessment
  - Location extraction & geocoding
  - Source agency identification
  - Timestamps

Usage:
    python ingest_bulk.py
"""

import os
import sys
from datetime import datetime, timezone
from tqdm import tqdm
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import IMAGE_FOLDER, TEXT_FILE, validate_config, ConfigError
from src.qdrant_manager import (
    get_qdrant_client, ensure_collections,
    upsert_text_points_batch, upsert_image_points_batch,
)
from src.embeddings import load_text_encoder, load_clip_model
from src.agents.triage_agent import triage_report
from src.utils.location_extractor import extract_location
from src.utils.logger import get_logger

logger = get_logger("ingest")


def ingest_text_logs():
    """Ingest text logs with automatic metadata enrichment."""
    if not os.path.exists(TEXT_FILE):
        logger.error(f"❌ Text file not found: {TEXT_FILE}")
        return 0
    
    with open(TEXT_FILE, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    logger.info(f"📄 Processing {len(lines)} text logs...")
    
    encoder = load_text_encoder()
    now = datetime.now(timezone.utc).isoformat()
    
    points_data = []
    
    for line in tqdm(lines, desc="Encoding text logs", unit="log"):
        # Encode
        vector = encoder.encode(line).tolist()
        
        # Triage: classify disaster type and severity
        triage = triage_report(line)
        
        # Extract location
        location_name, coordinates = extract_location(line)
        
        # Build enriched metadata
        metadata = {
            "disaster_type": triage["disaster_type"],
            "severity": triage["severity"],
            "severity_color": triage["severity_color"],
            "severity_icon": triage["severity_icon"],
            "source_agency": triage["source_agency"],
            "region": location_name or "Unknown",
            "timestamp": now,
        }
        
        if coordinates:
            metadata["lat"] = coordinates[0]
            metadata["lon"] = coordinates[1]
        
        points_data.append({
            "text": line,
            "vector": vector,
            "role": "system_report",
            "metadata": metadata,
        })
    
    # Batch upsert
    upsert_text_points_batch(points_data)
    logger.info(f"✅ Ingested {len(points_data)} text logs with enriched metadata")
    
    # Print summary
    _print_text_summary(points_data)
    
    return len(points_data)


def ingest_images():
    """Ingest disaster images using CLIP embeddings."""
    if not os.path.exists(IMAGE_FOLDER):
        logger.error(f"❌ Image folder not found: {IMAGE_FOLDER}")
        return 0
    
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
    image_files = [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(valid_extensions)
    ]
    
    if not image_files:
        logger.warning("⚠️ No images found in folder")
        return 0
    
    logger.info(f"📸 Processing {len(image_files)} images...")
    
    clip = load_clip_model()
    points_data = []
    skipped = 0
    
    for img_file in tqdm(image_files, desc="Encoding images", unit="img"):
        try:
            img_path = os.path.join(IMAGE_FOLDER, img_file)
            
            # Generate description from filename
            clean_desc = (
                img_file.rsplit(".", 1)[0]
                .replace("_", " ")
                .replace("-", " ")
                .strip()
            )
            
            # Encode image with CLIP
            img = Image.open(img_path).convert("RGB")
            vector = clip.encode(img).tolist()
            
            # Extract location from filename/description
            location_name, coordinates = extract_location(clean_desc)
            
            metadata = {
                "region": location_name or "Unknown",
            }
            if coordinates:
                metadata["lat"] = coordinates[0]
                metadata["lon"] = coordinates[1]
            
            points_data.append({
                "vector": vector,
                "filename": img_path,
                "description": clean_desc,
                "metadata": metadata,
            })
        except Exception as e:
            logger.warning(f"⚠️ Skipped {img_file}: {e}")
            skipped += 1
    
    # Batch upsert
    if points_data:
        upsert_image_points_batch(points_data)
    
    logger.info(
        f"✅ Ingested {len(points_data)} images "
        f"({skipped} skipped)"
    )
    return len(points_data)


def _print_text_summary(points_data: list[dict]):
    """Print a summary of the ingested text data."""
    severity_counts = {}
    type_counts = {}
    region_counts = {}
    
    for p in points_data:
        meta = p.get("metadata", {})
        
        sev = meta.get("severity", "UNKNOWN")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        dtype = meta.get("disaster_type", "other")
        type_counts[dtype] = type_counts.get(dtype, 0) + 1
        
        region = meta.get("region", "Unknown")
        if region != "Unknown":
            region_counts[region] = region_counts.get(region, 0) + 1
    
    print("\n" + "═" * 50)
    print("📊 INGESTION SUMMARY")
    print("═" * 50)
    
    print("\n🏷️ Severity Distribution:")
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        count = severity_counts.get(sev, 0)
        bar = "█" * count
        print(f"  {sev:10s} │ {bar} {count}")
    
    print("\n🔥 Disaster Types:")
    for dtype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {dtype:15s} │ {count}")
    
    print(f"\n📍 Geolocated Regions: {len(region_counts)}")
    print("═" * 50 + "\n")


def main():
    """Run the full ingestion pipeline."""
    print("╔══════════════════════════════════════════════════╗")
    print("║   CRISIS INTELLIGENCE — DATA INGESTION ENGINE   ║")
    print("╚══════════════════════════════════════════════════╝\n")
    
    # Validate environment before doing any work (no secrets are printed).
    try:
        validate_config(require=("QDRANT_URL", "QDRANT_API_KEY"))
    except ConfigError as e:
        logger.error(str(e))
        sys.exit(1)

    # Initialize
    logger.info("Connecting to Qdrant...")
    _ = get_qdrant_client()
    
    logger.info("Ensuring collections exist...")
    ensure_collections()
    
    # Ingest
    text_count = ingest_text_logs()
    img_count = ingest_images()
    
    # Final summary
    print(f"\n🎯 TOTAL: {text_count} text logs + {img_count} images ingested")
    print("✅ Ready. Launch the app with: streamlit run app.py\n")


if __name__ == "__main__":
    main()
