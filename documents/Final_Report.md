# Project Report: Crisis Intelligence Command Center
## A Multimodal RAG System for National Disaster Response

**Submitted by:** Aditya Sharma
**Hackathon:** Convolve 4.0 — Pan-IIT AI/ML Hackathon
**Problem Statement:** Qdrant — Search, Memory, and Recommendations for Societal Impact
**Date:** January 2026

---

## 1. Executive Summary

In the chaotic aftermath of natural disasters, decision-makers are paralyzed not by a lack of information, but by its **fragmentation**. Visual evidence from drones sits in one system while urgent radio logs from ground teams remain in another. The **Crisis Intelligence Command Center** is a Decision Support System (DSS) designed to bridge this cognitive gap.

Leveraging **Retrieval-Augmented Generation (RAG)** and **Multimodal Vector Search** powered by **Qdrant**, this system creates a unified intelligence layer that can *see*, *read*, and *remember*. By encoding images and text into a shared high-dimensional vector space, responders can query complex datasets using natural language and receive evidence-grounded, citation-backed responses synthesized by **Google Gemini 2.5 Flash**.

The system implements a **multi-agent architecture** with three specialized agents (Triage, Retrieval, Synthesis) and features an interactive GIS crisis map, analytics dashboard, and evolving memory with recency-based decay — directly addressing the challenge requirements for *search, memory, and recommendations for societal impact*.

---

## 2. Problem Statement

### 2.1 The Societal Issue: Information Fragmentation

India is highly vulnerable to natural disasters — from annual urban floods in Guwahati and Mumbai to cyclones in Odisha and landslides in the Himalayas. A key challenge in managing these crises is the "fog of war" created by disconnected data streams.

In a typical control room scenario:

- **Visual Data**: CCTV feeds, drone footage, and citizen photos are viewed on separate monitors.
- **Textual Data**: SOPs, live radio transcripts, and social media distress calls are text-based.
- **Human Bottleneck**: An operator must *manually* correlate a photo of a flooded street with a text report received 20 minutes earlier. This manual correlation is slow, error-prone, and impossible to scale during peak crisis moments.

### 2.2 Why It Matters

The inability to instantly synthesize visual and textual data leads to:

1. **Delayed Response**: Critical minutes are lost verifying reports.
2. **Resource Misallocation**: Rescue boats are sent to drained areas while critically submerged areas are ignored.
3. **Loss of Institutional Memory**: Situational awareness is lost during shift changes.

Our solution addresses these by creating a **persistent, multimodal memory** that provides instant situational awareness to any operator, at any time.

---

## 3. System Design & Architecture

### 3.1 Architecture Overview

The system follows a **Dual-Stream RAG** architecture with a **multi-agent pipeline**. Unlike standard LLM applications, our system "grounds" its answers in a dynamic database of local evidence.

The architecture consists of four layers:

**Layer 1 — Ingestion (Sensory Input):**
- Text Pathway: Emergency logs → `all-MiniLM-L6-v2` → 384-dimensional vectors
- Visual Pathway: Images → `CLIP ViT-B-32` → 512-dimensional vectors
- Metadata Enrichment: Triage Agent auto-classifies disaster type, severity, source agency, and geocodes locations
- Both streams indexed into separate Qdrant collections

**Layer 2 — Vector Storage (Qdrant Memory):**
- `user_episodic_memory`: Text logs with structured metadata payloads (384d, Cosine)
- `disaster_multimodal`: Image embeddings with descriptions (512d, Cosine)

**Layer 3 — Multi-Agent Retrieval:**
- **Retrieval Agent**: Orchestrates parallel search across both collections with metadata filtering
- **Triage Agent**: Classifies incoming reports by disaster type and severity
- **Synthesis Agent**: Generates evidence-cited responses using Gemini 2.5 Flash

**Layer 4 — Presentation:**
- Streamlit dashboard with three views: Command Chat, Situation Map, Analytics
- Custom dark theme with crisis-appropriate color system

### 3.2 Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Streamlit + Custom CSS | Real-time crisis dashboard |
| Vector DB | Qdrant Cloud | High-performance similarity search |
| Text Embeddings | all-MiniLM-L6-v2 (384d) | Semantic text encoding |
| Visual Embeddings | CLIP ViT-B-32 (512d) | Cross-modal image-text search |
| LLM | Google Gemini 2.5 Flash | Evidence synthesis & reasoning |
| GIS | Folium + CartoDB Dark | Interactive crisis map |
| Analytics | Plotly | Data visualization |
| Geocoding | Custom location dictionary | Indian city/state coordinate mapping |

### 3.3 Why Qdrant is Critical

Qdrant is the **linchpin** of this solution. A traditional SQL database cannot understand that "deluge" and "flood" are synonyms, nor can it match a photo of fire to the word "blaze."

Qdrant enables:

- **Vector Agnosticism**: We store different vector sizes (384d and 512d) in parallel collections.
- **Payload Filtering**: Critical for metadata-driven search — filter by disaster type, severity, region, or source agency.
- **Scored Retrieval**: Cosine similarity scores enable threshold-based filtering to prevent hallucinated connections.
- **Point Lifecycle**: The "Safe Reset" protocol uses targeted deletions based on metadata tags, preserving base data while clearing conversation history.
- **Low Latency**: HNSW index provides millisecond-latency searches — non-negotiable in disaster scenarios.

---

## 4. Multimodal Strategy & Vector Embeddings

### 4.1 Text Embeddings (The Semantic Stream)

We utilize the `all-MiniLM-L6-v2` transformer model:

- **Process**: Converts sentences like *"Severe water-logging at GS Road"* into dense 384-dimensional vectors.
- **Semantic Understanding**: In this space, the vector for "water-logging" is mathematically closer to "flooding" than to "earthquake," enabling intent-based retrieval over keyword matching.

### 4.2 Visual Embeddings (The CLIP Stream)

We utilize OpenAI's **CLIP (Contrastive Language-Image Pre-Training)** model:

- **Why CLIP?** CLIP aligns images and text in the **same** vector space. The vector for an image of a dog and the text "dog" land in nearly the same 512-dimensional position.
- **Application**: A commander typing *"Show me the damage"* (text) retrieves a photo of a landslide (image) — *without any manual tagging*.

### 4.3 Cross-Modal Search Mechanism

The dual-encoding strategy enables a powerful capability:

```
User Query: "Show me flooding in Assam"
     │
     ├──→ MiniLM (384d) ──→ Search text collection ──→ Matching text logs
     │
     └──→ CLIP (512d) ───→ Search image collection ──→ Matching photographs
```

Both search paths execute in parallel, and results are unified into a single context block for the LLM.

---

## 5. Search, Memory & Recommendation Logic

### 5.1 The Lifecycle of a Query

The retrieval process is a multi-step algorithm designed to minimize noise and maximize relevance:

1. **User Input**: The commander types: *"Is there flooding in the north?"*
2. **Dual Encoding**: The query is encoded into both a 384d vector (text search) and a 512d vector (image search).
3. **Parallel Execution**: Both vectors are sent to Qdrant simultaneously via the Retrieval Agent.
4. **Metadata Filtering**: Optional filters by disaster type, severity level, or geographic region narrow results.
5. **Threshold Filtering**:
   - Text Filter (>0.35): Ignores text logs with similarity below threshold.
   - Visual Filter (>0.22): Retrieves images only if confidence exceeds threshold.
   - Negative Constraints: Keywords like "don't show" suppress visual output.
6. **Recency Re-ranking**: Results are re-ranked with a decay factor — memories older than 48 hours receive a 0.7x score multiplier.
7. **Synthesis**: Valid results are formatted with source citations and sent to Gemini 2.5 Flash.

### 5.2 Memory Management

The system implements two distinct memory types:

**Episodic Memory (Short-Term):**
- Every user question and AI response is stored with `role="user"` or `role="assistant"`
- Includes timestamps for recency-based decay
- Enables follow-up questions and conversational context

**Semantic Memory (Long-Term):**
- Core disaster reports stored with `role="system_report"`
- Enriched with structured metadata: disaster_type, severity, region, coordinates, source_agency
- Persists across scenario resets

**Safe Reset Protocol:**
When the commander clicks "Start New Scenario":
```
DELETE FROM user_episodic_memory WHERE role IN ['user', 'assistant']
```
This surgically wipes conversation history while preserving all base disaster intelligence.

### 5.3 Metadata-Driven Recommendations

The Triage Agent automatically enriches each ingested report with:

| Field | Example | Source |
|-------|---------|--------|
| `disaster_type` | "flood" | Keyword classification |
| `severity` | "CRITICAL" | Heuristic scoring (casualty/trapped → CRITICAL) |
| `region` | "Assam" | Location extraction |
| `lat`, `lon` | 26.14, 91.73 | Coordinate lookup |
| `source_agency` | "NDRF Team 4" | Prefix extraction |

This enables **filtered retrieval** — a commander can ask about floods in a specific region and only see relevant results.

---

## 6. Simulated Operational Scenario

To demonstrate grounding capabilities, we present four crisis events detected by the Command Center:

### Phase 1: Structural Integrity Assessment (Gujarat)

- **Commander Query**: *"Report on the structural collapse in Bhuj."*
- **Text Retrieval**: `[Source 1]` Gujarat Control Room: Structural collapse in Bhuj old town; masonry collapse of heritage structure, collapse zone cordoned, USAR canine teams deployed.
  - Score: 0.847 | Severity: HIGH | Region: Bhuj
- **Visual Retrieval**: `bhuj-gujarat-india-january-a-ruined-stone-structure...jpg` (Score: 0.68)
- **AI Synthesis**: "Confirmed structural failure in Bhuj Old Town. Retrieved visual evidence shows a multi-story masonry structure with significant facade collapse. USAR canine teams are deployed; recommend structural shoring before interior search."

### Phase 2: Critical Infrastructure Failure (Karnataka)

- **Commander Query**: *"Show me the road damage report for Yellapur."*
- **Text Retrieval**: `[Source 1]` Karnataka SDRF: Flash flood in Yellapur taluk; bridge SC-B collapsed at span 3, two vehicles submerged.
  - Score: 0.812 | Severity: CRITICAL | Region: Yellapur
- **Visual Retrieval**: `dangerous-landslide-in-road-of-yellapur-karnataka-india.jpg` (Score: 0.72)
- **AI Synthesis**: "Major infrastructure severance detected. Visual evidence confirms complete roadway washout. Route is impassable; alternate routes via Hubli are recommended."

### Phase 3: Multi-State Flood Assessment

- **Commander Query**: *"Status of rescue operations in the flood zones?"*
- **Text Retrieval**: Multiple sources retrieved:
  - `[Source 1]` Kerala Control Room: Emergency water-rescue in Idukki (Score: 0.79)
  - `[Source 2]` Assam Rifles: Riverbank erosion at Dhubri sector (Score: 0.74)
- **Visual Retrieval**: `guwahati-india-october-ndrf-personnel-rescue-people.jpg` (Score: 0.61)
- **AI Synthesis**: "Active rescue operations confirmed in multiple states. NDRF personnel utilizing inflatable boats for urban evacuation. Priority: elderly and injured civilians."

### Phase 4: Industrial Hazard (Telangana)

- **Commander Query**: *"Any chemical or industrial incidents?"*
- **Text Retrieval**: `[Source 1]` Telangana EMS: Industrial fire at Ranga Reddy chemical warehouse; hazmat team on scene, evacuation radius 500m.
  - Score: 0.83 | Severity: CRITICAL | Region: Ranga Reddy
- **AI Synthesis**: "Active HAZMAT situation. Atmospheric monitoring for toxic plume is critical. 500m evacuation radius must be maintained. Medical teams should prepare for chemical exposure cases."

---

## 7. Multi-Agent Architecture

### 7.1 Agent Design

The system implements three specialized agents:

**Triage Agent** (`triage_agent.py`):
- Classifies disaster type using keyword frequency analysis across 9 categories
- Assesses severity using tiered keyword scoring (CRITICAL → HIGH → MEDIUM → LOW)
- Extracts source agency from report prefix
- Runs during ingestion to enrich Qdrant payloads

**Retrieval Agent** (`retrieval_agent.py`):
- Orchestrates parallel search across both Qdrant collections
- Applies metadata filters (disaster type, severity, region)
- Handles negative constraints (image suppression)
- Returns structured context package for synthesis

**Synthesis Agent** (`synthesis_agent.py`):
- Constructs structured prompts with retrieved context and citation protocol
- Invokes Gemini 2.5 Flash for evidence-grounded generation
- Handles API errors gracefully with fallback messaging

### 7.2 Data Flow

```
User Query
    ↓
Retrieval Agent
    ├── encode(query) → MiniLM → text_vector
    ├── encode(query) → CLIP   → image_vector
    ├── search(text_vector, filters) → Qdrant episodic → text_hits
    ├── search(image_vector) → Qdrant multimodal → image_hits
    ├── apply_recency_decay(text_hits)
    └── build_context_package()
    ↓
Synthesis Agent
    ├── build_prompt(context, visual_evidence, query)
    ├── invoke_gemini(prompt) → response
    └── format_with_citations(response)
    ↓
UI Layer
    ├── display_response()
    ├── show_evidence_panel()
    ├── show_reasoning_trace()
    └── store_in_memory()
```

---

## 8. Limitations & Ethics

### 8.1 Known Failure Modes

- **Semantic Ambiguity**: With sparse data, the model may force matches. If asked for "fire" when only "flood" images exist, a low threshold might retrieve irrelevant results. Strict threshold tuning at 0.35 (text) and 0.22 (image) mitigates this.
- **Latency Dependency**: The system relies on external APIs (Gemini, Qdrant Cloud). In grid-down scenarios, inference would fail, though cached models and a local Qdrant instance could serve as fallback.
- **Keyword-Based Triage**: The current triage agent uses keyword heuristics rather than ML classification. Complex or ambiguous reports may be misclassified.

### 8.2 Ethical Considerations

- **Algorithmic Bias**: CLIP is trained on internet-scale data that underrepresents disasters in the Global South. Retrieval accuracy for rural Indian landscapes may be lower than for Western urban settings.
- **Data Privacy**: Uploading disaster victim images to cloud services poses privacy risks. A production deployment must comply with the **Digital Personal Data Protection (DPDP) Act, 2023**, implementing anonymization and edge processing.
- **Human-in-the-Loop**: This system is an **assistance tool, not an automation**. All AI suggestions must be verified by a human commander before deploying resources. The system explicitly labels outputs as retrieved evidence, not ground truth.

---

## 9. Conclusion & Future Scope

The Crisis Intelligence Command Center demonstrates that modern AI can serve as a reliable partner in high-stakes environments. By integrating Multimodal RAG, multi-agent orchestration, and persistent vector memory via Qdrant, we address the critical problem of information fragmentation in disaster response.

**Key Achievements:**
- Dual-stream vector architecture processing 61 text logs + 26 images
- Multi-agent pipeline with automatic triage, retrieval, and synthesis
- Interactive GIS map with geocoded disaster locations
- Analytics dashboard with real-time collection health monitoring
- Evidence-cited responses with full reasoning traces

**Future Roadmap:**
1. **Voice Interface**: Speech-to-Text for radio-based querying
2. **Local Deployment**: Migrate from Gemini API to locally hosted Llama 3 for offline operation
3. **Real-Time Streaming**: Integration with social media APIs and IoT sensor data
4. **Satellite Integration**: Automated damage assessment from satellite imagery
5. **Multi-Language**: Regional language support for diverse disaster communication

This project serves as a foundational step toward a smarter, faster, and more resilient national disaster response infrastructure.

---

*Built with Qdrant, Google Gemini, OpenAI CLIP, and Streamlit.*
