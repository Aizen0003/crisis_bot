"""
Chat interface component for the Crisis Intelligence Command Center.
Handles message display, user input, and the full RAG interaction loop.
"""

import streamlit as st
from src.graph import run_crisis_graph
import os


def init_chat_state():
    """Initialize session state for chat."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_message_history():
    """Render all previous messages with their images and sources."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show retrieved images
            if message.get("images"):
                for img_info in message["images"]:
                    if os.path.exists(img_info["filename"]):
                        st.image(
                            img_info["filename"],
                            caption=f"📸 {img_info['description']} (Score: {img_info['score']:.2f})",
                            use_container_width=True,
                        )
            
            # Show evidence panel
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("🔍 Evidence & Reasoning Trace", expanded=False):
                    _render_evidence_panel(message)


def _render_evidence_panel(message: dict):
    """Render the expandable evidence panel for an assistant message."""
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("**📄 Text Sources Retrieved:**")
        if message.get("sources"):
            for i, src in enumerate(message["sources"], 1):
                severity = src.get("metadata", {}).get("severity", "")
                icon = ""
                if severity == "CRITICAL":
                    icon = "🔴"
                elif severity == "HIGH":
                    icon = "🟠"
                elif severity == "MEDIUM":
                    icon = "🟡"
                elif severity == "LOW":
                    icon = "🟢"
                    
                st.caption(
                    f"{icon} **[Source {i}]** (Score: {src['score']:.3f})\n"
                    f"{src['text'][:150]}..."
                )
        else:
            st.caption("No relevant text sources found.")
    
    with col2:
        st.markdown("**🖼️ Visual Evidence:**")
        if message.get("images"):
            for img_info in message["images"]:
                if os.path.exists(img_info["filename"]):
                    st.image(img_info["filename"], width=200)
                    st.caption(f"Match: {img_info['score']:.2f} — {img_info['description']}")
        else:
            st.caption("No matching images found.")
    
    # Reasoning trace
    if message.get("reasoning_trace"):
        st.markdown("**🧠 Reasoning Trace:**")
        st.markdown(
            f'<div class="reasoning-trace">{message["reasoning_trace"]}</div>',
            unsafe_allow_html=True,
        )


def handle_chat_input(disaster_filter: str = None, severity_filter: str = None,
                      region_filter: str = None):
    """
    Handle user chat input, execute RAG pipeline, and display response.
    """
    prompt = st.chat_input("Ask about a disaster scenario...")
    
    if not prompt:
        return
    
    # ── Display user message ──────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # ── Run the LangGraph RAG workflow (retrieval + synthesis + memory) ──
    with st.spinner("🧠 Running crisis intelligence workflow..."):
        retrieval_results = run_crisis_graph(
            query=prompt,
            filters={
                "disaster": disaster_filter,
                "severity": severity_filter,
                "region": region_filter,
            },
        )
    response_text = retrieval_results["response"]

    # Surface any actionable backend error to the operator.
    if retrieval_results.get("error"):
        st.error(retrieval_results["error"])

    # ── Display assistant response ────────────────────────────────
    with st.chat_message("assistant"):
        st.markdown(response_text)
        
        # Show images if relevant and not suppressed
        image_infos = []
        if retrieval_results["show_images"] and retrieval_results["image_results"]:
            for img in retrieval_results["image_results"]:
                filename = img["filename"]
                if os.path.exists(filename):
                    st.image(
                        filename,
                        caption=f"📸 {img['description']} (Score: {img['score']:.2f})",
                        use_container_width=True,
                    )
                    image_infos.append(img)
        elif not retrieval_results["show_images"]:
            st.info("📷 Image display suppressed by user request.")
        
        # Evidence panel
        with st.expander("🔍 Evidence & Reasoning Trace", expanded=False):
            _render_evidence_panel({
                "sources": retrieval_results["text_results"],
                "images": image_infos,
                "reasoning_trace": retrieval_results["reasoning_trace"],
            })
    
    # ── Store in session state ────────────────────────────────────
    # (Episodic memory in Qdrant is persisted inside the LangGraph workflow's
    # persist_memory node, so we do not write it again here.)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "images": image_infos,
        "sources": retrieval_results["text_results"],
        "reasoning_trace": retrieval_results["reasoning_trace"],
    })

    return retrieval_results
