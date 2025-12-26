#!/usr/bin/env python3
"""
Medical RAG Frontend - Streamlit UI
====================================
Interactive chat interface for the transplant RAG system.
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import Optional

# Page config
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .source-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .metric-card {
        background-color: #e9ecef;
        padding: 0.5rem;
        border-radius: 4px;
        text-align: center;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


def get_token(username: str, password: str) -> Optional[str]:
    """Authenticate and get JWT token"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/token",
            json={"username": username, "password": password},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()["access_token"]
        return None
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return None


def query_api(token: str, query: str, model: str = "gemma3:1b", temperature: float = 0.1, 
              max_tokens: int = 512, top_k: int = 4, answer_mode: str = "clinical",
              confidence_threshold: float = 0.55) -> Optional[dict]:
    """Send query to RAG API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "query": query,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_k": top_k,
                "answer_mode": answer_mode,
                "confidence_threshold": confidence_threshold
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            st.error("Session expired. Please login again.")
            st.session_state.clear()
            st.rerun()
        else:
            st.error(f"Query failed: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Query error: {e}")
        return None


def get_health_status() -> dict:
    """Check API health"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "unreachable"}
    except:
        return {"status": "unreachable"}


def render_confidence_badge(confidence: str, score: float):
    """Render colored confidence badge"""
    if confidence == "High":
        css_class = "confidence-high"
        emoji = "🟢"
    elif confidence == "Medium":
        css_class = "confidence-medium"
        emoji = "🟡"
    else:
        css_class = "confidence-low"
        emoji = "🔴"
    
    return f'{emoji} <span class="{css_class}">{confidence}</span> ({score:.3f})'


def render_sources(sources: list):
    """Render source citations"""
    st.markdown("### 📚 Sources")
    for idx, source in enumerate(sources, 1):
        with st.container():
            st.markdown(f"""
            <div class="source-card">
                <strong>[{idx}] {source['document']}</strong><br>
                <em>Section:</em> {source.get('section', 'N/A')}<br>
                <em>Relevance:</em> {source['similarity_score']:.3f} | 
                <em>Tokens:</em> {source.get('token_count', 0)}<br>
                <details>
                <summary>Preview</summary>
                {source.get('text_preview', 'No preview available')}
                </details>
            </div>
            """, unsafe_allow_html=True)


def login_page():
    """Render login page"""
    st.markdown('<div class="main-header">🏥 Medical RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown("### Transplant Knowledge Base")
    
    # Health status in sidebar
    with st.sidebar:
        st.markdown("### System Status")
        health = get_health_status()
        status = health.get("status", "unknown")
        
        if status == "healthy":
            st.success("✅ System Online")
        elif status == "degraded":
            st.warning("⚠️ Degraded Performance")
        else:
            st.error("❌ System Offline")
        
        if status != "unreachable":
            st.metric("Indexed Chunks", health.get("chunks_indexed", 0))
            st.metric("ChromaDB", "✓" if health.get("chroma_connected") else "✗")
            st.metric("Ollama", "✓" if health.get("model_available") else "✗")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="admin@transplant.ai")
            password = st.text_input("Password", type="password", placeholder="admin123")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    with st.spinner("Authenticating..."):
                        token = get_token(username, password)
                        if token:
                            st.session_state.token = token
                            st.session_state.username = username
                            st.success("✅ Login successful!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
        
        st.markdown("---")
        st.info("""
        **Demo Credentials:**
        - Admin: `admin@transplant.ai` / `admin123`
        - Researcher: `researcher@transplant.ai` / `research123`
        """)


def chat_page():
    """Render main chat interface"""
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="main-header">🏥 Medical RAG Assistant</div>', unsafe_allow_html=True)
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        st.markdown("---")
        
        st.markdown("### ⚙️ Query Settings")
        model = st.selectbox("Model", ["gemma3:1b", "phi3:mini"], index=0)
        answer_mode = st.selectbox(
            "Answer Mode",
            ["brief", "clinical", "detailed"],
            index=1,
            help="Brief: 2-3 sentences | Clinical: Structured with bullets | Detailed: Comprehensive"
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
        max_tokens = st.slider("Max Tokens", 128, 2048, 512, 64)
        top_k = st.slider("Top K Sources", 1, 10, 4, 1)
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.55, 0.05,
            help="Reject answers below this confidence (0.55 = hospital-grade)"
        )
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        health = get_health_status()
        status_emoji = "✅" if health.get("status") == "healthy" else "⚠️"
        st.metric("Status", f"{status_emoji} {health.get('status', 'unknown').title()}")
        st.metric("Chunks", health.get("chunks_indexed", 0))
        
        st.markdown("---")
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                st.markdown(message["content"])
                
                # Display metadata for assistant responses
                if "metadata" in message:
                    meta = message["metadata"]
                    
                    # Metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f'<div class="metric-card">⏱️<br>{meta["total_time"]:.2f}s</div>', 
                                  unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div class="metric-card">📝<br>{meta["total_tokens"]} tokens</div>', 
                                  unsafe_allow_html=True)
                    with col3:
                        confidence_html = render_confidence_badge(meta["confidence"], meta["confidence_score"])
                        st.markdown(f'<div class="metric-card">{confidence_html}</div>', 
                                  unsafe_allow_html=True)
                    with col4:
                        st.markdown(f'<div class="metric-card">📚<br>{meta["chunks_used"]} sources</div>', 
                                  unsafe_allow_html=True)
                    
                    # Sources
                    with st.expander(f"📚 View {len(meta['sources'])} Source Citations"):
                        render_sources(meta["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask about transplant medicine..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching knowledge base..."):
                result = query_api(
                    st.session_state.token, 
                    prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_k=top_k,
                    answer_mode=answer_mode,
                    confidence_threshold=confidence_threshold
                )
                
                if result:
                    # Display answer
                    st.markdown(result["answer"])
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f'<div class="metric-card">⏱️<br>{result["total_time"]:.2f}s</div>', 
                                  unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div class="metric-card">📝<br>{result["total_tokens"]} tokens</div>', 
                                  unsafe_allow_html=True)
                    with col3:
                        confidence_html = render_confidence_badge(result["confidence"], result["confidence_score"])
                        st.markdown(f'<div class="metric-card">{confidence_html}</div>', 
                                  unsafe_allow_html=True)
                    with col4:
                        st.markdown(f'<div class="metric-card">📚<br>{result["chunks_used"]} sources</div>', 
                                  unsafe_allow_html=True)
                    
                    # Sources
                    with st.expander(f"📚 View {len(result['sources'])} Source Citations"):
                        render_sources(result["sources"])
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "metadata": {
                            "confidence": result["confidence"],
                            "confidence_score": result["confidence_score"],
                            "total_time": result["total_time"],
                            "total_tokens": result["total_tokens"],
                            "chunks_used": result["chunks_used"],
                            "sources": result["sources"]
                        }
                    })
                else:
                    error_msg = "Failed to get response from API. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


def main():
    """Main app entry point"""
    # Check if logged in
    if "token" not in st.session_state:
        login_page()
    else:
        chat_page()


if __name__ == "__main__":
    main()