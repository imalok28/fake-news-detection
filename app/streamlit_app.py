import os
import re
import requests
import streamlit as st

# Page configurations
st.set_page_config(
    page_title="VeriTalk | Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Read API URL from environment variable, fallback to localhost
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #718096;
        margin-bottom: 2rem;
    }
    
    .result-card-real {
        background-color: rgba(46, 204, 113, 0.1);
        border: 2px solid #2ecc71;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    
    .result-card-fake {
        background-color: rgba(231, 76, 60, 0.1);
        border: 2px solid #e74c3c;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-top: 10px;
    }
    
    .highlight-container {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        padding: 20px;
        border-radius: 8px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        line-height: 1.8;
        max-height: 450px;
        overflow-y: auto;
        color: #2d3748;
    }
    
    .legend-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 10px;
    }
    
    .legend-real {
        background-color: rgba(46, 204, 113, 0.2);
        color: #27ae60;
        border: 1px solid #2ecc71;
    }
    
    .legend-fake {
        background-color: rgba(231, 76, 60, 0.2);
        color: #c0392b;
        border: 1px solid #e74c3c;
    }
    
    .bar-label {
        font-weight: 500;
        font-size: 0.9rem;
        width: 120px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .custom-bar-real {
        background: linear-gradient(90deg, #2ecc71 0%, #27ae60 100%);
        height: 18px;
        border-radius: 4px;
    }
    
    .custom-bar-fake {
        background: linear-gradient(90deg, #e74c3c 0%, #c0392b 100%);
        height: 18px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

def highlight_text(text: str, influential_words):
    """Highlights keywords in the text with matching red/green background colors."""
    if not text:
        return ""
    
    # Map words to their details
    word_map = {w["word"].lower(): w for w in influential_words}
    
    # Split text by word boundaries
    tokens = re.split(r"(\b\w+\b)", text)
    highlighted_tokens = []
    
    for token in tokens:
        token_lower = token.lower()
        if token_lower in word_map:
            info = word_map[token_lower]
            if info["influence"] == "REAL":
                bg_color = "rgba(46, 204, 113, 0.25)"
                border_color = "#2ecc71"
                text_color = "#27ae60"
            else:
                bg_color = "rgba(231, 76, 60, 0.25)"
                border_color = "#e74c3c"
                text_color = "#c0392b"
                
            highlighted_tokens.append(
                f'<span style="background-color: {bg_color}; border: 1px solid {border_color}; '
                f'color: {text_color}; padding: 2px 4px; border-radius: 3px; font-weight: 600;" '
                f'title="Weight: {info["weight"]:.4f} ({info["influence"]})">{token}</span>'
            )
        else:
            highlighted_tokens.append(token)
            
    return "".join(highlighted_tokens)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/news.png", width=80)
    st.markdown("### System Settings")
    api_url = st.text_input("FastAPI Endpoint URL", value=API_URL)
    
    st.markdown("---")
    st.markdown("### About VeriTalk")
    st.write(
        "VeriTalk uses a dual-engine architecture to detect misleading news. "
        "It evaluates articles using TF-IDF Machine Learning baselines and "
        "fine-tuned deep Transformer models (DistilBERT). Explainability is "
        "powered by LIME."
    )
    
    # Fetch health status of backend
    try:
        r = requests.get(f"{api_url}/health", timeout=3)
        if r.status_code == 200:
            health = r.json()
            if health.get("model_loaded"):
                st.success(f"● Connected ({health.get('model_type')})")
            else:
                st.warning("● Connected (No Model)")
        else:
            st.error("● API Error")
    except Exception:
        st.error("● Disconnected from API")

# Main Page
st.markdown('<div class="main-header">📰 VeriTalk News Verifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Production-Grade Fake News Detection and Explainability System</div>', unsafe_allow_html=True)

# Input Layout
article_text = st.text_area(
    "Paste the full news article text below (minimum 10 characters):",
    height=250,
    placeholder="Type or paste the news article content here..."
)

col_check, col_clear = st.columns([1, 8])
with col_check:
    check_btn = st.button("Check Article", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear Input", use_container_width=False):
        st.rerun()

if check_btn:
    if not article_text or len(article_text.strip()) < 10:
        st.error("Please enter a valid news article containing at least 10 characters.")
    else:
        with st.spinner("Analyzing article text and running explanations..."):
            try:
                response = requests.post(
                    f"{api_url}/predict",
                    json={"text": article_text},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    label = data["label"]
                    score = data["authenticity_score"]
                    words = data["top_influential_words"]
                    model_used = data["model_used"]
                    
                    # Columns for results
                    col_result, col_weights = st.columns([1, 1])
                    
                    with col_result:
                        st.subheader("Authenticity Report")
                        
                        card_class = "result-card-real" if label == "REAL" else "result-card-fake"
                        label_text = "VERIFIED REAL" if label == "REAL" else "CONFIRMED FAKE"
                        desc_text = (
                            "This article matches the linguistic footprint of factual and official reporting." 
                            if label == "REAL" else 
                            "This article exhibits strong indicators of sensationalism or fabricated news."
                        )
                        
                        st.markdown(f"""
                        <div class="{card_class}">
                            <h3 style="margin: 0; color: inherit;">{label_text}</h3>
                            <p style="margin: 8px 0 0 0; font-size: 0.95rem; opacity: 0.8;">{desc_text}</p>
                            <div class="metric-value">{score * 100:.1f}%</div>
                            <div style="font-size: 0.85rem; opacity: 0.7; margin-top: 5px;">Confidence Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add a clean custom visual bar for the score
                        color = "#2ecc71" if label == "REAL" else "#e74c3c"
                        st.markdown(f"""
                        <div style="width: 100%; background-color: #e2e8f0; height: 10px; border-radius: 5px; overflow: hidden; margin-bottom: 24px;">
                            <div style="width: {score * 100}%; background-color: {color}; height: 100%;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.info(f"Analysis engine utilized: **{model_used}**")
                        
                    with col_weights:
                        st.subheader("Key Influential Words")
                        st.write("These words/phrases had the highest impact on the decision:")
                        
                        if not words:
                            st.write("No strong features detected.")
                        else:
                            # Render custom horizontal bar chart for LIME features
                            max_weight = max([abs(w["weight"]) for w in words]) if words else 1.0
                            for w in words:
                                word_name = w["word"]
                                weight_val = w["weight"]
                                influence = w["influence"]
                                
                                # Percentage width relative to max weight
                                width_pct = min(100, int((abs(weight_val) / max_weight) * 100))
                                bar_class = "custom-bar-real" if influence == "REAL" else "custom-bar-fake"
                                
                                st.markdown(f"""
                                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                    <div class="bar-label" title="{word_name}">{word_name}</div>
                                    <div style="flex-grow: 1; margin: 0 10px; background-color: #edf2f7; border-radius: 4px; height: 18px;">
                                        <div class="{bar_class}" style="width: {width_pct}%;"></div>
                                    </div>
                                    <div style="font-size: 0.85rem; font-weight: 600; width: 50px; text-align: right;">{weight_val:+.3f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            st.markdown("""
                            <div style="margin-top: 15px; display: flex;">
                                <span class="legend-badge legend-real">REAL (+ weight)</span>
                                <span class="legend-badge legend-fake">FAKE (- weight)</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Highlighted original text
                    st.markdown("---")
                    st.subheader("Highlighted Article Text")
                    st.write("Review where the influential words appear in your article:")
                    
                    highlighted_html = highlight_text(article_text, words)
                    st.markdown(f'<div class="highlight-container">{highlighted_html}</div>', unsafe_allow_html=True)
                    
                else:
                    st.error(f"Error from prediction API (Status {response.status_code}): {response.text}")
                    
            except Exception as e:
                st.error(f"Failed to connect to prediction API. Make sure the backend server is running. (Error: {e})")
# streamlit run app/streamlit_app.py