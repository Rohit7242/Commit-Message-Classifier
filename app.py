import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time
import random
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------
# 🚀 PAGE CONFIG (must be first Streamlit command)
# -------------------------------------------------
st.set_page_config(
    page_title="🚀 Commit Classifier 9000",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# 🎨 CUSTOM CSS STYLES
# -------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 4rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 900;
        margin-bottom: 2rem;
        animation: rainbow 2s ease-in-out infinite;
    }
    @keyframes rainbow {
        0% { filter: hue-rotate(0deg); }
        100% { filter: hue-rotate(360deg); }
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff0000, #ffff00, #00ff00, #00ffff, #0000ff, #ff00ff);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# 📦 LOAD MODELS
# -------------------------------------------------
@st.cache_resource
def load_models():
    with st.spinner('🚀 Activating neural networks...'):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('commit_classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        st.balloons()
        return vectorizer, classifier

# -------------------------------------------------
# 🤖 MAIN HEADER
# -------------------------------------------------
st.markdown('<h1 class="main-header">🤖 COMMIT CLASSIFIER 9000</h1>', unsafe_allow_html=True)

# -------------------------------------------------
# 🎛️ SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.title("⚡ Control Panel")

    theme = st.selectbox("🎨 Choose Your Vibe:", ["Cyberpunk", "Neon Dreams", "Matrix", "Retro Wave", "Space Odyssey"])
    intensity = st.slider("💥 Animation Intensity", 1, 10, 5)
    sound_effects = st.checkbox("🔊 Enable Sound Effects", value=True)

    if st.checkbox("🕵️‍♂️ Activate Secret Agent Mode"):
        st.success("Mission accepted! Classifying with extra stealth...")

    if st.button("🚨 EMERGENCY STOP"):
        st.error("SYSTEM HALTED!")
        st.stop()

# -------------------------------------------------
# 🎯 MAIN CONTENT
# -------------------------------------------------
col1, col2 = st.columns([2, 1])

# -------------------- LEFT SIDE -------------------
with col1:
    st.subheader("🎯 Enter Your Commit Message")

    # Text input stored in session state (so Random Generator works)
    if "commit_message" not in st.session_state:
        st.session_state.commit_message = ""

    commit_message = st.text_area(
        "Type your commit message here:",
        key="commit_message",
        height=100,
        placeholder="feat: add quantum encryption module for enhanced security...",
        help="Make it descriptive! The AI loves good commit messages! 🤓"
    )

    # Prediction button
    if st.button("🧠 ANALYZE COMMIT", use_container_width=True):
        if commit_message.strip():
            with st.spinner('🔮 Consulting the AI oracle...'):
                time.sleep(1.5)

                vectorizer, classifier = load_models()
                features = vectorizer.transform([commit_message])
                probabilities = classifier.predict_proba(features)[0]
                class_names = classifier.classes_
                predicted_class = class_names[np.argmax(probabilities)]
                max_prob = max(probabilities)
