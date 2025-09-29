import streamlit as st
import pickle
import numpy as np
import random
import time
from datetime import datetime

# -------------------------------------------------
# 🚀 PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="🚀 Commit Classifier 9000",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# 🎨 CUSTOM CSS
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
# 🤖 HEADER
# -------------------------------------------------
st.markdown('<h1 class="main-header">🤖 COMMIT CLASSIFIER 9000</h1>', unsafe_allow_html=True)

# -------------------------------------------------
# SIDEBAR
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
# MAIN CONTENT
# -------------------------------------------------
col1, col2 = st.columns([2,1])

# -------------------- LEFT -------------------
with col1:
    st.subheader("🎯 Enter Your Commit Message")

    if "commit_message" not in st.session_state:
        st.session_state.commit_message = ""

    commit_message = st.text_area(
        "Type your commit message here:",
        key="commit_message",
        height=100,
        placeholder="feat: add quantum encryption module for enhanced security...",
        help="Make it descriptive! The AI loves good commit messages! 🤓"
    )

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

                # Display result
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🎉 PREDICTION: {predicted_class.upper()}</h2>
                    <h3>Confidence: {max_prob*100:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)

                # Confidence breakdown
                st.subheader("📊 Confidence Breakdown")
                for class_name, prob in zip(class_names, probabilities):
                    st.write(f"{class_name}: {prob*100:.1f}%")
                    st.progress(float(prob))

                # Fun facts
                fun_facts = {
                    'enhance file': "✨ Your commit is enhancing the matrix!",
                    'fix issues': "🐛 Squashing bugs like a pro!",
                    'update docs': "📚 Documentation is love, documentation is life!",
                    'refactor payment': "💳 Making money moves!",
                    'add email': "📧 Email me maybe!",
                    'optimize database': "⚡ Speedy Gonzalez mode activated!",
                    'implement jwt': "🔐 Security level: Fort Knox!",
                    'fix memory': "🧠 Brain gains!",
                    'update api': "🌐 Connecting the dots!",
                    'add unit': "🧪 Science, genius style!"
                }
                st.info(f"💡 **Fun Fact**: {fun_facts.get(predicted_class, 'You\'re making the world a better place, one commit at a time!')}")
        else:
            st.warning("⚠️ Please enter a commit message to analyze!")

# -------------------- RIGHT -------------------
with col2:
    st.subheader("🎲 Quick Commit Ideas")
    sample_commits = [
        "fix: resolve memory leak in user service",
        "feat: add jwt authentication for users",
        "docs: update installation guide",
        "refactor: optimize database queries",
        "test: add unit tests for payment module"
    ]
    if st.button("🎪 Random Commit Generator"):
        st.session_state.commit_message = random.choice(sample_commits)

    st.subheader("📈 Today's Stats")
    col3, col4, col5 = st.columns(3)
    with col3: st.metric("Commits Analyzed", "1,337", "+42")
    with col4: st.metric("Accuracy", "98.7%", "+0.2%")
    with col5: st.metric("AI Awesomeness", "∞", "MAX")

# -------------------- FOOTER -------------------
st.markdown("---")
col6, col7, col8 = st.columns(3)
with col6:
    st.write("### 🤖 AI Status")
    st.success("Neural Networks: ONLINE")
    st.info("Quantum Processor: ACTIVATED")
    st.warning("Caffeine Levels: CRITICAL")
with col7:
    st.write("### 🎯 Mission")
    st.write("Classifying commits with the power of **MACHINE LEARNING** and **PURE AWESOMENESS**!")
    st.write(f"🕐 System Time: {datetime.now().strftime('%H:%M:%S')}")
with col8:
    st.write("### 🚀 Next Level")
    if st.button("ACTIVATE TURBO MODE"):
        st.balloons()
        st.success("TURBO MODE ACTIVATED! Preparing for hyperspace...")
        time.sleep(2)
        st.rerun()
