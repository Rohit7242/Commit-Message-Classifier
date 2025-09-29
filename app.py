import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time
import random
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="🚀 Commit Classifier 9000",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 900;
        margin-bottom: 2rem;
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
    .metric-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models
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
        
        st.success("✅ Models loaded successfully!")
        return vectorizer, classifier

# Header
st.markdown('<h1 class="main-header">🤖 COMMIT CLASSIFIER 9000</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚡ Control Panel")
    
    theme = st.selectbox(
        "🎨 Choose Your Vibe:",
        ["Cyberpunk", "Neon Dreams", "Matrix", "Retro Wave", "Space Odyssey"]
    )
    
    intensity = st.slider("💥 Animation Intensity", 1, 10, 5)
    
    if st.button("🚨 EMERGENCY STOP"):
        st.error("SYSTEM HALTED!")
        st.stop()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎯 Enter Your Commit Message")
    
    commit_message = st.text_area(
        "Type your commit message here:",
        height=100,
        placeholder="feat: add quantum encryption module for enhanced security...",
        help="Make it descriptive! The AI loves good commit messages! 🤓"
    )
    
    if st.button("🧠 ANALYZE COMMIT", use_container_width=True):
        if commit_message:
            with st.spinner('🔮 Consulting the AI oracle...'):
                time.sleep(1.5)
                
                vectorizer, classifier = load_models()
                features = vectorizer.transform([commit_message])
                prediction = classifier.predict(features)[0]
                probabilities = classifier.predict_proba(features)[0]
                
                class_names = classifier.classes_
                max_prob = max(probabilities)
                predicted_class = class_names[np.argmax(probabilities)]
                
                # Display results
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🎉 PREDICTION: {predicted_class.upper()}</h2>
                    <h3>Confidence: {max_prob*100:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence breakdown
                st.subheader("📊 Confidence Breakdown")
                for class_name, prob in zip(class_names, probabilities):
                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        st.write(f"**{class_name}:**")
                    with col_b:
                        st.progress(prob, text=f"{prob*100:.1f}%")
                
                # Simple bar chart using native Streamlit
                st.subheader("📈 Probability Chart")
                prob_data = pd.DataFrame({
                    'Category': class_names,
                    'Probability': probabilities
                }).sort_values('Probability', ascending=True)
                
                st.bar_chart(prob_data.set_index('Category')['Probability'])
                
                # Fun facts
                fun_facts = {
                    'enhance file': "✨ Your commit is enhancing the matrix!",
                    'fix issues': "🐛 Squashing bugs like a pro!",
                    'update docs': "📚 Documentation is love, documentation is life!",
                    'refactor payment': "💳 Making money moves!",
                    'add email': "📧 Email me maybe?",
                    'optimize database': "⚡ Speedy Gonzalez mode activated!",
                    'implement jwt': "🔐 Security level: Fort Knox!",
                    'fix memory': "🧠 Brain gains!",
                    'update api': "🌐 Connecting the dots!",
                    'add unit': "🧪 Science time!"
                }
                
                default_fact = "You're making the world a better place, one commit at a time!"
                fact = fun_facts.get(predicted_class, default_fact)
                st.info(f"💡 **Fun Fact**: {fact}")
                
        else:
            st.warning("⚠️ Please enter a commit message to analyze!")

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
        random_commit = random.choice(sample_commits)
        st.text_area("Try this one:", random_commit, height=80)
    
    # Stats section
    st.subheader("📈 Today's Stats")
    
    st.metric("Commits Analyzed", "1,337", "+42")
    st.metric("Accuracy", "98.7%", "+0.2%")
    st.metric("AI Awesomeness", "∞", "MAX")
    
    # Leaderboard without Plotly
    st.subheader("🏆 Top Categories")
    categories = ['fix issues', 'enhance file', 'update docs', 'refactor payment', 'add email']
    counts = [45, 32, 28, 19, 15]
    
    leaderboard_data = pd.DataFrame({
        'Category': categories,
        'Count': counts
    })
    
    st.dataframe(leaderboard_data.sort_values('Count', ascending=False), 
                 use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
col3, col4, col5 = st.columns(3)

with col3:
    st.write("### 🤖 AI Status")
    st.success("Neural Networks: ONLINE")
    st.info("Quantum Processor: ACTIVATED")

with col4:
    st.write("### 🎯 Mission")
    st.write("Classifying commits with **MACHINE LEARNING** power! 🚀")

with col5:
    st.write("### 🚀 Next Level")
    if st.button("ACTIVATE TURBO MODE"):
        st.balloons()
        st.success("TURBO MODE ACTIVATED! 🚀")

# Secret Easter egg
if st.sidebar.checkbox("👑 Enable Royal Mode"):
    st.sidebar.success("Your majesty! The AI bows before you! 👑")
    st.balloons()
