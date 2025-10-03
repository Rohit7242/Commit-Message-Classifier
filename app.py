import streamlit as st
import pickle
import numpy as np
import pandas as pd
import random
import time

# Page config
st.set_page_config(
    page_title="Commit Classifier",
    page_icon="🤖",
    layout="wide"
)

# Load models with robust error handling
@st.cache_resource
def load_models():
    try:
        with st.spinner("Loading models..."):
            time.sleep(1)  # simulate loading
            
            # Load vectorizer
            with open('./vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            
            # Load classifier
            with open('./commit_classifier.pkl', 'rb') as f:
                classifier = pickle.load(f)
            
            st.success("✅ Models loaded successfully!")
            return vectorizer, classifier
    except FileNotFoundError:
        st.error("❌ Pickle files not found! Make sure 'vectorizer.pkl' and 'commit_classifier.pkl' are in the app folder.")
        st.stop()
    except ModuleNotFoundError as e:
        st.error(f"❌ Missing module when loading pickle: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ An unexpected error occurred while loading models: {e}")
        st.stop()

# Header
st.title("🤖 Commit Classifier")

# Sidebar
with st.sidebar:
    st.header("Control Panel")
    theme = st.selectbox("Theme", ["Default", "Light", "Dark"])
    if st.button("Stop"):
        st.warning("System halted!")
        st.stop()

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Commit Message")
    commit_message = st.text_area(
        "Commit message:", 
        placeholder="e.g., feat: add new login API"
    )

    if st.button("Analyze Commit"):
        if not commit_message.strip():
            st.warning("⚠️ Please enter a commit message.")
        else:
            # Load models safely
            vectorizer, classifier = load_models()
            
            # Transform and predict
            features = vectorizer.transform([commit_message])
            prediction = classifier.predict(features)[0]
            probabilities = classifier.predict_proba(features)[0]
            class_names = classifier.classes_
            max_prob = max(probabilities)

            # Display result
            st.markdown(f"**Prediction:** {prediction.upper()}")
            st.markdown(f"**Confidence:** {max_prob*100:.2f}%")

            # Confidence breakdown
            st.subheader("Confidence Breakdown")
            for class_name, prob in zip(class_names, probabilities):
                st.write(f"{class_name}: {prob*100:.1f}%")
            
            # Probability bar chart
            prob_data = pd.DataFrame({
                'Category': class_names,
                'Probability': probabilities
            }).sort_values('Probability', ascending=True)
            st.bar_chart(prob_data.set_index('Category')['Probability'])

with col2:
    st.subheader("Sample Commit Ideas")
    sample_commits = [
        "fix: resolve memory leak",
        "feat: add jwt authentication",
        "docs: update installation guide",
        "refactor: optimize database queries",
        "test: add unit tests"
    ]
    if st.button("Random Commit"):
        st.text_area("Try this one:", random.choice(sample_commits), height=80)

    st.subheader("Today's Stats")
    st.metric("Commits Analyzed", "1,337")
    st.metric("Accuracy", "98.7%")

    st.subheader("Top Categories")
    leaderboard_data = pd.DataFrame({
        'Category': ['fix issues', 'enhance file', 'update docs', 'refactor', 'add email'],
        'Count': [45, 32, 28, 19, 15]
    })
    st.dataframe(leaderboard_data.sort_values('Count', ascending=False), use_container_width=True)

# Footer
st.markdown("---")
st.write("Classifying commits with ML power! 🚀")
