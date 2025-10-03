import streamlit as st
import re
import numpy as np
import pandas as pd

# ------------------------------
# Helper function to clean text
# ------------------------------
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"[^a-z0-9\s]", "", text)  # remove special chars
    text = text.strip()
    return text

# ------------------------------
# Load models safely
# ------------------------------
@st.cache_resource
def load_models():
    try:
        model = joblib.load("commit_classifier.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Pickle files not found! Make sure 'commit_classifier.pkl' and 'vectorizer.pkl' are in the app folder.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(
    page_title="Commit Quality Classifier",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Commit Quality Classifier")
st.write("Type your commit message below to see if it is high or low quality.")

# Load models
model, vectorizer = load_models()

# User input
commit_msg = st.text_area("Enter your commit message:", height=100, placeholder="feat: add user authentication...")

if st.button("Predict Quality"):
    if commit_msg.strip() == "":
        st.warning("⚠️ Please enter a commit message!")
    else:
        cleaned_msg = clean_text(commit_msg)
        msg_vec = vectorizer.transform([cleaned_msg])
        pred = model.predict(msg_vec)[0]
        result = "High Quality ✅" if pred == 1 else "Low Quality 🚩"
        st.success(f"Result --> {result}")

# Optional: Show multiple predictions in a loop-like style
st.subheader("Try multiple commit messages")
multi_msgs = st.text_area("Enter one commit per line:", height=150, placeholder="feat: add login\nfix: resolve bug in payment\n...")
if st.button("Predict Multiple"):
    if multi_msgs.strip() == "":
        st.warning("⚠️ Please enter at least one commit message!")
    else:
        lines = multi_msgs.strip().split("\n")
        results = []
        for msg in lines:
            cleaned_msg = clean_text(msg)
            vec = vectorizer.transform([cleaned_msg])
            pred = model.predict(vec)[0]
            results.append({"Commit": msg, "Prediction": "High Quality ✅" if pred==1 else "Low Quality 🚩"})
        st.table(pd.DataFrame(results))
