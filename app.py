import streamlit as st
import pickle

# Load the trained model and vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("commit_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit app
st.set_page_config(page_title="Commit Message Classifier", page_icon="💻", layout="centered")

st.title("💻 Commit Message Classifier")
st.markdown("Enter a commit message to check if it’s a **good quality commit message**.")

# Text input from user
user_input = st.text_area("✍️ Commit Message", placeholder="e.g., Fix bug in user login validation")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a commit message before classifying.")
    else:
        # Transform input using vectorizer
        X = vectorizer.transform([user_input])

        # Prediction
        pred = model.predict(X)[0]

        # Show result
        if pred == 1:
            st.success("✅ This looks like a **good commit message**!")
        else:
            st.error("❌ This seems like a **low-quality commit message**. Try making it clearer and more descriptive.")
