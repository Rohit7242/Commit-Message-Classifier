import streamlit as st
import joblib  # or pickle

# Load model and vectorizer
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("commit_classifier.pkl")

# Streamlit app
st.set_page_config(page_title="Commit Classifier", page_icon="📝")

st.title("Commit Message Quality Classifier")
st.markdown("Enter a commit message to check if it's high-quality or low-quality.")

# Input textbox
commit_msg = st.text_area("Commit Message", "")

# Predict button
if st.button("Predict"):
    if commit_msg.strip() == "":
        st.warning("Please enter a commit message!")
    else:
        # Transform and predict
        msg_vec = vectorizer.transform([commit_msg])
        prediction = model.predict(msg_vec)[0]
        st.success(f"Prediction: {prediction}")
