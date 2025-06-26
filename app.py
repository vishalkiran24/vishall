import streamlit as st
import pickle
from preprocess import preprocess_text

# Load trained model and vectorizer
with open('model.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

st.title("AI-based Resume Screening System" )

uploaded_file = st.file_uploader("Upload your resume (TXT format)", type=["txt"])
if uploaded_file is not None:
    resume_text = uploaded_file.read().decode("utf-8")
    clean_text = preprocess_text(resume_text)
    features = vectorizer.transform([clean_text])
    prediction = clf.predict(features)
    st.write(f"Predicted Job Category: *{prediction[0]}*")