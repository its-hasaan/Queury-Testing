pip install streamlit transformers torch
import streamlit as st
from transformers import pipeline

# Load models (do this once)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="google/long-t5-tglobal-base")

@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

summarizer = load_summarizer()
zero_shot_classifier = load_classifier()

candidate_labels = [
    "Feature request",
    "Complaint",
    "Bug report",
    "General feedback",
    "Sales inquiry"
]

def summarize_email(email_text):
    summary = summarizer(email_text, max_length=500, min_length=15, do_sample=False)
    return summary[0]['summary_text']

def classify_intent(email_text):
    result = zero_shot_classifier(email_text, candidate_labels)
    return result["labels"][0], result["scores"][0], result["labels"], result["scores"]

# Streamlit UI
st.title("Email Insight Tool")

email = st.text_area("Paste your email here:", height=300)

if st.button("Analyze"):
    if email.strip():
        with st.spinner("Summarizing..."):
            summary = summarize_email(email)
        st.subheader("Summary")
        st.write(summary)

        with st.spinner("Classifying intent..."):
            intent, confidence, labels, scores = classify_intent(email)
        st.subheader("Predicted Intent")
        st.write(f"{intent} (confidence: {confidence:.2f})")

        st.subheader("Intent Scores")
        st.table({"Intent": labels, "Score": [f"{s:.2f}" for s in scores]})
    else:
        st.warning("Please enter an email to analyze.")