import streamlit as st
import numpy as np
import joblib

from sentence_transformers import SentenceTransformer

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize


# --------- CACHED LOADING ---------
@st.cache_resource
def load_models():
    # Download NLTK tokenizers once
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    # Load sentence embedding model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load trained logistic regression model (binary Human vs AI)
    clf_bin = joblib.load("ai_vs_human_logreg.pkl")

    return embed_model, clf_bin


def compute_stylo(text: str):
    """
    Same stylometric features as in Colab:
    n_words, n_sents, avg_word_len, avg_sent_len, type_token_ratio, n_commas, n_semicolons
    """
    words = word_tokenize(text)
    sents = sent_tokenize(text)

    words_alpha = [w for w in words if w.isalpha()]
    n_words = len(words_alpha)
    n_sents = len(sents) if len(sents) > 0 else 1

    avg_word_len = sum(len(w) for w in words_alpha) / n_words if n_words > 0 else 0
    avg_sent_len = n_words / n_sents if n_sents > 0 else 0
    ttr = len(set(w.lower() for w in words_alpha)) / n_words if n_words > 0 else 0

    n_commas = text.count(",")
    n_semicolons = text.count(";")

    return np.array([
        n_words,
        n_sents,
        avg_word_len,
        avg_sent_len,
        ttr,
        n_commas,
        n_semicolons,
    ], dtype=float).reshape(1, -1)


def make_prediction(text: str, embed_model, clf_bin):
    # Compute embedding
    emb = embed_model.encode([text])  # shape (1, 384)

    # Compute stylometric features
    stylo = compute_stylo(text)       # shape (1, 7)

    # Combine
    X = np.hstack([emb, stylo])       # shape (1, 391)

    # Predict label and probability
    proba = clf_bin.predict_proba(X)[0]
    label = clf_bin.predict(X)[0]

    # clf_bin.classes_ gives order of probabilities
    classes = clf_bin.classes_
    prob_dict = {cls: float(p) for cls, p in zip(classes, proba)}

    return label, prob_dict


# --------- STREAMLIT UI ---------

st.set_page_config(
    page_title="AI vs Human Abstract Detector",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI vs Human Scientific Abstract Detector")
st.write(
    "Paste a scientific abstract below, and this app will estimate whether it was written by a **Human** or **AI (ChatGPT/DeepSeek)** "
    "using a machine-learning model trained on your 150-abstract dataset."
)

st.markdown("---")

# Load models once
with st.spinner("Loading models..."):
    embed_model, clf_bin = load_models()

# Text input
example_text = """Type or paste your abstract here..."""

user_text = st.text_area(
    "Abstract text:",
    height=300,
    value=""
)

if st.button("Analyze abstract"):
    if not user_text.strip():
        st.warning("Please paste or type an abstract first.")
    else:
        with st.spinner("Analyzing..."):
            label, prob_dict = make_prediction(user_text, embed_model, clf_bin)

        st.markdown("### 🔎 Prediction")
        if label == "AI":
            st.error(f"Predicted source: **{label}**")
        else:
            st.success(f"Predicted source: **{label}**")

        st.markdown("### 📊 Probabilities")
        for cls, p in prob_dict.items():
            st.write(f"- **{cls}**: {p:.3f}")
