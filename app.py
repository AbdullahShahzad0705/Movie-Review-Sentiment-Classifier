"""
Streamlit app: live sentiment prediction + model report.
Run locally:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from preprocess import clean_text

ART = Path("artifacts")

st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading model ...")
def load_model():
    return joblib.load(ART / "model.joblib")


@st.cache_data
def load_metrics():
    with open(ART / "metrics.json") as f:
        return json.load(f)


if not (ART / "model.joblib").exists():
    st.error("`artifacts/model.joblib` not found. Run `python train.py` first.")
    st.stop()

bundle = load_model()
tfidf, rf, class_names = bundle["tfidf"], bundle["rf"], bundle["class_names"]
metrics = load_metrics()

st.title("🎬 Movie Review Sentiment Classifier")
st.caption("Random Forest + TF-IDF trained on the IMDB 50k reviews dataset")

tab_predict, tab_results, tab_features = st.tabs(
    ["🔮 Predict", "📊 Model results", "🔑 Top features"])

# ---------------------------------------------------------------- Predict
with tab_predict:
    examples = {
        "Positive example": "An absolutely brilliant film. The acting was superb, the story gripping, "
                            "and the ending left me in tears. Highly recommended!",
        "Negative example": "A complete waste of time. The plot made no sense, the acting was terrible "
                            "and I was bored from start to finish.",
    }
    choice = st.selectbox("Load an example (optional)", ["—"] + list(examples))
    default = examples.get(choice, "")
    text = st.text_area("Write or paste a movie review", value=default, height=180)

    if st.button("Classify", type="primary"):
        cleaned = clean_text(text)
        if not cleaned.strip():
            st.warning("Please enter a review with some real words in it.")
        else:
            X = tfidf.transform([cleaned])
            proba = rf.predict_proba(X)[0]
            pred = int(np.argmax(proba))
            label = class_names[pred]
            (st.success if label == "positive" else st.error)(
                f"Prediction: **{label.upper()}**  ({proba[pred]:.1%} confidence)")
            st.bar_chart(pd.DataFrame({"probability": proba}, index=class_names))
            with st.expander("What the model actually saw (after preprocessing)"):
                st.write(cleaned)

# ---------------------------------------------------------------- Results
with tab_results:
    rep = metrics["report"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", f"{metrics['test_accuracy']:.2%}")
    c2.metric("Weighted F1", f"{rep['weighted avg']['f1-score']:.2%}")
    c3.metric("Train accuracy", f"{metrics['train_accuracy']:.2%}")
    c4.metric("Best CV F1", f"{metrics['search']['best_cv_f1_weighted']:.2%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Precision / recall / F1 (test set)")
        rows = {k: rep[k] for k in class_names + ["weighted avg"]}
        df = pd.DataFrame(rows).T[["precision", "recall", "f1-score", "support"]]
        st.dataframe(df.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                      "f1-score": "{:.3f}", "support": "{:.0f}"}))
        st.subheader("Settings used")
        st.json({"tfidf": metrics["tfidf_params"],
                 "best_random_forest_params": metrics["search"]["best_params"]})
    with right:
        st.subheader("Confusion matrix")
        st.image(str(ART / "confusion_matrix.png"))

    st.subheader("Dataset overview")
    st.image(str(ART / "eda.png"))

# ---------------------------------------------------------------- Features
with tab_features:
    st.subheader("Top 20 most important features")
    st.image(str(ART / "feature_importance.png"))
    st.dataframe(pd.DataFrame(metrics["top_features"]), width="stretch")
