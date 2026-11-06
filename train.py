"""
Random Forest text classifier (IMDB movie-review sentiment).

Pipeline:
  1. Load data + exploratory analysis
  2. Preprocess text (lowercase, strip HTML/punctuation/numbers, stop words, lemmatize)
  3. TF-IDF features
  4. Stratified 80/20 split (random_state=42)
  5. RandomizedSearchCV over RandomForest hyper-parameters (cross-validated)
  6. Evaluate on the held-out test set (accuracy, P/R/F1, confusion matrix)
  7. Top-20 feature importances
  8. Save model + metrics + plots for the Streamlit app

Run:  python train.py                 # default settings
      python train.py --tune-sample 0 # tune on the full training set (slow)
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")                       # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from preprocess import clean_text

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
DATA_PATH = Path("data/IMDB Dataset.csv")
CLEAN_CACHE = Path("data/cleaned.csv")      # cache so preprocessing runs once
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(exist_ok=True)

# TF-IDF settings (documented in README and shown in the app)
TFIDF_PARAMS = dict(
    max_features=5000,      # keep the 5k most frequent terms -> small, fast model
    ngram_range=(1, 2),     # unigrams + bigrams ("not good", "waste time")
    min_df=5,               # ignore terms in < 5 documents (typos, rare names)
    max_df=0.9,             # ignore terms in > 90 % of documents
    sublinear_tf=True,      # 1 + log(tf) dampens very frequent words
)

# Hyper-parameter search space
PARAM_GRID = {
    "rf__n_estimators": [100, 200, 300],
    "rf__max_depth": [20, 40, 80, None],
    "rf__min_samples_split": [2, 5, 10],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cv", type=int, default=3, help="CV folds")
    p.add_argument("--n-iter", type=int, default=8,
                   help="random combinations to try")
    p.add_argument("--tune-sample", type=int, default=10000,
                   help="rows of the TRAIN set used for the search "
                        "(0 = use all). The final model is always refit on "
                        "the full training set.")
    return p.parse_args()


# ----------------------------------------------------------------------------
# 1. Load + EDA
# ----------------------------------------------------------------------------
def load_and_explore():
    df = pd.read_csv(DATA_PATH)
    print("=" * 70, "\n1. EXPLORATORY ANALYSIS\n" + "=" * 70)
    print(f"Shape            : {df.shape}")
    print(f"Columns          : {df.columns.tolist()}")
    print(f"Missing values   :\n{df.isna().sum().to_string()}")
    n_dup = df.duplicated().sum()
    print(f"Duplicate rows   : {n_dup}")
    print(f"Class distribution:\n{df['sentiment'].value_counts().to_string()}")
    words = df["review"].str.split().str.len()
    print(f"Words per review : mean={words.mean():.0f}, median={words.median():.0f}, "
          f"max={words.max()}")

    # Duplicates would leak between train and test -> drop them.
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"After dropping duplicates: {df.shape}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    df["sentiment"].value_counts().plot.bar(ax=ax[0], color=["#4c9f70", "#d9534f"])
    ax[0].set_title("Class distribution"); ax[0].set_xlabel(""); ax[0].tick_params(axis="x", rotation=0)
    words.clip(upper=1000).plot.hist(bins=40, ax=ax[1], color="#5b8def")
    ax[1].set_title("Review length (words, clipped at 1000)")
    plt.tight_layout(); plt.savefig(OUT_DIR / "eda.png", dpi=130); plt.close()
    return df, {"n_rows_raw": int(len(df) + n_dup), "n_duplicates": int(n_dup),
                "n_rows_used": int(len(df))}


# ----------------------------------------------------------------------------
# 2. Preprocess
# ----------------------------------------------------------------------------
def preprocess(df):
    print("\n" + "=" * 70, "\n2. PREPROCESSING\n" + "=" * 70)
    if CLEAN_CACHE.exists():
        cached = pd.read_csv(CLEAN_CACHE)
        if len(cached) == len(df):
            print(f"Loaded cleaned text from cache: {CLEAN_CACHE}")
            df["clean"] = cached["clean"].fillna("").values
            return df
    t0 = time.time()
    df["clean"] = df["review"].apply(clean_text)
    print(f"Cleaned {len(df)} reviews in {time.time() - t0:.0f}s")
    df[["clean"]].to_csv(CLEAN_CACHE, index=False)
    print("Example before:", df["review"].iloc[0][:150].replace("\n", " "), "...")
    print("Example after :", df["clean"].iloc[0][:150], "...")
    return df


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    df, eda_info = load_and_explore()
    df = preprocess(df)
    y = (df["sentiment"] == "positive").astype(int).values
    class_names = ["negative", "positive"]

    # 4. Split BEFORE fitting TF-IDF so no test information leaks in --------
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"].to_numpy(dtype=object), y, test_size=0.20, stratify=y,
        random_state=RANDOM_STATE)
    print("\n" + "=" * 70, "\n4. TRAIN/TEST SPLIT\n" + "=" * 70)
    print(f"Train: {len(X_train)}  Test: {len(X_test)}  "
          f"(train positive share = {y_train.mean():.3f}, test = {y_test.mean():.3f})")

    # 3 + 5. TF-IDF + RF inside ONE Pipeline so CV re-fits TF-IDF on each
    # training fold only (no leakage across folds). ---------------------------
    print("\n" + "=" * 70, "\n3/5. TF-IDF + HYPERPARAMETER SEARCH\n" + "=" * 70)
    print("TF-IDF settings:", TFIDF_PARAMS)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("rf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    if args.tune_sample and args.tune_sample < len(X_train):
        X_tune, _, y_tune, _ = train_test_split(
            X_train, y_train, train_size=args.tune_sample, stratify=y_train,
            random_state=RANDOM_STATE)
        print(f"Tuning on a stratified subsample of {len(X_tune)} training rows")
    else:
        X_tune, y_tune = X_train, y_train
        print("Tuning on the full training set")

    search = RandomizedSearchCV(
        pipe, PARAM_GRID, n_iter=args.n_iter, cv=args.cv, scoring="f1_weighted",
        random_state=RANDOM_STATE, n_jobs=1, verbose=2, refit=False)
    t0 = time.time()
    search.fit(X_tune, y_tune)
    print(f"Search finished in {(time.time() - t0) / 60:.1f} min")
    best_params = {k.replace("rf__", ""): v for k, v in search.best_params_.items()}
    print(f"Best params: {best_params}")
    print(f"Best CV weighted-F1: {search.best_score_:.4f}")

    cv_table = (pd.DataFrame(search.cv_results_)
                [["params", "mean_test_score", "std_test_score", "rank_test_score"]]
                .sort_values("rank_test_score"))
    print(cv_table.to_string(index=False))

    # Refit the best configuration on the FULL training set
    print("\nRefitting best model on full training set ...")
    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **best_params)
    t0 = time.time(); rf.fit(Xtr, y_train)
    print(f"Final fit took {time.time() - t0:.0f}s")

    # 6. Evaluate -----------------------------------------------------------
    print("\n" + "=" * 70, "\n6. TEST-SET EVALUATION\n" + "=" * 70)
    y_pred = rf.predict(Xte)
    acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, rf.predict(Xtr))
    report = classification_report(y_test, y_pred, target_names=class_names,
                                   output_dict=True, digits=4)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test  accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    plt.tight_layout(); plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=130); plt.close()

    # 7. Top-20 features ----------------------------------------------------
    names = tfidf.get_feature_names_out()
    imp = rf.feature_importances_
    top_idx = np.argsort(imp)[::-1][:20]
    # Direction of each feature: mean TF-IDF in positive vs negative docs
    pos_mean = np.asarray(Xtr[y_train == 1].mean(axis=0)).ravel()
    neg_mean = np.asarray(Xtr[y_train == 0].mean(axis=0)).ravel()
    top = pd.DataFrame({
        "feature": names[top_idx],
        "importance": imp[top_idx],
        "leans": np.where(pos_mean[top_idx] > neg_mean[top_idx], "positive", "negative"),
    })
    print("\n" + "=" * 70, "\n7. TOP 20 FEATURES\n" + "=" * 70)
    print(top.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = top["leans"].map({"positive": "#4c9f70", "negative": "#d9534f"})
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1])
    ax.set_title("Top 20 features (green = leans positive, red = leans negative)")
    ax.set_xlabel("Gini importance")
    plt.tight_layout(); plt.savefig(OUT_DIR / "feature_importance.png", dpi=130); plt.close()

    # 8. Persist everything for the app --------------------------------------
    joblib.dump({"tfidf": tfidf, "rf": rf, "class_names": class_names},
                OUT_DIR / "model.joblib", compress=("xz", 3))
    size_mb = (OUT_DIR / "model.joblib").stat().st_size / 1e6
    print(f"\nSaved model.joblib ({size_mb:.1f} MB)")

    metrics = {
        "eda": eda_info,
        "tfidf_params": {k: list(v) if isinstance(v, tuple) else v
                         for k, v in TFIDF_PARAMS.items()},
        "search": {"cv": args.cv, "n_iter": args.n_iter,
                   "tune_sample": args.tune_sample,
                   "best_params": best_params,
                   "best_cv_f1_weighted": float(search.best_score_)},
        "train_accuracy": float(train_acc),
        "test_accuracy": float(acc),
        "report": report,
        "confusion_matrix": cm.tolist(),
        "top_features": top.to_dict(orient="records"),
        "model_size_mb": round(size_mb, 1),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved artifacts/metrics.json and plots.")


if __name__ == "__main__":
    main()
