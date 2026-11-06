# 🎬 IMDB Sentiment Classifier — Random Forest + TF-IDF

Binary sentiment classification (positive / negative) of 50,000 IMDB movie reviews using
**scikit-learn's `RandomForestClassifier`**, with a **Streamlit** app for live predictions.

## Results (held-out test set, 9,917 reviews)

| Metric | Negative | Positive | Weighted avg |
|---|---|---|---|
| Precision | 0.859 | 0.852 | 0.856 |
| Recall | 0.850 | 0.862 | 0.856 |
| F1-score | 0.854 | 0.857 | 0.856 |

**Accuracy: 85.56 %** · Confusion matrix `[[4197, 743], [689, 4288]]`

## Pipeline

| Step | Choice |
|---|---|
| Data | IMDB Dataset of 50K Movie Reviews (balanced 25k/25k, no missing values, 418 duplicates dropped) |
| Cleaning | lowercase → strip HTML (`<br />`) → remove punctuation & numbers → remove stop words (negations like *not/no* kept) → WordNet lemmatization |
| TF-IDF | `max_features=5000`, `ngram_range=(1,2)`, `min_df=5`, `max_df=0.9`, `sublinear_tf=True` |
| Split | 80/20, stratified, `random_state=42` (split happens *before* fitting TF-IDF → no leakage) |
| Tuning | `RandomizedSearchCV`, 3-fold CV, 6 candidates over `n_estimators` {100,200,300}, `max_depth` {20,40,80,None}, `min_samples_split` {2,5,10}; scoring = weighted F1; TF-IDF is inside the CV pipeline |
| Best params | `n_estimators=300`, `max_depth=None`, `min_samples_split=10` (CV F1 = 0.849) |

## Run it

```bash
pip install -r requirements.txt

# 1. Download the dataset (Kaggle: "IMDB Dataset of 50K Movie Reviews")
#    and place it at  data/IMDB Dataset.csv

# 2. Train (creates artifacts/model.joblib, metrics.json and plots)
python train.py
#    More thorough search:  python train.py --n-iter 20 --tune-sample 0

# 3. Launch the app
streamlit run app.py
```

The trained model (`artifacts/model.joblib`, ~31 MB) is included, so you can run the app
without retraining.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (the `artifacts/` folder must be committed).
2. Go to <https://share.streamlit.io> → **New app** → select the repo, branch `main`, main file `app.py`.
3. In *Advanced settings* choose **Python 3.12** (the model was pickled with scikit-learn 1.9.1,
   pinned in `requirements.txt`).

## Files

```
├── app.py              # Streamlit app
├── train.py            # full training + evaluation pipeline
├── preprocess.py       # text cleaning (shared by train.py and app.py)
├── requirements.txt
├── artifacts/          # model.joblib, metrics.json, confusion_matrix.png, feature_importance.png, eda.png
└── README.md
```

## Interpretation & ideas for improvement

* The Random Forest reaches 100 % train accuracy but ~85.6 % test accuracy: it **overfits** (typical for
  fully-grown trees), and its CV score was still rising with more trees / deeper trees.
* Errors are symmetric (743 false positives vs 689 false negatives); the classes are balanced, so accuracy is a fair metric.
* Top features are intuitive sentiment words (*bad, worst, great, awful, waste, excellent*). Words like
  *plot* and *even* lean negative because they show up in critical, plot-focused reviews.
* Random Forests handle high-dimensional sparse text poorly compared with linear models. **Logistic Regression
  or Linear SVM on the same TF-IDF features typically reaches ~89–90 %** and trains in seconds.
* Other options: larger `max_features` (20k+) with trigrams, tune on the full training set with more candidates,
  Naive Bayes-SVM, or fine-tuned transformers (DistilBERT ≈ 92–93 %).
