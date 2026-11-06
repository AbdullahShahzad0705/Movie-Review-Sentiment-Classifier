"""
Text preprocessing shared by train.py and app.py.

Keeping this in ONE place guarantees the app cleans user input exactly the
same way the model saw the training data (no train/serve skew).
"""
import re
from functools import lru_cache

import nltk


def _ensure_nltk():
    """Download the NLTK resources we need (no-op if already present)."""
    for pkg, path in [("stopwords", "corpora/stopwords"),
                      ("wordnet", "corpora/wordnet"),
                      ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk()

from nltk.corpus import stopwords          # noqa: E402
from nltk.stem import WordNetLemmatizer    # noqa: E402

# Negations carry the sentiment ("not good" != "good"), so we deliberately
# KEEP them even though NLTK lists them as stop words.
NEGATIONS = {"no", "nor", "not", "don", "didn", "doesn", "isn", "wasn", "weren",
             "aren", "couldn", "shouldn", "wouldn", "hadn", "hasn", "haven",
             "won", "ain", "mustn", "needn", "shan", "mightn"}
STOP_WORDS = set(stopwords.words("english")) - NEGATIONS

_lemmatizer = WordNetLemmatizer()
_HTML = re.compile(r"<[^>]+>")           # IMDB reviews contain <br /> tags
_NON_LETTERS = re.compile(r"[^a-z\s]")   # removes punctuation AND digits


@lru_cache(maxsize=200_000)
def _lemma(word: str) -> str:
    return _lemmatizer.lemmatize(word)


def clean_text(text: str) -> str:
    """lowercase -> strip HTML -> drop punctuation/numbers -> drop stop words -> lemmatize"""
    text = str(text).lower()
    text = _HTML.sub(" ", text)
    text = text.replace("'", "")          # "don't" -> "dont" instead of "don t"
    text = _NON_LETTERS.sub(" ", text)
    tokens = [_lemma(t) for t in text.split()
              if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)
