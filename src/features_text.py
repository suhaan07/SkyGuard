"""Text-modality features.

Two encoders to compare, plus a missingness indicator that's independent of
which encoder is used (~28% of notes are empty, and the empty rate near the
base rate, so it's a weak feature alone, but cheap and robust).

TF-IDF must be fit inside each CV fold (`tfidf_features` takes separate
train/val text lists and fits only on train) to keep the fold honest, even
though with only 24 unique templates in this dataset it barely matters.
Sentence embeddings come from a frozen pretrained model, so there's nothing to
fit and no leakage risk -- they can be computed once for the whole dataset.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_ST_MODEL = None  # lazy-loaded singleton, sentence-transformers model load is slow


def missing_indicator(texts) -> np.ndarray:
    return np.array([1.0 if t.strip() == "" else 0.0 for t in texts])


def tfidf_features(train_texts, val_texts, max_features: int = 200):
    vec = TfidfVectorizer(max_features=max_features, min_df=1)
    train_X = vec.fit_transform(train_texts)
    val_X = vec.transform(val_texts)
    return train_X, val_X


def embed_notes(texts, model_name: str = "BAAI/bge-large-en-v1.5") -> np.ndarray:
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer(model_name)
    return _ST_MODEL.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
