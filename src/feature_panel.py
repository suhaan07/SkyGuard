"""Combine the three per-modality feature sets into one panel.

This is the shared input both Stage 2's dropout ablation and Stage 3's
mid-level fusion model consume: one row per flight_id, tabular + signal +
text-embedding columns side by side, plus a `text_missing` indicator.

Text is represented by frozen BGE embeddings rather than TF-IDF here --
they tied on the Stage 1 baseline, but embeddings give Stage 3 a fixed-size
dense vector that's trivial to concatenate with the other two modalities
and trivial to "blank out" consistently (the empty string's own embedding),
which matters for the modality-dropout work in this stage.
"""
import numpy as np
import pandas as pd

from .features_tabular import build_tabular_features
from .features_signal import build_signal_features
from .features_text import missing_indicator, embed_notes


def _text_embedding_frame(note_text: pd.Series) -> pd.DataFrame:
    unique_texts = note_text.unique()
    unique_emb = embed_notes(unique_texts.tolist())
    emb_lookup = dict(zip(unique_texts, unique_emb))
    emb = np.vstack(note_text.map(emb_lookup).values)
    cols = [f"text_emb_{i}" for i in range(emb.shape[1])]
    out = pd.DataFrame(emb, columns=cols, index=note_text.index)
    out["text_missing"] = missing_indicator(note_text).astype(np.float64)
    return out, cols


def build_feature_panel(tab: pd.DataFrame, note_text: pd.Series, signals: np.ndarray, channel_names: np.ndarray):
    """Returns (panel, categorical_cols, text_embedding_cols).

    `tab`, `note_text`, and `signals` must all be row-aligned (same order,
    same length) -- exactly what `src.data.load_split` already guarantees.
    """
    X_tab, cat_cols = build_tabular_features(tab)
    X_sig = build_signal_features(signals, channel_names)
    X_text, text_cols = _text_embedding_frame(note_text.reset_index(drop=True))

    panel = pd.concat(
        [X_tab.reset_index(drop=True), X_sig.reset_index(drop=True), X_text.reset_index(drop=True)],
        axis=1,
    )
    return panel, cat_cols, text_cols
