"""Synthetic modality dropout for training a fusion model robustly.

Only `signal` and `text` are droppable -- tabular telemetry has zero
missingness anywhere in this dataset (no NaNs, always logged), so there's no
real-world condition synthetic tabular dropout would simulate. Sensor dropout
and empty notes are both real, so those are what the fusion model needs to
survive without becoming over-reliant on one of them.

Dropout is implemented at the *raw* level (zero the (128,8) tensor; blank the
note string) and then fed back through the normal feature builders, rather
than hand-faking feature values. That guarantees a synthetically-dropped row
produces *exactly* the same representation a naturally-missing one already
does (e.g. `vibration_is_dropout=1`, or the embedding of the empty string) --
no second "missing" convention to invent or get inconsistent with the first.
"""
import numpy as np
import pandas as pd


def sample_dropout_indices(idx: np.ndarray, p: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.random(len(idx)) < p
    return idx[mask]


def apply_modality_dropout(
    signals: np.ndarray,
    note_text: pd.Series,
    train_idx: np.ndarray,
    p_signal: float = 0.15,
    p_text: float = 0.15,
    seed: int = 42,
):
    """Returns (signals_aug, note_text_aug, signal_dropped_idx, text_dropped_idx).

    Only rows in `train_idx` are eligible to be dropped -- validation/test
    rows are always passed through untouched, since dropout simulates a
    training-time augmentation, not a real test-time condition.
    """
    signal_dropped_idx = sample_dropout_indices(train_idx, p_signal, seed)
    text_dropped_idx = sample_dropout_indices(train_idx, p_text, seed + 1)

    signals_aug = signals.copy()
    signals_aug[signal_dropped_idx] = 0.0

    note_text_aug = note_text.copy()
    note_text_aug.iloc[text_dropped_idx] = ""

    return signals_aug, note_text_aug, signal_dropped_idx, text_dropped_idx
