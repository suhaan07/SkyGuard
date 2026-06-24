"""Hand-engineered features from the (N, 128, 8) signal tensor.

Per channel: dropout handling, time-domain stats, and full-spectrum frequency
features. "Full-spectrum" is deliberate -- the EDA for this project showed
bearing-mode flights have *elevated low-frequency* vibration power and higher
overall variance, not the high-frequency shift the naive mechanical-wear story
suggests. Hand-picking one band would have baked in the wrong prior; binning
the whole spectrum and letting the model pick the band is more robust, and is
also what generalizes better to airframe model "E", which has no training
examples to calibrate a hand-picked threshold against.

Dropout: a channel is flagged if >10% of its timesteps are exact zero. Those
spans are linearly interpolated from the channel's own non-dropout timesteps
before computing any stat -- otherwise a "missing" reading and a genuine
"voltage=0" catastrophic reading would look identical to every downstream
feature.
"""
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

DROPOUT_FRAC_THRESHOLD = 0.10
N_BANDS = 3


def _impute_dropout(sig: np.ndarray, zero_mask: np.ndarray) -> np.ndarray:
    """sig: (N, T) for one channel. Interpolates zeroed spans per-row from that row's own data."""
    out = sig.copy()
    t = np.arange(sig.shape[1])
    for i in np.where(zero_mask.any(axis=1))[0]:
        known = ~zero_mask[i]
        if known.sum() >= 2:
            out[i, ~known] = np.interp(t[~known], t[known], sig[i, known])
        # if the whole row is zero, leave as-is: there's nothing to interpolate from
    return out


def _spectral_features(sig: np.ndarray) -> dict[str, np.ndarray]:
    """sig: (N, T) for one channel, already imputed."""
    fft = np.fft.rfft(sig, axis=1)
    power = np.abs(fft) ** 2
    power = power[:, 1:]  # drop DC bin
    total = power.sum(axis=1) + 1e-9

    n_bins = power.shape[1]
    edges = np.linspace(0, n_bins, N_BANDS + 1, dtype=int)
    band_energy = {}
    for b in range(N_BANDS):
        lo, hi = edges[b], edges[b + 1]
        band_energy[f"band{b}_energy"] = np.log1p(power[:, lo:hi].sum(axis=1))

    dominant_freq = power.argmax(axis=1).astype(np.float64)
    p = power / total[:, None]
    entropy = -(p * np.log(p + 1e-12)).sum(axis=1)

    return {
        **band_energy,
        "dominant_freq": dominant_freq,
        "spectral_entropy": entropy,
        "total_power": np.log1p(total),
    }


def build_signal_features(signals: np.ndarray, channel_names: np.ndarray) -> pd.DataFrame:
    n = signals.shape[0]
    feats = {}

    imputed = {}
    for c, name in enumerate(channel_names):
        chan = signals[:, :, c]
        zero_mask = chan == 0
        frac_zero = zero_mask.mean(axis=1)
        feats[f"{name}_frac_zero"] = frac_zero
        feats[f"{name}_is_dropout"] = (frac_zero > DROPOUT_FRAC_THRESHOLD).astype(np.float64)

        clean = _impute_dropout(chan, zero_mask)
        imputed[name] = clean

        feats[f"{name}_mean"] = clean.mean(axis=1)
        feats[f"{name}_std"] = clean.std(axis=1)
        feats[f"{name}_skew"] = skew(clean, axis=1)
        feats[f"{name}_kurtosis"] = kurtosis(clean, axis=1)
        feats[f"{name}_min"] = clean.min(axis=1)
        feats[f"{name}_max"] = clean.max(axis=1)
        feats[f"{name}_range"] = feats[f"{name}_max"] - feats[f"{name}_min"]

        t = np.arange(clean.shape[1], dtype=np.float64)
        t -= t.mean()
        denom = (t ** 2).sum()
        feats[f"{name}_slope"] = (clean * t).sum(axis=1) / denom

        for k, v in _spectral_features(clean).items():
            feats[f"{name}_{k}"] = v

    # cross-channel: motor current imbalance between the two motors
    if "motor_current_1" in imputed and "motor_current_2" in imputed:
        diff = imputed["motor_current_1"] - imputed["motor_current_2"]
        feats["current_imbalance_mean"] = diff.mean(axis=1)
        feats["current_imbalance_std"] = diff.std(axis=1)

    return pd.DataFrame(feats, index=np.arange(n))
