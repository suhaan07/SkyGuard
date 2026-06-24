"""AUPRC scoring and per-fold reporting."""
import numpy as np
from sklearn.metrics import average_precision_score

BASE_RATE = 0.125  # ~Pr(y=1); the floor a random ranker should score


def auprc(y_true, y_score) -> float:
    return average_precision_score(y_true, y_score)


def report_folds(fold_scores: list[float], label: str) -> dict:
    mean, std = float(np.mean(fold_scores)), float(np.std(fold_scores))
    print(f"{label}: fold AUPRCs = {[round(s, 4) for s in fold_scores]}")
    print(f"{label}: mean={mean:.4f}  std={std:.4f}  (floor~{BASE_RATE})")
    return {"fold_scores": fold_scores, "mean": mean, "std": std}
