"""Cross-validation that respects the drone_id grouping.

Plain GroupKFold partitions groups into folds balancing *group count*, not
label rate -- with ~620 drones of very different sizes and a 48%/52% split
between drones that ever fail and drones that never do, that can leave folds
with noticeably different positive rates. StratifiedGroupKFold (sklearn) does
the same group-disjointness guarantee but balances label rate across folds too,
so per-fold AUPRC is a fairer apples-to-apples comparison.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def get_folds(tab: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    """Yields (train_idx, val_idx) arrays positional into `tab`, grouped by drone_id."""
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = tab["failure_within_horizon"].values
    groups = tab["drone_id"].values
    yield from skf.split(np.zeros(len(tab)), y, groups)


def assert_disjoint_groups(tab: pd.DataFrame, train_idx, val_idx):
    train_drones = set(tab["drone_id"].iloc[train_idx])
    val_drones = set(tab["drone_id"].iloc[val_idx])
    overlap = train_drones & val_drones
    assert not overlap, f"drone_id leaked across fold: {overlap}"
