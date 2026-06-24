"""Loading and aligning the three SkyGuard modalities.

Every function returns rows in **tabular row order** for a given split: the
tabular CSV is treated as the canonical ordering, and notes / signals are
reindexed onto it. Downstream feature/CV code can then assume `tab.iloc[i]`,
`notes.iloc[i]` and `signals[i]` all describe the same flight.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Known category universes, taken from data_dictionary.md rather than scanned
# from the files. `model` includes "E", which has zero training rows but ~31%
# of test rows -- if we let pandas infer categories from train alone, LightGBM
# would have no stable code for "E" at predict time. Declaring the full
# universe up front keeps category->code mappings identical across train/test.
CATEGORY_LEVELS = {
    "model": ["A", "B", "C", "D", "E"],
    "motor_type": ["M1", "M2", "M3"],
    "firmware_version": ["v3.1", "v3.2", "v4.0"],
    "operator_region": ["north", "south", "east", "west"],
}


def load_tabular(split: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / split / f"{split}_tabular.csv")
    for col, levels in CATEGORY_LEVELS.items():
        df[col] = pd.Categorical(df[col], categories=levels)
    return df


def load_notes(split: str, flight_ids: pd.Series) -> pd.DataFrame:
    notes = pd.read_csv(DATA_DIR / split / f"{split}_notes.csv")
    notes = notes.set_index("flight_id").reindex(flight_ids).reset_index()
    notes["maintenance_note"] = notes["maintenance_note"].fillna("").str.strip()
    return notes


def load_signals(split: str, flight_ids: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Returns (signals, channel_names) with `signals` reindexed to flight_ids order."""
    z = np.load(DATA_DIR / split / f"{split}_signals.npz", allow_pickle=True)
    pos = pd.Series(np.arange(len(z["flight_id"])), index=z["flight_id"])
    signals = z["signals"][pos.loc[flight_ids].values]
    return signals, z["channel_names"]


def load_split(split: str) -> dict:
    """One call to get everything for a split, aligned on tabular row order."""
    tab = load_tabular(split)
    notes = load_notes(split, tab["flight_id"])
    signals, channel_names = load_signals(split, tab["flight_id"])
    return {"tab": tab, "notes": notes, "signals": signals, "channel_names": channel_names}
