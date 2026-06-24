# SkyGuard — Data Dictionary

All three modalities join on **`flight_id`** (unique integer per flight). One row / one signal
slice / one note = one completed flight.

---

## 1. Tabular files — `train_tabular.csv`, `test_tabular.csv`

| Column | Type | Description |
|---|---|---|
| `flight_id` | int | Unique flight identifier. Join key across all modalities. |
| `drone_id` | str | Aircraft identifier. **All flights of one aircraft are in the same split** (train *or* test, never both). |
| `model` | cat | Airframe model (`A`–`E`). Affects structural resonance and dynamics. |
| `motor_type` | cat | Motor variant (`M1`/`M2`/`M3`). Different wear characteristics and sensor response. |
| `firmware_version` | cat | Flight-controller firmware (`v3.1`/`v3.2`/`v4.0`). |
| `battery_capacity_mAh` | int | Installed battery pack capacity. |
| `max_payload_g` | float | Rated maximum payload of the airframe. |
| `propeller_in` | float | Propeller diameter (inches). |
| `manufacture_batch` | int | Production batch index (0–11). |
| `operator_region` | cat | Operating region (`north`/`south`/`east`/`west`). |
| `flight_index` | int | Sequence number of this flight in the aircraft's life (1 = first flight). |
| `payload_g` | float | Payload carried on this flight. |
| `ambient_temp_C` | float | Ambient temperature at takeoff. |
| `wind_speed_ms` | float | Mean wind speed during the flight. |
| `flight_duration_min` | float | Flight duration. |
| `avg_throttle` | float | Mean throttle setting (0–1). |
| `num_aggressive_maneuvers` | int | Count of high-load maneuvers detected this flight. |
| `cumulative_flight_hours` | float | Total airframe flight hours up to and including this flight. |
| `battery_cycles` | int | Charge cycles on the current pack up to this flight. |
| `failure_mode` | cat | **TRAIN ONLY.** Dominant degradation path for labelled failures: `bearing`, `battery`, or `none`. Auxiliary — use freely in training; absent from test. |
| `failure_within_horizon` | int | **TRAIN ONLY. Target.** 1 if the aircraft fails within its next 5 flights, else 0. |

## 2. Signal files — `train_signals.npz`, `test_signals.npz`

Loaded with `numpy.load(path)`:

| Key | Shape | Description |
|---|---|---|
| `flight_id` | `(N,)` int | Flight IDs, **row-aligned** to `signals`. |
| `signals` | `(N, 128, 8)` float32 | 128 time-steps × 8 channels per flight (resampled to a fixed length). |
| `channel_names` | `(8,)` str | Channel order (below). |

**Channel order (axis 2):**

| Idx | Channel | Notes |
|---|---|---|
| 0 | `accel_x` | IMU acceleration. |
| 1 | `accel_y` | IMU acceleration. |
| 2 | `accel_z` | IMU acceleration. |
| 3 | `gyro_z` | Yaw-rate gyro. |
| 4 | `motor_current_1` | Motor 1 current draw. |
| 5 | `motor_current_2` | Motor 2 current draw. |
| 6 | `vibration` | Dedicated structural vibration sensor. |
| 7 | `battery_voltage` | Battery bus voltage (normalised). |

> **Sensor dropout:** on a minority of flights, one or more channels are zeroed for a
> contiguous span (a real sensor-dropout artefact). Handle it; do not assume every channel is
> always present.

**Loading example**

```python
import numpy as np, pandas as pd
z = np.load("data/train/train_signals.npz")
sig = z["signals"]               # (N, 128, 8)
sig_index = pd.Series(np.arange(len(z["flight_id"])), index=z["flight_id"])
# row for a given flight_id:
row = sig[sig_index.loc[flight_id]]
```

## 3. Notes files — `train_notes.csv`, `test_notes.csv`

| Column | Type | Description |
|---|---|---|
| `flight_id` | int | Join key. |
| `maintenance_note` | str | Free-text post-flight note. **May be an empty string** (≈ 28% of flights). |

## 4. `sample_submission.csv`

| Column | Type | Description |
|---|---|---|
| `flight_id` | int | Every test `flight_id`. |
| `failure_within_horizon` | float | Your predicted probability in `[0, 1]`. |
