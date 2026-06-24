# SkyGuard — Multimodal UAV Failure Prognosis

Predict, for each drone flight, the probability that the
aircraft suffers a **critical powertrain failure within its next 5 flights**, by fusing three
modalities: **tabular telemetry**, **raw onboard signals** (IMU / motor current / vibration /
battery), and **free-text maintenance notes**.

## Start here

1. Read **`PROBLEM\_STATEMENT.md`** — scenario, task, submission format, evaluation.
2. Read **`data\_dictionary.md`** — every field, the 8 signal channels, file formats.

## Contents

```
PROBLEM\_STATEMENT.md      the brief
data\_dictionary.md        field-by-field reference
data/
  train/  train\_tabular.csv · train\_notes.csv · train\_signals.npz   (labelled)
  test/   test\_tabular.csv  · test\_notes.csv  · test\_signals.npz    (unlabelled)
  sample\_submission.csv
```

## Quickstart

```python
import numpy as np, pandas as pd

tab   = pd.read\_csv("data/train/train\_tabular.csv")          # tabular + label
notes = pd.read\_csv("data/train/train\_notes.csv")            # flight\_id -> text (may be empty)
z     = np.load("data/train/train\_signals.npz")              # raw signals
sig, sig\_id = z\["signals"], z\["flight\_id"]                   # (N,128,8), aligned ids
print(z\["channel\_names"])

y = tab\["failure\_within\_horizon"]                            # target (rare: \~12%)
```

## Evaluation

* **Metric:** Average Precision (AUPRC) on the hidden test set.
* **Submit:** `submission.csv` with `flight\_id,failure\_within\_horizon` (probability) for every test flight.

Three modalities, one prediction.

