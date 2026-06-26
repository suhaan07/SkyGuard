# SkyGuard Failure Prognosis Walkthrough

This document summarizes the implementations, validations, and results for the three stages of multi-modal failure prognosis modeling: **Stage 3 (Fusion)**, **Stage 4 (Multitask Auxiliary)**, and **Stage 5 (Calibration)**.

---

## Stage 3: Mid-Level Concatenation Fusion

### Implementation
* **Notebook:** [03_fusion.ipynb](file:///Users/adityagupta/Documents/BCLUB/ANALYTICS/HACKATHON%201/SkyGuard/notebooks/03_fusion.ipynb)
* **Strategy:** Concatenated all three modalities (tabular, signals, and text embeddings) into a wide feature panel of 1181 columns. Trained a LightGBM binary classifier using a 5-fold Stratified Group CV scheme (grouped by `drone_id`) with 15% synthetic modality dropout during training.
* **Results:**
  * **Fold AUPRCs:** `[0.5262, 0.4993, 0.5022, 0.5358, 0.5252]`
  * **Mean Validation AUPRC:** **0.5177** (std: `0.0144`, baseline floor ~`0.125`)
  * **Pooled OOF AUPRC:** **0.5155**

---

## Stage 4: Multi-Task Auxiliary Signal (6-Class GBDT)

### Implementation
* **Notebook:** [04_multitask.ipynb](file:///Users/adityagupta/Documents/BCLUB/ANALYTICS/HACKATHON%201/SkyGuard/notebooks/04_multitask.ipynb)
* **Strategy:** Formulated joint prediction of the binary target `failure_within_horizon` ($y \in \{0, 1\}$) and the auxiliary train-only multi-class target `failure_mode` ($m \in \{\text{none}, \text{bearing}, \text{battery}\}$) as a single **6-class classification problem**.
  * Class 0: $y=0, m=\text{none}$
  * Class 1: $y=0, m=\text{battery}$
  * Class 2: $y=0, m=\text{bearing}$
  * Class 3: $y=1, m=\text{none}$
  * Class 4: $y=1, m=\text{battery}$
  * Class 5: $y=1, m=\text{bearing}$
* **Regularization:** Forces the shared tree structures in LightGBM to split on features that capture modality-specific degradation paths (like vibration shifts or voltage sag) rather than just an undifferentiated "something's wrong" signal.
* **Inference:** Test set probabilities are marginalized to obtain $P(y=1) = p_3 + p_4 + p_5$.
* **Results:**
  * **Fold AUPRCs:** `[0.5114, 0.5035, 0.4994, 0.5331, 0.5247]`
  * **Mean Validation AUPRC:** **0.5144** (std: `0.0127`, baseline floor ~`0.125`)
  * **Pooled OOF AUPRC:** **0.5134**

---

## Stage 5: Validation and Calibration

### Implementation
* **Notebook:** [05_calibration.ipynb](file:///Users/adityagupta/Documents/BCLUB/ANALYTICS/HACKATHON%201/SkyGuard/notebooks/05_calibration.ipynb)
* **Strategy:** Evaluated two probability calibration methods fitted on the pooled out-of-fold (OOF) predictions:
  1. **Isotonic Regression:** Non-parametric method. Improves Brier score significantly from `0.08817` to `0.08170`, but introduces flat-bin ties that slightly decrease AUPRC to `0.49940`.
  2. **Platt Scaling (Logistic Regression):** Parametric method. Improves Brier score to `0.08591` and preserves the relative ranking perfectly, maintaining the high OOF AUPRC of **0.51341** with zero ties.
* **Selection:** To maximize leaderboard performance (evaluated strictly on AUPRC ranking) while delivering valid probability outputs, we chose **Platt Scaling** as the final calibrator.
* **Inference:** The fitted Platt scaling sigmoid calibrator was applied to the ensemble test predictions.

---

## Submission & Verification Results

### Final Output File
* **Path:** [submission.csv](file:///Users/adityagupta/Documents/BCLUB/ANALYTICS/HACKATHON%201/SkyGuard/submission.csv) (generated using the Stage 5 Platt-calibrated ensemble model predictions).

### Verification Checks
* **Row Count:** Contains exactly 7,534 prediction rows (7,535 lines including the header), matching [sample_submission.csv](file:///Users/adityagupta/Documents/BCLUB/ANALYTICS/HACKATHON%201/SkyGuard/data/sample_submission.csv) exactly.
* **Value Integrity:** No null or missing prediction values exist.
* **Prediction Distribution:** Bounded strictly in $[0, 1]$:
  * **Min:** `0.07658` (representing the background failure probability under clean conditions)
  * **Mean:** `0.12714` (aligns with the ~12.5% training set prior probability)
  * **Median (50%):** `0.08114`
  * **Max:** `0.94066`
