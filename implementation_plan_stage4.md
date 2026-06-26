# Stage 4: Multi-Task Auxiliary Signal Implementation Plan

This plan details the implementation of **Stage 4: Multi-Task Auxiliary Signal** modeling. We will build a model that jointly predicts the primary binary target `failure_within_horizon` ($y$) and the auxiliary train-only multi-class target `failure_mode` ($m$). This regularizes the model's representation by forcing it to learn modality-specific degradation patterns (e.g. vibration shifts for bearings vs. voltage sags for batteries).

We propose two alternative modeling strategies to achieve this:

## Proposed Modeling Options

### Option 1: Joint State-Space Multi-Class GBDT (LightGBM) — *Recommended*
Since LightGBM is highly optimized for mixed tabular, statistical, and embedding feature panels, we can formulate the joint task as a single **6-class classification problem**:
* We define a combined target variable representing all non-empty combinations of $y \in \{0, 1\}$ and $m \in \{\text{none}, \text{bearing}, \text{battery}\}$:
  1. Class 0: $y=0, m=\text{none}$ (Healthy / No Degradation)
  2. Class 1: $y=0, m=\text{battery}$ (Battery wear, but $>5$ flights from terminal failure)
  3. Class 2: $y=0, m=\text{bearing}$ (Bearing wear, but $>5$ flights from terminal failure)
  4. Class 3: $y=1, m=\text{none}$ (Failure within 5 flights, unspecified/other path)
  5. Class 4: $y=1, m=\text{battery}$ (Battery failure within 5 flights)
  6. Class 5: $y=1, m=\text{bearing}$ (Bearing failure within 5 flights)
* **Training:** Train a standard multi-class LightGBM classifier on this 6-class target.
* **Inference:** At test time, output the 6 class probabilities $[p_0, p_1, p_2, p_3, p_4, p_5]$ and obtain the primary prediction by marginalizing out the auxiliary target:
  $$P(\text{failure\_within\_horizon} = 1) = p_3 + p_4 + p_5$$
* **Pros:** Leverages GBDTs' strong performance on wide tabular feature panels directly; does not require complex neural network tuning or learning rate/scaling setup.

### Option 2: Shared-Trunk Multi-Task Neural Network (PyTorch MLP)
* **Architecture:** A PyTorch Multi-Layer Perceptron (MLP) with a shared trunk (Linear layers + BatchNorm + ReLU + Dropout) splitting into two heads:
  1. **Binary Head:** Outputs 1 logit for `failure_within_horizon` (loss: BCEWithLogitsLoss).
  2. **Multi-Class Head:** Outputs 3 logits for `failure_mode` (loss: CrossEntropyLoss).
* **Training:** Jointly minimize: $Loss = Loss_{\text{binary}} + \alpha \cdot Loss_{\text{multiclass}}$.
* **Inference:** Drop the multi-class head and use only the binary head prediction.
* **Pros:** Elegant, classic neural implementation of shared representation learning.
* **Cons:** Requires strict preprocessing (feature scaling, imputation, categorical one-hot encoding) and is more sensitive to hyperparameters (learning rate, weight decay, epoch counts).

---

## User Review Required

We recommend **Option 1 (Joint State-Space Multi-Class GBDT)** because GBDTs are inherently more robust and perform significantly better on the heterogeneous feature panel (1181 columns including raw statistics, categorical codes, and embeddings) without needing extensive normalization and learning rate tuning.

> [!TIP]
> We will evaluate both options or implement the recommended Option 1, verify its out-of-fold AUPRC performance compared to the Stage 3 fusion baseline, and generate the final predictions.

---

## Proposed Changes

### Modeling and Notebooks

#### [NEW] [04_multitask.ipynb](file:///Users/adityagupta/Documents/BCLUB/ANALYTICS/HACKATHON%201/SkyGuard/notebooks/04_multitask.ipynb)
A new notebook in the `notebooks` directory containing:
1. Loading the train and test splits, and combined clean feature panels.
2. Mapping the train split targets ($y$ and $m$) to the 6-class joint target.
3. Training the multi-class LightGBM model using the 5-fold Stratified Group CV scheme.
4. Marginalizing the class predictions to compute validation AUPRC and comparing against the Stage 3 baseline.
5. Exporting test split predictions to `submission.csv`.

---

## Verification Plan

### Automated Verification
We will run a temporary script in the scratch directory:
* `/Users/adityagupta/.gemini/antigravity-ide/brain/5acadd8a-0985-4a4d-91bf-5a0caed2eb93/scratch/run_multitask.py`

This script will execute the training pipeline, report fold-level AUPRC, generate the final `submission.csv`, and check:
1. Shape and structure of `submission.csv`.
2. Absence of NaNs and check that probabilities are in $[0, 1]$.
3. Match against the sample submission IDs.
