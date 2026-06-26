# Stage 3 Fusion Implementation Plan

This plan outlines the steps to build a final **Mid-Level Concatenation Fusion model** for the SkyGuard failure prognosis challenge. We will create a new notebook to train a LightGBM model with synthetic modality dropout on the combined feature panel, evaluate out-of-fold AUPRC performance, and generate the final `submission.csv` on the test split.

## User Review Required

> [!NOTE]
> We will train the final model using a 5-fold ensemble (averaging predictions of the 5 models trained on different cross-validation folds). This is standard best practice for Kaggle-style tabular/multimodal setups because:
> 1. It increases model robustness and reduces prediction variance on the unseen test set.
> 2. It avoids the need to select a single epoch or risk overfitting when training on 100% of the data without validation feedback.
> 3. Each fold model is evaluated on a group-disjoint validation set to measure out-of-fold (OOF) performance.

> [!IMPORTANT]
> The test set has some rows for airframe model `E` (which is rare/absent in the training split). Our category encoding in [data.py](file:///Users/adityagupta/Documents/BCLUB/ANALYTICS/HACKATHON%201/SkyGuard/src/data.py) pre-defines category levels to handle this gracefully, and the mid-level fusion features must be processed for the test split exactly like they were for the train split.

## Open Questions

None. The task is fully specified: we will use a mid-level concatenation technique with LightGBM, including the modality dropout augmentation during training to maintain robustness against sensor dropout and missing notes.

---

## Proposed Changes

### Modeling and Notebooks

#### [NEW] [03_fusion.ipynb](file:///Users/adityagupta/Documents/BCLUB/ANALYTICS/HACKATHON%201/SkyGuard/notebooks/03_fusion.ipynb)
A new notebook in the `notebooks` directory containing:
1. Loading the train and test splits using `src.data.load_split`.
2. Building the combined feature panels (`panel_train` and `panel_test`) by concatenating tabular, signal, and text embedding features.
3. Defining the training hyperparameters for LightGBM, matching the baseline parameters used in previous notebooks.
4. Setting up the cross-validation loops using the group-disjoint folds from `src.cv.get_folds`.
5. Running the training loop with synthetic modality dropout on the training folds and evaluating out-of-fold (OOF) AUPRC to verify fusion quality.
6. Generating predictions for the test split by averaging the outputs from the 5 fold-trained models.
7. Checking the shape, bounds, and distribution of the predictions, and saving the final output to `submission.csv` in the repository root.

---

## Verification Plan

### Automated Verification
We will write and run a temporary verification script in the scratch directory:
* `/Users/adityagupta/.gemini/antigravity-ide/brain/5acadd8a-0985-4a4d-91bf-5a0caed2eb93/scratch/run_fusion.py`

This script will run the entire data loading, panel building, ensemble training, and prediction generation logic using `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`. It will:
1. Check that the training out-of-fold AUPRC matches or exceeds the baseline performance.
2. Confirm `submission.csv` is created.
3. Validate `submission.csv` properties:
   - Exactly 7,536 lines (including the header).
   - Column names are exactly `flight_id` and `failure_within_horizon`.
   - Predictions are valid float values in $[0, 1]$ (i.e. no NaNs, infs, or out-of-bounds values).
   - Matches the IDs of `sample_submission.csv` exactly.

Once verified, the script will programmatically write the final Jupyter Notebook cells into `notebooks/03_fusion.ipynb`.
