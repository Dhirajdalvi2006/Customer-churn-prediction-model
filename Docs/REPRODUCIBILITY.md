# Reproducibility Baseline

## Environment Information
- **Python Version:** 3.11+
- **Major Dependency Versions:**
  - `scikit-learn`: >= 1.3.0
  - `pandas`: >= 2.0.0
  - `numpy`: >= 1.24.0
  - `streamlit`: >= 1.25.0
  - `shap`: >= 0.42.0
  - `pytest`: >= 7.4.0

## Phase 3B Experimental Configuration
- **Dataset Source:** `data/telco_customer_churn_cleaned.csv`
- **Data Splitting Strategy:** `train_test_split(target='Churn', test_size=0.2, random_state=42, stratify='Churn')`
- **Cross Validation Strategy:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Calibration Split (Phase 3C):** Inherited strictly from the Development set (80%), subdivided further (0.2).

## Production Baseline (Phase 4 Freeze)
### 1. Frozen Legacy V1 Artifacts
*Used for backward compatibility and shadow audit.*
- `models/bundles/v1/model.pkl` (LogisticRegression)
- `models/bundles/v1/preprocessor.pkl`

### 2. Frozen Active V2 Artifacts (Production Champion)
- `models/bundles/v2/model.pkl` (GradientBoostingClassifier Pipeline)
- `models/bundles/v2/manifest.json`

### 3. Active Pointer
- The production boundary reads `models/active_bundle.json`.
- It dynamically targets `v2` as of Phase 4 Step 12D.

### 4. Continuous Integration Checks
- **Test Command:** `python -m pytest -q`
- **Result:** Passes (381 tests, 0 failures) against dynamically inspected V2 bundle properties.
- **Compilation Check:** `python -m compileall backend frontend tests app.py` exits clean.
- **Hashes:** Cryptographic `sha256` integrity verified in `tests/test_prediction_service.py` to prevent unauthorized file mutation post-freeze.

### 5. Protected Artifacts Directory
By design, all `*.pkl` artifacts in `models/`, `models/experiments/`, and `models/bundles/` are strictly read-only as of Phase 4 freeze.

### 6. Code Generation
Any updates post-freeze that seek to adjust models must generate a new explicitly bound directory (`models/bundles/v3`) instead of overwriting existing paths or pointers without a shadow audit.