import pytest
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from backend.experiment import get_preprocessor, run_experiment
from backend.model import load_data, TARGET, DROP_COLS

def test_split_reproducibility():
    df = load_data()
    X = df.drop(columns=DROP_COLS + [TARGET])
    y = df[TARGET]
    
    # Check that split is consistent for the same random_state
    from sklearn.model_selection import train_test_split
    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X3, X4, y3, y4 = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pd.testing.assert_frame_equal(X1, X3)
    pd.testing.assert_series_equal(y1, y3)

def test_cv_folds():
    # Verify 5 folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    assert cv.get_n_splits() == 5

def test_experiment_runs():
    results = run_experiment()
    assert len(results) == 4
    assert "LogisticRegression" in results
    assert "test_metrics" in results["LogisticRegression"]
    assert "ROC-AUC" in results["LogisticRegression"]["test_metrics"]
    
    # Verify that the target labels are correctly handled
    df = load_data()
    unique_labels = df[TARGET].unique()
    assert set(unique_labels) == {'Yes', 'No'}
