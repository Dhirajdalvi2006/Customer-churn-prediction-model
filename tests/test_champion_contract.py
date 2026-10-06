import pytest
import pandas as pd
import numpy as np
import joblib
import os
from unittest.mock import MagicMock
from backend.champion_contract import ChampionInputContract

@pytest.fixture
def champion_pipeline():
    model_path = os.path.join("models", "experiments", "champion.pkl")
    return joblib.load(model_path)

@pytest.fixture
def contract(champion_pipeline):
    return ChampionInputContract(champion_pipeline)

@pytest.fixture
def sample_row():
    return {
        "gender": "Female", "Partner": "Yes", "Dependents": "No", "PhoneService": "No",
        "MultipleLines": "No phone service", "InternetService": "DSL", "OnlineSecurity": "No",
        "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "SeniorCitizen": 0,
        "tenure": 1, "MonthlyCharges": 29.85, "TotalCharges": 29.85
    }

def test_schema(contract, sample_row):
    contract.predict(sample_row)
    for col in contract.REQUIRED_COLS:
        row = sample_row.copy()
        del row[col]
        with pytest.raises(ValueError, match="Missing required columns"):
            contract.predict(row)
    row = sample_row.copy()
    row["extra_col"] = 1
    contract.predict(row)
    row = {k: sample_row[k] for k in reversed(list(sample_row.keys()))}
    contract.predict(row)

def test_categorical_validation(contract, sample_row):
    contract.predict(sample_row)
    row = sample_row.copy()
    row["gender"] = "InvalidGender"
    with pytest.raises(ValueError, match="Invalid category"):
        contract.predict(row)

@pytest.mark.parametrize("val", [0, 1, 1.0, "Yes", "No", "yes", "true", "0"])
def test_senior_citizen_accepted(contract, sample_row, val):
    row = sample_row.copy()
    row["SeniorCitizen"] = val
    contract.predict(row)

@pytest.mark.parametrize("val", [2, "maybe", np.nan, ""])
def test_senior_citizen_rejected(contract, sample_row, val):
    row = sample_row.copy()
    row["SeniorCitizen"] = val
    with pytest.raises(ValueError):
        contract.predict(row)

def test_total_charges(contract, sample_row):
    for val in [29.85, "29.85", np.nan, "abc", " "]:
        row = sample_row.copy()
        row["TotalCharges"] = val
        contract.predict(row)

@pytest.mark.parametrize("col", ["tenure", "MonthlyCharges"])
def test_numeric_nan_rejected(contract, sample_row, col):
    row = sample_row.copy()
    row[col] = np.nan
    with pytest.raises(ValueError, match="NaN not allowed"):
        contract.predict(row)

@pytest.mark.parametrize("col", ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"])
def test_infinity_rejected(contract, sample_row, col):
    row = sample_row.copy()
    row[col] = np.inf
    with pytest.raises(ValueError, match="Infinity detected"):
        contract.predict(row)

def test_class_semantics(contract, sample_row):
    assert "Yes" in contract.classes
    assert contract.classes[contract.pos_label_idx] == "Yes"

def test_probability_semantics(contract, sample_row):
    res = contract.predict(sample_row)
    assert pytest.approx(res["churn_probability"] + res["no_churn_probability"], 1) == 100

def test_label_semantics(contract, sample_row):
    res_no = contract.predict(sample_row)
    assert res_no["prediction"] in ["Yes", "No"]

def test_single_batch_parity(contract, sample_row):
    res_single = contract.predict(sample_row)
    res_batch_preds, res_batch_probs = contract.predict(pd.DataFrame([sample_row]))
    assert res_single["prediction"] == res_batch_preds[0]
    assert pytest.approx(res_single["churn_probability"], 0.1) == res_batch_probs[0] * 100

def test_determinism(contract, sample_row):
    res1 = contract.predict(sample_row)
    res2 = contract.predict(sample_row)
    assert res1 == res2

def test_pre_inference_rejection(contract, sample_row):
    contract.pipeline.predict = MagicMock()
    contract.pipeline.predict_proba = MagicMock()
    row = sample_row.copy()
    row["gender"] = "Invalid"
    with pytest.raises(ValueError):
        contract.predict(row)
    contract.pipeline.predict.assert_not_called()
    contract.pipeline.predict_proba.assert_not_called()

