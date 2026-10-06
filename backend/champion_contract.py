import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

class ChampionInputContract:
    """Internal champion-side preparation and validation path."""

    REQUIRED_COLS = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod", "SeniorCitizen", "tenure",
        "MonthlyCharges", "TotalCharges"
    ]

    def __init__(self, pipeline: Pipeline):
        if not isinstance(pipeline, Pipeline):
            raise TypeError("Expected a fitted sklearn.pipeline.Pipeline")
        self.pipeline = pipeline
        self._extract_metadata()

    def _extract_metadata(self):
        """Unwrap the fitted pipeline to extract validation and mapping metadata."""
        try:
            preprocessor = self.pipeline.named_steps["preprocessor"]
            cat_pipe = preprocessor.named_transformers_["cat"]
            encoder = cat_pipe.named_steps["onehot"] if isinstance(cat_pipe, Pipeline) else cat_pipe
            
            self.categorical_domains = {}
            for i, col in enumerate(self.REQUIRED_COLS[:15]):
                self.categorical_domains[col] = set(encoder.categories_[i])
            
            clf = self.pipeline.named_steps["classifier"]
            self.classes = clf.classes_.tolist()
            if "Yes" not in self.classes:
                raise ValueError(f"Expected 'Yes' in classifier classes, found {self.classes}")
            self.pos_label_idx = self.classes.index("Yes")
        except (KeyError, AttributeError, IndexError) as e:
            raise ValueError(f"Could not extract metadata: {e}")

    def _coerce_senior_citizen(self, val):
        """Coerce SeniorCitizen to 0/1 per production semantics."""
        if pd.isna(val) or val == "":
            raise ValueError("SeniorCitizen cannot be empty or NaN")
        
        if isinstance(val, str):
            low_val = val.lower().strip()
            if low_val in ("yes", "1", "true"): return 1
            if low_val in ("no", "0"): return 0
            raise ValueError(f"Invalid SeniorCitizen: {val}")
        
        try:
            num_val = float(val)
            if num_val == 1.0: return 1
            if num_val == 0.0: return 0
            raise ValueError(f"Invalid SeniorCitizen: {val}")
        except (ValueError, TypeError):
            raise ValueError(f"Invalid SeniorCitizen: {val}")

    def _prepare_frame(self, data):
        """Convert input to a validated DataFrame."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("Input must be a dict or a pandas DataFrame")

        missing = set(self.REQUIRED_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        df = df[self.REQUIRED_COLS]

        for col in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
            if np.isinf(pd.to_numeric(df[col], errors="coerce")).any():
                raise ValueError(f"Infinity detected in: {col}")

        df["SeniorCitizen"] = df["SeniorCitizen"].apply(self._coerce_senior_citizen)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        for col in ["tenure", "MonthlyCharges"]:
            if pd.to_numeric(df[col], errors="coerce").isna().any():
                raise ValueError(f"NaN not allowed in: {col}")

        for col, domain in self.categorical_domains.items():
            invalid = df[~df[col].isin(domain)][col].unique()
            if len(invalid) > 0:
                raise ValueError(f"Invalid category '{invalid[0]}' for {col}")

        return df

    def predict(self, data):
        """Prepared, validated prediction."""
        df = self._prepare_frame(data)
        
        if isinstance(data, dict):
            probs = self.pipeline.predict_proba(df)[0]
            pred_label = self.pipeline.predict(df)[0]
            
            return {
                "prediction": pred_label,
                "churn_probability": round(float(probs[self.pos_label_idx]) * 100, 2),
                "no_churn_probability": round(float(probs[1 - self.pos_label_idx]) * 100, 2),
            }
        
        preds = self.pipeline.predict(df)
        probs = self.pipeline.predict_proba(df)[:, self.pos_label_idx]
        return preds, probs
