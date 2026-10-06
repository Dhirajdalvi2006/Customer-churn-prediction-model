import os
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "telco_customer_churn_cleaned.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

NUMERICAL_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

TARGET = "Churn"
DROP_COLS = ["customerID"]

# The complete model input schema: 15 categorical + 4 numerical raw features.
# This single source of truth drives training, inference, batch scoring, the
# What-If simulator and the regression tests, so the feature contract cannot
# drift between entry points.
REQUIRED_COLS = CATEGORICAL_COLS + NUMERICAL_COLS

# Fallback imputation value used ONLY when loading model artifacts that were
# serialized before the imputation metadata existed. It reproduces the median of
# TotalCharges over the deterministic training split
# (train_test_split(test_size=0.2, random_state=42, stratify=y)), i.e. the exact
# value written by ChurnModel.prepare_data() -> ChurnModel.save().
FALLBACK_TOTAL_CHARGES_MEDIAN = 1394.925


def derive_total_charges(monthly_charges, tenure):
    """Proxy for the historical, cumulative ``TotalCharges`` feature.

    Why a proxy is required
    -----------------------
    During training ``TotalCharges`` is the *observed* lifetime spend recorded
    by the billing system. That value does not exist for a customer profile
    that a user assembles by hand in the predictor form or in the What-If
    simulator, so the UI cannot supply a genuine measurement.

    ``TotalCharges = MonthlyCharges * max(1, tenure)`` is therefore a proxy
    reconstruction used consistently by the predictor, the batch engine and the
    simulator. The ``max(1, tenure)`` floor keeps the value non-zero for
    zero-tenure customers instead of collapsing to 0.

    Caveat: this is a proxy, not the true historical feature. The fitted
    StandardScaler learned its mean/std from real observed values, so a small
    systematic bias is expected for profiles where the proxy diverges from
    actual spend. The rule is intentionally centralised here so the predictor
    and the simulator can never disagree about the same customer.
    """
    return float(monthly_charges) * max(1, int(tenure))


def load_data(path=None):
    if path is None:
        path = DATA_PATH
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Assign rather than fillna(..., inplace=True): chained inplace assignment
    # is a no-op under pandas Copy-on-Write.
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


def get_eda_stats(df):
    stats = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "describe": df.describe().to_dict(),
        "churn_counts": df[TARGET].value_counts().to_dict(),
        "churn_pct": (df[TARGET].value_counts(normalize=True) * 100).round(2).to_dict(),
    }
    return stats


def _coerce_senior_citizen(series):
    """Coerce SeniorCitizen to a clean 0/1 integer column.

    The training CSV stores this feature as ``int64`` (0 or 1). Uploaded files
    frequently deliver the same logical values as text ("0", "1", "Yes", "No")
    or as floats (0.0, 1.0), so the same tolerant coercion is applied on every
    inference path to keep behaviour identical between the predictor, the batch
    engine and the simulator. Values that are not a recognisable 0/1 raise a
    clear error instead of failing deep inside the StandardScaler.
    """
    def _one(value):
        if pd.isna(value):
            raise ValueError(
                f"Feature 'SeniorCitizen' received an empty value; expected 0, 1, "
                f"'Yes' or 'No'."
            )
        if isinstance(value, str):
            token = value.strip().lower()
            if token in ("1", "1.0", "yes", "true", "y", "t"):
                return 1
            if token in ("0", "0.0", "no", "false", "n", "f"):
                return 0
            raise ValueError(
                f"Feature 'SeniorCitizen' received invalid value {value!r}; "
                f"allowed values are 0, 1, 'Yes', 'No'."
            )
        numeric = float(value)
        if numeric == 1.0:
            return 1
        if numeric == 0.0:
            return 0
        raise ValueError(
            f"Feature 'SeniorCitizen' received invalid value {value!r}; "
            f"allowed values are 0, 1, 'Yes', 'No'."
        )

    return series.apply(_one).astype("int64")


def _validate_categoricals(df, known_categories):
    """Reject categorical values outside the trained encoder domain.

    The fitted OneHotEncoder uses ``handle_unknown="ignore"``, which silently
    maps an unseen level onto the drop-first baseline category. That turns a
    typo such as "month-to-month" (lowercase) or an out-of-domain value such as
    "Three year" into a confident but wrong risk score. Validation here makes
    malformed input fail loudly instead.

    ``known_categories`` maps a column name to the exact ordered list of
    categories observed during training.
    """
    errors = []
    for col, allowed in known_categories.items():
        if col not in df.columns:
            continue
        bad = sorted({str(v) for v in df[col].dropna().unique()
                      if str(v) not in allowed})
        if bad:
            errors.append(
                f"Feature {col!r} received invalid value(s) {bad}; "
                f"allowed values are {sorted(allowed)}."
            )
    if errors:
        raise ValueError("Invalid categorical input -> " + " | ".join(errors))


def prepare_frame(customer_data, preprocessor, total_charges_median):
    """The single authoritative preprocessing path for ALL inference.

    Used by :meth:`ChurnModel.predict_single`, the batch/CSV scoring engine and
    the What-If simulator so that a given customer is always turned into model
    input in exactly the same way, regardless of entry point.

    Responsibilities (all inference-time input hygiene):
      * build the dataframe
      * verify every required column is present
      * coerce SeniorCitizen to a clean 0/1 int64
      * coerce TotalCharges to numeric and impute NaN with the **persisted
        training median** (never a batch-computed median)
      * validate categorical values against the trained encoder categories

    Scaling and one-hot encoding are deliberately left to the already-fitted
    ``preprocessor`` (a ColumnTransformer). This function never fits anything.

    Parameters
    ----------
    customer_data : dict or pandas.DataFrame
        Raw customer input. Extra columns (e.g. ``customerID``, ``Churn``) are
        ignored rather than rejected, so labelled datasets can be scored.
    preprocessor : fitted ColumnTransformer
        Supplies the training-time category domain for validation.
    total_charges_median : float
        The median persisted from the training split.
    """
    if isinstance(customer_data, pd.DataFrame):
        df = customer_data.copy()
    elif isinstance(customer_data, dict):
        df = pd.DataFrame([customer_data])
    else:
        raise TypeError(
            f"customer_data must be a dict or DataFrame, got "
            f"{type(customer_data).__name__}."
        )

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required field(s): {missing}")

    # SeniorCitizen: identical tolerant coercion on every path.
    df["SeniorCitizen"] = _coerce_senior_citizen(df["SeniorCitizen"])

    # TotalCharges: coerce, then impute with the persisted training median so a
    # customer's score never depends on which other rows happen to share the
    # batch. Uses the median captured from the training split, not the incoming
    # data, which is what makes the result reproducible.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(float(total_charges_median))

    # Categorical domain check against the trained encoder.
    _validate_categoricals(df, encoder_categories(preprocessor))

    return df[REQUIRED_COLS]


def encoder_categories(preprocessor):
    """Extract the exact per-column category domain learned during training."""
    try:
        ohe = preprocessor.named_transformers_["cat"]
    except (AttributeError, KeyError):
        return {}
    return {col: [str(c) for c in cats]
            for col, cats in zip(ohe.feature_names_in_, ohe.categories_)}


def preprocess(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Assign rather than fillna(..., inplace=True): chained inplace assignment
    # is a no-op under pandas Copy-on-Write.
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    df.drop(columns=DROP_COLS, inplace=True, errors="ignore")
    return df


def build_preprocessor():
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )
    return preprocessor


class ChurnModel:
    def __init__(self):
        self.preprocessor = build_preprocessor()
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        # Median of TotalCharges over the TRAINING split, captured at
        # prepare_data() time and persisted by save(). It is the single
        # imputation value used by every inference path, so a customer's score
        # never depends on batch composition.
        self.total_charges_median = FALLBACK_TOTAL_CHARGES_MEDIAN

    def prepare_data(self, df=None, test_size=0.2, random_state=42):
        """Split, then fit the preprocessor on the TRAINING split only.

        Note (latent leakage risk, no current impact): preprocess() imputes
        TotalCharges using a median computed over the WHOLE frame, before the
        train/test split, so the imputation value is influenced by the test
        split. The shipped dataset has no missing TotalCharges values, so no
        leakage actually occurs today, but it should be corrected by imputing
        inside the training split only if the pipeline is ever retrained on
        data that does contain nulls.
        """
        if df is None:
            df = load_data()
        df = preprocess(df)
        X = df.drop(columns=[TARGET])
        y = df[TARGET]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self.preprocessor.fit(self.X_train)
        self.feature_names = self.preprocessor.get_feature_names_out()
        # Persist the imputation value from the TRAINING split only. compute()
        # derives its own from the incoming frame, but because prepare_data() has
        # already imputed TotalCharges, self.X_train carries exactly the value
        # the scaler was fitted on.
        self.total_charges_median = float(self.X_train["TotalCharges"].median())
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_all(self):
        model_dict = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }

        X_train_t = self.preprocessor.transform(self.X_train)
        X_test_t = self.preprocessor.transform(self.X_test)

        best_f1 = -1
        for name, model in model_dict.items():
            model.fit(X_train_t, self.y_train)
            y_pred = model.predict(X_test_t)
            y_prob = model.predict_proba(X_test_t)[:, 1]

            metrics = {
                "Accuracy": round(accuracy_score(self.y_test, y_pred), 4),
                "Precision": round(precision_score(self.y_test, y_pred), 4),
                "Recall": round(recall_score(self.y_test, y_pred), 4),
                "F1-Score": round(f1_score(self.y_test, y_pred), 4),
                "ROC-AUC": round(roc_auc_score(self.y_test, y_prob), 4),
            }
            cm = confusion_matrix(self.y_test, y_pred)
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)

            self.models[name] = model
            self.results[name] = {
                "metrics": metrics,
                "confusion_matrix": cm,
                "roc_curve": (fpr, tpr),
                "y_pred": y_pred,
                "y_prob": y_prob,
            }

            if metrics["F1-Score"] > best_f1:
                best_f1 = metrics["F1-Score"]
                self.best_model_name = name
                self.best_model = model

        return self.results

    def get_feature_importance(self, model_name=None):
        if model_name is None:
            model_name = self.best_model_name
        model = self.models.get(model_name, None)
        if model is None and model_name == self.best_model_name:
            model = self.best_model
        if model is None:
            return None

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return None

        feature_imp = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        return feature_imp

    def save(self, model_dir=None):
        if model_dir is None:
            model_dir = MODEL_DIR
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(self.preprocessor, os.path.join(model_dir, "preprocessor.pkl"))
        joblib.dump(self.best_model, os.path.join(model_dir, "best_model.pkl"))
        joblib.dump(self.best_model_name, os.path.join(model_dir, "best_model_name.pkl"))
        joblib.dump(self.feature_names, os.path.join(model_dir, "feature_names.pkl"))
        # Imputation metadata: training-split median of TotalCharges.
        joblib.dump(float(self.total_charges_median),
                    os.path.join(model_dir, "total_charges_median.pkl"))

        results_save = {}
        for name, res in self.results.items():
            results_save[name] = {
                "metrics": res["metrics"],
            }
        joblib.dump(results_save, os.path.join(model_dir, "results.pkl"))

    @classmethod
    def load(cls, model_dir=None):
        if model_dir is None:
            model_dir = MODEL_DIR
        instance = cls()
        instance.preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.pkl"))
        instance.best_model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
        instance.best_model_name = joblib.load(os.path.join(model_dir, "best_model_name.pkl"))
        instance.feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
        instance.results = joblib.load(os.path.join(model_dir, "results.pkl"))
        # Backward compatible: artifacts saved before this metadata existed fall
        # back to the documented training-split median, so the shipped models/
        # directory keeps working without retraining.
        median_path = os.path.join(model_dir, "total_charges_median.pkl")
        if os.path.exists(median_path):
            instance.total_charges_median = float(joblib.load(median_path))
        # Retained so callers can re-save sidecar metadata (e.g. the median)
        # without needing to know the artifact directory.
        instance.model_dir = model_dir
        return instance

    def _transform(self, customer_data):
        """Run the ONE shared preparation path, then the fitted preprocessor.

        Never fits anything: the ColumnTransformer is already fitted and is used
        in transform-only mode.
        """
        frame = prepare_frame(customer_data, self.preprocessor,
                              self.total_charges_median)
        return self.preprocessor.transform(frame)

    def predict_single(self, customer_data: dict):
        X_transformed = self._transform(customer_data)
        prediction = self.best_model.predict(X_transformed)[0]
        probability = self.best_model.predict_proba(X_transformed)[0]

        return {
            "prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(float(probability[1]) * 100, 2),
            "no_churn_probability": round(float(probability[0]) * 100, 2),
        }

    def predict_batch(self, customers_df: pd.DataFrame):
        """Score many customers through the exact same path as predict_single.

        Guarantees single/batch equivalence: identical preparation, identical
        imputation, identical encoding and identical scaling, so a customer's
        probability does not depend on which other rows share the batch.
        """
        if len(customers_df) == 0:
            return np.array([]), np.array([])
        X_transformed = self._transform(customers_df)
        preds = self.best_model.predict(X_transformed)
        probs = self.best_model.predict_proba(X_transformed)[:, 1]
        return preds, probs


def train_and_save():
    print("Loading data...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")

    churn_model = ChurnModel()
    print("Preparing data...")
    churn_model.prepare_data(df)

    print("Training models...")
    results = churn_model.train_all()

    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    for name, res in results.items():
        print(f"\n{name}:")
        for metric, value in res["metrics"].items():
            print(f"  {metric}: {value}")

    print(f"\nBest Model: {churn_model.best_model_name}")
    print(f"Best F1-Score: {results[churn_model.best_model_name]['metrics']['F1-Score']}")

    print("\nSaving model and preprocessor...")
    churn_model.save()
    print("Model saved successfully!")

    return churn_model


if __name__ == "__main__":
    train_and_save()
