import json
import os
import sys
import joblib
import hashlib
import platform
import sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

ROOT = r"C:\Users\dalvi\Downloads\data analusis\customer_churn_project"
os.chdir(ROOT)

print("python:", platform.python_version(), "| sklearn:", sklearn.__version__)

res = joblib.load("models/results.pkl")
print("results.pkl keys:", list(res.keys())[:12] if isinstance(res, dict) else type(res))
if isinstance(res, dict):
    for k, v in res.items():
        if isinstance(v, dict):
            print("   ", k, "->", list(v.keys()))

names = joblib.load("models/feature_names.pkl")
print("feature_names.pkl type:", type(names), "len:", len(names))
print("   first 5:", list(names)[:5])

bmn = None
p = "models/best_model_name.pkl"
if os.path.exists(p):
    bmn = joblib.load(p)
    print("best_model_name.pkl:", repr(bmn))

prep = joblib.load("models/preprocessor.pkl")
print("preprocessor type:", type(prep).__name__)
print("named_transformers_ keys:", list(prep.named_transformers_.keys()))
print("num block:", prep.named_transformers_["num"])
print("cat block:", type(prep.named_transformers_["cat"]).__name__)

v1f = list(prep.get_feature_names_out())
print("v1 features:", len(v1f))
print("   first 4:", v1f[:4])
print("   last 2 :", v1f[-2:])
print("EQUALS feature_names.pkl?", v1f == list(names))
