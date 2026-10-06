import os
import json
import platform
import datetime
import joblib
import numpy as np
import scipy
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from backend.model import load_data, CATEGORICAL_COLS, NUMERICAL_COLS, TARGET, DROP_COLS

EXPERIMENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "experiments")

def get_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_COLS),
            ('cat', categorical_transformer, CATEGORICAL_COLS)
        ])
    return preprocessor

def run_experiment(context=None):
    """Run the Phase 3B model comparison.

    If ``context`` is a dict, it is populated in-place with the objects and
    configuration actually used by the run, so results can be persisted
    verbatim without recomputing or re-deriving any value. It is a pure
    out-parameter: the return value and the methodology are unchanged.
    """
    df = load_data()
    X = df.drop(columns=DROP_COLS + [TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = get_preprocessor()

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42)
    }

    params = {
        "LogisticRegression": {"classifier__C": [0.1, 1.0, 10.0]},
        "DecisionTreeClassifier": {"classifier__max_depth": [3, 5, 10]},
        "RandomForestClassifier": {"classifier__n_estimators": [50, 100], "classifier__max_depth": [5, 10]},
        "GradientBoostingClassifier": {"classifier__n_estimators": [50, 100], "classifier__learning_rate": [0.1, 0.01]}
    }

    results = {}
    best_pipelines = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        grid = GridSearchCV(
            pipeline, params[name], cv=cv, scoring='roc_auc', n_jobs=-1
        )
        grid.fit(X_train, y_train)
        
        # Evaluation on test set
        best_pipe = grid.best_estimator_
        y_pred = best_pipe.predict(X_test)
        y_prob = best_pipe.predict_proba(X_test)[:, 1]
        
        metrics = {
            'ROC-AUC': roc_auc_score(y_test, y_prob),
            'PR-AUC': average_precision_score(y_test, y_prob, pos_label='Yes'),
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, pos_label='Yes'),
            'Recall': recall_score(y_test, y_pred, pos_label='Yes'),
            'F1': f1_score(y_test, y_pred, pos_label='Yes')
        }
        
        results[name] = {
            "best_params": grid.best_params_,
            "cv_score_mean": grid.best_score_,
            "cv_score_std": grid.cv_results_['std_test_score'][grid.best_index_],
            "test_metrics": metrics
        }
        best_pipelines[name] = best_pipe

    if context is not None:
        # Champion chosen on development/CV ROC-AUC only. The final test set
        # is never consulted for selection.
        champion_name = max(results, key=lambda n: results[n]["cv_score_mean"])
        context.update({
            "models": models,
            "params": params,
            "cv": cv,
            "scoring": 'roc_auc',
            "test_size": 0.2,
            "random_state": 42,
            "positive_label": 'Yes',
            "n_splits": cv.n_splits,
            "best_pipelines": best_pipelines,
            "champion_name": champion_name,
        })

    return results

def _json_safe(obj):
    """Convert numpy scalars/arrays to plain Python so json can serialise them."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_experiment_artifacts(results, context, experiment_dir=None):
    """Persist the Phase 3B experiment outputs to ``models/experiments/``.

    Every value written here comes from ``results``/``context`` produced by
    the real run; nothing is hardcoded or recomputed.
    """
    experiment_dir = experiment_dir or EXPERIMENT_DIR
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    champion_name = context["champion_name"]
    champion_pipeline = context["best_pipelines"][champion_name]

    # Cross-validation results for every model
    cv_results = {
        name: {
            "best_params": res["best_params"],
            "cv_roc_auc_mean": res["cv_score_mean"],
            "cv_roc_auc_std": res["cv_score_std"],
        }
        for name, res in results.items()
    }

    # Final-test metrics, reported for all models but never used to select
    test_metrics = {
        name: res["test_metrics"] for name, res in results.items()
    }

    metadata = {
        "experiment_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "random_state": context["random_state"],
        "test_size": context["test_size"],
        "train_size": f"{1 - context['test_size']:.0%} (stratified)",
        "positive_label": context["positive_label"],
        "cv_strategy": "StratifiedKFold(shuffle=True)",
        "cv_n_splits": context["n_splits"],
        "scoring": context["scoring"],
        "model_configurations": {
            name: {k: v for k, v in model.get_params().items()
                   if k in ("max_iter", "random_state", "C", "max_depth",
                            "n_estimators", "learning_rate")}
            for name, model in context["models"].items()
        },
        "model_random_states": {
            name: model.get_params().get("random_state")
            for name, model in context["models"].items()
        },
        # The three tree/ensemble models are stochastic and are explicitly seeded
        # with random_state. LogisticRegression is deterministic by construction
        # (lbfgs uses no randomness), so random_state=None is expected and correct.
        "stochastic_models_seeded": {
            name: model.get_params().get("random_state")
            for name, model in context["models"].items()
            if model.__class__.__name__ != "LogisticRegression"
        },
        "all_stochastic_models_seeded": all(
            model.get_params().get("random_state") == context["random_state"]
            for name, model in context["models"].items()
            if model.__class__.__name__ != "LogisticRegression"
        ),
        "deterministic_models": {
            name: "lbfgs solver uses no randomness; random_state not required"
            for name, model in context["models"].items()
            if model.__class__.__name__ == "LogisticRegression"
        },
        "param_grids": {name: {k: list(v) for k, v in grid.items()}
                        for name, grid in context["params"].items()},
        "cv_results": cv_results,
        "selected_champion": champion_name,
        "champion_selection_criterion": "highest mean CV ROC-AUC on training folds",
        "champion_cv_roc_auc_mean": results[champion_name]["cv_score_mean"],
        "champion_cv_roc_auc_std": results[champion_name]["cv_score_std"],
        "champion_test_metrics": results[champion_name]["test_metrics"],
        "final_test_metrics_untouched": (
            "The 20% test set was used for reporting only. Champion selection "
            "used development/CV ROC-AUC exclusively."
        ),
    }

    joblib.dump(results, os.path.join(experiment_dir, "cv_results.pkl"))
    joblib.dump(champion_pipeline, os.path.join(experiment_dir, "champion.pkl"))
    with open(os.path.join(experiment_dir, "test_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(_json_safe(test_metrics), fh, indent=2)
    with open(os.path.join(experiment_dir, "experiment_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(_json_safe(metadata), fh, indent=2)

    return experiment_dir


if __name__ == "__main__":
    context = {}
    results = run_experiment(context)
    saved_to = save_experiment_artifacts(results, context)
    print(f"Experiment completed and saved to: {saved_to}")
    print(f"Selected champion (highest mean CV ROC-AUC): {context['champion_name']}")
