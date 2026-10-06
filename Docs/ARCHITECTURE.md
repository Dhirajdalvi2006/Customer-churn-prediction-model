# Project Architecture

## Overall System Diagram

```text
       [ User ]
          |
          V
   ( Streamlit UI )  --------------------+
          |                              |
          V                              V
[ BundlePredictionService ]      [ Explainability ]
          |                              |
          | (reads pointer)              |
          V                              V
[ active_bundle.json: "v2" ]       ( SHAP TreeExplainer )
          |                              |
          V                              V
    [ ModelBundle ]              [ 45 Transformed Features ]
          |
          +--> Input Contract Validator
          |
          +--> v2/manifest.json (pipeline layout)
          |
          +--> Preprocessing Pipeline (45 feats)
          |
          +--> GradientBoostingClassifier
          |
          V
    [ Prediction Result ]
```

## Production Pointer Mechanism
The source-of-truth for what model governs live responses is **strictly** isolated to a single file: `models/active_bundle.json`.
The active production system reads this file and routes all subsequent loading through `backend.model_bundle.BundleLoader`. 

## Legacy (V1) Bundle
- Located at: `models/bundles/v1`
- Format: `estimator_only`
- Core inference relies on the `LogisticRegression` model + separated `preprocessor.pkl`. Emits 30 intermediate features.
- Persists as a fully operational fallback target.

## Active (V2) Bundle
- Located at: `models/bundles/v2`
- Format: `pipeline`
- Core inference relies on `GradientBoostingClassifier` natively bundled within a single Scikit-learn Pipeline alongside transformation logic. Emits 45 intermediate features via `OneHotEncoder(drop=None)`.
- Replaces legacy state without destructive overwriting.

## Explanation Architecture 
The application does not tie XAI assumptions directly to `LogisticRegression` or a fixed `30` feature layout. `backend/explainability.py` abstracts bundle execution via a unified `ModelView` interface, parsing the `manifest.json` metadata to route the estimator to the fundamentally correct explainer (`TreeExplainer` for V2) automatically, returning dynamically sized `shap_values` tailored strictly to the actual executed artifact.

## Safety & Integrity
- An unalterable `v2_promotion_record.json` logs the metadata delta between the shadow V1->V2 migration.
- All artifact payloads (9 distinct pkl/json files) are rigorously cryptographically hashed against frozen Phase 3B baselines in test suites to absolutely prohibit silent drift, unauthorized modification, or structural mutation.
- The `pytest` architecture operates *bundle-aware*, verifying metadata coherence dynamically around the current pointer state rather than brittle, constant assertions.