"""STEP 12 - Rollback Test"""
import json, os, sys
sys.path.insert(0, '.')
import backend.prediction_service as ps
from backend.prediction_service import get_prediction_service
from backend.model import load_data

POINTER = 'models/active_bundle.json'

print('=== 1. Starting at V2 ===')
with open(POINTER) as f:
    print('Initial pointer:', f.read().strip())

print('\n=== 2. Changing pointer V2 -> v1 ===')
with open(POINTER, 'w') as f:
    json.dump({'active_version': 'v1'}, f)

ps._SERVICE_CACHE.clear()
service_v1 = get_prediction_service('models')
meta_v1 = service_v1.metadata()
print('Loaded V1 bundle_version:', meta_v1['bundle_version'])
print('Model type:', meta_v1['model_type'])
print('Model format:', meta_v1['model_format'])
print('Positive class:', meta_v1['positive_class'])
print('Threshold:', meta_v1['decision_threshold'])
print('Transformed features:', meta_v1['transformed_feature_count'])

assert meta_v1['bundle_version'] == 'v1', 'Failed: not v1'
assert meta_v1['model_type'] == 'LogisticRegression', 'Failed: not LogisticRegression'
assert meta_v1['model_format'] == 'estimator_only', 'Failed: not estimator_only'
assert meta_v1['positive_class'] in ('1', 'Yes'), 'Failed: not Yes/1'
assert meta_v1['decision_threshold'] == 0.5, 'Failed: not 0.5'
assert meta_v1['transformed_feature_count'] == 30, 'Failed: not 30 features'

# Run deterministic prediction on V1
df = load_data().drop(columns=['Churn', 'customerID'])
row = df.iloc[0].to_dict()
r1 = service_v1.predict_single(row)
print(f'V1 prediction: {r1["prediction"]} (prob: {r1["churn_probability"]}%)')

print('\n=== 3. Restoring pointer v1 -> v2 ===')
with open(POINTER, 'w') as f:
    json.dump({'active_version': 'v2'}, f)

ps._SERVICE_CACHE.clear()
service_v2 = get_prediction_service('models')
meta_v2 = service_v2.metadata()
print('Restored V2 bundle_version:', meta_v2['bundle_version'])
print('Model type:', meta_v2['model_type'])
print('Model format:', meta_v2['model_format'])
print('Positive class:', meta_v2['positive_class'])
print('Threshold:', meta_v2['decision_threshold'])
print('Transformed features:', meta_v2['transformed_feature_count'])

assert meta_v2['bundle_version'] == 'v2', 'Failed to restore v2'
assert meta_v2['model_type'] == 'GradientBoostingClassifier', 'Failed: not GBDT'
assert meta_v2['transformed_feature_count'] == 45, 'Failed: not 45 features'

# Run prediction on V2 to verify
r2 = service_v2.predict_single(row)
print(f'V2 prediction: {r2["prediction"]} (prob: {r2["churn_probability"]}%)')

print('\nROLLBACK TEST: PASS')
