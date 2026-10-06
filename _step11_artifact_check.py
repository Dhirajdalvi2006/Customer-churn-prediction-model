"""STEP 11 - Verify all 9 protected artifacts remain byte-identical."""
import hashlib, json, os

BASE = os.path.dirname(os.path.abspath(__file__))
BASELINE_FILE = os.path.join(BASE, '_baseline_hashes.json')

artifacts = [
    'models/best_model.pkl',
    'models/preprocessor.pkl',
    'models/results.pkl',
    'models/feature_names.pkl',
    'models/total_charges_median.pkl',
    'models/experiments/champion.pkl',
    'models/bundles/v1/model.pkl',
    'models/bundles/v1/preprocessor.pkl',
    'models/bundles/v2/model.pkl',
]

# Rebuild baseline if missing
if not os.path.exists(BASELINE_FILE):
    baseline = {}
    for a in artifacts:
        path = os.path.join(BASE, a)
        baseline[a] = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    with open(BASELINE_FILE, 'w') as f:
        json.dump(baseline, f, indent=2)
    print('BASELINE REBUILT')
else:
    with open(BASELINE_FILE) as f:
        baseline = json.load(f)

all_ok = True
for a in artifacts:
    path = os.path.join(BASE, a)
    actual = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    expected = baseline.get(a)
    if actual == expected:
        print(f'  OK  {a}')
    else:
        print(f'  FAIL {a}: {actual} != {expected}')
        all_ok = False

print()
print('ARTIFACT INTEGRITY:', 'PASS' if all_ok else 'FAIL')