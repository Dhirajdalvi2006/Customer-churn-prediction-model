import pytest
import os
import json
import joblib
import pandas as pd
from backend.model_bundle import ModelBundle, BundleLoader

# --- Fixtures ---
@pytest.fixture
def bundles_dir():
    return os.path.join("models", "bundles")

def test_v1_exists(bundles_dir):
    assert os.path.exists(os.path.join(bundles_dir, "v1", "manifest.json"))
    assert os.path.exists(os.path.join(bundles_dir, "v1", "model.pkl"))
    assert os.path.exists(os.path.join(bundles_dir, "v1", "preprocessor.pkl"))

def test_v2_exists(bundles_dir):
    assert os.path.exists(os.path.join(bundles_dir, "v2", "manifest.json"))
    assert os.path.exists(os.path.join(bundles_dir, "v2", "model.pkl"))

def test_manifests_load(bundles_dir):
    v1 = ModelBundle(os.path.join(bundles_dir, "v1"))
    v2 = ModelBundle(os.path.join(bundles_dir, "v2"))
    assert v1.manifest["bundle_version"] == "v1"
    assert v2.manifest["bundle_version"] == "v2"

def test_formats(bundles_dir):
    v1 = ModelBundle(os.path.join(bundles_dir, "v1"))
    v2 = ModelBundle(os.path.join(bundles_dir, "v2"))
    assert v1.manifest["model_format"] == "estimator_only"
    assert v2.manifest["model_format"] == "pipeline"
    assert v1.preprocessor is not None
    assert v2.preprocessor is None

def test_active_pointer():
    """active_bundle.json must point to a known, loadable bundle.

    Phase 4 Step 12D: updated from a v1-specific assertion to a bundle-aware
    check. The pointer may now legitimately say v2 after the controlled
    promotion documented in models/experiments/v2_promotion_record.json.
    """
    with open("models/active_bundle.json", "r") as f:
        active = json.load(f)
    assert active["active_version"] in ("v1", "v2"), (
        f"active_bundle.json must point to a known bundle, "
        f"got {active['active_version']!r}"
    )

def test_production_artifacts_untouched():
    import hashlib
    def get_hash(path):
        return hashlib.sha256(open(path, "rb").read()).hexdigest().upper()
    
    expected = {
        "models/best_model.pkl": "66400195EB9CFEF8AFAAF9EE59EE83E19BAA1E9AFA70A4DDC9FC49CE80D9010E",
        "models/preprocessor.pkl": "FCC2D4E47FA218FCE6056F7904E4B54F7F9C58FDD6CCBD04FF20B7A5210E028F",
        "models/experiments/champion.pkl": "372DDF6C9EC977460D078A244A97B8781295394BC8F54ACB2DC8031F3B736589"
    }
    for path, hash_val in expected.items():
        assert get_hash(path) == hash_val


def test_v1_model_hash_matches_source():
    import hashlib
    v1_h = hashlib.sha256(open("models/bundles/v1/model.pkl", "rb").read()).hexdigest()
    src_h = hashlib.sha256(open("models/best_model.pkl", "rb").read()).hexdigest()
    assert v1_h == src_h

def test_v1_preprocessor_hash_matches_source():
    import hashlib
    v1_h = hashlib.sha256(open("models/bundles/v1/preprocessor.pkl", "rb").read()).hexdigest()
    src_h = hashlib.sha256(open("models/preprocessor.pkl", "rb").read()).hexdigest()
    assert v1_h == src_h

def test_v2_model_hash_matches_champion_source():
    import hashlib
    v2_h = hashlib.sha256(open("models/bundles/v2/model.pkl", "rb").read()).hexdigest()
    src_h = hashlib.sha256(open("models/experiments/champion.pkl", "rb").read()).hexdigest()
    assert v2_h == src_h

