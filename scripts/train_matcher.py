"""Train the pair classifier on all labelled data and save it for the API.

Usage: python scripts/train_matcher.py  ->  models/matcher.joblib
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdq.matching import FEATURES, candidate_pairs, label_pairs, pair_features, prepare  # noqa: E402

df = pd.read_csv("data/raw/companies.csv", dtype={"postal_code": str, "phone": str})
d = prepare(df)
feats = pair_features(d, sorted(candidate_pairs(d)))
y = label_pairs(d, feats)
model = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(feats[FEATURES], y)
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/matcher.joblib")
print(f"trained on {len(feats)} pairs ({y.sum()} matches) -> models/matcher.joblib")
