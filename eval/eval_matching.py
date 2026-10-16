"""Evaluate entity resolution: blocking quality + pairwise P/R/F1 for 3 strategies
(exact CIF -> fuzzy rules -> ML classifier), then clustering + golden records.

Usage (from repo root):  python eval/eval_matching.py
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdq.matching import (FEATURES, candidate_pairs, cluster, golden_records,  # noqa: E402
                          label_pairs, pair_features, predict_exact, predict_rules, prepare)

RAW = Path("data/raw/companies.csv")
OUT = Path("data/processed")


def true_pairs(df: pd.DataFrame) -> set[tuple[int, int]]:
    tp = set()
    for _, g in df.groupby("true_cluster_id"):
        tp.update(combinations(sorted(g.index), 2))
    return tp


def prf(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    return {"precision": round(pr, 3), "recall": round(rc, 3), "f1": round(f1, 3)}


def cluster_prf(df, cluster_ids, all_true):
    """Pairwise metrics after transitive closure (what the business actually gets)."""
    pred = set()
    for _, g in pd.Series(df.index, index=df.index).groupby(cluster_ids.values):
        pred.update(combinations(sorted(g.values), 2))
    tp = len(pred & all_true)
    p = tp / len(pred) if pred else 0
    r = tp / len(all_true) if all_true else 0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0}


def build(df):
    t0 = time.time()
    d = prepare(df.reset_index(drop=True))
    cands = candidate_pairs(d)
    feats = pair_features(d, sorted(cands))
    feats["y"] = label_pairs(d, feats)
    tp = true_pairs(d)
    n = len(d)
    block = {"records": n,
             "candidate_pairs": len(cands),
             "reduction_ratio": round(1 - len(cands) / (n * (n - 1) / 2), 4),
             "pair_completeness": round(len(cands & tp) / len(tp), 3),  # blocking recall
             "seconds": round(time.time() - t0, 1)}
    return d, feats, tp, block


def main():
    df = pd.read_csv(RAW, dtype={"postal_code": str, "phone": str})
    # split by TRUE entity so no entity leaks between train and test
    ents = df.true_cluster_id.unique()
    rng = np.random.default_rng(0)
    test_ents = set(rng.choice(ents, size=len(ents) // 2, replace=False))
    df_tr, df_te = df[~df.true_cluster_id.isin(test_ents)], df[df.true_cluster_id.isin(test_ents)]

    d_tr, f_tr, _, b_tr = build(df_tr)
    d_te, f_te, tp_te, b_te = build(df_te)

    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "gbm": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0),
    }
    results = {"blocking_test": b_te,
               "exact_cif": prf(f_te.y, predict_exact(f_te)),
               "fuzzy_rules": prf(f_te.y, predict_rules(f_te))}
    preds = {"exact_cif": predict_exact(f_te), "fuzzy_rules": predict_rules(f_te)}
    for name, m in models.items():
        m.fit(f_tr[FEATURES], f_tr.y)
        p = (m.predict_proba(f_te[FEATURES])[:, 1] >= 0.5).astype(int)
        results[f"ml_{name}"] = prf(f_te.y, p)
        preds[f"ml_{name}"] = p

    # Note: pairwise metrics above are on candidate pairs; true pairs lost in
    # blocking are counted as misses in the end-to-end cluster metrics below.
    results["end_to_end_clusters"] = {}
    for name, p in preds.items():
        cids = cluster(d_te, f_te, p)
        results["end_to_end_clusters"][name] = cluster_prf(d_te, cids, tp_te) | {
            "golden_records": int(cids.nunique()), "true_entities": int(d_te.true_cluster_id.nunique())}

    best = max(preds, key=lambda k: results["end_to_end_clusters"][k]["f1"])
    gold = golden_records(d_te, cluster(d_te, f_te, preds[best]))
    OUT.mkdir(parents=True, exist_ok=True)
    gold.to_csv(OUT / "golden_records_test.csv", index=False)
    f_te.assign(pred=preds[best]).to_csv(OUT / "pairs_test.csv", index=False)
    results["best_strategy"] = best
    (OUT / "matching_metrics.json").write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print("\n| Strategy | Precision | Recall | F1 | Golden records |\n|---|---|---|---|---|")
    for k, v in results["end_to_end_clusters"].items():
        print(f"| {k} | {v['precision']} | {v['recall']} | {v['f1']} | {v['golden_records']} / {v['true_entities']} true |")


if __name__ == "__main__":
    main()
