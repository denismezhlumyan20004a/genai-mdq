"""Day 2: entity resolution — blocking, similarity features, classifier,
clustering and golden record (survivorship)."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from .normalize import (normalize_address, normalize_cif, normalize_email,
                        normalize_name, normalize_phone)

# ---------- string similarity (rapidfuzz if installed, pure-python fallback) ----------
try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler

    def ratio(a, b): return fuzz.ratio(a, b) / 100
    def token_sort(a, b): return fuzz.token_sort_ratio(a, b) / 100
    def token_set(a, b): return fuzz.token_set_ratio(a, b) / 100
    def jaro_winkler(a, b): return JaroWinkler.similarity(a, b)
except ImportError:
    from difflib import SequenceMatcher

    def ratio(a, b): return SequenceMatcher(None, a, b).ratio() if a and b else 0.0
    def token_sort(a, b): return ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))

    def token_set(a, b):
        sa, sb = set(a.split()), set(b.split())
        inter = " ".join(sorted(sa & sb))
        da, db = " ".join(sorted(sa - sb)), " ".join(sorted(sb - sa))
        t1, t2 = (inter + " " + da).strip(), (inter + " " + db).strip()
        return max(ratio(inter, t1), ratio(inter, t2), ratio(t1, t2)) if inter else ratio(a, b)

    def jaro_winkler(a, b, p=0.1):
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        md = max(len(a), len(b)) // 2 - 1
        am, bm = [False] * len(a), [False] * len(b)
        m = 0
        for i, ca in enumerate(a):
            for j in range(max(0, i - md), min(len(b), i + md + 1)):
                if not bm[j] and b[j] == ca:
                    am[i] = bm[j] = True
                    m += 1
                    break
        if m == 0:
            return 0.0
        t, k = 0, 0
        for i in range(len(a)):
            if am[i]:
                while not bm[k]:
                    k += 1
                t += a[i] != b[k]
                k += 1
        jaro = (m / len(a) + m / len(b) + (m - t / 2) / m) / 3
        pre = 0
        for ca, cb in zip(a[:4], b[:4]):
            if ca != cb:
                break
            pre += 1
        return jaro + pre * p * (1 - jaro)


# ---------- 1. normalization ----------
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["n_name"] = out["name"].map(normalize_name)
    out["n_addr"] = out["address"].map(normalize_address)
    out["n_cif"] = out["cif"].map(normalize_cif)
    out["n_phone"] = out["phone"].map(normalize_phone)
    out["n_email"] = out["email"].map(normalize_email)
    out["n_pc"] = out["postal_code"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    out.loc[out["postal_code"].isna(), "n_pc"] = ""
    return out


# ---------- 2. blocking ----------
def blocking_keys(row) -> list[str]:
    keys = []
    if row.n_cif:
        keys.append("cif:" + row.n_cif)
    if row.n_phone:
        keys.append("ph:" + row.n_phone)
    tokens = sorted(row.n_name.split())
    if row.n_pc and tokens:
        keys.append(f"pc_tok:{row.n_pc}:{tokens[0][:4]}")
    if tokens:  # sorted-token prefix: robust to word order
        keys.append("tok2:" + "".join(t[:3] for t in tokens[:2]))
    if row.n_email:
        keys.append("em:" + row.n_email)
    return keys


def candidate_pairs(df: pd.DataFrame, max_block: int = 60) -> set[tuple[int, int]]:
    blocks = defaultdict(list)
    for idx, row in df.iterrows():
        for k in blocking_keys(row):
            blocks[k].append(idx)
    pairs = set()
    for members in blocks.values():
        if 1 < len(members) <= max_block:  # skip giant blocks (too generic keys)
            pairs.update(combinations(sorted(members), 2))
    return pairs


# ---------- 3. pair features ----------
FEATURES = ["name_ratio", "name_token_sort", "name_token_set", "name_jw", "addr_token_sort",
            "cif_eq", "cif_close", "cif_missing", "phone_eq", "email_eq", "pc_eq", "city_eq"]


def _cif_close(a: str, b: str) -> int:
    return int(len(a) == len(b) == 9 and sum(x != y for x, y in zip(a, b)) == 1)


def pair_features(df: pd.DataFrame, pairs) -> pd.DataFrame:
    rows = []
    for i, j in pairs:
        a, b = df.loc[i], df.loc[j]
        rows.append({
            "i": i, "j": j,
            "name_ratio": ratio(a.n_name, b.n_name),
            "name_token_sort": token_sort(a.n_name, b.n_name),
            "name_token_set": token_set(a.n_name, b.n_name),
            "name_jw": jaro_winkler(a.n_name, b.n_name),
            "addr_token_sort": token_sort(a.n_addr, b.n_addr),
            "cif_eq": int(bool(a.n_cif) and a.n_cif == b.n_cif),
            "cif_close": _cif_close(a.n_cif, b.n_cif),
            "cif_missing": int(not a.n_cif or not b.n_cif),
            "phone_eq": int(bool(a.n_phone) and a.n_phone == b.n_phone),
            "email_eq": int(bool(a.n_email) and a.n_email == b.n_email),
            "pc_eq": int(bool(a.n_pc) and a.n_pc == b.n_pc),
            "city_eq": int(a.city == b.city),
        })
    return pd.DataFrame(rows)


def label_pairs(df: pd.DataFrame, feats: pd.DataFrame) -> pd.Series:
    cid = df["true_cluster_id"]
    return (cid.loc[feats.i].values == cid.loc[feats.j].values).astype(int)


# ---------- 4. decision strategies ----------
def predict_exact(feats: pd.DataFrame) -> np.ndarray:
    """Baseline: same CIF -> match."""
    return feats["cif_eq"].values.astype(int)


def predict_rules(feats: pd.DataFrame, name_thr: float = 0.88) -> np.ndarray:
    """Hand-written fuzzy rules (interpretable)."""
    strong_id = (feats.cif_eq == 1) | (feats.phone_eq == 1) | (feats.email_eq == 1)
    fuzzy_name = feats.name_token_sort >= name_thr
    same_place = (feats.pc_eq == 1) | (feats.addr_token_sort >= 0.8)
    return ((strong_id & (feats.name_token_set >= 0.6)) | (fuzzy_name & same_place)).values.astype(int)


# ---------- 5. clustering + golden record ----------
def cluster(df: pd.DataFrame, feats: pd.DataFrame, pred: np.ndarray) -> pd.Series:
    g = nx.Graph()
    g.add_nodes_from(df.index)
    g.add_edges_from(zip(feats.i[pred == 1], feats.j[pred == 1]))
    cid = {}
    for k, comp in enumerate(nx.connected_components(g)):
        for n in comp:
            cid[n] = k
    return pd.Series(cid).reindex(df.index)


SOURCE_PRIORITY = {"ERP": 0, "Facturacion": 1, "CRM": 2, "Web": 3}


def golden_records(df: pd.DataFrame, cluster_ids: pd.Series) -> pd.DataFrame:
    """Survivorship: per field, take the value from the most trusted / most recent
    record that has it; for CIF prefer the most frequent valid value."""
    from .quality import is_valid_cif

    d = df.assign(pred_cluster=cluster_ids.values,
                  _prio=df["source"].map(SOURCE_PRIORITY).fillna(9),
                  _date=pd.to_datetime(df["updated_at"]))
    d = d.sort_values(["pred_cluster", "_prio", "_date"], ascending=[True, True, False])
    out = []
    for k, grp in d.groupby("pred_cluster", sort=False):
        rec = {"golden_id": f"G{k:05d}", "n_records": len(grp),
               "source_record_ids": ",".join(grp.record_id)}
        cifs = [c for c in grp.n_cif if c and is_valid_cif(c)]
        rec["cif"] = max(set(cifs), key=cifs.count) if cifs else None
        for col in ["name", "address", "postal_code", "city", "phone", "email"]:
            vals = grp[col].dropna()
            rec[col] = vals.iloc[0] if len(vals) else None
        out.append(rec)
    return pd.DataFrame(out)
