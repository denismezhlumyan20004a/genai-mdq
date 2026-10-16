"""Service layer: all business logic used by the API (framework-agnostic, unit-tested)."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from .extract import LLMExtractor, RegexExtractor
from .invoice_rules import SupplierIndex, validate_invoice
from .matching import FEATURES, candidate_pairs, cluster, golden_records, pair_features, predict_rules, prepare
from .quality import quality_report

RECORD_COLS = ["record_id", "name", "cif", "address", "postal_code", "city", "phone", "email",
               "source", "updated_at"]


@dataclass
class Settings:
    master_path: str = os.getenv("MDQ_MASTER_PATH", "data/raw/companies.csv")
    model_path: str = os.getenv("MDQ_MATCHER_PATH", "models/matcher.joblib")
    match_threshold: float = float(os.getenv("MDQ_MATCH_THRESHOLD", "0.5"))
    max_pdf_mb: float = float(os.getenv("MDQ_MAX_PDF_MB", "10"))


@dataclass
class MDQService:
    settings: Settings = field(default_factory=Settings)

    def __post_init__(self):
        p = Path(self.settings.model_path)
        self.model = joblib.load(p) if p.exists() else None  # None -> fuzzy-rule fallback
        mp = Path(self.settings.master_path)
        self.supplier_index = SupplierIndex(pd.read_csv(mp, dtype=str)) if mp.exists() else None
        self._llm = None

    # ------------------------------------------------------------ matching
    def match(self, records: list[dict]) -> dict:
        df = pd.DataFrame(records).reindex(columns=RECORD_COLS)
        df["record_id"] = df["record_id"].fillna(pd.Series([f"in_{i}" for i in range(len(df))]))
        df["source"] = df["source"].fillna("API")
        df["updated_at"] = df["updated_at"].fillna(pd.Timestamp.today().date().isoformat())
        d = prepare(df.reset_index(drop=True))
        pairs = sorted(candidate_pairs(d))
        feats = pair_features(d, pairs) if pairs else pd.DataFrame(columns=["i", "j"] + FEATURES)
        if len(feats) and self.model is not None:
            proba = self.model.predict_proba(feats[FEATURES])[:, 1]
            pred = (proba >= self.settings.match_threshold).astype(int)
            method = "ml_model"
        else:
            pred = predict_rules(feats).astype(int) if len(feats) else []
            proba = pred
            method = "fuzzy_rules"
        pred = pd.Series(pred, dtype=int).values
        cids = cluster(d, feats, pred)
        gold = golden_records(d, cids)
        matches = [{"record_a": d.record_id[i], "record_b": d.record_id[j], "score": round(float(s), 3)}
                   for i, j, s, p in zip(feats.i, feats.j, proba, pred) if p == 1]
        return {"method": method, "records_in": len(d), "golden_records_out": len(gold),
                "duplicates_found": len(d) - len(gold), "matches": matches,
                "golden_records": json.loads(gold.to_json(orient="records"))}  # native types for JSON

    # ------------------------------------------------------------ data quality
    def validate(self, records: list[dict]) -> dict:
        return quality_report(records)

    # ------------------------------------------------------------ invoices
    def extract_invoice(self, pdf_bytes: bytes, extractor: str = "regex") -> dict:
        if len(pdf_bytes) > self.settings.max_pdf_mb * 1024 * 1024:
            raise ValueError("file too large")
        if not pdf_bytes.startswith(b"%PDF"):
            raise ValueError("not a PDF file")
        ex = RegexExtractor() if extractor == "regex" else self._llm_extractor()
        # delete=False + manual cleanup: on Windows an open temp file cannot be reopened by pdfplumber
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            tmp = f.name
        try:
            inv, meta = ex.extract(tmp)
        finally:
            os.unlink(tmp)
        if inv is None:
            return {"extractor": ex.name, "invoice": None, "route": "needs_review",
                    "issues": ["extraction_failed"], "meta": meta}
        v = validate_invoice(inv, self.supplier_index)
        return {"extractor": ex.name, "invoice": inv.model_dump(mode="json"), **v, "meta": meta}

    def _llm_extractor(self) -> LLMExtractor:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("LLM extractor unavailable: ANTHROPIC_API_KEY not set")
        if self._llm is None:
            self._llm = LLMExtractor(input_mode=os.getenv("MDQ_LLM_INPUT", "text"), cache_dir=None)
        return self._llm
