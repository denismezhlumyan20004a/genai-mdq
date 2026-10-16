"""Evaluate invoice extraction + validation against ground truth.

Metrics per extractor:
  * field-level accuracy (10 scalar fields), line-item accuracy, % documents 100% correct
  * schema-valid rate, avg latency, cost per document
  * validation: recall/precision of injected arithmetic errors (total_mismatch)
  * routing: % auto-approved and "unsafe auto-approve" rate
    (auto-approved although extraction was wrong or the invoice had an error)
  * supplier linked to the correct master-data entity

Usage (repo root):
  python eval/eval_extraction.py                      # regex baseline only (no API key needed)
  python eval/eval_extraction.py --llm text pdf       # + Claude on text and on native PDF
  MDQ_MODEL=claude-sonnet-4-5 python eval/eval_extraction.py --llm text
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdq.extract import LLMExtractor, RegexExtractor, pdf_text  # noqa: E402
from mdq.invoice_rules import SupplierIndex, validate_invoice  # noqa: E402
from mdq.normalize import normalize_cif, normalize_name  # noqa: E402
from mdq.schemas import SCALAR_FIELDS  # noqa: E402

INV_DIR = Path("data/invoices")
OUT = Path("data/processed")
MONEY = {"subtotal", "vat_amount", "total", "vat_rate"}


def same(field: str, pred, true) -> bool:
    if pred is None or true is None:
        return pred is None and true is None
    if field in MONEY:
        return abs(float(pred) - float(true)) <= 0.01
    if field.endswith("_cif"):
        return normalize_cif(pred) == normalize_cif(true)
    if field.endswith("_name"):  # legal-form/accents/case-insensitive
        return normalize_name(pred) == normalize_name(true)
    if field == "issue_date":
        return str(pred) == str(true)
    return str(pred).strip().upper() == str(true).strip().upper()


def items_correct(pred_items, true_items) -> bool:
    if len(pred_items) != len(true_items):
        return False
    return all(abs(p["quantity"] - t["quantity"]) < 1e-6 and abs(p["unit_price"] - t["unit_price"]) <= 0.01
               and abs(p["line_total"] - t["line_total"]) <= 0.01 for p, t in zip(pred_items, true_items))


def evaluate(extractor, truth: list[dict], sidx: SupplierIndex, master_cif_to_id: dict) -> tuple[dict, pd.DataFrame]:
    rows = []
    for t in truth:
        pdf_path = INV_DIR / "pdf" / f"{t['doc_id']}.pdf"
        inv, meta = extractor.extract(pdf_path)
        r = {"doc_id": t["doc_id"], "template": t["template"], "has_arith_error": t["has_arith_error"],
             "schema_valid": inv is not None, "latency_s": meta.get("latency_s", 0),
             "cost_usd": meta.get("cost_usd", 0), "error": meta.get("error")}
        if inv is not None:
            p = inv.model_dump(mode="json")
            for f in SCALAR_FIELDS:
                r[f] = same(f, p[f], t[f])
            r["line_items"] = items_correct(p["line_items"], t["line_items"])
            v = validate_invoice(inv, sidx, today=date(2026, 12, 31), source_text=pdf_text(pdf_path))
            r["flag_total_mismatch"] = ("total_mismatch" in v["issues"]
                                        or "amount_not_in_source:total" in v["issues"])
            r["route"] = v["route"]
            r["issues"] = ";".join(v["issues"])
            r["supplier_link_ok"] = v["supplier_master_id"] == master_cif_to_id.get(normalize_cif(t["supplier_cif"]))
        else:
            for f in SCALAR_FIELDS + ["line_items"]:
                r[f] = False
            r.update(flag_total_mismatch=False, route="needs_review", issues="extraction_failed",
                     supplier_link_ok=False)
        r["all_fields_correct"] = all(r[f] for f in SCALAR_FIELDS + ["line_items"])
        rows.append(r)

    df = pd.DataFrame(rows)
    err = df[df.has_arith_error]
    flagged = df[df.flag_total_mismatch]
    auto = df[df.route == "auto_approve"]
    unsafe = auto[~auto.all_fields_correct | auto.has_arith_error]
    summary = {
        "documents": len(df),
        "schema_valid_rate": round(df.schema_valid.mean(), 3),
        "field_accuracy": round(df[SCALAR_FIELDS].values.mean(), 3),
        "line_item_accuracy": round(df.line_items.mean(), 3),
        "docs_100pct_correct": round(df.all_fields_correct.mean(), 3),
        "error_detection_recall": round(err.flag_total_mismatch.mean(), 3) if len(err) else None,
        "error_detection_precision": round(flagged.has_arith_error.mean(), 3) if len(flagged) else None,
        "auto_approve_rate": round(len(auto) / len(df), 3),
        "unsafe_auto_approve": int(len(unsafe)),
        "supplier_link_accuracy": round(df.supplier_link_ok.mean(), 3),
        "avg_latency_s": round(df.latency_s.mean(), 2),
        "cost_per_doc_usd": round(df.cost_usd.mean(), 5),
        "per_field": {f: round(df[f].mean(), 3) for f in SCALAR_FIELDS + ["line_items"]},
        "by_template": df.groupby("template").all_fields_correct.mean().round(3).to_dict(),
    }
    return summary, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", nargs="*", default=[], choices=["text", "pdf"],
                    help="also run the LLM extractor on extracted text and/or native PDF")
    a = ap.parse_args()

    truth = json.loads((INV_DIR / "ground_truth.json").read_text())
    master = pd.read_csv("data/raw/companies.csv", dtype=str)
    sidx = SupplierIndex(master)
    master_cif_to_id = {normalize_cif(c): int(i) for c, i in zip(master.cif, master.true_cluster_id)
                        if isinstance(c, str)}

    extractors = [RegexExtractor()] + [LLMExtractor(input_mode=m) for m in a.llm]
    results = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for ex in extractors:
        summary, df = evaluate(ex, truth, sidx, master_cif_to_id)
        results[ex.name] = summary
        df.to_csv(OUT / f"extraction_{ex.name}.csv", index=False)
    (OUT / "extraction_metrics.json").write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    keys = ["field_accuracy", "line_item_accuracy", "docs_100pct_correct", "error_detection_recall",
            "unsafe_auto_approve", "supplier_link_accuracy", "avg_latency_s", "cost_per_doc_usd"]
    print("\n| Extractor | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 1))
    for name, s in results.items():
        print(f"| {name} | " + " | ".join(str(s[k]) for k in keys) + " |")


if __name__ == "__main__":
    main()
