import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mdq.quality import check_record, quality_report  # noqa: E402
from mdq.service import MDQService, Settings  # noqa: E402

DUPES = [
    {"record_id": "a", "name": "Construcciones García S.L.", "cif": "B12345674", "address": "Calle Mayor, 12",
     "postal_code": "28013", "city": "Madrid", "phone": "912345678", "source": "ERP", "updated_at": "2025-01-01"},
    {"record_id": "b", "name": "CONSTRUCCIONES GARCIA SL", "cif": None, "address": "C/ Mayor 12",
     "postal_code": "28013", "city": "Madrid", "phone": "+34 912 34 56 78", "source": "CRM", "updated_at": "2025-06-01"},
    {"record_id": "c", "name": "Transportes Ruiz S.A.", "cif": "A87654321", "address": "Avda. de la Paz 3",
     "postal_code": "41001", "city": "Sevilla", "phone": "954000000", "source": "ERP", "updated_at": "2024-01-01"},
]


def svc():
    return MDQService(Settings(master_path=str(ROOT / "data/raw/companies.csv"),
                               model_path=str(ROOT / "models/matcher.joblib")))


def test_match_merges_duplicates_and_keeps_distinct():
    out = svc().match(DUPES)
    assert out["records_in"] == 3 and out["golden_records_out"] == 2
    assert {(m["record_a"], m["record_b"]) for m in out["matches"]} == {("a", "b")}
    merged = next(g for g in out["golden_records"] if g["n_records"] == 2)
    assert merged["name"] == "Construcciones García S.L."  # ERP wins survivorship


def test_match_single_record():
    out = svc().match(DUPES[:1])
    assert out["golden_records_out"] == 1 and out["matches"] == []


def test_quality_report():
    rep = quality_report([{"name": "X", "cif": "B0000000X", "email": "bad@", "phone": "123"}])
    issues = rep["per_record"][0]["issues"]
    assert {"invalid_cif", "invalid_email", "invalid_phone", "missing_address"} <= set(issues)
    assert 0 <= rep["quality_score"] <= 100
    assert check_record(DUPES[0]) == []


def test_extract_invoice_regex_and_guards():
    s = svc()
    pdf = next((ROOT / "data/invoices/pdf").glob("*.pdf"), None)
    if pdf is not None:
        out = s.extract_invoice(pdf.read_bytes(), "regex")
        assert out["route"] in ("auto_approve", "needs_review")
    for bad in [b"hello", b"%PDF" + b"0" * int(11 * 1024 * 1024)]:
        try:
            s.extract_invoice(bad)
            raise AssertionError("should fail")
        except ValueError:
            pass
