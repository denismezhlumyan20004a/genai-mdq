"""Validators (used by golden record now; full data-quality report on Day 4)."""
from __future__ import annotations

import re

from .generate import cif_control


def is_valid_cif(cif: str | None) -> bool:
    if not cif:
        return False
    c = re.sub(r"[^A-Z0-9]", "", str(cif).upper())
    if not re.fullmatch(r"[ABCDEFGHJKLMNPQRSUVW]\d{7}[0-9A-J]", c):
        return False
    expected = cif_control(c[0], c[1:8])
    if c[0] in "ABEH":
        return c[8] == expected
    if c[0] in "KPQSNW":
        return c[8] == expected
    return c[8] in (expected, "JABCDEFGHI"[int(expected)] if expected.isdigit() else expected)


# ---------------------------------------------------------------- record-level data quality
EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")
REQUIRED = ["name", "cif", "address", "postal_code", "city"]
OPTIONAL = ["phone", "email"]


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and v != v) or str(v).strip() == ""


def is_valid_email(v) -> bool:
    return bool(EMAIL_RE.match(str(v).strip())) if not _blank(v) else False


def is_valid_phone_es(v) -> bool:
    if _blank(v):
        return False
    d = re.sub(r"\D", "", str(v))
    d = d[4:] if d.startswith("0034") else d[2:] if d.startswith("34") and len(d) == 11 else d
    return len(d) == 9 and d[0] in "6789"


def is_valid_postal_code_es(v) -> bool:
    s = str(v).strip() if not _blank(v) else ""
    s = s[:-2] if s.endswith(".0") else s
    return len(s) == 5 and s.isdigit() and 1 <= int(s[:2]) <= 52


VALIDATORS = {"cif": is_valid_cif, "email": is_valid_email, "phone": is_valid_phone_es,
              "postal_code": is_valid_postal_code_es}


def check_record(rec: dict) -> list[str]:
    issues = [f"missing_{f}" for f in REQUIRED if _blank(rec.get(f))]
    for f, fn in VALIDATORS.items():
        if not _blank(rec.get(f)) and not fn(rec.get(f)):
            issues.append(f"invalid_{f}")
    return issues


def quality_report(records: list[dict]) -> dict:
    """Completeness / validity per column + overall score + per-record issues."""
    n = len(records) or 1
    cols = REQUIRED + OPTIONAL
    completeness = {c: round(sum(not _blank(r.get(c)) for r in records) / n, 3) for c in cols}
    validity = {}
    for c, fn in VALIDATORS.items():
        filled = [r.get(c) for r in records if not _blank(r.get(c))]
        validity[c] = round(sum(map(fn, filled)) / len(filled), 3) if filled else None
    per_record = [{"record_id": r.get("record_id", str(i)), "issues": check_record(r)} for i, r in enumerate(records)]
    clean = sum(not x["issues"] for x in per_record)
    req_compl = sum(completeness[c] for c in REQUIRED) / len(REQUIRED)
    val = [v for v in validity.values() if v is not None]
    score = round(100 * (0.5 * req_compl + 0.5 * (sum(val) / len(val) if val else 1)), 1)
    return {"records": len(records), "quality_score": score, "clean_records_pct": round(clean / n, 3),
            "completeness": completeness, "validity": validity, "per_record": per_record}
