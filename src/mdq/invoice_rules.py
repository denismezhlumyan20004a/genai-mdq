"""Day 6: business-rule validation of extracted invoices + link to master data.

Every extracted invoice gets a list of issues and a routing decision:
  auto_approve  -> no issues
  needs_review  -> human-in-the-loop queue (with the reasons)
"""
from __future__ import annotations

import re
from datetime import date

import pandas as pd

from .normalize import normalize_cif, normalize_name
from .quality import is_valid_cif
from .schemas import Invoice

TOL = 0.02  # euro cents tolerance for rounding
VALID_VAT = {0, 4, 10, 21}


class SupplierIndex:
    """Links an extracted supplier to the company master (by CIF, else fuzzy name)."""

    def __init__(self, master: pd.DataFrame):
        m = master.copy()
        m["n_cif"] = m["cif"].map(normalize_cif)
        m["n_name"] = m["name"].map(normalize_name)
        self.by_cif = dict(zip(m.n_cif[m.n_cif != ""], m.true_cluster_id[m.n_cif != ""]))
        self.names = m.drop_duplicates("n_name")[["n_name", "true_cluster_id"]].values.tolist()

    def link(self, cif: str | None, name: str | None) -> tuple[object, str]:
        c = normalize_cif(cif)
        if c and c in self.by_cif:
            return self.by_cif[c], "cif"
        from .matching import token_sort
        n = normalize_name(name)
        best = max(self.names, key=lambda x: token_sort(n, x[0]), default=None)
        if best and token_sort(n, best[0]) >= 0.92:
            return best[1], "fuzzy_name"
        return None, "not_found"


# whole-number boundaries so "1234.56" is not read as "234.56"; handles 1.234,56 / 1,234.56 / 1234.56
AMOUNT_RE = r"(?<![\d.,])(?:€\s?)?-?\d+(?:[.,]\d{3})*[.,]\d{2}(?![\d])"


def source_amounts(text: str) -> set[float]:
    """All money amounts printed in the document text (absolute values, 2 decimals)."""
    from .extract import parse_money
    out = set()
    for tok in re.findall(AMOUNT_RE, text or ""):
        try:
            out.add(round(abs(parse_money(tok)), 2))
        except ValueError:
            pass
    return out


def ungrounded_amounts(inv: Invoice, source_text: str) -> list[str]:
    """Deterministic grounding check: every extracted amount must literally appear in the source.
    Catches LLMs that 'helpfully' recompute a value instead of copying what is printed."""
    printed = source_amounts(source_text)
    fields = {"subtotal": inv.subtotal, "vat_amount": inv.vat_amount, "total": inv.total}
    fields |= {f"line_{k}_total": it.line_total for k, it in enumerate(inv.line_items)}
    return [f"amount_not_in_source:{name}" for name, val in fields.items()
            if not any(abs(round(abs(val), 2) - p) <= 0.005 for p in printed)]


def validate_invoice(inv: Invoice, supplier_index: SupplierIndex | None = None,
                     today: date | None = None, source_text: str | None = None) -> dict:
    """source_text (the document's own text) enables the grounding check on amounts."""
    today = today or date.today()
    issues = []
    if source_text is not None:
        issues += ungrounded_amounts(inv, source_text)
    if not is_valid_cif(inv.supplier_cif):
        issues.append("invalid_supplier_cif")
    if inv.customer_cif and not is_valid_cif(inv.customer_cif):
        issues.append("invalid_customer_cif")
    if inv.vat_rate not in VALID_VAT:
        issues.append("unusual_vat_rate")
    for k, it in enumerate(inv.line_items):
        if abs(it.quantity * it.unit_price - it.line_total) > TOL:
            issues.append(f"line_{k}_qty_x_price_mismatch")
    lines_sum = round(sum(it.line_total for it in inv.line_items), 2)
    if inv.line_items and not (0.5 * lines_sum - TOL <= inv.subtotal <= lines_sum + TOL):
        issues.append("subtotal_inconsistent_with_lines")
    if abs(inv.subtotal * inv.vat_rate / 100 - inv.vat_amount) > TOL:
        issues.append("vat_amount_mismatch")
    if abs(inv.subtotal + inv.vat_amount - inv.total) > TOL:
        issues.append("total_mismatch")
    if inv.issue_date > today:
        issues.append("future_date")
    link = None
    if supplier_index is not None:
        link, how = supplier_index.link(inv.supplier_cif, inv.supplier_name)
        if link is None:
            issues.append("unknown_supplier")
    return {"issues": issues, "route": "needs_review" if issues else "auto_approve",
            "supplier_master_id": None if link is None else int(link)}
