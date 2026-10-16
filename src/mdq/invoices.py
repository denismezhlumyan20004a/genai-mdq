"""Day 5: synthetic Spanish invoice PDFs with ground-truth JSON.

Four visually different templates, varied labels ("Nº Factura" / "Factura nº" /
"Número"), number formats (1.234,56 € vs 1234.56), date formats, VAT rates,
discounts, and ~20% invoices with an injected arithmetic error (total does not
match base + IVA) so the validation layer can be evaluated too.

Suppliers are taken from the master data (golden records) so extracted invoices
can later be linked to the MDM.

Usage:  PYTHONPATH=src python -m mdq.invoices --n 40
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .generate import make_cif

MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
          "septiembre", "octubre", "noviembre", "diciembre"]
PRODUCTS = [("Cortina lino beige 140x260", 35, 90), ("Estor enrollable blackout", 40, 120),
            ("Riel aluminio 2m", 12, 30), ("Barra cortina forja 160cm", 18, 45),
            ("Tela tapiceria terciopelo (m)", 14, 38), ("Cojin decorativo 45x45", 8, 22),
            ("Alfombra yute 160x230", 60, 180), ("Servicio de instalacion (h)", 25, 45),
            ("Transporte y montaje", 30, 90), ("Visillo voile blanco (m)", 6, 15),
            ("Panel japones 60x300", 28, 70), ("Mosquitera enrollable", 20, 55)]
CUSTOMERS = ["Decoraciones Martin S.L.", "Hotel Plaza Mayor S.A.", "Interiorismo Nova S.L.",
             "Reformas Integrales Sur S.L.", "Restaurante El Olivo S.L.", "Clinica Dental Sonrisa S.L.P."]

LABELS = {
    "number": ["Nº Factura", "Factura nº", "Número de factura", "FACTURA"],
    "date": ["Fecha", "Fecha de emisión", "F. factura", "Fecha factura"],
    "base": ["Base imponible", "Subtotal", "Base Imp.", "Importe neto"],
    "vat": ["IVA", "I.V.A.", "Cuota IVA", "Impuesto (IVA)"],
    "total": ["TOTAL", "Total factura", "Total a pagar", "IMPORTE TOTAL"],
}


def fmt_money(x: float, style: int) -> str:
    if style == 0:  # Spanish: 1.234,56 €
        s = f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{s} €"
    if style == 1:
        return f"{x:.2f} EUR"
    return f"€{x:,.2f}"


def fmt_date(d: date, style: int) -> str:
    return [d.strftime("%d/%m/%Y"), f"{d.day} de {MONTHS[d.month - 1]} de {d.year}",
            d.strftime("%d-%m-%Y"), d.isoformat()][style]


def make_invoice_data(i: int, rng: random.Random, suppliers: pd.DataFrame) -> dict:
    sup = suppliers.iloc[rng.randrange(len(suppliers))]
    n_items = rng.randint(1, 6)
    items = []
    for desc, lo, hi in rng.sample(PRODUCTS, n_items):
        q = rng.choice([1, 1, 2, 3, 4, 5, 10, 12, 20]) if "(h)" not in desc else rng.randint(1, 8)
        price = round(rng.uniform(lo, hi), 2)
        items.append({"description": desc, "quantity": q, "unit_price": price,
                      "line_total": round(q * price, 2)})
    gross = round(sum(it["line_total"] for it in items), 2)
    discount_pct = rng.choice([0, 0, 0, 5, 10])
    subtotal = round(gross * (1 - discount_pct / 100), 2)
    vat_rate = rng.choices([21, 10, 4], weights=[85, 10, 5])[0]
    vat_amount = round(subtotal * vat_rate / 100, 2)
    total = round(subtotal + vat_amount, 2)
    has_error = rng.random() < 0.2
    printed_total = round(total + rng.choice([-10, 10, 0.9, -1, 100]), 2) if has_error else total
    d = date(2025, 1, 1) + timedelta(days=rng.randint(0, 600))
    prefix = rng.choice(["F", "FV", "2025/", "A-", "FAC"])
    return {
        "doc_id": f"INV{i:03d}",
        "truth": {
            "invoice_number": f"{prefix}{rng.randint(100, 9999):04d}",
            "issue_date": d.isoformat(),
            "supplier_name": sup["name"],
            "supplier_cif": sup["cif"],
            "customer_name": rng.choice(CUSTOMERS),
            "customer_cif": make_cif(rng, "B"),
            "line_items": items,
            "subtotal": subtotal, "vat_rate": vat_rate, "vat_amount": vat_amount,
            "total": printed_total,  # what is PRINTED (extraction target)
            "currency": "EUR",
        },
        "meta": {"has_arith_error": has_error, "correct_total": total, "discount_pct": discount_pct,
                 "gross": gross, "supplier_address": sup.get("address"), "supplier_city": sup.get("city"),
                 "template": i % 4, "money_style": rng.randrange(3), "date_style": rng.randrange(4),
                 "labels": {k: rng.choice(v) for k, v in LABELS.items()}},
    }


def render_pdf(inv: dict, path: Path) -> None:
    t, m = inv["truth"], inv["meta"]
    ms, lab = m["money_style"], m["labels"]
    money = lambda x: fmt_money(x, ms)  # noqa: E731
    styles = getSampleStyleSheet()
    P = lambda s, st="Normal": Paragraph(s, styles[st])  # noqa: E731
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm)
    story = []
    supplier_block = (f"<b>{t['supplier_name']}</b><br/>CIF: {t['supplier_cif']}<br/>"
                      f"{m['supplier_address'] or ''}<br/>{m['supplier_city'] or ''}")
    customer_block = f"<b>Cliente:</b> {t['customer_name']}<br/>NIF/CIF: {t['customer_cif']}"
    header = (f"<b>{lab['number']}:</b> {t['invoice_number']}<br/>"
              f"<b>{lab['date']}:</b> {fmt_date(date.fromisoformat(t['issue_date']), m['date_style'])}")

    tpl = m["template"]
    if tpl == 0:  # classic: title, supplier left / invoice data right
        story += [P("FACTURA", "Title"),
                  Table([[P(supplier_block), P(header)]], colWidths=[100 * mm, 70 * mm]),
                  Spacer(1, 8), P(customer_block)]
    elif tpl == 1:  # customer first, header at the bottom of the block
        story += [P(customer_block), Spacer(1, 6), P(supplier_block), Spacer(1, 6),
                  P(header), Spacer(1, 6)]
    elif tpl == 2:  # compact one-line header
        story += [P(f"{t['supplier_name']} · CIF {t['supplier_cif']} · {m['supplier_city'] or ''}", "Heading3"),
                  P(f"Factura {t['invoice_number']} — emitida el "
                    f"{fmt_date(date.fromisoformat(t['issue_date']), m['date_style'])}"),
                  P(f"Facturar a: {t['customer_name']} ({t['customer_cif']})")]
    else:  # boxed header table
        story += [Table([["Emisor", "Cliente", "Documento"],
                         [P(supplier_block), P(customer_block), P(header)]],
                        colWidths=[62 * mm, 62 * mm, 50 * mm],
                        style=TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                                          ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))]
    story.append(Spacer(1, 10))

    head = ["Concepto", "Cant.", "Precio", "Importe"] if tpl != 2 else ["Descripción", "Uds", "P. unit.", "Total"]
    rows = [head] + [[it["description"], f"{it['quantity']:g}", money(it["unit_price"]), money(it["line_total"])]
                     for it in t["line_items"]]
    story.append(Table(rows, colWidths=[85 * mm, 20 * mm, 30 * mm, 35 * mm],
                       style=TableStyle([("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
                                         ("ALIGN", (1, 1), (-1, -1), "RIGHT")])))
    story.append(Spacer(1, 10))
    totals = []
    if m["discount_pct"]:
        totals += [["Suma", money(m["gross"])],
                   [f"Descuento {m['discount_pct']}%", "-" + money(round(m["gross"] - t["subtotal"], 2))]]
    totals += [[lab["base"], money(t["subtotal"])],
               [f"{lab['vat']} {t['vat_rate']}%", money(t["vat_amount"])],
               [lab["total"], money(t["total"])]]
    story.append(Table(totals, colWidths=[50 * mm, 35 * mm], hAlign="RIGHT",
                       style=TableStyle([("ALIGN", (1, 0), (1, -1), "RIGHT"),
                                         ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold")])))
    story += [Spacer(1, 14), P("Forma de pago: transferencia bancaria a 30 días. "
                               "IBAN ES76 2100 0418 4502 0005 1332", "Italic")]
    doc.build(story)


def main(n: int = 50, seed: int = 7, out: str = "data/invoices") -> None:
    rng = random.Random(seed)
    master = pd.read_csv("data/raw/companies.csv", dtype=str)
    # use clean records (valid CIF present) as the supplier universe
    suppliers = master[master["cif"].notna() & ~master["cif"].str.contains("-", na=False)]
    suppliers = suppliers.drop_duplicates("true_cluster_id").reset_index(drop=True)
    out_dir = Path(out)
    (out_dir / "pdf").mkdir(parents=True, exist_ok=True)
    truth = []
    for i in range(n):
        inv = make_invoice_data(i, rng, suppliers)
        render_pdf(inv, out_dir / "pdf" / f"{inv['doc_id']}.pdf")
        truth.append({"doc_id": inv["doc_id"], **inv["truth"],
                      "has_arith_error": inv["meta"]["has_arith_error"], "template": inv["meta"]["template"]})
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False))
    print(f"{n} invoices -> {out_dir}/pdf, ground truth -> {out_dir}/ground_truth.json "
          f"({sum(t['has_arith_error'] for t in truth)} with injected arithmetic errors)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main(a.n, a.seed)
