"""Day 5-6: invoice extraction.

Two extractors behind the same interface (doc -> Invoice + metadata):
  * RegexExtractor  — rule-based baseline (labels + regexes), no LLM.
  * LLMExtractor    — Claude with forced tool use => JSON matching the Pydantic
                      schema; input can be extracted text or the native PDF.
                      Self-correction: if the output fails Pydantic validation,
                      the error is sent back once for a retry.

Usage:  PYTHONPATH=src python -m mdq.extract data/invoices/pdf/INV000.pdf --mode llm
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from datetime import date
from pathlib import Path

import pdfplumber
from pydantic import ValidationError

from .schemas import Invoice, LineItem

MONTHS = {m: i + 1 for i, m in enumerate(
    ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
     "septiembre", "octubre", "noviembre", "diciembre"])}


def pdf_text(path: str | Path) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join((p.extract_text(layout=True) or "") for p in pdf.pages)


# ---------------------------------------------------------------- parsing helpers
MONEY_RE = r"(?:€\s?)?-?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})\s?(?:€|EUR)?"


def parse_money(s: str) -> float:
    s = s.replace("€", "").replace("EUR", "").strip()
    if re.search(r",\d{2}$", s):          # 1.234,56
        s = s.replace(".", "").replace(",", ".")
    else:                                   # 1,234.56 / 1234.56
        s = s.replace(",", "")
    return float(s)


def parse_date(s: str) -> date | None:
    s = s.strip().lower()
    for pat, fn in [
        (r"(\d{4})-(\d{2})-(\d{2})", lambda m: date(int(m[1]), int(m[2]), int(m[3]))),
        (r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", lambda m: date(int(m[3]), int(m[2]), int(m[1]))),
        (r"(\d{1,2}) de (\w+)\s+de (\d{4})", lambda m: date(int(m[3]), MONTHS[m[2]], int(m[1]))),
    ]:
        m = re.search(pat, s)
        if m:
            try:
                return fn(m)
            except (KeyError, ValueError):
                return None
    return None


# ---------------------------------------------------------------- regex baseline
class RegexExtractor:
    name = "regex_baseline"

    NUM_LABELS = r"(?:Nº Factura|Factura nº|Número de factura|FACTURA|Factura)"
    DATE_LABELS = r"(?:Fecha de emisión|Fecha factura|F\. factura|Fecha|emitida el)"
    BASE_LABELS = r"(?:Base imponible|Subtotal|Base Imp\.|Importe neto)"
    VAT_LABELS = r"(?:Cuota IVA|Impuesto \(IVA\)|I\.V\.A\.|IVA)"
    TOTAL_LABELS = r"(?:IMPORTE TOTAL|Total factura|Total a pagar|TOTAL)"

    def extract(self, path: str | Path) -> tuple[Invoice | None, dict]:
        t0 = time.time()
        text = pdf_text(path)
        try:
            inv = self._parse(text)
            err = None
        except Exception as e:  # noqa: BLE001 — baseline: any failure = no output
            inv, err = None, f"{type(e).__name__}: {e}"
        return inv, {"latency_s": time.time() - t0, "cost_usd": 0.0, "error": err}

    def _parse(self, text: str) -> Invoice:
        g = lambda pat, flags=0: (re.search(pat, text, flags) or [None, None])[1]  # noqa: E731
        number = g(self.NUM_LABELS + r"[:\s]+([A-Z]*[-/]?\d[\w/-]*)")
        d = parse_date(g(self.DATE_LABELS + r"[:\s]+([^\n]{6,25})") or "")
        sup_cif = g(r"(?<!/)CIF[:\s]+([A-Z]\d{7}[0-9A-J])")
        cus_cif = g(r"(?:NIF/CIF[:\s]+|\()([A-Z]\d{7}[0-9A-J])")
        # supplier name: line right before "CIF:" / start of compact header
        sup_name = g(r"^\s*([^\n·]+?)\s*(?:·\s*CIF|\n\s*CIF:)", re.M)
        cus_name = g(r"(?:Cliente:|Facturar a:)\s*([^\n(]+?)\s*(?:\(|\n|$)")
        money = lambda lab: parse_money(g(lab + r"[^\n\d€]*?(?:\d{1,2}%)?\s+(" + MONEY_RE + ")"))  # noqa: E731
        subtotal = money(self.BASE_LABELS)
        vat_amount = money(self.VAT_LABELS)
        total = money(r"\n\s*" + self.TOTAL_LABELS)
        vat_rate = float(g(self.VAT_LABELS + r"\s*(\d{1,2})%") or 0)
        items = []
        for m in re.finditer(r"^\s*([A-Za-zÁÉÍÓÚáéíóúñ][^\n]*?)\s{2,}(\d+)\s{2,}(" + MONEY_RE + r")\s{2,}("
                             + MONEY_RE + r")\s*$", text, re.M):
            items.append(LineItem(description=m[1].strip(), quantity=float(m[2]),
                                  unit_price=parse_money(m[3]), line_total=parse_money(m[4])))
        return Invoice(invoice_number=number, issue_date=d, supplier_name=sup_name, supplier_cif=sup_cif,
                       customer_name=cus_name, customer_cif=cus_cif, line_items=items, subtotal=subtotal,
                       vat_rate=vat_rate, vat_amount=vat_amount, total=total)


# ---------------------------------------------------------------- LLM extractor
SYSTEM_PROMPT = """You are a precise document-extraction engine for Spanish invoices (facturas).
Extract the fields EXACTLY as printed on the document by calling the `record_invoice` tool.
Rules:
- Supplier = the company that ISSUES the invoice (emisor); customer = cliente / "facturar a".
- Amounts: convert Spanish number format (1.234,56 €) to plain numbers (1234.56).
- subtotal = base imponible AFTER any discount; vat_rate as a percentage number (21, 10, 4).
- Dates in ISO format YYYY-MM-DD.
- CIF/NIF without spaces or dashes.
- NEVER fix or recompute amounts: if the printed total looks wrong, still return the printed value.
- If a field is not present, use null (only for optional fields)."""

# Prices per million tokens (USD). Check current pricing for the model you use.
PRICES = {"claude-haiku-4-5": (1.0, 5.0), "claude-sonnet-4-5": (3.0, 15.0)}


def _inline_refs(schema: dict) -> dict:
    """Tool input_schema without $ref/$defs (simpler for any provider)."""
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(dict(defs[node["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node
    return walk(schema)


TOOL = {"name": "record_invoice",
        "description": "Record the structured data extracted from the invoice.",
        "input_schema": _inline_refs(Invoice.model_json_schema())}


class LLMExtractor:
    def __init__(self, model: str | None = None, input_mode: str = "text",
                 cache_dir: str | None = "data/invoices/llm_cache", client=None):
        self.model = model or os.getenv("MDQ_MODEL", "claude-haiku-4-5")
        self.input_mode = input_mode  # "text" (pdfplumber text) or "pdf" (native PDF input)
        self.name = f"llm_{self.model}_{input_mode}"
        self.cache = Path(cache_dir) / self.name if cache_dir else None
        if client is None:
            import anthropic  # pip install anthropic ; needs ANTHROPIC_API_KEY
            client = anthropic.Anthropic()
        self.client = client

    def _content(self, path: Path) -> list[dict]:
        if self.input_mode == "pdf":
            data = base64.standard_b64encode(path.read_bytes()).decode()
            return [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": data}},
                    {"type": "text", "text": "Extract this invoice."}]
        return [{"type": "text", "text": f"<invoice>\n{pdf_text(path)}\n</invoice>\nExtract this invoice."}]

    def _call(self, messages: list[dict]):
        return self.client.messages.create(
            model=self.model, max_tokens=2000, system=SYSTEM_PROMPT,
            tools=[TOOL], tool_choice={"type": "tool", "name": "record_invoice"}, messages=messages)

    def extract(self, path: str | Path) -> tuple[Invoice | None, dict]:
        path = Path(path)
        cached = self.cache / f"{path.stem}.json" if self.cache else None
        if cached and cached.exists():
            c = json.loads(cached.read_text())
            return (Invoice.model_validate(c["invoice"]) if c["invoice"] else None), c["meta"]

        t0 = time.time()
        messages = [{"role": "user", "content": self._content(path)}]
        in_tok = out_tok = retries = 0
        inv, err = None, None
        for attempt in range(2):  # 1 retry with the validation error (self-correction)
            resp = self._call(messages)
            in_tok += resp.usage.input_tokens
            out_tok += resp.usage.output_tokens
            block = next(b for b in resp.content if b.type == "tool_use")
            try:
                inv, err = Invoice.model_validate(block.input), None
                break
            except ValidationError as e:
                err, retries = str(e), retries + 1
                messages += [{"role": "assistant", "content": resp.content},
                             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": block.id,
                                                           "is_error": True,
                                                           "content": f"Schema validation failed:\n{e}\nFix and call the tool again."}]}]
        p_in, p_out = next((v for k, v in PRICES.items() if self.model.startswith(k)), (0, 0))
        meta = {"latency_s": round(time.time() - t0, 2), "input_tokens": in_tok, "output_tokens": out_tok,
                "cost_usd": round((in_tok * p_in + out_tok * p_out) / 1e6, 6), "retries": retries, "error": err}
        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps({"invoice": inv.model_dump(mode="json") if inv else None, "meta": meta},
                                         ensure_ascii=False, indent=2))
        return inv, meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--mode", choices=["regex", "llm"], default="regex")
    ap.add_argument("--input", choices=["text", "pdf"], default="text")
    a = ap.parse_args()
    ex = RegexExtractor() if a.mode == "regex" else LLMExtractor(input_mode=a.input, cache_dir=None)
    inv, meta = ex.extract(a.pdf)
    print(json.dumps({"invoice": inv.model_dump(mode="json") if inv else None, "meta": meta},
                     indent=2, ensure_ascii=False))
