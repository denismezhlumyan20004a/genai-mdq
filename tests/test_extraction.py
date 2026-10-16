import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mdq.extract import TOOL, LLMExtractor, parse_date, parse_money  # noqa: E402
from mdq.invoice_rules import validate_invoice  # noqa: E402
from mdq.schemas import Invoice  # noqa: E402

SAMPLE = {
    "invoice_number": "F0001", "issue_date": "2025-03-15", "supplier_name": "Transportes Ruiz S.L.",
    "supplier_cif": "B12345674", "customer_name": "Hotel Plaza Mayor S.A.", "customer_cif": None,
    "line_items": [{"description": "Riel", "quantity": 2, "unit_price": 10.0, "line_total": 20.0}],
    "subtotal": 20.0, "vat_rate": 21, "vat_amount": 4.2, "total": 24.2, "currency": "EUR",
}


class FakeClient:
    """Mimics anthropic.Anthropic().messages.create; first answer is invalid to test the retry."""

    def __init__(self, answers):
        self.answers, self.calls = list(answers), []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        block = SimpleNamespace(type="tool_use", id=f"t{len(self.calls)}", input=self.answers.pop(0))
        return SimpleNamespace(content=[block], usage=SimpleNamespace(input_tokens=1000, output_tokens=200))


def test_parsers():
    assert parse_money("1.234,56 €") == 1234.56
    assert parse_money("1234.56 EUR") == 1234.56
    assert parse_money("€1,234.56") == 1234.56
    assert parse_date("8 de febrero de 2025") == date(2025, 2, 8)
    assert parse_date("12-07-2025") == date(2025, 7, 12)


def test_tool_schema_has_no_refs():
    assert "$ref" not in json.dumps(TOOL["input_schema"])


def test_llm_extractor_retries_on_schema_error(tmp_path):
    pdf = next((ROOT / "data/invoices/pdf").glob("*.pdf"), None)
    if pdf is None:
        return
    bad = dict(SAMPLE, issue_date="not a date")
    client = FakeClient([bad, SAMPLE])
    ex = LLMExtractor(model="claude-haiku-4-5", client=client, cache_dir=None)
    inv, meta = ex.extract(pdf)
    assert inv is not None and inv.total == 24.2
    assert meta["retries"] == 1 and len(client.calls) == 2
    assert client.calls[0]["tool_choice"] == {"type": "tool", "name": "record_invoice"}
    assert meta["cost_usd"] > 0


def test_validation_flags_total_mismatch():
    ok = validate_invoice(Invoice.model_validate(SAMPLE), today=date(2026, 1, 1))
    assert "total_mismatch" not in ok["issues"]
    bad = validate_invoice(Invoice.model_validate(dict(SAMPLE, total=30.0)), today=date(2026, 1, 1))
    assert "total_mismatch" in bad["issues"] and bad["route"] == "needs_review"
