"""Pydantic schemas: the contract between the LLM and the rest of the system."""
from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str = Field(description="Product or service description as written")
    quantity: float = Field(description="Quantity (number of units)")
    unit_price: float = Field(description="Unit price before VAT, in EUR")
    line_total: float = Field(description="Line amount before VAT, in EUR")


class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice number/identifier exactly as printed")
    issue_date: date = Field(description="Issue date (ISO format YYYY-MM-DD)")
    supplier_name: str = Field(description="Legal name of the issuing company (seller)")
    supplier_cif: Optional[str] = Field(None, description="Tax ID (CIF/NIF) of the supplier, no spaces or dashes")
    customer_name: str = Field(description="Legal name of the customer (buyer)")
    customer_cif: Optional[str] = Field(None, description="Tax ID (CIF/NIF) of the customer, no spaces or dashes")
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: float = Field(description="Taxable base (base imponible) before VAT, after discounts, in EUR")
    vat_rate: float = Field(description="VAT (IVA) rate in percent, e.g. 21")
    vat_amount: float = Field(description="VAT (IVA) amount in EUR")
    total: float = Field(description="Total amount payable in EUR")
    currency: str = Field("EUR", description="ISO currency code")


SCALAR_FIELDS = ["invoice_number", "issue_date", "supplier_name", "supplier_cif",
                 "customer_name", "customer_cif", "subtotal", "vat_rate", "vat_amount", "total"]
