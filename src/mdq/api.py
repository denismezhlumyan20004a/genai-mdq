"""Day 7: REST API.

Run locally:  PYTHONPATH=src uvicorn mdq.api:app --reload
Docs:         http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from .invoice_rules import validate_invoice
from .schemas import Invoice
from .service import MDQService

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mdq.api")


# ---------------------------------------------------------------- request / response models
class CompanyRecord(BaseModel):
    record_id: Optional[str] = None
    name: str
    cif: Optional[str] = None
    address: Optional[str] = None
    postal_code: Optional[str] = None
    city: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    source: Optional[str] = Field(None, description="ERP | Facturacion | CRM | Web (survivorship priority)")
    updated_at: Optional[str] = Field(None, description="ISO date, used for survivorship")


class RecordsIn(BaseModel):
    records: list[CompanyRecord] = Field(..., min_length=1, max_length=5000)


class MatchPair(BaseModel):
    record_a: str
    record_b: str
    score: float


class MatchOut(BaseModel):
    method: str
    records_in: int
    golden_records_out: int
    duplicates_found: int
    matches: list[MatchPair]
    golden_records: list[dict]


class QualityOut(BaseModel):
    records: int
    quality_score: float
    clean_records_pct: float
    completeness: dict
    validity: dict
    per_record: list[dict]


class InvoiceOut(BaseModel):
    extractor: Optional[str] = None
    invoice: Optional[dict]
    issues: list[str]
    route: Literal["auto_approve", "needs_review"]
    supplier_master_id: Optional[int] = None
    meta: dict = {}


# ---------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.svc = MDQService()  # load model + master data once
    log.info("service ready (matcher=%s)", "ml_model" if app.state.svc.model else "fuzzy_rules")
    yield


app = FastAPI(title="GenAI Master Data Quality API", version="0.1.0", lifespan=lifespan,
              description="Entity resolution, data-quality checks and LLM invoice extraction.")


@app.middleware("http")
async def access_log(request: Request, call_next):
    """Request id + latency. Bodies are never logged (they contain personal/company data)."""
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["x-request-id"] = rid
    log.info("rid=%s %s %s -> %s %.0fms", rid, request.method, request.url.path, response.status_code, ms)
    return response


def require_api_key(x_api_key: Optional[str] = Header(None)):
    expected = os.getenv("MDQ_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def svc(request: Request) -> MDQService:
    return request.app.state.svc


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health(request: Request):
    s = request.app.state.svc
    return {"status": "ok", "matcher": "ml_model" if s.model else "fuzzy_rules",
            "master_data_loaded": s.supplier_index is not None,
            "llm_available": bool(os.getenv("ANTHROPIC_API_KEY"))}


@app.post("/match", response_model=MatchOut, dependencies=[Depends(require_api_key)],
          summary="Deduplicate company records and return golden records")
def match(body: RecordsIn, s: MDQService = Depends(svc)):
    return s.match([r.model_dump() for r in body.records])


@app.post("/validate", response_model=QualityOut, dependencies=[Depends(require_api_key)],
          summary="Data-quality report: completeness, validity (CIF, email, phone, postal code)")
def validate(body: RecordsIn, s: MDQService = Depends(svc)):
    return s.validate([r.model_dump() for r in body.records])


@app.post("/extract", response_model=InvoiceOut, dependencies=[Depends(require_api_key)],
          summary="Extract a PDF invoice into structured JSON, validate it and route it")
def extract(file: UploadFile = File(...), extractor: Literal["regex", "llm"] = Query("regex"),
            s: MDQService = Depends(svc)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(415, "only PDF files are accepted")
    try:
        return s.extract_invoice(file.file.read(), extractor)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e


@app.post("/invoices/validate", response_model=InvoiceOut, dependencies=[Depends(require_api_key)],
          summary="Validate an already-structured invoice against business rules")
def validate_invoice_json(inv: Invoice, s: MDQService = Depends(svc)):
    v = validate_invoice(inv, s.supplier_index)
    return {"invoice": inv.model_dump(mode="json"), **v}
