# GenAI Master Data Quality & Document Intelligence

Personal project: ML entity resolution (golden records) for Spanish company master data, data-quality rules,
and LLM invoice extraction (Claude + Pydantic) with business-rule validation and evals, served as a FastAPI service.

**Highlights:** LLM extraction 99.6% field accuracy vs 72.6% regex baseline at <$0.005/invoice ·
entity resolution F1 0.985 vs 0.734 CIF-only · FastAPI + Docker + CI.

## Status
- [x] Day 1 — synthetic Spanish company data with ground truth (noisy duplicates + hard negatives)
- [x] Day 2 — entity resolution: normalization → blocking → similarity features → classifier → clusters → golden record
- [x] Day 4 — data-quality rules & report (completeness, validity, score 0–100)
- [x] Days 5–6 — LLM invoice extraction (Pydantic structured outputs, self-correcting retry) + business-rule validation + field-level evals
- [x] Day 7 — FastAPI service (/match, /validate, /extract, /invoices/validate) + Docker
- [x] Day 8 — tests + GitHub Actions CI (pytest, metrics guard, Docker build + smoke test)
- [ ] Next — multilingual embeddings as extra matching feature

## Quickstart
Linux / macOS:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
make data && make train && pytest -q
python eval/eval_matching.py
python eval/eval_extraction.py                       # regex baseline, no API key
ANTHROPIC_API_KEY=... python eval/eval_extraction.py --llm text pdf
```
Windows (CMD):
```bat
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-dev.txt
run_all.bat
set ANTHROPIC_API_KEY=...
python eval\eval_extraction.py --llm text pdf
```

## API
```bash
make data && make train          # synthetic data + matcher model
make run                         # http://localhost:8000/docs
# or
docker compose up --build
```
| Endpoint | What it does |
|---|---|
| `GET /health` | status, matcher type, LLM availability |
| `POST /match` | dedupe company records → matches with scores + golden records (survivorship) |
| `POST /validate` | data-quality report: completeness, validity (CIF checksum, email, phone, postal code), score 0–100 |
| `POST /extract?extractor=regex\|llm` | PDF invoice → structured JSON → business rules → `auto_approve` / `needs_review` |
| `POST /invoices/validate` | validate an already-structured invoice |

```bash
curl -X POST localhost:8000/match -H 'Content-Type: application/json' -d '{"records":[
 {"record_id":"a","name":"Construcciones García S.L.","cif":"B12345674","postal_code":"28013","phone":"912345678","source":"ERP"},
 {"record_id":"b","name":"CONSTRUCCIONES GARCIA SL","postal_code":"28013","phone":"+34 912 34 56 78","source":"CRM"}]}'
curl -X POST "localhost:8000/extract?extractor=llm" -F "file=@data/invoices/pdf/INV000.pdf"
```
Production touches: model + master data loaded once at startup (lifespan), optional `X-API-Key` auth,
request-id + latency logging **without request bodies** (PII), PDF type/size guards, LLM endpoint returns
503 when no API key is configured, multi-stage Docker image running as non-root with a healthcheck.

## Entity resolution results (test split, entities unseen in training)
3,523 records / 2,400 true entities; ~20% "hard negatives" (different companies with near-identical names).

Blocking: reduction ratio 99.8%, pair completeness 97.8%.

End-to-end (after clustering / transitive closure):

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| Exact CIF match (baseline) | 1.000 | 0.580 | 0.734 |
| Fuzzy rules (rapidfuzz + postal code / IDs) | 0.896 | 0.979 | 0.936 |
| Logistic regression on similarity features | 0.966 | 0.988 | 0.977 |
| Gradient boosting on similarity features | 0.983 | 0.988 | 0.985 |

## Design decisions
- **Split by true entity**, not by pair, to avoid leakage between train and test.
- **Multi-key blocking** (CIF, phone, email, postal code + name token, sorted-token prefix) — robust to missing CIFs and word order.
- **Survivorship**: most trusted source (ERP > Facturación > CRM > Web), then most recent; CIF = most frequent *valid* value (checksum-validated).
- End-to-end metrics are reported after transitive closure, since that is what the business receives.

## Invoice extraction (Document AI)
50 synthetic Spanish invoices, 4 layouts (label/number/date-format variation, discounts,
3 VAT rates), 20% with an injected arithmetic error. Suppliers come from the company master,
so each extracted invoice is linked back to a golden record.

Pipeline: `PDF → text (pdfplumber) or native PDF → Claude tool use (forced, JSON = Pydantic schema)
→ schema validation (1 self-correcting retry) → business rules → route: auto_approve | needs_review`.

Business rules: CIF checksum, qty×price = line total, base + IVA = total, IVA amount = base × rate,
valid IVA rate, no future dates, supplier exists in master data. On ground-truth data the rules catch
10/10 injected errors with 0 false positives.

| Extractor | Field acc. | Line items | Docs 100% correct | Error recall | Unsafe auto-approve | Supplier link | Latency | Cost/doc |
|---|---|---|---|---|---|---|---|---|
| Regex baseline | 0.726 | 0.74 | 0.62 | 0.70 | 5 / 50 | 0.74 | 0.08 s | $0 |
| Claude Haiku 4.5 (text) | 0.996 | 1.00 | 0.96 | 0.90 | 1 / 50 | 1.00 | 3.5 s | $0.0044 |
| Claude Haiku 4.5 (native PDF) | 0.998 | 1.00 | 0.98 | 0.90 | 1 / 50 | 1.00 | 4.4 s | $0.0056 |

**Findings**
- The LLM lifts field accuracy from 72.6% to 99.6–99.8% and fully-correct documents from 62% to 96–98%,
  for under half a cent per invoice. On the hardest (boxed) layout: regex 17% vs LLM 83–92%.
- Native PDF input is slightly more accurate than extracted text (+2 pp documents) at ~27% higher cost
  and ~0.8 s more latency, so text input is the default and PDF input is the fallback for complex layouts.
- Unsafe auto-approvals drop from 5 to 1. The remaining miss is an invoice with a wrong printed total:
  despite the prompt, the model returned the *mathematically correct* total, which hid the error from the
  business rules. Lesson: LLMs tend to "helpfully" fix values, so amounts must be cross-checked
  deterministically against the source text before auto-approval.
- Regex is ~25x faster and free, but brittle: it only works on layouts it was written for.

**Next steps:** deterministic cross-check of LLM amounts against the PDF text; customer master lookup;
confidence-based routing; Bedrock + Textract for scanned documents.
