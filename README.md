# GenAI Master Data Quality & Document Intelligence

Personal project: entity resolution (golden records) for Spanish company master data,
data-quality rules, and (next) LLM-based invoice extraction with evals, served via FastAPI.

## Status
- [x] Day 1 — synthetic Spanish company data with ground truth (noisy duplicates + hard negatives)
- [x] Day 2 — entity resolution: normalization → blocking → similarity features → classifier → clusters → golden record
- [ ] Day 3 — multilingual embeddings as extra feature
- [ ] Day 4 — data-quality rules & report
- [x] Days 5–6 — LLM invoice extraction (Pydantic structured outputs, self-correcting retry) + business-rule validation + field-level evals
- [x] Day 7 — FastAPI service (/match, /validate, /extract, /invoices/validate) + Docker
- [x] Day 8 — tests + GitHub Actions CI (pytest, metrics guard, Docker build + smoke test)

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m mdq.generate --n 2000 --dup-rate 0.3
python eval/eval_matching.py
# invoices (days 5-6)
PYTHONPATH=src python -m mdq.invoices --n 50
python eval/eval_extraction.py                 # regex baseline, no API key
export ANTHROPIC_API_KEY=...                   # then:
python eval/eval_extraction.py --llm text pdf  # Claude on extracted text vs native PDF
pytest -q
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
| Claude (text) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Claude (native PDF) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Regex works on known layouts (85% docs correct on template 0) but collapses on the boxed layout
(17%). Finding: arithmetic rules cannot catch wrong *text* fields (5 unsafe auto-approvals were
truncated customer names) → next step: cross-check customer against the customer master and add
LLM confidence / dual-extraction agreement before auto-approving.
