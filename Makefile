.PHONY: data train test run docker
data:   ; PYTHONPATH=src python -m mdq.generate && PYTHONPATH=src python -m mdq.invoices
train:  ; python scripts/train_matcher.py
test:   ; pytest -q
run:    ; PYTHONPATH=src uvicorn mdq.api:app --reload
docker: ; docker compose up --build
