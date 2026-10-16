@echo off
REM Windows: run from the genai-mdq folder with the venv activated
set PYTHONPATH=src
python -m mdq.generate || exit /b 1
python -m mdq.invoices || exit /b 1
python scripts\train_matcher.py || exit /b 1
python eval\eval_matching.py || exit /b 1
python eval\eval_extraction.py || exit /b 1
python -m pytest -q
