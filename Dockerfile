# ---- build stage: install deps, train the matcher, generate demo data ----
FROM python:3.12-slim AS build
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt
ENV PYTHONPATH=/install/lib/python3.12/site-packages:/app/src
COPY src ./src
COPY scripts ./scripts
COPY data/raw ./data/raw
RUN python scripts/train_matcher.py

# ---- runtime stage: slim image, non-root user ----
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src
COPY --from=build /install /usr/local
COPY --from=build /app/models ./models
COPY src ./src
COPY data/raw ./data/raw
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "mdq.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
