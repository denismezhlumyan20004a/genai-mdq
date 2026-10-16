"""API tests (need fastapi + httpx: pip install -r requirements-dev.txt)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from mdq.api import app  # noqa: E402
from test_service import DUPES  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert "x-request-id" in r.headers


def test_match(client):
    r = client.post("/match", json={"records": DUPES})
    assert r.status_code == 200 and r.json()["golden_records_out"] == 2


def test_validate(client):
    r = client.post("/validate", json={"records": DUPES})
    assert r.status_code == 200 and 0 <= r.json()["quality_score"] <= 100


def test_match_rejects_empty(client):
    assert client.post("/match", json={"records": []}).status_code == 422


def test_extract_pdf(client):
    pdf = next((ROOT / "data/invoices/pdf").glob("*.pdf"))
    r = client.post("/extract", files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")})
    assert r.status_code == 200 and r.json()["route"] in ("auto_approve", "needs_review")


def test_extract_rejects_non_pdf(client):
    r = client.post("/extract", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_api_key(client, monkeypatch):
    monkeypatch.setenv("MDQ_API_KEY", "secret")
    assert client.post("/validate", json={"records": DUPES}).status_code == 401
    assert client.post("/validate", json={"records": DUPES}, headers={"X-API-Key": "secret"}).status_code == 200
