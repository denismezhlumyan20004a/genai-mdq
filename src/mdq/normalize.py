"""Normalization of company names, addresses, CIF, phone and email."""
from __future__ import annotations

import re
import unicodedata

try:
    from unidecode import unidecode as _strip_accents
except ImportError:  # fallback without extra dependency
    def _strip_accents(s: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

LEGAL_FORM_PATTERNS = [
    r"sociedad limitada unipersonal", r"sociedad limitada", r"sociedad anonima",
    r"sociedad cooperativa", r"\bs\.?\s?l\.?\s?u\.?$", r"\bs\.?\s?l\.?$", r"\bs\.?\s?a\.?$",
    r"\bs\.?\s?coop\.?$", r"\bscoop$",
]
STOPWORDS = {"de", "del", "la", "las", "los", "el", "y", "e"}
STREET_TYPE_MAP = {
    "c": "calle", "cl": "calle", "calle": "calle",
    "av": "avenida", "avda": "avenida", "avd": "avenida", "avenida": "avenida",
    "pza": "plaza", "pl": "plaza", "plaza": "plaza",
    "po": "paseo", "ps": "paseo", "paseo": "paseo",
    "pol": "poligono", "pg": "poligono", "poligono": "poligono",
}


def basic(s: str | None) -> str:
    if s is None or (isinstance(s, float)):
        return ""
    s = _strip_accents(str(s)).lower().replace("º", "o")
    return re.sub(r"\s+", " ", s).strip()


def normalize_name(name: str | None) -> str:
    s = basic(name)
    for pat in LEGAL_FORM_PATTERNS:
        s = re.sub(pat, "", s).strip(" .,")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    tokens = [t for t in s.split() if t not in STOPWORDS]
    return " ".join(tokens)


def normalize_address(addr: str | None) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", basic(addr).replace("/", " "))
    tokens = s.split()
    if tokens and tokens[0] in STREET_TYPE_MAP:
        tokens[0] = STREET_TYPE_MAP[tokens[0]]
    return " ".join(t for t in tokens if t not in STOPWORDS)


def normalize_cif(cif: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", basic(cif).upper())


def normalize_phone(phone: str | None) -> str:
    digits = re.sub(r"\D", "", basic(phone))
    if digits.startswith("0034"):
        digits = digits[4:]
    elif digits.startswith("34") and len(digits) == 11:
        digits = digits[2:]
    return digits


def normalize_email(email: str | None) -> str:
    return basic(email)
