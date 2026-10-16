"""Day 1: synthetic Spanish company master data with ground truth.

Creates N unique companies, then adds noisy duplicates (typos, legal-form
variants, abbreviations, missing fields). Every record keeps `true_cluster_id`
so matching quality can be measured exactly.

Also adds "hard negatives": different companies with very similar names
(e.g. same surname + sector in another city) so the problem is not trivial.

Usage:  python -m mdq.generate --n 2000 --dup-rate 0.3 --seed 42
"""
from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

import pandas as pd

SURNAMES = ["Garcia", "Martinez", "Lopez", "Sanchez", "Gonzalez", "Rodriguez", "Fernandez",
            "Perez", "Gomez", "Ruiz", "Diaz", "Moreno", "Alvarez", "Romero", "Navarro",
            "Torres", "Dominguez", "Vazquez", "Ramos", "Gil", "Serrano", "Molina", "Ortega",
            "Delgado", "Castro", "Ortiz", "Rubio", "Marin", "Sanz", "Iglesias", "Medina",
            "Garrido", "Cortes", "Castillo", "Santos", "Lozano", "Guerrero", "Cano", "Prieto"]
SECTORS = ["Construcciones", "Transportes", "Distribuciones", "Inmobiliaria", "Consultoria",
           "Talleres", "Instalaciones", "Reformas", "Logistica", "Asesores", "Alimentacion",
           "Climatizacion", "Electricidad", "Carpinteria", "Suministros", "Ingenieria",
           "Textil", "Mobiliario", "Decoracion", "Servicios Informaticos", "Hosteleria"]
SUFFIXES = ["", "", "", "Hermanos", "e Hijos", "Iberica", "Global", "del Sur", "Norte",
            "Integral", "Grupo", "Levante", "Centro"]
CITIES = {  # city: postal-code prefix
    "Madrid": "280", "Barcelona": "080", "Valencia": "460", "Sevilla": "410",
    "Zaragoza": "500", "Malaga": "290", "Murcia": "300", "Bilbao": "480",
    "Alicante": "030", "Cordoba": "140", "Valladolid": "470", "Vigo": "362",
    "Getafe": "289", "Alcala de Henares": "288", "Granada": "180",
}
STREETS = ["Mayor", "Real", "de la Constitucion", "del Sol", "San Juan", "de Alcala",
           "Gran Via", "de Espana", "del Mar", "de los Olivos", "de la Paz", "Toledo",
           "Castellana", "de Andalucia", "del Prado", "Industrial"]
STREET_TYPES = ["Calle", "Avenida", "Plaza", "Paseo", "Poligono"]
LEGAL_FORMS = {  # canonical -> CIF letter
    "S.L.": "B", "S.A.": "A", "S.L.U.": "B", "S.Coop.": "F",
}
LEGAL_VARIANTS = {
    "S.L.": ["SL", "S.L", "S L", "Sociedad Limitada", "s.l.", ""],
    "S.A.": ["SA", "S.A", "Sociedad Anonima", "s.a.", ""],
    "S.L.U.": ["SLU", "S.L.U", "Sociedad Limitada Unipersonal", ""],
    "S.Coop.": ["S. Coop.", "Sociedad Cooperativa", "SCOOP", ""],
}
STREET_ABBR = {"Calle": ["C/", "C.", "Cl"], "Avenida": ["Avda.", "Av.", "Avd"],
               "Plaza": ["Pza.", "Pl."], "Paseo": ["Pº", "Ps."], "Poligono": ["Pol.", "Pg."]}


# ---------- valid Spanish CIF ----------
def cif_control(letter: str, digits7: str) -> str:
    even = sum(int(d) for d in digits7[1::2])
    odd = 0
    for d in digits7[0::2]:
        x = int(d) * 2
        odd += x // 10 + x % 10
    c = (10 - (even + odd) % 10) % 10
    if letter in "KPQSNW":
        return "JABCDEFGHI"[c]
    return str(c)


def make_cif(rng: random.Random, letter: str) -> str:
    digits = "".join(rng.choice(string.digits) for _ in range(7))
    return f"{letter}{digits}{cif_control(letter, digits)}"


# ---------- noise functions ----------
def typo(s: str, rng: random.Random) -> str:
    if len(s) < 4:
        return s
    i = rng.randrange(1, len(s) - 1)
    op = rng.choice(["swap", "del", "sub", "dup"])
    if op == "swap":
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if op == "del":
        return s[:i] + s[i + 1:]
    if op == "sub":
        return s[:i] + rng.choice(string.ascii_lowercase) + s[i + 1:]
    return s[:i] + s[i] + s[i:]


def accentize(s: str, rng: random.Random) -> str:
    table = {"Garcia": "García", "Martinez": "Martínez", "Lopez": "López", "Sanchez": "Sánchez",
             "Gonzalez": "González", "Rodriguez": "Rodríguez", "Fernandez": "Fernández",
             "Perez": "Pérez", "Gomez": "Gómez", "Diaz": "Díaz", "Logistica": "Logística",
             "Consultoria": "Consultoría", "Ingenieria": "Ingeniería", "Iberica": "Ibérica",
             "Climatizacion": "Climatización", "Electricidad": "Electricidad",
             "Alimentacion": "Alimentación", "Decoracion": "Decoración", "Informaticos": "Informáticos"}
    for k, v in table.items():
        if k in s and rng.random() < 0.7:
            s = s.replace(k, v)
    return s


def noisy_copy(rec: dict, rng: random.Random) -> dict:
    r = dict(rec)
    # name
    name = r["base_name"]
    if rng.random() < 0.5:
        name = accentize(name, rng)
    if rng.random() < 0.35:
        name = typo(name, rng)
    if rng.random() < 0.15:
        name = name.upper()
    if rng.random() < 0.10:  # word order (e.g. "Garcia Construcciones")
        parts = name.split()
        if len(parts) >= 2:
            parts[0], parts[1] = parts[1], parts[0]
            name = " ".join(parts)
    legal = rng.choice(LEGAL_VARIANTS[r["legal_form"]]) if rng.random() < 0.7 else r["legal_form"]
    r["name"] = f"{name} {legal}".strip()
    # address
    st = r["street_type"]
    stype = rng.choice(STREET_ABBR[st]) if rng.random() < 0.5 else st
    street = r["street"] if rng.random() > 0.2 else typo(r["street"], rng)
    num = r["number"]
    if rng.random() < 0.12:  # company moved: completely different address & postal code
        street, num = rng.choice(STREETS), rng.randint(1, 250)
        stype = rng.choice(STREET_TYPES)
        r["postal_code"] = CITIES[r["city"]] + f"{rng.randint(1, 99):02d}"
    r["address"] = f"{stype} {street}, {num}" if rng.random() < 0.6 else f"{stype} {street} {num}"
    # CIF: missing / formatted / one-char typo
    p = rng.random()
    if p < 0.35:
        r["cif"] = None
    elif p < 0.50:
        r["cif"] = f"{r['cif'][0]}-{r['cif'][1:]}"
    elif p < 0.55:
        c = list(r["cif"])
        c[rng.randrange(1, 8)] = rng.choice(string.digits)
        r["cif"] = "".join(c)
    # phone / email
    if r["phone"] and rng.random() < 0.4:
        ph = r["phone"]
        r["phone"] = rng.choice([f"+34 {ph}", f"{ph[:3]} {ph[3:5]} {ph[5:7]} {ph[7:]}", f"0034{ph}"])
    q = rng.random()
    if q < 0.25:
        r["phone"] = None
    elif q < 0.45:  # another phone (mobile of a contact, new line)
        r["phone"] = "6" + "".join(rng.choice(string.digits) for _ in range(8))
    q = rng.random()
    if q < 0.30:
        r["email"] = None
    elif q < 0.50:  # personal / generic mailbox
        r["email"] = f"{rng.choice(['admin', 'contabilidad', 'pedidos'])}{rng.randint(1, 99)}@{rng.choice(['gmail.com', 'hotmail.com', 'yahoo.es'])}"
    elif r["email"] and rng.random() < 0.2:
        r["email"] = r["email"].upper()
    if rng.random() < 0.05:
        r["postal_code"] = None
    r["updated_at"] = (pd.Timestamp(r["updated_at"]) + pd.Timedelta(days=rng.randint(-900, 900))).date().isoformat()
    r["source"] = rng.choice(["ERP", "CRM", "Facturacion", "Web"])
    return r


def make_entity(i: int, rng: random.Random, base_name: str | None = None, city: str | None = None) -> dict:
    surname = rng.choice(SURNAMES)
    sector = rng.choice(SECTORS)
    suffix = rng.choice(SUFFIXES)
    if base_name is None:
        style = rng.random()
        if style < 0.6:
            base_name = f"{sector} {surname} {suffix}".strip()
        elif style < 0.85:
            base_name = f"{surname} {rng.choice(SURNAMES)} {sector}"
        else:
            base_name = f"{sector} {rng.choice(['Alfa', 'Delta', 'Orion', 'Atlas', 'Nova', 'Iris', 'Lince'])} {suffix}".strip()
    legal = rng.choices(list(LEGAL_FORMS), weights=[70, 15, 10, 5])[0]
    city = city or rng.choice(list(CITIES))
    words = base_name.lower().split()
    domain = "".join(ch for ch in "".join(w[:6] for w in words) if ch.isalpha())[:24]
    return {
        "true_cluster_id": i,
        "base_name": base_name,
        "legal_form": legal,
        "name": f"{base_name} {legal}",
        "cif": make_cif(rng, LEGAL_FORMS[legal]),
        "street_type": rng.choice(STREET_TYPES),
        "street": rng.choice(STREETS),
        "number": rng.randint(1, 250),
        "city": city,
        "postal_code": CITIES[city] + f"{rng.randint(1, 99):02d}",
        "phone": (pre := rng.choice(["91", "93", "96", "95", "6", "7"])) + "".join(rng.choice(string.digits) for _ in range(9 - len(pre))),
        "email": f"info@{domain}.es" if rng.random() < 0.8 else None,
        "updated_at": (pd.Timestamp("2022-01-01") + pd.Timedelta(days=rng.randint(0, 1000))).date().isoformat(),
        "source": "ERP",
    }


def generate(n: int = 2000, dup_rate: float = 0.3, hard_neg_rate: float = 0.20, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    entities = [make_entity(i, rng) for i in range(n)]
    # hard negatives: different company, near-identical name, other city
    for j in range(int(n * hard_neg_rate)):
        src = rng.choice(entities)
        same_city = rng.random() < 0.5
        city = src["city"] if same_city else rng.choice([c for c in CITIES if c != src["city"]])
        variant = src["base_name"] + rng.choice([" Hermanos", " Grupo", " 2000", " Sur", ""])
        e = make_entity(n + j, rng, base_name=variant, city=city)
        if e["email"]:
            e["email"] = e["email"].replace("@", f"{rng.randint(10, 99)}@")
        entities.append(e)

    records = []
    for e in entities:
        clean = dict(e)
        clean["address"] = f"{e['street_type']} {e['street']}, {e['number']}"
        records.append(clean)
        if rng.random() < dup_rate:
            for _ in range(rng.choices([1, 2, 3], weights=[65, 25, 10])[0]):
                records.append(noisy_copy(clean, rng))

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "record_id", [f"R{i:05d}" for i in range(len(df))])
    cols = ["record_id", "name", "cif", "address", "postal_code", "city", "phone", "email",
            "source", "updated_at", "true_cluster_id"]
    return df[cols]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--dup-rate", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/raw/companies.csv")
    a = ap.parse_args()
    df = generate(a.n, a.dup_rate, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} records, {df.true_cluster_id.nunique()} true entities -> {a.out}")
