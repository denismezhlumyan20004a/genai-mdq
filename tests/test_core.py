import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdq.generate import make_cif  # noqa: E402
from mdq.normalize import normalize_address, normalize_name, normalize_phone  # noqa: E402
from mdq.quality import is_valid_cif  # noqa: E402


def test_normalize_name_removes_legal_form_and_accents():
    assert normalize_name("Construcciones García S.L.") == "construcciones garcia"
    assert normalize_name("CONSTRUCCIONES GARCIA Sociedad Limitada") == "construcciones garcia"
    assert normalize_name("Transportes Ruiz SA") == "transportes ruiz"


def test_normalize_address_abbreviations():
    assert normalize_address("C/ Mayor, 12") == normalize_address("Calle Mayor 12")
    assert normalize_address("Avda. de la Paz 3") == "avenida paz 3"


def test_normalize_phone():
    assert normalize_phone("+34 912 34 56 78") == "912345678"
    assert normalize_phone("0034912345678") == "912345678"


def test_cif_validator():
    rng = random.Random(0)
    for letter in "ABF":
        assert is_valid_cif(make_cif(rng, letter))
    good = make_cif(rng, "B")
    bad = good[:-1] + str((int(good[-1]) + 1) % 10)  # wrong control digit
    assert is_valid_cif(good[0] + "-" + good[1:])
    assert not is_valid_cif(bad)
    assert not is_valid_cif("X1234")
    assert not is_valid_cif(None)
