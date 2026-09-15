"""Invariants of the shared amino-acid alphabet and BLOSUM62 transcription.

The table itself is not restated here. Digests pin the original values; structure
and a handful of published entries catch a swap; legacy imports keep the public
names that already existed; a subprocess import proves the constants module does
not pull in torch or arms.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.transfer.amino_acids import (  # noqa: E402
    AA20,
    BLOSUM62_ORDER,
    BLOSUM62_ROWS,
    blosum62_ncbi_rows,
)

#: SHA-256 of the original constants, taken from the 2026-09-14 cleanup baseline
#: snapshot before this module existed.
_AA20_DIGEST = "5a52efc76a4a4ceb3c992ff17426b3545634646080bb6acec132c47c278c9846"
_ORDER_DIGEST = "c5d93ccc6d713461cf1bdcf27d148bd17d25488f4c7a5dda7af3818e49a49879"
_NUMERIC_DIGEST = "d81640694dba3049bc0136cd1688098b0ab77feb97f16c6bc5c673e4845b69b5"
_NCBI_STRING_DIGEST = "18a9c3f0be65dfe7b2ceabbca64832698bd79b2d4429f74a6b913434eadc2ac1"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def test_alphabet_and_order_match_original_digests() -> None:
    assert len(AA20) == 20
    assert len(set(AA20)) == 20
    assert AA20 == "".join(sorted(AA20))
    assert sorted(BLOSUM62_ORDER) == sorted(AA20)
    assert _sha256(AA20) == _AA20_DIGEST
    assert _sha256(BLOSUM62_ORDER) == _ORDER_DIGEST


def test_blosum62_numeric_digest_and_published_entries() -> None:
    assert len(BLOSUM62_ROWS) == 20
    assert all(len(row) == 20 for row in BLOSUM62_ROWS)
    assert all(type(value) is int for row in BLOSUM62_ROWS for value in row)
    assert BLOSUM62_ROWS == tuple(
        tuple(BLOSUM62_ROWS[column][row] for column in range(20)) for row in range(20)
    )
    encoded = ",".join(str(value) for row in BLOSUM62_ROWS for value in row)
    assert _sha256(encoded) == _NUMERIC_DIGEST

    def score(left: str, right: str) -> int:
        return BLOSUM62_ROWS[BLOSUM62_ORDER.index(left)][BLOSUM62_ORDER.index(right)]

    assert score("A", "A") == 4
    assert score("W", "W") == 11
    assert score("C", "C") == 9
    assert score("P", "W") == -4


def test_ncbi_adapter_is_lossless_and_matches_historical_digest() -> None:
    rows = blosum62_ncbi_rows()
    assert type(rows) is tuple
    assert type(rows[0]) is str
    assert _sha256("\n".join(rows)) == _NCBI_STRING_DIGEST
    parsed = tuple(tuple(int(value) for value in row.split()) for row in rows)
    assert parsed == BLOSUM62_ROWS


def test_legacy_alphabet_aliases_keep_public_names() -> None:
    from src.transfer.arms import AA20 as arms_aa20
    from src.transfer.context_homologue import COMPOSITION_ALPHABET
    from src.transfer.kmer_background import ALPHABET
    from src.transfer.profiles import AA20 as profiles_aa20

    assert arms_aa20 is AA20
    assert profiles_aa20 is AA20
    assert ALPHABET is AA20
    assert COMPOSITION_ALPHABET is AA20
    assert type(ALPHABET) is str


def test_legacy_blosum_public_names_keep_types() -> None:
    from src.transfer import alphabet_chemistry as ac
    from src.transfer import circuits

    assert ac.BLOSUM62_ROWS is BLOSUM62_ROWS
    assert ac.BLOSUM62_ORDER is BLOSUM62_ORDER
    assert type(ac.BLOSUM62_ROWS[0]) is tuple
    assert circuits.BLOSUM62_ORDER is BLOSUM62_ORDER
    assert type(circuits.BLOSUM62_ROWS) is tuple
    assert type(circuits.BLOSUM62_ROWS[0]) is str
    assert circuits.BLOSUM62_ROWS == blosum62_ncbi_rows()


def test_constants_module_does_not_import_torch_or_arms() -> None:
    script = (
        "import sys\n"
        "from src.transfer.amino_acids import AA20, BLOSUM62_ROWS\n"
        "assert AA20 == 'ACDEFGHIKLMNPQRSTVWY'\n"
        "assert len(BLOSUM62_ROWS) == 20\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'src.transfer.arms' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
