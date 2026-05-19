"""Vocabulary loader — re-derives the same 96-sign list from
docs/VOCABULARY.md that the Postgres seed migration uses.

The migration is authoritative; this module exists so the Python
training pipeline does not need a database connection to know the
vocabulary list. It parses the markdown table at import time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VocabItem:
    sign_id: str
    gloss: str
    category: str
    lifeprint_lesson: str
    asl_lex_code: str | None
    asl_lex_match: str
    wlasl_clip_count: int
    static_or_movement: str  # "static" or "movement"
    flippable: bool


_VOCAB: list[VocabItem] = []
SIGN_IDS: set[str] = set()


def _slug(gloss: str) -> str:
    return gloss.lower().replace("-", "_")


def load_vocabulary(path: Path | None = None) -> list[VocabItem]:
    global _VOCAB, GLOSS_TO_ID
    if _VOCAB:
        return _VOCAB

    if path is None:
        path = Path(__file__).resolve().parents[2] / "docs" / "VOCABULARY.md"

    text = path.read_text()
    row_re = re.compile(
        r"^\|\s*(\d+)\s*"
        r"\|\s*([A-Z][A-Z0-9\-]*)\s*"
        r"\|\s*([a-z\-]+)\s*"
        r"\|\s*(L\d+)\s*"
        r"\|\s*([A-Z]_\d+_\d+|)\s*"
        r"\|\s*([a-z0-9_]+)\s*"
        r"\|\s*(\d+)\s*"
        r"\|\s*PENDING\s*"
        r"\|\s*(static|movement)\s*"
        r"\|\s*(yes|no)\s*\|"
    )
    items: list[VocabItem] = []
    for line in text.splitlines():
        m = row_re.match(line)
        if not m:
            continue
        _, gloss, cat, lesson, code, match, clips, mov, flip = m.groups()
        sid = _slug(gloss)
        items.append(
            VocabItem(
                sign_id=sid,
                gloss=gloss,
                category=cat,
                lifeprint_lesson=lesson,
                asl_lex_code=code or None,
                asl_lex_match=match,
                wlasl_clip_count=int(clips),
                static_or_movement=mov,
                flippable=(flip == "yes"),
            )
        )

    if len(items) != 96:
        raise RuntimeError(
            f"docs/VOCABULARY.md parse produced {len(items)} rows; expected 96. "
            "Did the table format change?"
        )

    _VOCAB.clear()
    _VOCAB.extend(items)
    SIGN_IDS.clear()
    SIGN_IDS.update(item.sign_id for item in items)
    return _VOCAB


def get_all() -> list[VocabItem]:
    if not _VOCAB:
        load_vocabulary()
    return _VOCAB
