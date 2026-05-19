"""Generate Layer A (pre-attempt) and Layer C (generic failure) hint copy.

Per ARCHITECTURE §5 and ADR 0004. For slice 1, hint copy is authored
against ASL-LEX 2.0 phonological data without Deaf-signer review; the
slice-2 production-deployment work (ADR 0004) replaces it with
instructor-validated copy.

Layer A — pre-attempt parameter card — primes attention on the five
sign parameters before the learner attempts the sign. Cognitive Load
Theory (Sweller 1988, 2010): reduce extraneous load *before* the
attempt so more working memory goes to germane load.

Layer C — generic per-sign failure hint — fired when the model has
no confusion-pair-specific hint to serve. Names 1-2 likely failure
modes for the sign's phonological profile.

Output: SQL migration that UPDATEs vocabulary_items.pre_attempt_hint
and generic_failure_hint for each of our 96 signs.

Usage:

    python -m training.data.author_hints \\
        --asl-lex training/data/sources/asl_lex_signdata.csv \\
        --output supabase/migrations/20260519120000_seed_hints.sql
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.data.vocabulary import get_all, VocabItem  # noqa: E402


# Map ASL-LEX phonological codes to plain English. The mappings below
# are derived from ASL-LEX 2.0's published coding scheme (Sevcikova
# Sehyr et al. 2021) and capture the parameters most useful for
# pre-attempt priming.

HANDSHAPE_NAMES = {
    "1": "an index-finger ‘1’ handshape",
    "5": "an open ‘5’ handshape",
    "A": "a fist ‘A’ handshape",
    "B": "a flat ‘B’ handshape",
    "Flat-B": "a flat ‘B’ handshape",
    "Bent-B": "a bent ‘B’ handshape",
    "Bent-5": "a clawed ‘5’ handshape",
    "Curved-5": "a curved ‘5’ handshape",
    "Curved-B": "a curved ‘B’ handshape",
    "C": "a ‘C’ handshape",
    "F": "an ‘F’ handshape",
    "G": "a ‘G’ handshape",
    "H": "an ‘H’ handshape",
    "I": "an ‘I’ handshape",
    "K": "a ‘K’ handshape",
    "L": "an ‘L’ handshape",
    "O": "an ‘O’ handshape",
    "S": "an ‘S’ handshape (closed fist)",
    "U": "a ‘U’ handshape",
    "V": "a ‘V’ handshape",
    "W": "a ‘W’ handshape",
    "Y": "a ‘Y’ handshape",
    "3": "a ‘3’ handshape",
    "8": "an ‘8’ handshape",
    "10": "a thumbs-up ‘10’ handshape",
    "Open-A": "an open-‘A’ handshape (thumb extended)",
    "Open-B": "an open-‘B’ handshape",
    "Open-F": "an open-‘F’ handshape",
    "Crooked-1": "a crooked-‘1’ handshape",
    "Bent-V": "a bent-‘V’ handshape",
    "Flat-O": "a flat-‘O’ handshape",
    "Baby-O": "a baby-‘O’ handshape",
    "ILY": "an ILY handshape (I-Love-You)",
    "NA": "the documented handshape (see reference)",
}

LOCATION_NAMES = {
    "Head": "near the head",
    "Forehead": "near the forehead",
    "Eye": "at eye level",
    "Cheek": "near the cheek",
    "Chin": "near the chin",
    "Mouth": "near the mouth",
    "Nose": "near the nose",
    "Ear": "near the ear",
    "Body": "near the body",
    "Chest": "at chest level",
    "Neck": "near the neck",
    "Stomach": "near the stomach",
    "Trunk": "at chest level",
    "Shoulder": "near the shoulder",
    "Arm": "near the opposite arm",
    "Forearm": "near the forearm",
    "Hand": "on the non-dominant hand",
    "Wrist": "near the wrist",
    "Neutral": "in neutral space in front of you",
    "NA": "where the reference video shows",
}

MOVEMENT_NAMES = {
    "Straight": "moves in a straight line",
    "Curved": "follows a curved path",
    "Circular": "moves in a circular pattern",
    "Arc": "follows an arc",
    "Wiggling": "with fingers wiggling",
    "Twist": "with a twisting motion",
    "Bend": "with a bending motion at the joints",
    "Rub": "with a rubbing motion",
    "Static": "is held with no movement",
    "None": "is held with no movement",
    "Hooking": "with a hooking motion",
    "BackAndForth": "moves back and forth",
    "NA": "matches the reference video's movement",
}


def _strip(v: str) -> str:
    return (v or "").strip().strip('"')


def _human_handshape(code: str) -> str:
    return HANDSHAPE_NAMES.get(_strip(code), HANDSHAPE_NAMES["NA"])


def _human_location(major: str, minor: str) -> str:
    minor_s = _strip(minor)
    if minor_s and minor_s in LOCATION_NAMES:
        return LOCATION_NAMES[minor_s]
    major_s = _strip(major)
    return LOCATION_NAMES.get(major_s, LOCATION_NAMES["NA"])


def _human_movement(mov: str, repeated: str) -> str:
    base = MOVEMENT_NAMES.get(_strip(mov), MOVEMENT_NAMES["NA"])
    rep = _strip(repeated)
    if rep and rep not in ("0", "None"):
        return base + " and is repeated"
    return base


def _layer_a(item: VocabItem, row: dict[str, str]) -> str:
    """Pre-attempt parameter card."""
    hs = _human_handshape(row.get("Handshape.2.0", ""))
    loc = _human_location(row.get("MajorLocation.2.0", ""), row.get("MinorLocation.2.0", ""))
    mov = _human_movement(row.get("Movement.2.0", ""), row.get("RepeatedMovement.2.0", ""))
    sign_type = _strip(row.get("SignType.2.0", ""))
    handedness = "two-handed" if sign_type.startswith(("Symm", "Asymm")) else "one-handed"

    return (
        f"Before signing {item.gloss}, set up: {hs}, {loc}, the hand {mov}. "
        f"This is a {handedness} sign."
    )


def _layer_c(item: VocabItem, row: dict[str, str]) -> str:
    """Generic per-sign failure hint."""
    sign_type = _strip(row.get("SignType.2.0", ""))
    movement = _strip(row.get("Movement.2.0", ""))
    contact = _strip(row.get("Contact.2.0", ""))

    bits: list[str] = []
    if sign_type.startswith("Asymm"):
        bits.append(
            "common failure: the non-dominant hand drops position or rotates; keep it steady"
        )
    elif sign_type.startswith("Symm"):
        bits.append(
            "common failure: the two hands fall out of mirror symmetry; both hands should move identically"
        )
    if movement and movement not in ("Static", "None"):
        bits.append("common failure: the movement is cut short; let the full motion complete")
    else:
        bits.append("common failure: the hand drifts during the hold; keep position stable")
    if contact and contact not in ("None", "0", ""):
        bits.append(
            "common failure: missing the contact point; the hand should actually touch the indicated location"
        )

    # Always close with parameter framing.
    bits.append(
        "if unsure, watch the reference and check: handshape, palm orientation, location, movement, facial expression"
    )
    # Capitalize each bit and join with periods.
    sentences = [b[0].upper() + b[1:] + "." for b in bits]
    return " ".join(sentences)


def _esc(s: str) -> str:
    return s.replace("'", "''")


def author(asl_lex_csv: Path, output: Path) -> None:
    # Load ASL-LEX rows by Code for fast lookup.
    by_code: dict[str, dict[str, str]] = {}
    with asl_lex_csv.open(encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("Code") or "").strip()
            if code:
                by_code[code] = row

    items = get_all()
    print(f"loaded {len(by_code)} ASL-LEX rows; authoring hints for {len(items)} signs")

    output.parent.mkdir(parents=True, exist_ok=True)
    out_lines: list[str] = []
    out_lines.append(
        "-- Layer A (pre-attempt) + Layer C (generic failure) hints for the 96 slice-1 signs."
    )
    out_lines.append(
        "-- Authored mechanically from ASL-LEX 2.0 phonological data (Sevcikova Sehyr et al. 2021)."
    )
    out_lines.append(
        "-- Slice-1 only per ADR 0004; slice-2 production-deployment work replaces with"
    )
    out_lines.append("-- Deaf-signer-reviewed copy. Re-run is idempotent (UPDATEs in place).")
    out_lines.append("")

    missing: list[str] = []
    for item in items:
        row = by_code.get(item.asl_lex_code or "")
        if not row:
            missing.append(item.gloss)
            # Fall back to a parameter-agnostic but still informative hint.
            la = (
                f"Before signing {item.gloss}, watch the reference once. "
                "Note the handshape, the starting location, and how the hand moves."
            )
            lc = (
                "Common failure: rushing the movement before the handshape is set. "
                "Slow down, form the handshape first, then perform the movement."
            )
        else:
            la = _layer_a(item, row)
            lc = _layer_c(item, row)
        out_lines.append(
            f"update public.vocabulary_items set pre_attempt_hint = '{_esc(la)}', "
            f"generic_failure_hint = '{_esc(lc)}' where id = '{item.sign_id}';"
        )

    out_lines.append("")
    if missing:
        out_lines.append(f"-- ASL-LEX code missing for: {', '.join(missing)}")
        print(f"WARN: {len(missing)} signs had no ASL-LEX match; using generic copy")

    output.write_text("\n".join(out_lines) + "\n")
    print(f"wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asl-lex",
        type=Path,
        default=Path("training/data/sources/asl_lex_signdata.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("supabase/migrations/20260519120000_seed_hints.sql"),
    )
    args = parser.parse_args()
    author(args.asl_lex, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
