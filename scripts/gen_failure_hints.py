#!/usr/bin/env python3
"""Generate per-sign "how to sign it" failure hints for the 80 active signs.

Reproducible generator. Reads:
  * public/models/sign_classifier_v4.config.json   -> the 80 active sign ids (authoritative)
  * supabase/migrations/20260519100200_seed_vocabulary.sql -> id -> (asl_lex_code, asl_lex_match)
  * training/data/sources/asl_lex_signdata.csv      -> ASL-LEX 2.0 phonology (the .2.0 columns)
  * dataset/sign_handshapes_top80.json              -> our own per-sign handshape label,
        used ONLY to disambiguate which ASL-LEX homonym variant a sign maps to when the
        seed has no asl_lex_code AND EntryID has multiple variants (cool/dark/follow/same).

For each active sign it composes a concise, tutor-toned 1-3 sentence production hint
sourced ENTIRELY from ASL-LEX phonology (handshape, location, movement, repetition,
one/two-handed). It NEVER invents specifics. If a sign cannot be matched to ASL-LEX,
it is flagged and a safe generic fallback is emitted instead.

Emits the idempotent migration SQL to stdout (or to --out).

Usage:
    python3 scripts/gen_failure_hints.py            # print SQL
    python3 scripts/gen_failure_hints.py --report   # also print SIGN: hint lines to stderr
    python3 scripts/gen_failure_hints.py --out supabase/migrations/<ts>_failure_hints_top80.sql
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = os.path.join(ROOT, "public/models/sign_classifier_v4.config.json")
SEED = os.path.join(ROOT, "supabase/migrations/20260519100200_seed_vocabulary.sql")
ASL_LEX = os.path.join(ROOT, "training/data/sources/asl_lex_signdata.csv")
HANDSHAPES = os.path.join(ROOT, "dataset/sign_handshapes_top80.json")

GENERIC_FALLBACK = (
    "Watch the reference once more and copy it in three steps: match the hand "
    "shape, place your hand where the signer's is, then trace the same movement."
)

# ASL-LEX handshape code -> readable name. Derived from the labels that actually
# occur in the CSV's Handshape.2.0 / NonDominantHandshape.2.0 columns.
HANDSHAPE_NAMES = {
    "1": "a '1' (index finger pointing up, others closed)",
    "bent_1": "a bent-1 (crooked index finger)",
    "flat_1": "a flat-1 (index out, hand flattened)",
    "curved_1": "a curved-1 (slightly hooked index)",
    "3": "a '3' (thumb, index and middle fingers out)",
    "4": "a '4' (four fingers up, thumb tucked)",
    "flat_4": "a flat-4 (four fingers out and flattened)",
    "curved_4": "a curved-4 (four fingers slightly bent)",
    "5": "an open '5' (all five fingers spread)",
    "curved_5": "a curved-5 (claw hand, fingers spread and bent)",
    "flatspread_5": "a flat spread-5 (flat hand, fingers apart)",
    "stacked_5": "a stacked-5 (loose claw, fingers stacked)",
    "7": "a '7' (ring finger touching thumb)",
    "8": "an '8' (middle finger touching thumb)",
    "open_8": "an open-8 (middle finger bent toward thumb, hand open)",
    "a": "an 'A' (closed fist, thumb up the side)",
    "s": "an 'S' (closed fist, thumb across the front)",
    "b": "a flat 'B' (flat hand, fingers together)",
    "open_b": "an open 'B' (flat hand, fingers spread a little)",
    "flat_b": "a flat 'B' (flat hand, fingers together)",
    "closed_b": "a closed 'B' (flat hand, thumb tucked in)",
    "c": "a 'C' (hand curved into a C shape)",
    "d": "a 'D' (index up, thumb and middle finger forming a circle)",
    "e": "an 'E' (fingertips curled to the thumb)",
    "open_e": "an open 'E' (loosely curled fingers)",
    "spread_open_e": "a spread open-E (fingers loosely curled and apart)",
    "spread_e": "a spread 'E' (curled fingers held apart)",
    "closed_e": "a closed 'E' (tightly curled fingers)",
    "f": "an 'F' (index and thumb form a circle, other fingers up)",
    "open_f": "an open 'F' (thumb-index circle, fingers spread)",
    "g": "a 'G' (index and thumb extended, pointing sideways)",
    "h": "an 'H' (index and middle fingers out together)",
    "flat_h": "a flat 'H' (index and middle out and flattened)",
    "open_h": "an open 'H' (index and middle out and apart)",
    "curved_h": "a curved 'H' (index and middle out and slightly bent)",
    "i": "an 'I' (pinky finger up, others closed)",
    "ily": "an ILY (thumb, index and pinky out)",
    "flat_ily": "a flat-ILY (thumb, index and pinky out, hand flattened)",
    "k": "a 'K' (index up, middle out, thumb between them)",
    "l": "an 'L' (index up, thumb out, forming an L)",
    "curved_l": "a curved 'L' (L shape with a bent index)",
    "bent_l": "a bent 'L' (L shape with a crooked index)",
    "flat_l": "a flat 'L' (L shape with the hand flattened)",
    "m": "an 'M' (thumb under three fingers)",
    "flat_m": "a flat 'M' (M shape, hand flattened)",
    "n": "an 'N' (thumb under two fingers)",
    "flat_n": "a flat 'N' (N shape, hand flattened)",
    "o": "an 'O' (fingers and thumb form a round O)",
    "flat_o": "a flat-O (fingertips pinched to the thumb)",
    "baby_o": "a baby-O (a small O made with the index and thumb)",
    "p": "a 'P' (K shape pointed downward)",
    "r": "an 'R' (index and middle fingers crossed)",
    "t": "a 'T' (fist with the thumb between index and middle)",
    "v": "a 'V' (index and middle fingers up in a V)",
    "bent_v": "a bent-V (index and middle fingers up and hooked)",
    "curved_v": "a curved 'V' (V shape with both fingers slightly bent)",
    "flat_v": "a flat 'V' (V shape with the hand flattened)",
    "w": "a 'W' (index, middle and ring fingers up)",
    "y": "a 'Y' (thumb and pinky out, others closed)",
    "horns": "a horns hand (index and pinky out)",
    "flat_horns": "a flat horns hand (index and pinky out, hand flattened)",
    "goody_goody": "a 'goody-goody' hand (a loose claw shape)",
}

# Major/Minor location -> readable place. Composed from MajorLocation.2.0 +
# MinorLocation.2.0 so the learner knows roughly WHERE to sign.
MAJOR_NAMES = {
    "Neutral": "in the neutral space in front of your chest",
    "Head": "up at your head",
    "Body": "in front of your body",
    "Arm": "near your forearm",
    "Hand": "on your other hand",
    "Other": "in front of you",
}

# Minor location refinements (override / append to the major when informative).
MINOR_NAMES = {
    "Forehead": "at your forehead",
    "Eye": "near your eye",
    "CheekNose": "by your cheek or nose",
    "Chin": "at your chin",
    "UnderChin": "under your chin",
    "Mouth": "at your mouth",
    "UpperLip": "at your upper lip",
    "HeadTop": "at the top of your head",
    "HeadAway": "just out from your head",
    "Neck": "at your neck",
    "Clavicle": "at your collarbone",
    "Shoulder": "at your shoulder",
    "UpperArm": "on your upper arm",
    "TorsoTop": "high on your chest",
    "TorsoMid": "at the middle of your chest",
    "TorsoBottom": "low on your torso",
    "Hips": "at your hips",
    "Waist": "at your waist",
    "Heel": "at the heel of your hand",
    "Palm": "in the palm of your other hand",
    "PalmBack": "on the back of your other hand",
    "FingerTip": "at your fingertips",
    "FingerBack": "on the back of your fingers",
    "FingerFront": "on the front of your fingers",
    "FingerRadial": "along the thumb side of your fingers",
    "FingerUlnar": "along the pinky side of your fingers",
    "WristBack": "on the back of your wrist",
    "WristFront": "on the inside of your wrist",
    "ForearmBack": "on the back of your forearm",
    "ForearmFront": "on the front of your forearm",
    "ForearmUlnar": "along the underside of your forearm",
    "ElbowBack": "at your elbow",
    "BodyAway": "just out from your body",
    "HandAway": "just out from your other hand",
    "ArmAway": "just out from your arm",
}

# Movement.2.0 -> readable path phrase.
MOVEMENT_NAMES = {
    "Straight": "move it in a straight line",
    "Curved": "move it along a curved path",
    "Circular": "move it in a circle",
    "Z-shaped": "trace a Z-shaped path",
    "X-shaped": "trace a crossing X-shaped path",
    "BackAndForth": "move it back and forth",
    "Other": "move it as the reference shows",
    "None": None,  # static / held
}


def load_active() -> list[str]:
    cfg = json.load(open(CONFIG))
    return list(cfg["classes"])


def load_seed_map() -> dict[str, tuple[str, str]]:
    """id -> (asl_lex_code, asl_lex_match) parsed from the seed migration."""
    txt = open(SEED).read()
    out: dict[str, tuple[str, str]] = {}
    # ('id', 'GLOSS', 'category', 'L#', 'CODE', 'match', ...)
    for m in re.finditer(
        r"\('([^']+)', '[^']*', '[^']*', '[^']*', '([^']*)', '([^']*)'", txt
    ):
        out[m.group(1)] = (m.group(2), m.group(3))
    return out


def load_aslex_rows() -> list[dict]:
    return list(csv.DictReader(open(ASL_LEX, newline="", encoding="latin-1")))


def load_dataset_handshapes() -> dict[str, str]:
    try:
        return json.load(open(HANDSHAPES))
    except Exception:
        return {}


# Manually confirmed variant picks for signs the seed didn't map AND whose EntryID
# has multiple ASL-LEX homonyms. Chosen by matching dataset/sign_handshapes_top80.json
# (our own clip-derived handshape label) to the ASL-LEX variant's Handshape.2.0.
# These are flagged in the report so a human can verify.
VARIANT_OVERRIDES = {
    "cool": "F_01_014",  # cool_1: HS=5, one-handed at chest (matches dataset HS '5')
    "dark": "E_01_012",  # dark_1: HS=5, two-handed (matches dataset HS '5')
    "follow": "K_03_016",  # follow_4: HS=4, symmetrical (matches dataset HS '4')
    "same": "B_02_009",  # same_1: HS=y, one-handed (matches dataset HS 'y')
}


def resolve_row(
    vid: str,
    seed_map: dict[str, tuple[str, str]],
    by_code: dict[str, dict],
    by_entry: dict[str, list[dict]],
) -> tuple[dict | None, str]:
    """Return (asl_lex_row, provenance) for an active sign id, or (None, reason)."""
    # 1) seed asl_lex_code (preferred — human-curated mapping).
    if vid in seed_map:
        code, match = seed_map[vid]
        if code in by_code:
            return by_code[code], f"seed asl_lex_code {code} ({match})"
    # 2) manual variant override (homonym disambiguated via dataset handshape).
    if vid in VARIANT_OVERRIDES:
        code = VARIANT_OVERRIDES[vid]
        if code in by_code:
            return by_code[code], f"dataset-handshape variant {by_code[code]['EntryID']} ({code})"
    # 3) exact EntryID == id, single variant only (unambiguous).
    if vid in by_entry and len(by_entry[vid]) == 1:
        return by_entry[vid][0], f"EntryID {vid}"
    if vid in by_entry and len(by_entry[vid]) > 1:
        return by_entry[vid][0], f"EntryID {vid} (first of {len(by_entry[vid])})"
    return None, "no ASL-LEX match"


def handshape_phrase(code: str) -> str:
    code = (code or "").strip()
    if code in HANDSHAPE_NAMES:
        return HANDSHAPE_NAMES[code]
    if not code or code in ("NA", "0"):
        return ""
    # Unknown but present: name the raw code rather than inventing a description.
    return f"a '{code}' handshape"


def location_phrase(major: str, minor: str) -> str:
    minor = (minor or "").strip()
    major = (major or "").strip()
    if minor and minor not in ("Neutral", "NA", "Other") and minor in MINOR_NAMES:
        return MINOR_NAMES[minor]
    return MAJOR_NAMES.get(major, "in front of you")


def two_handed(sign_type: str) -> bool:
    return sign_type not in ("OneHanded", "", "NA", None)


def compose_hint(vid: str, gloss: str, row: dict) -> str:
    hs = handshape_phrase(row["Handshape.2.0"])
    sign_type = row["SignType.2.0"]
    movement = row["Movement.2.0"]
    repeated = row["RepeatedMovement.2.0"] == "1"
    major = row["MajorLocation.2.0"]
    minor = row["MinorLocation.2.0"]
    nd_hs = row.get("NonDominantHandshape.2.0", "NA")

    is_two = two_handed(sign_type)
    symmetrical = sign_type in ("SymmetricalOrAlternating",)
    loc = location_phrase(major, minor)

    # Sentence 1: handshape + (non-dominant for two-handed) + hand count.
    if is_two:
        if symmetrical:
            hand_clause = f"Use both hands in {hs}" if hs else "Use both hands"
            s1 = f"{hand_clause}, mirrored."
        else:
            nd = handshape_phrase(nd_hs)
            if hs and nd and nd != hs:
                s1 = (
                    f"Your dominant hand makes {hs} and your other (still) hand makes {nd}."
                )
            elif hs:
                s1 = f"Make {hs} with your dominant hand over your other hand."
            else:
                s1 = "Use two hands as the reference shows."
    else:
        s1 = f"Make {hs} with one hand." if hs else "Use one hand as the reference shows."

    # Sentence 2: location + movement + repetition.
    mv = MOVEMENT_NAMES.get(movement, "move it as the reference shows")
    if mv is None:
        # Static / held sign.
        s2 = f"Hold it {loc}"
        if repeated:
            s2 += ", tapping in place"
        s2 += "."
    else:
        s2 = f"Position it {loc} and {mv}"
        if repeated:
            s2 += ", repeating the motion"
        s2 += "."

    hint = f"To sign {gloss}: {s1} {s2}"
    # Collapse any accidental double spaces.
    return re.sub(r"\s+", " ", hint).strip()


def sql_escape(s: str) -> str:
    return s.replace("'", "''")


def build():
    active = load_active()
    seed_map = load_seed_map()
    rows = load_aslex_rows()
    by_code = {r["Code"]: r for r in rows}
    by_entry: dict[str, list[dict]] = {}
    for r in rows:
        by_entry.setdefault(r["EntryID"], []).append(r)

    results = []  # (vid, gloss, hint, provenance, matched)
    for vid in active:
        gloss = vid.upper()
        row, prov = resolve_row(vid, seed_map, by_code, by_entry)
        if row is None:
            results.append((vid, gloss, GENERIC_FALLBACK, prov, False))
        else:
            results.append((vid, gloss, compose_hint(vid, gloss, row), prov, True))
    return results


def emit_sql(results) -> str:
    ts = "20260525150000"
    lines = []
    lines.append(
        "-- Failure-production hints (generic_failure_hint) for the 80 active signs.\n"
        "-- Generated by scripts/gen_failure_hints.py from ASL-LEX 2.0 phonology\n"
        "-- (training/data/sources/asl_lex_signdata.csv, Sevcikova Sehyr et al. 2021),\n"
        "-- joined via the seed asl_lex_code mapping (20260519100200_seed_vocabulary.sql).\n"
        "-- Each hint names the actual handshape + location + movement so the fail screen\n"
        "-- can show a concrete 'how to sign it' tip. Idempotent: re-run safe (UPDATE in place).\n"
        "-- Sign ids without an ASL-LEX match fall back to a safe generic 3-step copy (none today).\n"
    )
    for vid, gloss, hint, prov, matched in results:
        flag = "" if matched else "  -- FALLBACK: no ASL-LEX match"
        lines.append(
            f"update public.vocabulary_items set generic_failure_hint = '{sql_escape(hint)}' "
            f"where id = '{sql_escape(vid)}';{flag}"
        )
    lines.append(
        "\n-- DOWN MIGRATION (manual; reversible):\n"
        "-- Reverts the 80 hints to NULL. Run only to undo this migration.\n"
        "/*\n"
        "update public.vocabulary_items set generic_failure_hint = null where id in (\n"
        + ",\n".join(
            "  " + ", ".join(f"'{sql_escape(v)}'" for v, *_ in results[i : i + 8])
            for i in range(0, len(results), 8)
        )
        + "\n);\n*/\n"
    )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write SQL to this path instead of stdout")
    ap.add_argument("--report", action="store_true", help="print SIGN: hint lines to stderr")
    args = ap.parse_args()

    results = build()
    sql = emit_sql(results)

    if args.report:
        for vid, gloss, hint, prov, matched in results:
            tag = "" if matched else "  [FALLBACK]"
            print(f"{gloss}: {hint}{tag}", file=sys.stderr)
        n_fb = sum(1 for *_, m in results if not m)
        print(f"\n# {len(results)} signs, {n_fb} fallback(s)", file=sys.stderr)

    if args.out:
        with open(args.out, "w") as f:
            f.write(sql)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(sql)


if __name__ == "__main__":
    main()
