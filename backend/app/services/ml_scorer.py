#!/usr/bin/env python3
"""
ml_scorer.py - SkillProof AI baseline technical scoring + vulnerability (trap) detection.

Inputs (CSV):
    JDS Skill Traits.csv : id, big_data_skills, maths-stats_skills, coding_skills,
                           ai_and_ml_skills, dashboard_and_storytelling_skills,
                           salary_hike_high_or_low
    SDS Personality.csv  : id, neuroticism, extraversion, openness_to_experience,
                           agreeableness, conscientiousness,
                           success_ classification_ high_low

Output:
    candidate_scores.json - one record per candidate (see build_records()).

Usage:
    python ml_scorer.py
    python ml_scorer.py --skills "JDS Skill Traits.csv" --personality "SDS Personality.csv" \
                        --output candidate_scores.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SKILL_COLS = [
    "big_data_skills",
    "maths-stats_skills",
    "coding_skills",
    "ai_and_ml_skills",
    "dashboard_and_storytelling_skills",
]
PERSONALITY_COLS = [
    "neuroticism",
    "extraversion",
    "openness_to_experience",
    "agreeableness",
    "conscientiousness",
]
SKILLS_REQUIRED = ["id"] + SKILL_COLS
PERSONALITY_REQUIRED = ["id"] + PERSONALITY_COLS

HIGH_THRESHOLD = 7   # score >= 7 counts as "high"
LOW_THRESHOLD = 3    # score <= 3 counts as "low"
SCORE_SCALE = 10     # mean of 0-10 skills * 10 -> 0-100

# Deterministic trap rules: (code, high_col, low_col, description)
TRAP_RULES = [
    (
        "AIML_WITHOUT_MATH_STATS",
        "ai_and_ml_skills",
        "maths-stats_skills",
        "High AI/ML skill paired with low Maths/Stats foundation.",
    ),
    (
        "BIGDATA_WITHOUT_CODING",
        "big_data_skills",
        "coding_skills",
        "High Big Data skill paired with low Coding ability.",
    ),
    (
        "HIGH_NEUROTICISM_LOW_CONSCIENTIOUSNESS",
        "neuroticism",
        "conscientiousness",
        "High Neuroticism combined with low Conscientiousness.",
    ),
]


class ScorerError(Exception):
    """Raised for recoverable, user-facing input problems."""


# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #
def clean_header(name: str) -> str:
    """Strip leading/trailing whitespace; also collapse internal spaces so that
    'success_ classification_ high_low' -> 'success_classification_high_low'."""
    return "".join(str(name).strip().split())


def clean_id_value(value) -> str:
    """Normalise a candidate id so Excel/CSV formatting cannot break the join.

    Lowercase, strip whitespace, and drop a trailing '.0' when Excel stored
    the id as a float (e.g. 101.0 -> '101').
    """
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
        if text.endswith(".0"):
            stem = text[:-2]
            if stem.replace("-", "", 1).isdigit():
                text = stem
    return text.strip().lower()


def load_csv(path: Path, required: list, label: str) -> pd.DataFrame:
    if not path.exists():
        raise ScorerError(f"{label} file not found: '{path}'")
    if not path.is_file():
        raise ScorerError(f"{label} path is not a file: '{path}'")

    try:
        df = pd.read_excel(path)
    except pd.errors.EmptyDataError:
        raise ScorerError(f"{label} file is empty: '{path}'")
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ScorerError(f"{label} file could not be parsed ('{path}'): {exc}")

    df.columns = [clean_header(c) for c in df.columns]

    id_rename = {c: "id" for c in df.columns if str(c).lower() == "id" and c != "id"}
    if id_rename:
        df = df.rename(columns=id_rename)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ScorerError(
            f"{label} file is missing required column(s): {missing}. "
            f"Found: {list(df.columns)}"
        )
    if df.empty:
        raise ScorerError(f"{label} file contains no data rows: '{path}'")

    df["id"] = df["id"].map(clean_id_value)
    return df


def coerce_numeric(df: pd.DataFrame, cols: list, label: str, warnings: list) -> pd.DataFrame:
    """Force score columns to numeric; drop rows where any is non-numeric/missing."""
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = df[cols].isna().any(axis=1)
    if bad.any():
        warnings.append(
            f"{label}: dropped {int(bad.sum())} row(s) with missing/non-numeric score values."
        )
        df = df.loc[~bad].copy()
    return df


def drop_duplicate_ids(df: pd.DataFrame, label: str, warnings: list) -> pd.DataFrame:
    dupes = df.duplicated(subset="id", keep="first")
    if dupes.any():
        warnings.append(
            f"{label}: {int(dupes.sum())} duplicate id(s) found; kept first occurrence."
        )
        df = df.loc[~dupes].copy()
    return df


# --------------------------------------------------------------------------- #
# Scoring & trap engine
# --------------------------------------------------------------------------- #
def add_suitability_score(df: pd.DataFrame) -> pd.DataFrame:
    """Requirement 3: mean of 5 skill columns * 10 -> 0-100."""
    df["suitability_score"] = (df[SKILL_COLS].mean(axis=1) * SCORE_SCALE).round(2)
    df["suitability_score"] = df["suitability_score"].clip(0, 100)
    return df


def detect_vulnerabilities(row: pd.Series) -> list:
    """Requirement 4: deterministic rule-based trap engine."""
    flags = []
    for code, high_col, low_col, description in TRAP_RULES:
        high_val, low_val = row[high_col], row[low_col]
        if high_val >= HIGH_THRESHOLD and low_val <= LOW_THRESHOLD:
            flags.append(
                {
                    "code": code,
                    "description": description,
                    "evidence": {high_col: _num(high_val), low_col: _num(low_val)},
                }
            )
    return flags


def _num(x):
    """Return int when the value is whole, else float (clean JSON)."""
    x = float(x)
    return int(x) if x.is_integer() else round(x, 4)


# --------------------------------------------------------------------------- #
# Output assembly
# --------------------------------------------------------------------------- #
def build_records(df: pd.DataFrame) -> list:
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "candidate_id": row["id"],
                "suitability_score": float(row["suitability_score"]),
                "skill_scores": {c: _num(row[c]) for c in SKILL_COLS},
                "personality_scores": {c: _num(row[c]) for c in PERSONALITY_COLS},
                "flagged_vulnerabilities": detect_vulnerabilities(row),
            }
        )
    return records


def print_summary(records: list, stats: dict, warnings: list, output_path: Path) -> None:
    n = len(records)
    flagged = [r for r in records if r["flagged_vulnerabilities"]]
    scores = [r["suitability_score"] for r in records]

    rule_counts = {code: 0 for code, *_ in TRAP_RULES}
    for r in records:
        for v in r["flagged_vulnerabilities"]:
            rule_counts[v["code"]] += 1

    line = "=" * 62
    print(line)
    print(" SkillProof AI - ML Scorer Summary")
    print(line)
    print(f" Skills rows loaded       : {stats['skills_rows']}")
    print(f" Personality rows loaded  : {stats['personality_rows']}")
    print(f" Candidates after join    : {n}")
    print(f" Unmatched (dropped)      : skills={stats['skills_unmatched']}, "
          f"personality={stats['personality_unmatched']}")
    if n:
        print(f" Suitability score        : mean={sum(scores)/n:.2f}  "
              f"min={min(scores):.2f}  max={max(scores):.2f}")
        print(f" Candidates flagged       : {len(flagged)} ({len(flagged)/n:.1%})")
    print(" Flags by rule:")
    for code, count in rule_counts.items():
        print(f"   - {code:<42} {count}")
    if warnings:
        print(" Warnings:")
        for w in warnings:
            print(f"   ! {w}")
    print(f" Output written to        : {output_path}")
    print(line)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def run(skills_path: Path, personality_path: Path, output_path: Path) -> list:
    warnings = []

    skills = load_csv(skills_path, SKILLS_REQUIRED, "Skills (JDS)")
    personality = load_csv(personality_path, PERSONALITY_REQUIRED, "Personality (SDS)")

    skills = coerce_numeric(skills, SKILL_COLS, "Skills (JDS)", warnings)
    personality = coerce_numeric(personality, PERSONALITY_COLS, "Personality (SDS)", warnings)

    # Pair rows by position (the two files use disjoint candidate id spaces).
    # Keep the skills file 'id' as the primary candidate id.
    skills_df = skills.reset_index(drop=True)
    personality_df = personality.drop(columns=["id"]).reset_index(drop=True)
    merged = pd.concat([skills_df, personality_df], axis=1)
    merged = merged.dropna(subset=SKILL_COLS + PERSONALITY_COLS).copy()
    if merged.empty:
        raise ScorerError("Positional join produced no rows - one or both files had no usable data.")

    merged = drop_duplicate_ids(merged, "Skills (JDS)", warnings)

    stats = {
        "skills_rows": len(skills),
        "personality_rows": len(personality),
        "skills_unmatched": max(0, len(skills) - len(merged)),
        "personality_unmatched": max(0, len(personality) - len(merged)),
    }

    merged = add_suitability_score(merged)
    records = build_records(merged)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise ScorerError(f"Could not write output file '{output_path}': {exc}")

    print_summary(records, stats, warnings, output_path)
    return records


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SkillProof AI baseline scorer & trap engine.")
    p.add_argument("--skills", default="data/JDS Skill Traits.xlsx", help="Path to skills Excel.")
    p.add_argument("--personality", default="data/SDS Personality Traits.xlsx", help="Path to personality Excel.")
    p.add_argument("--output", default="candidate_scores.json", help="Path for JSON output.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    base_dir = Path(__file__).resolve().parent
    
    skills_path = (base_dir / args.skills).resolve() if not Path(args.skills).is_absolute() else Path(args.skills)
    personality_path = (base_dir / args.personality).resolve() if not Path(args.personality).is_absolute() else Path(args.personality)
    output_path = (base_dir / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    
    try:
        run(skills_path, personality_path, output_path)
    except ScorerError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[UNEXPECTED ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())