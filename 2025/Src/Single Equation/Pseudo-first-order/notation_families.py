# notation_families.py
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Dict, Tuple, List

# Your internal "canonical" variable tokens used in the dataframe symbol part
# (the part before _(...)).
CANONICAL_TOKENS = ["qmax", "qe", "qt", "KL", "k2", "RL"]

# Realistic notation families (no mixing inside a family)
# Keys MUST match CANONICAL_TOKENS.
NOTATION_FAMILIES: Dict[str, Dict[str, str]] = {
    # Environmental / Water treatment (most common)
    "env_standard": {
        "qmax": "q_max",
        "qe":   "q_e",
        "qt":   "q_t",
        "KL":   "K_L",
        "k2":   "k_2",
        "RL":   "R_L",
    },

    # Chemical Engineering style (capital Q, b for Langmuir)
    "chem_eng": {
        "qmax": "Q_m",
        "qe":   "Q_e",
        "qt":   "Q_t",
        "KL":   "b",
        "k2":   "k_2",
        "RL":   "R_L",
    },

    # Materials / surface chemistry style
    "materials": {
        "qmax": "Q_max",
        "qe":   "Q_e",
        "qt":   "Q_t",
        "KL":   "K",
        "k2":   "k_ps",
        "RL":   "R_L",
    },

    # Soil / geochem style (slightly simplified)
    "soil_geo": {
        "qmax": "q_m",
        "qe":   "q_e",
        "qt":   "q_t",
        "KL":   "K",
        "k2":   "k_2",
        "RL":   "R",
    },
}

_COL_RE = re.compile(r"^(.+?)_\((.+)\)$")

def split_col(col: str) -> Tuple[str, str]:
    """
    Split a column like 'qmax_(mg/g)' -> ('qmax', 'mg/g')
    """
    m = _COL_RE.match(col)
    if not m:
        return col, ""
    return m.group(1), m.group(2)

def join_col(sym: str, unit: str) -> str:
    if unit:
        return f"{sym}_({unit})"
    return sym

@dataclass(frozen=True)
class NotationChoice:
    family_name: str
    mapping: Dict[str, str]  # canonical token -> chosen symbol

def choose_family(experiment: dict, rng) -> NotationChoice:
    """
    Picks one family per report.
    If experiment['notation_family'] is set to a valid family name, it is forced.
    Otherwise randomly samples from experiment['notation_families'] (or all).
    """
    forced = (experiment.get("notation_family") or "").strip()
    if forced:
        if forced not in NOTATION_FAMILIES:
            raise ValueError(f"Unknown notation_family='{forced}'. Valid: {list(NOTATION_FAMILIES)}")
        return NotationChoice(forced, NOTATION_FAMILIES[forced])

    allowed = experiment.get("notation_families")
    if not allowed:
        allowed = list(NOTATION_FAMILIES.keys())

    allowed = [x for x in allowed if x in NOTATION_FAMILIES]
    if not allowed:
        raise ValueError("No valid notation families configured.")

    family_name = rng.choice(allowed)
    return NotationChoice(family_name, NOTATION_FAMILIES[family_name])

def apply_notation_to_dataframe(df, notation: NotationChoice):
    """
    Renames df columns by replacing ONLY the symbol token (before _(unit)).
    Example: 'qmax_(mg/g)' -> 'Q_m_(mg/g)' if family maps qmax->Q_m.
    """
    rename = {}
    for c in df.columns:
        if c == "run_id":
            continue
        sym, unit = split_col(c)
        if sym in notation.mapping:
            rename[c] = join_col(notation.mapping[sym], unit)
    return df.rename(columns=rename), rename


def choose_family_sequential(experiment: dict, i: int) -> NotationChoice:
    allowed = experiment.get("notation_families")
    if not allowed:
        allowed = list(NOTATION_FAMILIES.keys())

    # keep only valid
    allowed = [x for x in allowed if x in NOTATION_FAMILIES]
    if not allowed:
        raise ValueError("No valid notation families configured.")

    family_name = allowed[(i - 1) % len(allowed)]
    return NotationChoice(family_name, NOTATION_FAMILIES[family_name])


def save_mapping_json(out_dir, paper_id: str, notation: NotationChoice, col_rename_map: Dict[str, str], report_seed: int | None):
    """
    Saves the family + canonical->chosen mapping + actual col renames.
    """
    payload = {
        "paper_id": paper_id,
        "notation_family": notation.family_name,
        "canonical_to_chosen_symbol": notation.mapping,
        "column_rename_map": col_rename_map,
        "report_seed": report_seed,
    }
    path = out_dir / f"{paper_id}_notation.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path