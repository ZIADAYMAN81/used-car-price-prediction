"""Turning messy free-text listing fields into clean features."""
from __future__ import annotations

import re

import numpy as np

_SIZE = re.compile(r"(\d{1,2}(?:\.\d)?)\s*L\b", re.I)
_CYL = re.compile(r"\b(?:[IVHW])-?(\d{1,2})\b|\b(\d{1,2})[ -]?cyl", re.I)
_GEARS = re.compile(r"(\d{1,2})[- ]?speed", re.I)
_MPG = re.compile(r"^\s*(\d{1,3})\s*-\s*(\d{1,3})\s*$")

# First pass: the colour name itself as a whole word, in this order, so "Platinum White Pearl" is white,
# "Black Sapphire" is black and "Lunar Silver" is silver. Second pass: marketing synonyms ("Granite Crystal").
BASE_COLOURS = [
    ("White", ("white", "ivory", "snow", "pearl white", "chalk", "glacier")),
    ("Black", ("black", "ebony", "onyx", "obsidian", "jet", "raven", "panther")),
    ("Gray", ("gray", "grey", "graphite", "gun", "granite", "charcoal", "slate", "steel", "magnetic",
              "carbon", "tungsten", "lunar", "storm", "cement", "nardo")),
    ("Silver", ("silver", "platinum", "aluminum", "chrome", "titanium")),
    ("Blue", ("blue", "navy", "sapphire", "azure", "cobalt", "indigo", "aqua", "teal")),
    ("Red", ("red", "ruby", "crimson", "maroon", "burgundy", "cherry", "garnet", "rosso", "cayenne")),
    ("Green", ("green", "olive", "emerald", "jade", "forest", "army", "sage")),
    ("Brown", ("brown", "bronze", "mocha", "espresso", "chestnut", "copper", "walnut", "sandstone", "java")),
    ("Beige", ("beige", "tan", "sand", "champagne", "khaki", "cream", "desert", "sahara")),
    ("Gold", ("gold",)),
    ("Orange", ("orange", "sunset", "tangerine")),
    ("Yellow", ("yellow", "lime")),
    ("Purple", ("purple", "violet", "plum")),
]


def engine_size(text: str) -> float:
    """'2.0L I4 16V GDI DOHC Turbo' -> 2.0. Sizes outside 0.6-8.5 L are treated as parsing noise."""
    m = _SIZE.search(text or "")
    if not m:
        return np.nan
    size = float(m.group(1))
    return size if 0.6 <= size <= 8.5 else np.nan


def cylinders(text: str) -> float:
    """'3.6L V6 24V' -> 6, 'I-4' -> 4, '6 Cyl' -> 6."""
    m = _CYL.search(text or "")
    if not m:
        return np.nan
    n = int(m.group(1) or m.group(2))
    return float(n) if n in (2, 3, 4, 5, 6, 8, 10, 12, 16) else np.nan


def forced_induction(text: str) -> int:
    t = (text or "").lower()
    return int(any(k in t for k in ("turbo", "supercharg", "intercool")))


def transmission_type(text: str) -> str:
    t = (text or "").lower()
    if not t:
        return "Unknown"
    if "cvt" in t or "variable" in t:
        return "CVT"
    if "dual" in t or "dct" in t or "pdk" in t:
        return "Dual-Clutch"
    if "manual" in t:
        return "Manual"
    if "auto" in t:
        return "Automatic"
    return "Other"


def gears(text: str) -> float:
    m = _GEARS.search(text or "")
    if not m:
        return np.nan
    n = int(m.group(1))
    return float(n) if 1 <= n <= 10 else np.nan


def fuel(text: str, engine: str = "") -> str:
    t = f"{text or ''} {engine or ''}".lower()
    if "plug-in" in t:
        return "Plug-In Hybrid"
    if "hybrid" in t:
        return "Hybrid"
    if "electric" in t:
        return "Electric"
    if "diesel" in t:
        return "Diesel"
    if "flex" in t or "e85" in t:
        return "Flex Fuel"
    if "gas" in t or "unleaded" in t or "petrol" in t:
        return "Gasoline"
    return "Unknown"


def drivetrain(text: str) -> str:
    t = (text or "").lower().replace("-", " ")
    if t.startswith("all") or t == "awd":
        return "AWD"
    if t.startswith("four") or t in ("4wd", "4x4"):
        return "4WD"
    if t.startswith("front") or t == "fwd":
        return "FWD"
    if t.startswith("rear") or t == "rwd":
        return "RWD"
    return "Unknown"


def base_colour(text: str) -> str:
    t = (text or "").lower()
    if not t:
        return "Unknown"
    for name, _ in BASE_COLOURS:
        if re.search(rf"\b{name.lower()}\b", t) or (name == "Gray" and re.search(r"\bgrey\b", t)):
            return name
    for name, words in BASE_COLOURS:
        if any(w in t for w in words):
            return name
    return "Other"


def finish(text: str) -> str:
    t = (text or "").lower()
    if "matte" in t or "frozen" in t:
        return "Matte"
    if "pearl" in t or "mica" in t or "tricoat" in t or "tri-coat" in t:
        return "Pearl"
    if "metallic" in t or "metal" in t:
        return "Metallic"
    return "Solid"


def mpg(text: str) -> tuple[float, float]:
    """'19-26' -> (19, 26): city and highway miles per gallon."""
    m = _MPG.match(text or "")
    if not m:
        return np.nan, np.nan
    city, highway = float(m.group(1)), float(m.group(2))
    return (city, highway) if 5 <= city <= 160 and 5 <= highway <= 160 else (np.nan, np.nan)
