"""Cleaning the cars.com listings and building the model table."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import parsing

SNAPSHOT_YEAR = 2026  # every listing in the dataset was collected in October 2026

CATEGORICAL = ["make", "make_model", "make_model_trim", "body_type", "drivetrain", "fuel", "transmission_type",
               "exterior_colour", "exterior_finish", "interior_colour", "region"]
NUMERIC = ["year", "age", "mileage", "log_mileage", "miles_per_year", "engine_size", "cylinders",
           "forced_induction", "gears", "mpg_city", "mpg_highway", "certified_pre_owned"]
TARGET = "log_price"

RULES = {"min_price": 1_000, "max_price": 200_000, "min_year": 1990, "max_year": SNAPSHOT_YEAR + 1,
         "max_mileage": 400_000}


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep used cars with full specifications and plausible values; one row per vehicle (VIN)."""
    steps = {"raw": len(raw)}
    df = raw[(raw["stock_type"] == "Used") & (raw["is_enriched"] == 1)]
    steps["used_with_specifications"] = len(df)
    df = df.sort_values("last_checked_at").drop_duplicates("vin", keep="last")
    steps["after_removing_duplicate_vins"] = len(df)
    df = df[df["price_usd"].between(RULES["min_price"], RULES["max_price"])
            & df["year"].between(RULES["min_year"], RULES["max_year"])
            & df["mileage_miles"].between(1, RULES["max_mileage"])
            & (df["make"].str.strip() != "") & (df["model"].str.strip() != "")]
    steps["after_plausibility_rules"] = len(df)
    return df.reset_index(drop=True), steps


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["make"] = df["make"].str.strip()
    out["make_model"] = out["make"] + " " + df["model"].str.strip()
    out["make_model_trim"] = out["make_model"] + " " + df["trim"].fillna("").str.strip()
    out["body_type"] = df["body_type"].replace("", "Unknown")
    out["drivetrain"] = df["drivetrain"].map(parsing.drivetrain)
    out["fuel"] = [parsing.fuel(f, e) for f, e in zip(df["fuel"], df["engine"])]
    out["transmission_type"] = df["transmission"].map(parsing.transmission_type)
    out["exterior_colour"] = df["exterior_color"].map(parsing.base_colour)
    out["exterior_finish"] = df["exterior_color"].map(parsing.finish)
    out["interior_colour"] = df["interior_color"].map(parsing.base_colour)
    zips = df["seller_zip"].fillna("").astype(str).str.strip()
    out["region"] = np.where(zips.str.match(r"^\d"), "zip-" + zips.str[:1], "Unknown")

    out["year"] = df["year"].astype(float)
    out["age"] = (SNAPSHOT_YEAR - df["year"]).clip(lower=0).astype(float)
    out["mileage"] = df["mileage_miles"].astype(float)
    out["log_mileage"] = np.log1p(out["mileage"])
    out["miles_per_year"] = out["mileage"] / out["age"].clip(lower=0.5)
    out["engine_size"] = df["engine"].map(parsing.engine_size)
    out["cylinders"] = df["engine"].map(parsing.cylinders)
    out["forced_induction"] = df["engine"].map(parsing.forced_induction)
    out["gears"] = df["transmission"].map(parsing.gears)
    city_highway = df["mpg"].map(parsing.mpg)
    out["mpg_city"] = [c for c, _ in city_highway]
    out["mpg_highway"] = [h for _, h in city_highway]
    out["certified_pre_owned"] = df["cpo"].astype(float)

    if "price_usd" in df:
        out["price"] = df["price_usd"].astype(float)
        out[TARGET] = np.log(out["price"])
    return out
