"""Command line.

    python -m carprice train --data data/usa_used_cars_carscom_1_6m.parquet
    python -m carprice predict --make Toyota --model Camry --trim SE --year 2021 --mileage 40000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from . import data, model  # noqa: E402


def gpu_available() -> bool:
    try:
        from xgboost import XGBRegressor

        XGBRegressor(device="cuda", n_estimators=1).fit(np.zeros((4, 1)), [0, 1, 0, 1])
        return True
    except Exception:
        return False


def train(args) -> dict:
    out, model_dir = Path(args.out), Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    clean, steps = data.clean(data.load_raw(args.data))
    if args.sample and len(clean) > args.sample:
        clean = clean.sample(args.sample, random_state=0).reset_index(drop=True)
    table = data.build_features(clean)
    train_df, rest = train_test_split(table, test_size=0.2, random_state=0)
    valid_df, test_df = train_test_split(rest, test_size=0.5, random_state=0)
    y = {k: d[data.TARGET] for k, d in (("train", train_df), ("valid", valid_df), ("test", test_df))}
    device = "cuda" if (args.device == "auto" and gpu_available()) or args.device == "cuda" else "cpu"
    print(f"{len(table):,} cars ({len(train_df):,} train / {len(valid_df):,} validation / {len(test_df):,} test), "
          f"device {device}")

    results = {}
    gm = model.GroupMedian().fit(train_df, y["train"])
    results["Median price of same make, model and year"] = model.metrics(y["test"], gm.predict(test_df))
    rg = model.ridge().fit(train_df, y["train"])
    results["Ridge regression (one-hot)"] = model.metrics(y["test"], rg.predict(test_df))
    t1 = time.time()
    pm = model.PriceModel(device, args.trees).fit(train_df, y["train"], valid_df, y["valid"])
    fit_seconds = time.time() - t1
    pred = pm.predict(test_df)
    results["XGBoost (target-encoded models)"] = model.metrics(y["test"], pred)
    for name, m in results.items():
        print(f"{name:45s} R2(log) {m['r2_log']:.3f}  R2($) {m['r2_dollars']:.3f}  MAE ${m['mae_dollars']:,.0f}  "
              f"median error {m['median_abs_pct_error']:.1%}")
    pm.save(model_dir)

    importance = pd.Series(pm.model.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    deals = model.find_deals(test_df, pred, pm.training_listings(test_df))
    bands = band_errors(test_df["price"], np.exp(pred))
    plot_actual_vs_predicted(test_df["price"], np.exp(pred), out / "actual_vs_predicted.png")
    plot_importance(importance, out / "feature_importance.png")
    plot_depreciation(table, out / "depreciation.png")
    summary = {
        "data": "cars.com used-car listings, October 2026 (KTHESAT/usa-used-car-market-cars-com, CC BY 4.0)",
        "cleaning_steps": steps, "rows_used": len(table),
        "split": {"train": len(train_df), "validation": len(valid_df), "test": len(test_df)},
        "device": device, "trees": int(pm.model.best_iteration + 1), "xgboost_fit_seconds": round(fit_seconds, 1),
        "results": {k: {m: round(v, 4) for m, v in r.items()} for k, r in results.items()},
        "error_by_price_band": bands,
        "top_features_by_gain": {k: round(float(v), 2) for k, v in importance.head(12).items()},
        "deals_in_test_set": deals.round(3).to_dict(orient="records"),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def band_errors(price, pred) -> list:
    bands = [0, 10_000, 20_000, 30_000, 50_000, 80_000, 200_001]
    d = pd.DataFrame({"price": np.asarray(price), "ape": np.abs(np.asarray(pred) - np.asarray(price)) / np.asarray(price)})
    d["band"] = pd.cut(d["price"], bands, right=False)
    g = d.groupby("band", observed=True)["ape"].agg(["size", "median"])
    return [{"band": f"${int(i.left):,}-{int(i.right) - 1:,}", "cars": int(r["size"]),
             "median_abs_pct_error": round(float(r["median"]), 4)} for i, r in g.iterrows()]


def plot_actual_vs_predicted(price, pred, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    hb = ax.hexbin(price, pred, gridsize=70, bins="log", xscale="log", yscale="log", cmap="Blues", mincnt=1)
    lims = [1_000, 200_000]
    ax.plot(lims, lims, color="#c0392b", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Listed price ($, log scale)")
    ax.set_ylabel("Predicted price ($, log scale)")
    ax.set_title("Test set: predicted vs listed price")
    ax.legend(loc="upper left")
    fig.colorbar(hb, label="Cars (log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(importance, path) -> None:
    top = importance.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top.index, top.values / importance.sum(), color="#2f6fb0")
    ax.set_xlabel("Share of total gain")
    ax.set_title("What determines a used car's price")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_depreciation(table, path, models_to_show=4) -> None:
    # The most-listed model of four different body types, so the chart is not four pickup trucks.
    popular = [table.loc[table["body_type"] == b, "make_model"].value_counts().index[0]
               for b in ("SUV", "Sedan", "Truck", "Hatchback") if (table["body_type"] == b).any()]
    if not popular:
        popular = table["make_model"].value_counts().head(models_to_show).index.tolist()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mm in popular:
        d = table[(table["make_model"] == mm) & (table["age"] <= 10)]
        med = d.groupby("age")["price"].median()
        ax.plot(med.index, med.values / med.iloc[0] * 100, marker="o", label=mm)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Median price, % of newest model year")
    ax.set_title("Depreciation: most-listed SUV, sedan, truck and hatchback")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def predict(args) -> float:
    pm = model.PriceModel.load(args.model_dir)
    raw = pd.DataFrame([{
        "make": args.make, "model": args.model, "trim": args.trim or "", "year": args.year,
        "mileage_miles": args.mileage, "body_type": args.body or "", "drivetrain": args.drivetrain or "",
        "fuel": args.fuel or "", "engine": args.engine or "", "transmission": args.transmission or "",
        "exterior_color": args.colour or "", "interior_color": "", "mpg": "", "cpo": int(args.certified),
        "seller_zip": args.zip or "",
    }])
    price = float(np.exp(pm.predict(data.build_features(raw))[0]))
    print(f"Estimated price: ${price:,.0f}")
    return price


def main(argv=None):
    p = argparse.ArgumentParser(description="Used car price prediction")
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/usa_used_cars_carscom_1_6m.parquet")
    t.add_argument("--out", default="results")
    t.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    t.add_argument("--trees", type=int, default=5000, help="maximum trees (early stopping picks the number)")
    t.add_argument("--sample", type=int, default=0, help="train on a random sample of this many cars")
    pr = sub.add_parser("predict")
    for name in ("make", "model"):
        pr.add_argument(f"--{name}", required=True)
    pr.add_argument("--year", type=int, required=True)
    pr.add_argument("--mileage", type=float, required=True)
    for name in ("trim", "body", "drivetrain", "fuel", "engine", "transmission", "colour", "zip"):
        pr.add_argument(f"--{name}")
    pr.add_argument("--certified", action="store_true")
    for s in (t, pr):
        s.add_argument("--model-dir", default="models")
    args = p.parse_args(argv)
    return train(args) if args.command == "train" else predict(args)


if __name__ == "__main__":
    main()
