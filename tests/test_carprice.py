import numpy as np
import pandas as pd
import pytest

from carprice import data, model, parsing
from carprice.__main__ import main


@pytest.mark.parametrize("text, size, cyl, turbo", [
    ("2.0L I4 16V GDI DOHC Turbo", 2.0, 4, 1),
    ("Pentastar 3.6L V-6 DOHC, variable valve control", 3.6, 6, 0),
    ("2L I-4 direct injection, DOHC, variable valve control, intercool", 2.0, 4, 1),
    ("6.2L 8 Cyl", 6.2, 8, 0),
    ("Electric", np.nan, np.nan, 0),
    ("", np.nan, np.nan, 0),
])
def test_engine_parsing(text, size, cyl, turbo):
    assert parsing.engine_size(text) == pytest.approx(size, nan_ok=True)
    assert parsing.cylinders(text) == pytest.approx(cyl, nan_ok=True)
    assert parsing.forced_induction(text) == turbo


@pytest.mark.parametrize("text, kind, n", [
    ("8-Speed Automatic with Tiptronic", "Automatic", 8), ("Automatic CVT", "CVT", np.nan),
    ("7-Speed Dual Clutch", "Dual-Clutch", 7), ("Manual", "Manual", np.nan), ("Automatic 6-Speed", "Automatic", 6),
    ("", "Unknown", np.nan),
])
def test_transmission_parsing(text, kind, n):
    assert parsing.transmission_type(text) == kind
    assert parsing.gears(text) == pytest.approx(n, nan_ok=True)


@pytest.mark.parametrize("text, base, fin", [
    ("Platinum White Pearl", "White", "Pearl"), ("Black Sapphire Metallic", "Black", "Metallic"),
    ("Lunar Silver Metallic", "Silver", "Metallic"), ("Granite Crystal Metallic Clearcoat", "Gray", "Metallic"),
    ("Grey", "Gray", "Solid"), ("Rosso Competizione Tri-Coat", "Red", "Pearl"), ("", "Unknown", "Solid"),
])
def test_colour_parsing(text, base, fin):
    assert parsing.base_colour(text) == base
    assert parsing.finish(text) == fin


def test_small_parsers():
    assert parsing.mpg("19-26") == (19, 26)
    assert all(np.isnan(v) for v in parsing.mpg("n/a"))
    assert parsing.fuel("", "Electric") == "Electric" and parsing.fuel("Gas/Electric Mild Hybrid") == "Hybrid"
    assert parsing.fuel("Regular unleaded") == "Gasoline" and parsing.fuel("") == "Unknown"
    assert parsing.drivetrain("Four-wheel drive") == "4WD" and parsing.drivetrain("AWD") == "AWD"
    assert parsing.drivetrain("") == "Unknown"


def raw_listings(n=3000, seed=0):
    """Synthetic listings with the real schema; price depends on model, age and mileage."""
    rng = np.random.default_rng(seed)
    models_ = {"Camry": 30000, "Corolla": 24000, "F-150": 48000, "CX-5": 32000, "Civic": 26000}
    names = rng.choice(list(models_), n)
    year = rng.integers(2010, 2027, n)
    miles = rng.integers(100, 150_000, n)
    base = np.array([models_[m] for m in names])
    price = base * 0.88 ** (2026 - year) * (1 - miles / 600_000) * rng.lognormal(0, 0.05, n)
    return pd.DataFrame({
        "vin": [f"VIN{i:06d}" for i in range(n)], "make": np.where(np.isin(names, ["F-150"]), "Ford", "Toyota"),
        "model": names, "trim": rng.choice(["SE", "LE", ""], n), "year": year, "price_usd": price.round(),
        "mileage_miles": miles, "fuel": "Gasoline", "transmission": "8-Speed Automatic",
        "engine": "2.5L I4 16V GDI DOHC", "drivetrain": "Front-wheel Drive", "mpg": "28-39", "body_type": "Sedan",
        "exterior_color": rng.choice(["Super White", "Midnight Black Metallic"], n), "interior_color": "Black",
        "stock_type": "Used", "cpo": 0, "seller_zip": "90210", "is_enriched": 1,
        "last_checked_at": pd.Timestamp("2026-10-17"),
    })


def test_cleaning_rules():
    raw = raw_listings(20)
    raw.loc[0, "stock_type"] = "New"
    raw.loc[1, "is_enriched"] = 0
    raw.loc[2, "price_usd"] = 500
    raw.loc[3, "mileage_miles"] = 0
    raw.loc[4, "year"] = 1970
    dup = raw.iloc[[10]].assign(price_usd=12345, last_checked_at=pd.Timestamp("2026-10-18"))
    clean, steps = data.clean(pd.concat([raw, dup], ignore_index=True))
    assert len(clean) == 15 and clean["vin"].is_unique
    assert clean.loc[clean["vin"] == "VIN000010", "price_usd"].item() == 12345  # the latest copy is kept
    assert steps["raw"] == 21 and steps["after_plausibility_rules"] == 15


def test_features():
    f = data.build_features(raw_listings(5))
    assert set(data.NUMERIC + data.CATEGORICAL + [data.TARGET]) <= set(f.columns)
    assert (f["age"] == 2026 - f["year"]).all()
    assert f["make_model"].iloc[0].startswith(f["make"].iloc[0] + " ")
    assert f["region"].iloc[0] == "zip-9" and f["mpg_city"].iloc[0] == 28


def test_group_median_falls_back_to_broader_groups():
    X = pd.DataFrame({"make": ["A", "A", "B"], "make_model": ["A x", "A x", "B y"], "year": [2020, 2020, 2019]})
    gm = model.GroupMedian().fit(X, [10.0, 12.0, 20.0])
    new = pd.DataFrame({"make": ["A", "A", "C"], "make_model": ["A x", "A z", "C q"], "year": [2020, 2018, 2020]})
    np.testing.assert_allclose(gm.predict(new), [11.0, 11.0, 12.0])


def test_metrics():
    y = np.log([10_000, 20_000, 40_000])
    perfect = model.metrics(y, y)
    assert perfect["r2_log"] == 1 and perfect["mae_dollars"] == 0 and perfect["within_10pct"] == 1
    off = model.metrics(y, y + np.log(1.2))
    assert off["median_abs_pct_error"] == pytest.approx(0.2)


def test_train_save_load_predict(tmp_path):
    clean, _ = data.clean(raw_listings())
    t = data.build_features(clean)
    tr, va, te = t.iloc[:2000], t.iloc[2000:2500], t.iloc[2500:]
    pm = model.PriceModel("cpu", n_estimators=300).fit(tr, tr[data.TARGET], va, va[data.TARGET])
    pred = pm.predict(te)
    assert model.metrics(te[data.TARGET], pred)["r2_log"] > 0.9
    pm.save(tmp_path)
    np.testing.assert_allclose(model.PriceModel.load(tmp_path).predict(te), pred, rtol=1e-6)

    # A newer, low-mileage car must be worth more than an older, high-mileage one.
    one = te.iloc[[0]].copy()
    newer = one.assign(year=2025.0, age=1.0, mileage=5000.0, log_mileage=np.log1p(5000), miles_per_year=5000.0)
    older = one.assign(year=2012.0, age=14.0, mileage=140000.0, log_mileage=np.log1p(140000),
                       miles_per_year=10000.0)
    assert pm.predict(newer)[0] > pm.predict(older)[0]


def test_find_deals_ignores_rare_models_and_cheap_cars():
    listings = pd.DataFrame({"make_model_trim": ["A", "B", "C", "D"], "year": [2020] * 4, "mileage": [1000] * 4,
                             "price": [10_000, 10_000, 3_000, 19_000]})
    predicted = np.log([20_000, 20_000, 9_000, 20_000])
    deals = model.find_deals(listings, predicted, training_listings=[500, 3, 500, 500])
    assert deals["make_model_trim"].tolist() == ["A"]  # B is rare, C is under $5,000, D is only 5% below


def test_command_line_end_to_end(tmp_path, capsys):
    path = tmp_path / "listings.parquet"
    raw_listings().to_parquet(path)
    summary = main(["train", "--data", str(path), "--out", str(tmp_path / "results"), "--model-dir",
                    str(tmp_path / "models"), "--device", "cpu", "--trees", "300"])
    assert summary["results"]["XGBoost (target-encoded models)"]["r2_log"] > 0.9
    price = main(["predict", "--make", "Toyota", "--model", "Camry", "--year", "2022", "--mileage", "20000",
                  "--model-dir", str(tmp_path / "models")])
    assert 10_000 < price < 40_000
    assert "Estimated price" in capsys.readouterr().out
