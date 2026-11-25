"""Baselines, the XGBoost model and evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC, TARGET


def metrics(y_log_true, y_log_pred) -> dict:
    """Errors on the log scale (what the model optimises) and in dollars (what a buyer cares about)."""
    y_log_true, y_log_pred = np.asarray(y_log_true), np.asarray(y_log_pred)
    price, pred = np.exp(y_log_true), np.exp(y_log_pred)
    ape = np.abs(pred - price) / price

    def r2(a, b):
        return 1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)

    return {
        "rmse_log": float(np.sqrt(np.mean((y_log_true - y_log_pred) ** 2))),
        "r2_log": float(r2(y_log_true, y_log_pred)),
        "r2_dollars": float(r2(price, pred)),
        "mae_dollars": float(np.mean(np.abs(pred - price))),
        "median_abs_pct_error": float(np.median(ape)),
        "within_10pct": float(np.mean(ape <= 0.10)),
    }


class GroupMedian:
    """Baseline: median log price of the same make and model and year, falling back to make+model, then make."""

    levels = (["make_model", "year"], ["make_model"], ["make"])

    def fit(self, X: pd.DataFrame, y) -> "GroupMedian":
        d = X[["make", "make_model", "year"]].astype({"make": str, "make_model": str}).assign(y=np.asarray(y))
        self.tables = [d.groupby(cols)["y"].median() for cols in self.levels]
        self.global_ = float(np.median(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        d = X[["make", "make_model", "year"]].astype({"make": str, "make_model": str})
        pred = pd.Series(np.nan, index=d.index)
        for cols, table in zip(self.levels, self.tables):
            keys = pd.MultiIndex.from_frame(d[cols]) if len(cols) > 1 else d[cols[0]]
            looked_up = pd.Series(table.reindex(keys).to_numpy(), index=d.index)
            pred = pred.fillna(looked_up)
        return pred.fillna(self.global_).to_numpy()


def ridge() -> object:
    """Linear baseline on one-hot categories (make, model, trim, ...) plus scaled numeric features."""
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CATEGORICAL),
    ])
    return make_pipeline(prep, Ridge(alpha=1.0))


def xgboost(device: str = "cpu", n_estimators: int = 5000, seed: int = 0):
    from xgboost import XGBRegressor

    return XGBRegressor(n_estimators=n_estimators, learning_rate=0.05, max_depth=8, min_child_weight=3,
                        subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, max_cat_to_onehot=16,
                        enable_categorical=True, tree_method="hist", device=device, early_stopping_rounds=200,
                        eval_metric="rmse", random_state=seed)


HIGH_CARDINALITY = ["make", "make_model", "make_model_trim"]
LOW_CARDINALITY = [c for c in CATEGORICAL if c not in HIGH_CARDINALITY]


class PriceModel:
    """XGBoost on numeric features, out-of-fold target encoding for make / model / trim (thousands of values),
    and native categories for the low-cardinality columns.

    Target encoding replaces each model name by its average log price. Fitting it with 5-fold cross-fitting
    means no row is encoded with its own price, which would leak the answer. How many training listings each
    make, model and trim has is also a feature, so the model knows when an average is based on few cars.
    """

    def __init__(self, device: str = "cpu", n_estimators: int = 5000):
        self.device, self.n_estimators = device, n_estimators

    def _matrix(self, df: pd.DataFrame, fit_target=None) -> pd.DataFrame:
        X = df[NUMERIC].astype(float).copy()
        high = df[HIGH_CARDINALITY].astype(str)
        enc = self.encoder.fit_transform(high, fit_target) if fit_target is not None else self.encoder.transform(high)
        for i, c in enumerate(HIGH_CARDINALITY):
            X["avg_price_" + c] = enc[:, i]
            X["listings_" + c] = high[c].map(self.counts[c]).fillna(0).to_numpy()
        for c in LOW_CARDINALITY:
            X[c] = pd.Categorical(df[c], categories=self.categories[c])
        return X

    def fit(self, X_train, y_train, X_valid, y_valid) -> "PriceModel":
        from sklearn.preprocessing import TargetEncoder

        self.encoder = TargetEncoder(target_type="continuous", cv=5, random_state=0)
        self.counts = {c: X_train[c].astype(str).value_counts() for c in HIGH_CARDINALITY}
        self.categories = {c: sorted(X_train[c].dropna().unique()) for c in LOW_CARDINALITY}
        Xt = self._matrix(X_train, np.asarray(y_train))
        Xv = self._matrix(X_valid)
        self.model = xgboost(self.device, self.n_estimators)
        self.model.fit(Xt, y_train, eval_set=[(Xv, y_valid)], verbose=False)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(self._matrix(X))

    def training_listings(self, X) -> np.ndarray:
        """How many training listings share each car's make and model."""
        return X["make_model"].astype(str).map(self.counts["make_model"]).fillna(0).to_numpy()

    def save(self, folder: str | Path) -> None:
        import joblib

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save_model(folder / "xgboost_price_model.ubj")
        joblib.dump({"encoder": self.encoder, "counts": self.counts, "categories": self.categories},
                    folder / "encoders.joblib")

    @classmethod
    def load(cls, folder: str | Path) -> "PriceModel":
        import joblib
        from xgboost import XGBRegressor

        folder = Path(folder)
        obj = cls()
        parts = joblib.load(folder / "encoders.joblib")  # only load model files you created yourself
        obj.encoder, obj.counts, obj.categories = parts["encoder"], parts["counts"], parts["categories"]
        obj.model = XGBRegressor(enable_categorical=True)
        obj.model.load_model(folder / "xgboost_price_model.ubj")
        return obj


def find_deals(listings: pd.DataFrame, predicted_log: np.ndarray, training_listings, min_discount: float = 0.25,
               min_listings: int = 50, min_price: float = 5_000, top: int = 15) -> pd.DataFrame:
    """Listings priced well below what the model expects for the same car: either bargains or listings with a
    problem the data does not show (accident history, salvage title), so they are worth a closer look."""
    d = listings.copy()
    d["predicted_price"] = np.exp(predicted_log).round(-1)
    d["discount"] = 1 - d["price"] / d["predicted_price"]
    # Only judge cars the model has seen plenty of: rare models (a 1999 Plymouth Prowler) get unreliable estimates.
    d = d[(np.asarray(training_listings) >= min_listings) & (d["price"] >= min_price)]
    cols = ["make_model_trim", "year", "mileage", "price", "predicted_price", "discount"]
    return d.loc[d["discount"] >= min_discount, cols].sort_values("discount", ascending=False).head(top)


__all__ = ["metrics", "GroupMedian", "ridge", "PriceModel", "find_deals", "TARGET"]
