#!/usr/bin/env python3
"""
Three-model trainer for pricing:
- RandomForestRegressor
- HistGradientBoostingRegressor
- GradientBoostingRegressor

✔ Uses synthetic data if --data is omitted (schema matches predict.py)
✔ RandomizedSearch with safe param spaces (no 'auto', no broken fixes)
✔ Robust metrics (RMSE via sqrt, no squared= kw)
✔ Saves bundle compatible with model/predict.py:
   {"version":1,"model":estimator,"feature_order":FEATURES,...}
"""

import argparse, json, os, pickle, time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from scipy.stats import randint, uniform, loguniform

# EXACT feature order used by predict.py
FEATURES: List[str] = [
    "distance_miles", "container_size_encoded", "minute_of_day",
    "day_of_week", "is_weekend", "hour", "month", "quarter"
]
TARGET = "price"


# ---------- data helpers ----------
def ensure_dirs() -> None:
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)


def synth(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    distance = rng.uniform(5, 300, n)
    size = rng.integers(0, 2, n)
    minute = rng.integers(0, 24 * 60, n)
    dow = rng.integers(0, 7, n)
    month = rng.integers(1, 13, n)

    hour = minute // 60
    quarter = (month - 1) // 3 + 1
    is_weekend = (dow >= 5).astype(int)

    price = (
        45
        + 0.85 * distance
        + 35 * size
        + 0.9 * hour
        + 4 * is_weekend
        + 2.5 * quarter
        + 0.15 * np.sqrt(distance) * (1 + 0.2 * size)
        + rng.normal(0, 9.0, n)
    )

    return pd.DataFrame({
        "distance_miles": distance,
        "container_size_encoded": size,
        "minute_of_day": minute,
        "day_of_week": dow,
        "is_weekend": is_weekend,
        "hour": hour,
        "month": month,
        "quarter": quarter,
        "price": price,
    })


def load_or_create_data(data_path: Optional[str], samples: int, seed: int) -> pd.DataFrame:
    if data_path:
        df = pd.read_csv(data_path)
        missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
        if missing:
            raise ValueError(f"Data file is missing columns: {missing}")
        return df
    df = synth(samples, seed)
    Path("data/synthetic_pricing.csv").write_text(df.to_csv(index=False))
    return df


# ---------- modeling ----------
def build_preprocessor() -> ColumnTransformer:
    to_scale = ["distance_miles", "minute_of_day", "hour", "month"]
    passthrough = [c for c in FEATURES if c not in to_scale]
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), to_scale),
            ("keep", "passthrough", passthrough),
        ],
        verbose_feature_names_out=False,
        remainder="drop",
    )


def candidate_spaces(seed: int) -> Dict[str, Tuple[Pipeline, Dict[str, Any]]]:
    pre = build_preprocessor()

    rf = Pipeline([
        ("pre", pre),
        ("est", RandomForestRegressor(random_state=seed, n_jobs=-1))
    ])
    rf_space = {
        "est__n_estimators": randint(300, 901),          # 300..900
        "est__max_depth": [None] + list(range(6, 31, 2)),
        "est__min_samples_split": randint(2, 11),
        "est__min_samples_leaf": randint(1, 6),
        "est__max_features": ["sqrt", "log2", 0.6, 0.8], # no 'auto'
        "est__bootstrap": [True, False],
    }

    hgb = Pipeline([
        ("pre", pre),
        ("est", HistGradientBoostingRegressor(
            random_state=seed, early_stopping=True, validation_fraction=0.1))
    ])
    hgb_space = {
        "est__max_depth": randint(3, 17),                # 3..16
        "est__learning_rate": loguniform(1e-3, 3e-1),
        "est__max_iter": randint(200, 801),              # 200..800
        "est__l2_regularization": loguniform(1e-8, 1e-2),
        "est__min_samples_leaf": randint(10, 61),        # 10..60
    }

    gbr = Pipeline([
        ("pre", pre),
        ("est", GradientBoostingRegressor(random_state=seed))
    ])
    gbr_space = {
        "est__n_estimators": randint(200, 901),          # 200..900
        "est__max_depth": randint(2, 7),                 # 2..6
        "est__learning_rate": loguniform(5e-3, 5e-1),
        "est__subsample": uniform(0.6, 0.4),             # 0.6..1.0
        "est__min_samples_leaf": randint(1, 21),         # 1..20
    }

    return {
        "RandomForest": (rf, rf_space),
        "HistGradientBoosting": (hgb, hgb_space),
        "GradientBoosting": (gbr, gbr_space),
    }


def fit_best_model(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    cv_folds: int,
    n_iter: int,
    n_jobs: int,
    scoring: str,
) -> Tuple[str, Any, Dict[str, Any]]:
    spaces = candidate_spaces(seed)
    results: List[Dict[str, Any]] = []

    for name, (pipe, space) in spaces.items():
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=space,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            random_state=seed,
            verbose=0,
            error_score=np.nan,  # don't crash on a bad combo
        )
        search.fit(X, y)
        results.append({
            "name": name,
            "best_score": float(search.best_score_),
            "best_params": search.best_params_,
            "best_estimator": search.best_estimator_,
        })

    best = max(results, key=lambda r: r["best_score"])
    summary = {
        "candidates": [
            {"name": r["name"], "best_cv_score": r["best_score"], "best_params": r["best_params"]}
            for r in results
        ],
        "winner": {"name": best["name"], "best_cv_score": best["best_score"], "best_params": best["best_params"]},
        "scoring": scoring,
        "cv_folds": cv_folds,
        "n_iter": n_iter,
    }
    return best["name"], best["best_estimator"], summary


def evaluate(model, Xte: pd.DataFrame, yte: pd.Series) -> Dict[str, float]:
    preds = model.predict(Xte)
    r2 = float(r2_score(yte, preds))
    mae = float(mean_absolute_error(yte, preds))
    rmse = float(np.sqrt(mean_squared_error(yte, preds)))  # no squared kw
    return {"r2": r2, "mae": mae, "rmse": rmse}


# ---------- cli ----------
def main():
    ap = argparse.ArgumentParser(description="Train 3 models and pick the best.")
    ap.add_argument("--data", type=str, default=None, help="CSV with FEATURES + price; if omitted, synthetic is generated.")
    ap.add_argument("--samples", type=int, default=15000, help="Rows for synthetic data (when --data omitted).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", type=int, default=3, help="CV folds")
    ap.add_argument("--n-iter", type=int, default=20, help="RandomizedSearch iterations per model")
    ap.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers for CV")
    ap.add_argument("--scoring", type=str, default="r2")
    args = ap.parse_args()

    ensure_dirs()
    df = load_or_create_data(args.data, args.samples, args.seed)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    winner_name, best_est, cv_summary = fit_best_model(
        Xtr, ytr, seed=args.seed, cv_folds=args.cv, n_iter=args.n_iter,
        n_jobs=args.n_jobs, scoring=args.scoring
    )
    metrics = evaluate(best_est, Xte, yte)

    ts = time.strftime("%Y%m%d_%H%M%S")
    model_path = Path("models") / f"pricing_best_{ts}.pkl"
    card_path = Path("models") / f"pricing_best_{ts}.md"

    bundle = {
        "version": 1,
        "created_at": ts,
        "framework": "scikit-learn",
        "model_name": winner_name,
        "model": best_est,
        "feature_order": FEATURES,
        "target": TARGET,
        "metrics": metrics,
        "cv_summary": cv_summary,
        "train_shape": [int(Xtr.shape[0]), int(Xtr.shape[1])],
        "test_shape": [int(Xte.shape[0]), int(Xte.shape[1])],
        "random_state": args.seed,
    }
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    card = f"""# Pricing Model — {winner_name}
Created: {ts}

## Data
- Train rows: {Xtr.shape[0]}
- Test rows: {Xte.shape[0]}
- Features: {json.dumps(FEATURES)}

## CV
- Scoring: {cv_summary['scoring']}
- Folds: {cv_summary['cv_folds']}
- Iters/model: {cv_summary['n_iter']}

### Candidates
{os.linesep.join([f"- {c['name']}: CV {c['best_cv_score']:.4f}" for c in cv_summary['candidates']])}

**Winner:** {cv_summary['winner']['name']} (CV={cv_summary['winner']['best_cv_score']:.4f})

## Test Metrics
- R²: {metrics['r2']:.4f}
- MAE: {metrics['mae']:.3f}
- RMSE: {metrics['rmse']:.3f}

Saved bundle: `{model_path.name}`
"""
    card_path.write_text(card, encoding="utf-8")

    print(f"✅ Saved model -> {model_path}")
    print(f"📊 Test: R2={metrics['r2']:.4f}  MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}")
    print(f"📝 Model card -> {card_path}")


if __name__ == "__main__":
    main()
