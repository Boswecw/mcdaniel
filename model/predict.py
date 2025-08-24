#!/usr/bin/env python3
import argparse
import glob
import os
import pickle
from typing import Optional, Any, Tuple, List

import pandas as pd

# Default feature order if the bundle doesn't include one
FEATURES: List[str] = [
    "distance_miles", "container_size_encoded", "minute_of_day",
    "day_of_week", "is_weekend", "hour", "month", "quarter"
]

def make_row(distance: float, size: int, minute: int, dow: int, month: int) -> pd.DataFrame:
    hour = minute // 60
    quarter = (month - 1) // 3 + 1
    is_weekend = 1 if dow >= 5 else 0
    return pd.DataFrame([{
        "distance_miles": distance,
        "container_size_encoded": size,
        "minute_of_day": minute,
        "day_of_week": dow,
        "is_weekend": is_weekend,
        "hour": hour,
        "month": month,
        "quarter": quarter
    }])

def find_latest_model() -> Optional[str]:
    """
    Search common locations for the newest model file.
    Supports both 'pricing_best_*.pkl' (new) and 'pricing_rf_*.pkl' (old).
    """
    search_patterns = [
        os.path.join("models", "pricing_best_*.pkl"),
        os.path.join("models", "pricing_rf_*.pkl"),
        os.path.join("model", "models", "pricing_best_*.pkl"),
        os.path.join("model", "models", "pricing_rf_*.pkl"),
        os.path.join("models", "*.pkl"),
        os.path.join("model", "models", "*.pkl"),
    ]
    candidates: List[str] = []
    for pat in search_patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        return None
    return max(candidates, key=lambda p: os.path.getmtime(p))

def load_pickle(path: str) -> Any:
    # Try pickle first
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        # Fallback: some files were saved via joblib (still pickle under the hood),
        # but try importing joblib in case of issues.
        try:
            import joblib  # type: ignore
            return joblib.load(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model file '{path}': {e}")

def unwrap_model(obj: Any) -> Tuple[Any, List[str]]:
    """
    Support both bundle dicts (with 'model' and optional 'feature_order')
    and raw estimators/pipelines.
    """
    if isinstance(obj, dict) and "model" in obj:
        est = obj["model"]
        order = obj.get("feature_order", FEATURES)
        # defensive: ensure list of strings
        order = [str(c) for c in order]
        return est, order
    return obj, FEATURES

def main():
    ap = argparse.ArgumentParser(description="Predict price using a trained pricing model.")
    ap.add_argument("--model", help="Path to *.pkl (if omitted, auto-select newest)")
    ap.add_argument("--distance", type=float, required=True)
    ap.add_argument("--size", type=int, choices=[0, 1], required=True, help="0=20ft, 1=40ft")
    ap.add_argument("--minute", type=int, default=9*60+30, help="0..1439 (default 570 = 9:30 AM)")
    ap.add_argument("--dow", type=int, default=2, help="0=Mon .. 6=Sun (default 2=Wed)")
    ap.add_argument("--month", type=int, default=11, help="1..12 (default 11=Nov)")
    ap.add_argument("--verbose", action="store_true", help="Print model path & feature order")
    args = ap.parse_args()

    model_path = args.model or find_latest_model()
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            "Could not find a trained model file. "
            "Pass --model /path/to/pricing_best_*.pkl or run training first."
        )

    loaded = load_pickle(model_path)
    model, feature_order = unwrap_model(loaded)

    row = make_row(args.distance, args.size, args.minute, args.dow, args.month)

    # Ensure we only keep columns the model expects, in the right order.
    # If any are missing, raise a clear error.
    missing = [c for c in feature_order if c not in row.columns]
    if missing:
        raise ValueError(f"Input row is missing expected features: {missing}")

    row = row[feature_order]

    # Predict
    pred = model.predict(row)
    try:
        price = float(pred[0])
    except Exception:
        # Some sklearn regressors return shape (1,) numpy arrays of dtype object
        price = float(pd.Series(pred).iloc[0])

    if args.verbose:
        print(f"MODEL: {model_path}")
        print(f"FEATURE ORDER: {feature_order}")

    print(f"${price:,.2f}")

if __name__ == "__main__":
    main()
