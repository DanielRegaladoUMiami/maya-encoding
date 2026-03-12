"""VFD Benchmark: Compare Vigesimal Feature Decomposition against baseline encodings.

Experiments:
- V1: California Housing (regression)
- V2: Digits dataset (classification, sklearn built-in)

Encodings: Decimal normalized, Binary, VFD-lite, VFD-full
Models: LinearRegression/LogisticRegression, RandomForest, XGBoost, MLPRegressor/Classifier
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing, load_digits
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

warnings.filterwarnings("ignore")

# Optional: XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from maya_encoding import VFDEncoder  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# --- Encoding Strategies ---


def binary_encode(X: np.ndarray) -> np.ndarray:
    """Encode each value as binary features (16 bits)."""
    X_int = np.round(np.abs(X)).astype(np.int64)
    n_bits = 16
    result = np.zeros((X_int.shape[0], X_int.shape[1] * n_bits), dtype=np.float64)
    for col in range(X_int.shape[1]):
        for bit in range(n_bits):
            result[:, col * n_bits + bit] = (X_int[:, col] >> bit) & 1
    return result


def get_encoders():
    """Return dict of encoding name -> sklearn transformer."""
    encoders = {
        "decimal_norm": Pipeline([("scale", MinMaxScaler())]),
        "binary": Pipeline([
            ("scale_int", FunctionTransformer(
                lambda X: np.round(MinMaxScaler().fit_transform(X) * 1000).astype(float)
            )),
            ("binary", FunctionTransformer(binary_encode)),
        ]),
        "vfd_lite": VFDEncoder(components="lite"),
        "vfd_full": VFDEncoder(components="full"),
    }
    return encoders


def get_regression_models():
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "MLP": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=RANDOM_STATE),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=100, random_state=RANDOM_STATE, verbosity=0
        )
    return models


def get_classification_models():
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=RANDOM_STATE),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, random_state=RANDOM_STATE, verbosity=0, use_label_encoder=False
        )
    return models


# --- Experiment V1: California Housing ---


def run_california_housing():
    print("\n" + "=" * 70)
    print("EXPERIMENT V1: California Housing (Regression)")
    print("=" * 70)

    data = fetch_california_housing()
    X, y = data.data, data.target

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

    encoders = get_encoders()
    models = get_regression_models()

    results = {}
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for enc_name, encoder in encoders.items():
        results[enc_name] = {}
        print(f"\n--- Encoding: {enc_name} ---")

        for model_name, model in models.items():
            start = time.time()

            try:
                pipe = Pipeline([("encode", encoder), ("model", model)])

                rmses = []
                maes = []

                for train_idx, test_idx in kf.split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_test)

                    rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
                    maes.append(mean_absolute_error(y_test, y_pred))

                elapsed = time.time() - start
                result = {
                    "rmse_mean": float(np.mean(rmses)),
                    "rmse_std": float(np.std(rmses)),
                    "mae_mean": float(np.mean(maes)),
                    "mae_std": float(np.std(maes)),
                    "time_s": round(elapsed, 2),
                }
                results[enc_name][model_name] = result

                print(
                    f"  {model_name:20s} | RMSE: {result['rmse_mean']:.4f} "
                    f"± {result['rmse_std']:.4f} | MAE: {result['mae_mean']:.4f} "
                    f"± {result['mae_std']:.4f} | Time: {elapsed:.1f}s"
                )

            except Exception as e:
                print(f"  {model_name:20s} | ERROR: {e}")
                results[enc_name][model_name] = {"error": str(e)}

    return results


# --- Experiment V2: Digits Classification ---


def run_digits_classification():
    print("\n" + "=" * 70)
    print("EXPERIMENT V2: Digits (Classification)")
    print("=" * 70)

    data = load_digits()
    X, y = data.data, data.target

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

    encoders = get_encoders()
    models = get_classification_models()

    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for enc_name, encoder in encoders.items():
        results[enc_name] = {}
        print(f"\n--- Encoding: {enc_name} ---")

        for model_name, model in models.items():
            start = time.time()

            try:
                pipe = Pipeline([("encode", encoder), ("model", model)])

                f1s = []

                for train_idx, test_idx in skf.split(X, y):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_test)

                    f1s.append(f1_score(y_test, y_pred, average="weighted"))

                elapsed = time.time() - start
                result = {
                    "f1_mean": float(np.mean(f1s)),
                    "f1_std": float(np.std(f1s)),
                    "time_s": round(elapsed, 2),
                }
                results[enc_name][model_name] = result

                print(
                    f"  {model_name:20s} | F1: {result['f1_mean']:.4f} ± {result['f1_std']:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

            except Exception as e:
                print(f"  {model_name:20s} | ERROR: {e}")
                results[enc_name][model_name] = {"error": str(e)}

    return results


# --- Main ---


def main():
    print("VFD BENCHMARK SUITE")
    print(f"XGBoost available: {HAS_XGBOOST}")

    all_results = {}

    all_results["california_housing"] = run_california_housing()
    all_results["digits_classification"] = run_digits_classification()

    # Save results
    output_file = RESULTS_DIR / "vfd_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Best encoding per model (California Housing RMSE)")
    print("=" * 70)

    cal_results = all_results["california_housing"]
    models = set()
    for enc in cal_results.values():
        models.update(enc.keys())

    for model_name in sorted(models):
        best_enc = None
        best_rmse = float("inf")
        for enc_name, enc_results in cal_results.items():
            if model_name in enc_results and "rmse_mean" in enc_results[model_name]:
                rmse = enc_results[model_name]["rmse_mean"]
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_enc = enc_name
        if best_enc:
            print(f"  {model_name:20s} | Best: {best_enc:15s} | RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    main()
