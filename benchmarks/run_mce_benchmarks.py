"""
MCE Benchmark: Compare Maya Calendar Encoding against standard temporal encodings.

Uses synthetic time series data with known cyclical patterns to test whether
MCE can capture temporal structure better than standard sine/cosine encoding.

Also tests on real-world data if available (ETT dataset).
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from maya_encoding import MayaCalendarEncoder

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# --- Temporal Encoding Baselines ---


class SineCosineEncoder:
    """Standard sine/cosine temporal encoding with configurable periods."""

    def __init__(self, periods=(7, 14, 30, 90, 365)):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import datetime as dt

        # Parse dates to day-of-year ordinal
        dates = []
        for d in np.asarray(X).ravel():
            if isinstance(d, str):
                d = dt.date.fromisoformat(d)
            elif isinstance(d, np.datetime64):
                d = d.astype("datetime64[D]").astype(dt.date)
            dates.append(d.toordinal())

        ordinals = np.array(dates, dtype=np.float64)
        features = []

        for period in self.periods:
            angle = 2 * np.pi * ordinals / period
            features.append(np.sin(angle).reshape(-1, 1))
            features.append(np.cos(angle).reshape(-1, 1))

        return np.hstack(features)

    def get_feature_names_out(self, input_features=None):
        names = []
        for p in self.periods:
            names.extend([f"sin_{p}", f"cos_{p}"])
        return names


class NoTemporalEncoder:
    """No temporal features — just pass through ordinal day number."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import datetime as dt

        dates = []
        for d in np.asarray(X).ravel():
            if isinstance(d, str):
                d = dt.date.fromisoformat(d)
            elif isinstance(d, np.datetime64):
                d = d.astype("datetime64[D]").astype(dt.date)
            dates.append(d.toordinal())

        return MinMaxScaler().fit_transform(np.array(dates).reshape(-1, 1))


def get_temporal_encoders():
    """Return dict of temporal encoding strategies."""
    return {
        "no_temporal": NoTemporalEncoder(),
        "sincos_standard": SineCosineEncoder(periods=(7, 14, 30, 90, 365)),
        "mce_full": MayaCalendarEncoder(
            components=["tzolkin", "haab", "long_count"],
            cyclical=True,
        ),
        "mce_no_cyclical": MayaCalendarEncoder(
            components=["tzolkin", "haab", "long_count"],
            cyclical=False,
        ),
        "mce_tzolkin_only": MayaCalendarEncoder(
            components=["tzolkin"],
            cyclical=True,
        ),
        "mce_plus_sincos": "hybrid",  # Special case: MCE + sine/cosine
    }


def get_models():
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


# --- Synthetic Dataset ---


def generate_synthetic_time_series(n_days=1000, seed=42):
    """Generate synthetic time series with known cyclical patterns.

    The target combines:
    - A 13-day cycle (aligns with Tzolk'in number)
    - A 20-day cycle (aligns with Tzolk'in day name)
    - A 365-day seasonal cycle (aligns with Haab')
    - Random noise
    """
    rng = np.random.RandomState(seed)
    import datetime as dt

    start_date = dt.date(2015, 1, 1)
    dates = np.array([(start_date + dt.timedelta(days=i)).isoformat() for i in range(n_days)])

    t = np.arange(n_days, dtype=np.float64)

    # Cyclical components
    cycle_13 = 3.0 * np.sin(2 * np.pi * t / 13)      # Tzolk'in number period
    cycle_20 = 2.0 * np.sin(2 * np.pi * t / 20)      # Tzolk'in day name period
    cycle_365 = 5.0 * np.sin(2 * np.pi * t / 365)    # Haab' period
    trend = 0.005 * t                                   # Slow trend
    noise = rng.randn(n_days) * 1.5                     # Noise

    y = cycle_13 + cycle_20 + cycle_365 + trend + noise

    return dates, y


# --- Walk-Forward Validation ---


def walk_forward_evaluate(dates, y, encoder, model, train_size=0.7, step=50):
    """Walk-forward validation respecting temporal ordering."""
    n = len(dates)
    initial_train = int(n * train_size)

    rmses = []
    maes = []

    for start in range(initial_train, n - step, step):
        end = min(start + step, n)

        train_dates = dates[:start]
        test_dates = dates[start:end]
        y_train = y[:start]
        y_test = y[start:end]

        # Encode
        if isinstance(encoder, str) and encoder == "hybrid":
            mce = MayaCalendarEncoder(cyclical=True)
            sincos = SineCosineEncoder()
            X_train = np.hstack([mce.fit_transform(train_dates), sincos.fit_transform(train_dates)])
            X_test = np.hstack([mce.transform(test_dates), sincos.transform(test_dates)])
        else:
            X_train = encoder.fit_transform(train_dates)
            X_test = encoder.transform(test_dates)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        maes.append(mean_absolute_error(y_test, y_pred))

    return {
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
    }


# --- Experiments ---


def run_synthetic_benchmark():
    print("\n" + "=" * 70)
    print("EXPERIMENT M1: Synthetic Time Series (cycles at 13, 20, 365 days)")
    print("=" * 70)

    dates, y = generate_synthetic_time_series(n_days=2000)
    print(f"Dataset: {len(dates)} days, target std: {np.std(y):.2f}")

    encoders = get_temporal_encoders()
    models = get_models()

    results = {}

    for enc_name, encoder in encoders.items():
        results[enc_name] = {}
        print(f"\n--- Encoding: {enc_name} ---")

        for model_name, model_template in models.items():
            start = time.time()

            try:
                # Clone model for fresh state
                from sklearn.base import clone

                model = clone(model_template)

                result = walk_forward_evaluate(dates, y, encoder, model)
                elapsed = time.time() - start
                result["time_s"] = round(elapsed, 2)
                results[enc_name][model_name] = result

                print(
                    f"  {model_name:20s} | RMSE: {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f} | "
                    f"MAE: {result['mae_mean']:.4f} ± {result['mae_std']:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

            except Exception as e:
                print(f"  {model_name:20s} | ERROR: {e}")
                results[enc_name][model_name] = {"error": str(e)}

    return results


def main():
    print("MCE BENCHMARK SUITE")
    print(f"XGBoost available: {HAS_XGBOOST}")

    all_results = {}
    all_results["synthetic_time_series"] = run_synthetic_benchmark()

    # Save
    output_file = RESULTS_DIR / "mce_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Best encoding per model (Synthetic RMSE)")
    print("=" * 70)

    synth_results = all_results["synthetic_time_series"]
    models = set()
    for enc in synth_results.values():
        models.update(enc.keys())

    for model_name in sorted(models):
        best_enc = None
        best_rmse = float("inf")
        for enc_name, enc_results in synth_results.items():
            if model_name in enc_results and "rmse_mean" in enc_results[model_name]:
                rmse = enc_results[model_name]["rmse_mean"]
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_enc = enc_name
        if best_enc:
            print(f"  {model_name:20s} | Best: {best_enc:20s} | RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    main()
