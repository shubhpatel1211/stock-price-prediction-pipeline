from pathlib import Path

import pandas as pd

from src.model import build_model
from src.preprocessing import create_features, split_features_target
from src.utils import get_ticker_paths


def test_build_model_random_forest():
    config = {
        "model": {
            "type": "random_forest",
            "params": {
                "n_estimators": 10,
                "max_depth": 4,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1,
            },
        }
    }

    model = build_model(config)
    assert model is not None


def test_feature_creation_and_split():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "Open": list(range(100, 160)),
            "High": list(range(101, 161)),
            "Low": list(range(99, 159)),
            "Close": list(range(100, 160)),
            "Volume": [1000 + i * 10 for i in range(60)],
        }
    )

    config = {
        "features": {
            "lags": [1, 2, 3],
            "rolling_windows": [3, 5],
            "target_column": "target",
        }
    }

    features_df = create_features(df, config)
    X, y = split_features_target(features_df, "target")

    assert not features_df.empty
    assert "Close" in X.columns
    assert "close_lag_1" in X.columns
    assert "return_lag_1" in X.columns
    assert "rolling_mean_3" in X.columns
    assert "momentum_3" in X.columns
    assert "next_close" not in X.columns
    assert len(X) == len(y)


def test_get_ticker_paths():
    config = {
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
        },
        "output": {
            "model_dir": "models",
            "plot_dir": "assets/generated",
        },
    }

    paths = get_ticker_paths("aapl", config)

    assert paths["raw_path"] == Path("data/raw/AAPL_stock_data.csv")
    assert paths["processed_path"] == Path("data/processed/AAPL_features.csv")
    assert paths["model_path"] == Path("models/AAPL_model.pkl")
    assert paths["plot_path"] == Path("assets/generated/aapl_prediction.png")