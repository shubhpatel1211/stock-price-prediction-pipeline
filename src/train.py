from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data_loader import download_stock_data, load_config, save_dataframe
from src.model import build_model
from src.preprocessing import create_features, split_features_target
from src.utils import ensure_directories, get_ticker_paths


def evaluate_model(y_true: pd.Series, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(mse ** 0.5)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_plot(y_true: pd.Series, y_pred, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(y_true.reset_index(drop=True), label="Actual")
    plt.plot(pd.Series(y_pred).reset_index(drop=True), label="Predicted")
    plt.title("Actual vs Predicted Next Closing Price")
    plt.xlabel("Samples")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_training(ticker: str, config_path: str = "config/config.yaml") -> None:
    config = load_config(config_path)
    ticker = ticker.upper()

    paths = get_ticker_paths(ticker, config)
    ensure_directories(paths)

    raw_df = download_stock_data(
        ticker=ticker,
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    save_dataframe(raw_df, str(paths["raw_path"]))

    feature_df = create_features(raw_df, config)
    save_dataframe(feature_df, str(paths["processed_path"]))

    X, y = split_features_target(
        feature_df,
        target_column=config["features"]["target_column"],
    )

    next_close = feature_df["next_close"]

    X_train, X_test, y_train, y_test, next_close_train, next_close_test = train_test_split(
        X,
        y,
        next_close,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        shuffle=False,
    )

    model = build_model(config)
    model.fit(X_train, y_train)

    predicted_returns = model.predict(X_test)

    # Convert predicted returns back into next-day price predictions
    current_close_test = X_test["Close"]
    predicted_next_close = current_close_test * (1.0 + predicted_returns)
    actual_next_close = next_close_test

    # Naive baseline: tomorrow's close = today's close
    baseline_next_close = current_close_test

    metrics = evaluate_model(actual_next_close, predicted_next_close)
    baseline_metrics = evaluate_model(actual_next_close, baseline_next_close)

    joblib.dump(model, paths["model_path"])
    save_plot(actual_next_close, predicted_next_close, paths["plot_path"])

    print(f"Training complete for {ticker}.")
    print(f"Raw data saved to: {paths['raw_path']}")
    print(f"Processed features saved to: {paths['processed_path']}")
    print(f"Model saved to: {paths['model_path']}")
    print(f"Plot saved to: {paths['plot_path']}")

    print("Model metrics (price space):")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    print("Baseline metrics (predict next close = current close):")
    for metric_name, value in baseline_metrics.items():
        print(f"  {metric_name}: {value:.4f}")