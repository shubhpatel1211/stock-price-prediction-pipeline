from __future__ import annotations

import joblib
import pandas as pd

from src.data_loader import download_stock_data, load_config, save_dataframe
from src.preprocessing import create_features, split_features_target
from src.utils import ensure_directories, get_ticker_paths


def run_prediction(ticker: str, config_path: str = "config/config.yaml") -> None:
    config = load_config(config_path)
    ticker = ticker.upper()

    paths = get_ticker_paths(ticker, config)
    ensure_directories(paths)

    if not paths["model_path"].exists():
        raise FileNotFoundError(
            f"No trained model found for {ticker}. Expected at: {paths['model_path']}"
        )

    model = joblib.load(paths["model_path"])

    latest_df = download_stock_data(
        ticker=ticker,
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    save_dataframe(latest_df, str(paths["raw_path"]))

    features_df = create_features(latest_df, config)
    save_dataframe(features_df, str(paths["processed_path"]))

    X, y = split_features_target(
        features_df,
        target_column=config["features"]["target_column"],
    )

    latest_features = X.tail(1)
    latest_close = float(latest_features["Close"].iloc[0])

    predicted_return = float(model.predict(latest_features)[0])
    predicted_next_close = latest_close * (1.0 + predicted_return)

    latest_date = pd.to_datetime(features_df["Date"].iloc[-1]).date()

    print(f"Latest feature date: {latest_date}")
    print(f"Using model: {paths['model_path']}")
    print(f"Updated raw data file: {paths['raw_path']}")
    print(f"Updated processed features file: {paths['processed_path']}")
    print(f"Latest known close for {ticker}: {latest_close:.2f}")
    print(f"Predicted next-day return for {ticker}: {predicted_return:.4%}")
    print(f"Predicted next closing price for {ticker}: {predicted_next_close:.2f}")
    print(f"Most recent known next close in dataset: {features_df['next_close'].iloc[-1]:.2f}")