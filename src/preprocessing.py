from __future__ import annotations

from typing import Any

import pandas as pd


def create_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data = df.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            "_".join([str(part) for part in col if part not in (None, "")]).strip("_")
            for col in data.columns.to_flat_index()
        ]
    else:
        data.columns = [str(col) for col in data.columns]

    if "Date" not in data.columns:
        data = data.reset_index()

    for candidate in ["Date", "date", "Datetime", "index"]:
        if candidate in data.columns:
            data = data.rename(columns={candidate: "Date"})
            break

    required_price_cols = ["Open", "High", "Low", "Close", "Volume"]

    for required_col in required_price_cols:
        if required_col not in data.columns:
            for candidate in data.columns:
                if candidate == required_col or candidate.startswith(f"{required_col}_") or required_col in candidate:
                    data[required_col] = data[candidate]
                    break

    if "Date" not in data.columns:
        raise ValueError(
            f"Input data must include a 'Date' column. Found columns: {list(data.columns)}"
        )

    if "Close" not in data.columns:
        raise ValueError(
            f"Input data must include a 'Close' column. Found columns: {list(data.columns)}"
        )

    lags = config["features"]["lags"]
    rolling_windows = config["features"]["rolling_windows"]

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data["daily_return"] = data["Close"].pct_change()

    for lag in lags:
        data[f"close_lag_{lag}"] = data["Close"].shift(lag)
        data[f"return_lag_{lag}"] = data["daily_return"].shift(lag)

    for window in rolling_windows:
        rolling_mean = data["Close"].rolling(window).mean()
        rolling_std = data["Close"].rolling(window).std()

        data[f"rolling_mean_{window}"] = rolling_mean
        data[f"rolling_std_{window}"] = rolling_std
        data[f"price_vs_rolling_mean_{window}"] = data["Close"] / rolling_mean - 1.0
        data[f"momentum_{window}"] = data["Close"] / data["Close"].shift(window) - 1.0

    if {"High", "Low"}.issubset(data.columns):
        data["high_low_range"] = (data["High"] - data["Low"]) / data["Close"]

    if {"Open", "Close"}.issubset(data.columns):
        data["open_close_diff"] = (data["Close"] - data["Open"]) / data["Open"]

    if "Volume" in data.columns:
        data["volume_change"] = data["Volume"].pct_change()

    # Predict next-day return instead of raw next-day price
    data["target"] = data["Close"].shift(-1) / data["Close"] - 1.0

    # Keep actual next close for evaluation/reporting
    data["next_close"] = data["Close"].shift(-1)

    data = data.dropna().reset_index(drop=True)
    return data


def split_features_target(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    excluded_columns = {"Date", target_column, "next_close"}
    feature_columns = [col for col in df.columns if col not in excluded_columns]
    X = df[feature_columns]
    y = df[target_column]
    return X, y