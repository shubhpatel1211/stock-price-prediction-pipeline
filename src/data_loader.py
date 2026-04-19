from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    data.reset_index(inplace=True)
    return data


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def load_dataframe(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(csv_path)