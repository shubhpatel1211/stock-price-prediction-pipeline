from __future__ import annotations

from pathlib import Path
from typing import Any


def get_ticker_paths(ticker: str, config: dict[str, Any]) -> dict[str, Path]:
    ticker = ticker.upper()

    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    model_dir = Path(config["output"]["model_dir"])
    plot_dir = Path(config["output"]["plot_dir"])

    return {
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "model_dir": model_dir,
        "plot_dir": plot_dir,
        "raw_path": raw_dir / f"{ticker}_stock_data.csv",
        "processed_path": processed_dir / f"{ticker}_features.csv",
        "model_path": model_dir / f"{ticker}_model.pkl",
        "plot_path": plot_dir / f"{ticker.lower()}_prediction.png",
    }


def ensure_directories(paths: dict[str, Path]) -> None:
    for key in ["raw_dir", "processed_dir", "model_dir", "plot_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)