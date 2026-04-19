from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def build_model(config: dict[str, Any]):
    model_type = config["model"]["type"]
    params = config["model"].get("params", {})

    if model_type == "random_forest":
        return RandomForestRegressor(**params)

    if model_type == "linear_regression":
        return LinearRegression(**params)

    raise ValueError(
        f"Unsupported model type: {model_type}. "
        "Use 'random_forest' or 'linear_regression'."
    )
