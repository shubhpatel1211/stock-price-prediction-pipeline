import argparse

from src.predict import run_prediction
from src.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock price predictor")

    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        required=True,
        help="Run training or prediction",
    )

    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol, for example: TSLA, AAPL, MSFT",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ticker = args.ticker.upper().strip()

    if args.mode == "train":
        run_training(ticker=ticker)
    elif args.mode == "predict":
        run_prediction(ticker=ticker)