# 📈 Stock Price Prediction Pipeline

A modular **machine learning pipeline** for forecasting the **next-day stock return** and converting it into a **next-day closing price estimate** using engineered time-series features and regression models.

The project is designed as a **reproducible CLI-based workflow** with configurable experiments, automated feature engineering, model training, evaluation, artifact generation, and unit testing.

---

# 🚀 Features

* Download historical stock data using `yfinance`
* Automated time-series feature engineering
* Predict next-day stock **returns**
* Convert predicted returns into next-day closing price estimates
* Config-driven experimentation via YAML
* CLI-based training and prediction workflow
* Automatic dataset, model, and plot generation
* Baseline comparison against naive predictor
* Modular project architecture
* Unit testing with `pytest`

---

# 🧠 Tech Stack

* Python
* pandas
* scikit-learn
* yfinance
* matplotlib
* PyYAML
* joblib
* pytest

---

# 📂 Project Structure

```
stock-price-prediction-pipeline/
│
├── assets/generated/          # Prediction plots (auto-created)
├── config/config.yaml         # Pipeline configuration
├── data/raw/                  # Downloaded stock data (auto-created)
├── data/processed/            # Engineered features (auto-created)
├── models/                    # Saved trained models (auto-created)
│
├── notebooks/
│   └── exploration.ipynb      # Exploratory data analysis
│
├── src/
│   ├── data_loader.py         # Data download utilities
│   ├── preprocessing.py       # Feature engineering pipeline
│   ├── model.py               # Model builder
│   ├── train.py               # Training workflow
│   └── predict.py             # Prediction workflow
│
├── tests/
│   ├── conftest.py
│   └── test_model.py
│
├── main.py                    # CLI entry point
├── requirements.txt
├── pytest.ini
├── README.md
└── LICENSE
```

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/shubhpatel1211/stock-price-prediction-pipeline.git
cd stock-price-prediction-pipeline
```

Create a virtual environment:

### Windows

```
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Usage

Run the pipeline via CLI:

Supported modes:

```
train
predict
```

Both require a ticker symbol.

---

# 🏋️ Train Model

Example:

```
python main.py --mode train --ticker AAPL
```

Pipeline steps:

1. Download historical stock data
2. Generate engineered time-series features
3. Train regression model
4. Compare performance against baseline predictor
5. Save trained model
6. Generate prediction plot

Example output:

```
Training complete for AAPL.
Raw data saved to: data/raw/AAPL_stock_data.csv
Processed features saved to: data/processed/AAPL_features.csv
Model saved to: models/AAPL_model.pkl
Plot saved to: assets/generated/aapl_prediction.png
```

---

# 🔮 Predict Next-Day Closing Price

Example:

```
python main.py --mode predict --ticker AAPL
```

Example output:

```
Predicted next-day return for AAPL: 0.41%
Predicted next closing price for AAPL: 192.47
```

---

# 📊 Generated Outputs

Running training automatically creates:

```
data/raw/{TICKER}_stock_data.csv
data/processed/{TICKER}_features.csv
models/{TICKER}_model.pkl
assets/generated/{ticker}_prediction.png
```

Example:

```
python main.py --mode train --ticker NFLX
```

Produces:

```
data/raw/NFLX_stock_data.csv
data/processed/NFLX_features.csv
models/NFLX_model.pkl
assets/generated/nflx_prediction.png
```

---

# ⚙️ Configuration

Pipeline behavior is controlled via:

```
config/config.yaml
```

Configurable parameters include:

* training date range
* lag feature windows
* rolling statistics windows
* regression model selection
* train/test split ratio
* output directories

Example:

```
model:
  type: random_forest
  params:
    n_estimators: 500
    max_depth: 6
```

Supported models:

* RandomForestRegressor
* LinearRegression

---

# 📊 Feature Engineering

Automatically generated features include:

### Lag Features

```
close_lag_1
close_lag_2
close_lag_3
return_lag_1
return_lag_2
return_lag_3
```

### Rolling Statistics

```
rolling_mean_5
rolling_std_5
rolling_mean_10
rolling_std_10
```

### Momentum Indicators

```
momentum_5
momentum_10
price_vs_rolling_mean_5
price_vs_rolling_mean_10
```

### Volatility & Market Signals

```
daily_return
high_low_range
open_close_diff
volume_change
```

### Target Variable

```
Next-day return
```

Predictions are converted into **next-day closing price estimates** for evaluation and visualization.

---

# 📉 Baseline Comparison

Model performance is compared against a naive baseline:

```
next_day_close ≈ current_close
```

This helps determine whether the trained model improves over a simple persistence forecast.

---

# 🧪 Run Tests

Execute:

```
pytest
```

Tests validate:

* model creation
* feature engineering pipeline
* dataset transformation logic
* ticker-based file path generation

---

# 📈 Example Workflow

Train Netflix model:

```
python main.py --mode train --ticker NFLX
```

Predict next closing price:

```
python main.py --mode predict --ticker NFLX
```

---

# 🔬 Exploratory Analysis Notebook

The notebook:

```
notebooks/exploration.ipynb
```

demonstrates:

* dataset inspection
* trend visualization
* return-based target creation
* feature engineering experiments
* correlation analysis

before integrating logic into the pipeline.

---

# 🌱 Future Improvements

Potential extensions:

* walk-forward time-series validation
* model comparison framework
* feature importance visualization
* experiment tracking (MLflow)
* gradient boosting models
* deep learning forecasting (LSTM)

---

# 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
