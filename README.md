# 📈 Stock Predictor

This project is a machine learning-based stock prediction tool built with Python. It aims to estimate the probability distribution of 7-day returns for selected stocks using historical price data and technical features.

## 🚀 Features

- Fetches stock price data using `yfinance`
- Labels future returns into classes (e.g., positive, neutral, negative)
- Trains a calibrated classification model to output probabilities
- Supports batch predictions from a list of stocks
- Simulates forward expected return distributions

## 🧠 Technologies Used

- Python 3
- pandas
- numpy
- scikit-learn
- yfinance
- matplotlib

## 📁 File Overview

| File Name              | Description |
|------------------------|-------------|
| `get_data.py`          | Downloads historical stock price data |
| `label_data.py`        | Labels each data point based on future returns |
| `train_model.py`       | Trains a calibrated ML model for classification |
| `simulate_forward.py`  | Simulates forward return probabilities |
| `screeners.py`         | Screens stocks based on probability thresholds |
| `features.py`          | Generates technical indicators as features |
| `top100_midas_stocks.txt` | List of stocks to analyze |

## 🧪 How to Use

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   python get_data.py
   python label_data.py
   python train_model.py
   python simulate_forward.py

📌 Notes
This project is for educational and experimental purposes only. It does not provide financial advice or guarantees of return.
