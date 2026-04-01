# 📈 Stock Price Predictor

A machine learning project that predicts future stock prices using
Linear Regression trained on historical data fetched via Yahoo Finance.

## 🚀 Features
- Downloads real-time historical stock data using `yfinance`
- Predicts next 25 days of stock prices
- Visualizes actual vs predicted prices on a graph
- Saves prediction plot as an image

## 🛠️ Tech Stack
- Python
- yfinance
- Pandas & NumPy
- Scikit-learn (Linear Regression)
- Matplotlib



Install dependencies:
pip install yfinance pandas numpy scikit-learn matplotlib

## ▶️ Usage
python stock_predictor.py

## 📊 Output
- Prints predicted stock prices for the next 25 days
- Displays a graph of actual vs predicted prices
- Saves the graph as `stock_prediction.png`

## 📌 Example
Predicting AAPL (Apple Inc.) stock prices from 2020 to 2024

## 🔮 Future Improvements
- Add LSTM deep learning model
- Build Streamlit dashboard
- Add live price predictions
- Support multiple stock tickers

## 📄 License
MIT License