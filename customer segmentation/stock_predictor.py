import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Script started...")

    # Step 1: Download stock data
    print("Downloading data...")
    raw = yf.download('AAPL', start='2020-01-01', end='2024-01-01', auto_adjust=True)

    # Step 2: Fix multi-level columns (common yfinance issue)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    data = raw[['Close']].copy()
    print(f"Data loaded: {len(data)} rows")
    print(data.tail())

    # Step 3: Create prediction column (shift by future days)
    future_days = 25
    data['Prediction'] = data['Close'].shift(-future_days)

    # Step 4: Prepare dataset
    X = np.array(data[['Close']])[:-future_days]
    y = np.array(data['Prediction'])[:-future_days]

    # Step 5: Train model
    print("Training model...")
    model = LinearRegression()
    model.fit(X, y)
    print(f"Model trained. R² score: {model.score(X, y):.4f}")

    # Step 6: Predict future prices
    future_input = np.array(data[['Close']])[-future_days:]
    predictions = model.predict(future_input)

    # Step 7: Print predictions
    print("\nNext 25 days predicted prices:")
    for i, price in enumerate(predictions, 1):
        print(f"  Day {i:2d}: ${price:.2f}")

    # Step 8: Plot actual + predicted prices
    plt.figure(figsize=(14, 6))

    # Plot historical close prices
    plt.plot(data.index, data['Close'], label='Actual Price', color='blue')

    # Plot predicted prices starting after historical data
    last_date = data.index[-future_days]
    future_dates = data.index[-future_days:]
    plt.plot(future_dates, predictions, label='Predicted Price',
             color='red', linestyle='--', marker='o', markersize=3)

    plt.title('AAPL Stock Price Prediction (Linear Regression)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('stock_prediction.png')  # Save in case plt.show() doesn't work
    print("\nPlot saved as stock_prediction.png")
    plt.show()

    print("Done!")