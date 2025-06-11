import yfinance as yf
import pandas as pd
import datetime

def get_stock_data(ticker, start="2015-01-01", end=None):
    """
    Downloads historical stock data using yfinance.
    Auto-adjusts for splits/dividends and ensures single-level columns.
    """
    if end is None:
        end = datetime.datetime.today().strftime('%Y-%m-%d')

    print(f"📥 Fetching data for {ticker} from {start} to {end}")
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)

    # Flatten multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Close" not in data.columns or data.empty:
        print("⚠️ No 'Close' data found.")
        return pd.DataFrame()

    # Ensure 'Close' is a Series
    if isinstance(data["Close"], pd.DataFrame):
        data["Close"] = data["Close"].iloc[:, 0]

    data.dropna(inplace=True)
    print("✅ Data downloaded successfully.")
    return data

# ------------------------------------------------------
# Optional Standalone Test (Safe from Circular Imports)
# ------------------------------------------------------
if __name__ == "__main__":
    from features import add_indicators
    from label_data import add_target
    from train_model import train_bucket_classifier

    # Read tickers from file
    with open("top100_midas_stocks.txt", "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    for ticker in tickers:
        print(f"\n🔍 Testing {ticker}...")

        try:
            df = get_stock_data(ticker)
            if df.empty:
                print(f"⚠️ No data for {ticker}, skipping.")
                continue

            df = add_indicators(df)
            df = add_target(df)

            model = train_bucket_classifier(df, show_report=False)
            latest = df[["RSI", "MACD", "BB_Width"]].iloc[[-1]]
            probs = model.predict_proba(latest)[0]

            print(f"📈 Probability Distribution for {ticker}:")
            for i, p in enumerate(probs):
                print(f"  Bin {i}: {p:.2%}")

        except Exception as e:
            print(f"❌ Error with {ticker}: {e}")
