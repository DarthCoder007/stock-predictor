from get_data import get_stock_data
from features import add_indicators
from label_data import add_target
from train_model import train_bucket_classifier
import pandas as pd

# ------------------------------
# Load ticker list from file
# ------------------------------
try:
    with open("top100_midas_stocks.txt", "r") as file:
        tickers = [line.strip() for line in file.readlines() if line.strip()]
except FileNotFoundError:
    print("❌ Error: 'top100_midas_stocks.txt' not found. Please create it.")
    tickers = []

if tickers:
    print(f"\n🔎 Starting Weekly Screener for {len(tickers)} Stocks...\n")

# ------------------------------
# CHANGE: Bin ranges are updated for our new weekly model.
# These labels correspond to the bins in train_model.py
# ------------------------------
bin_ranges = [
    "Significant Drop (< -5%)",
    "Slight Drop (-5% to 0%)",
    "Neutral (0% to 3%)",
    "Modest Gain (3% to 6%)",
    "Strong Gain (6% to 10%)",
    "Exceptional Gain (> 10%)",
]

# ------------------------------
# Loop through each stock
# ------------------------------
results = []
for ticker in tickers:
    print(f"🔍 Evaluating {ticker}...")

    try:
        # Get the last 3 years of data for a reasonably sized training set
        df = get_stock_data(ticker, start="2021-01-01")

        if df.empty or len(df) < 100: # Need enough data to train
            print(f"⚠️ Not enough data for {ticker}, skipping.\n")
            continue

        # Feature Engineering and Labeling (now using weekly logic)
        df_featured = add_indicators(df)
        df_labeled = add_target(df_featured)

        # Train the model on this stock's historical data
        model = train_bucket_classifier(df_labeled, show_report=False)

        # Predict using the most recent data point
        latest_features = df_labeled[model.feature_names_in_].iloc[[-1]]
        probabilities = model.predict_proba(latest_features)[0]
        
        # Store result
        results.append({
            'ticker': ticker,
            'probabilities': probabilities,
            'top_bin': probabilities.argmax(),
            'confidence': probabilities.max()
        })
        
        print(f"✅ Evaluation for {ticker} complete.\n")

    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}\n")

# ------------------------------
# Display ranked results
# ------------------------------
if results:
    # Sort by the most optimistic bin, then by the confidence in that prediction
    sorted_results = sorted(results, key=lambda x: (x['top_bin'], x['confidence']), reverse=True)

    print("="*50)
    print("🏆 Weekly Trading Watchlist 🏆")
    print("="*50)

    for res in sorted_results:
        ticker = res['ticker']
        top_bin = res['top_bin']
        confidence = res['confidence']
        
        action = "--- SKIP ---"
        if top_bin >= 4: # Bins for "Strong Gain" or "Exceptional Gain"
            action = "✅ STRONG BUY"
        elif top_bin == 3: # Bin for "Modest Gain"
            action = "🟡 CONSIDER"

        print(f"\n{action}: {ticker}")
        print(f"Most Likely Outcome: '{bin_ranges[top_bin]}' with {confidence:.1%} confidence.")
        # Optional: uncomment to see full probability distribution
        # for i, p in enumerate(res['probabilities']):
        #     print(f"  - {bin_ranges[i]}: {p:.1%}")

else:
    print("No stocks were processed.")

print("\n✅ Screener Finished.")
