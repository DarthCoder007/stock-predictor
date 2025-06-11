import pandas as pd
from features import add_indicators
from label_data import add_target

def simulate_future(df, clf, reg, days=5):
    """
    Simulates future price and indicator updates for `days` steps.
    For each step:
    - Predicts buy/sell using classifier
    - Predicts expected return using regressor
    - Simulates closing price update
    - Recomputes technical indicators
    """

    if df.empty or "Close" not in df.columns:
        raise ValueError("Input dataframe must contain 'Close' column.")

    simulated_results = []
    df_sim = df.copy()
    last_close = df_sim["Close"].iloc[-1]

    for i in range(1, days + 1):
        # Get latest indicators
        df_sim = add_indicators(df_sim)
        latest_features = df_sim[["RSI", "MACD", "BB_Width"]].iloc[-1:]

        # Predict
        pred_class = clf.predict(latest_features)[0]
        pred_return = reg.predict(latest_features)[0]

        # Simulate future close price based on predicted return
        future_close = last_close * (1 + pred_return)

        simulated_results.append({
            "Day": f"Day +{i}",
            "Predicted Return": round(pred_return, 4),
            "Simulated Close": round(future_close, 2),
            "BUY Signal": bool(pred_class)
        })

        # Append simulated close to df
        new_row = pd.DataFrame({
            "Close": [future_close]
        }, index=[df_sim.index[-1] + pd.Timedelta(days=1)])

        df_sim = pd.concat([df_sim, new_row])
        last_close = future_close  # update for next day

    return pd.DataFrame(simulated_results)
