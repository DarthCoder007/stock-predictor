import pandas as pd

# CHANGE: The default for days_forward is now 5, and the default bins are wider.
def add_target(df, days_forward=5, threshold=0.03, bins=[-0.2, -0.05, 0, 0.03, 0.06, 0.1, 0.2, 0.5]):
    """
    Adds target columns to the DataFrame for a weekly forecast.
    
    - 'Target': Binary classification label (1 if future return > threshold, else 0)
    - 'Target_Return': Continuous % return after `days_forward` days
    - 'Return_Bin': Discretized return class (used for bucketed regression)
    
    Parameters:
        df (pd.DataFrame): DataFrame with at least a 'Close' column.
        days_forward (int): How many days into the future to compare.
        threshold (float): Minimum return to classify as BUY.
        bins (list): Bucket thresholds for return categories.
    """
    df = df.copy()

    # This will now get the close price 5 days in the future
    df["Future_Close"] = df["Close"].shift(-days_forward)

    # This now calculates the return over 5 days
    df["Future_Return"] = (df["Future_Close"] - df["Close"]) / df["Close"]

    # Binary classification target (BUY vs SKIP)
    df["Target"] = (df["Future_Return"] > threshold).astype(int)

    # Continuous return value
    df["Target_Return"] = df["Future_Return"]

    # Bucketed class label
    labels = list(range(len(bins) - 1))  # [0, 1, 2, ...]
    df["Return_Bin"] = pd.cut(df["Future_Return"], bins=bins, labels=labels)

    # Remove any rows with missing data
    df.dropna(inplace=True)

    return df