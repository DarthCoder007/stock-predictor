from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np

# CHANGE: The bins are now wider to capture typical weekly price swings.
# We've expanded the range to better categorize weekly movements.
bins = [-np.inf, -0.05, 0, 0.03, 0.06, 0.10, np.inf] 
bin_labels = [0, 1, 2, 3, 4, 5]

def train_bucket_classifier(df, show_report=True):
    """
    Trains a classifier to predict the return bucket for the next 5 days.
    """
    df = df.copy()
    features = ["RSI", "MACD", "BB_Width"]
    target = "Target_Return"

    if not all(col in df.columns for col in features + [target]):
        raise ValueError("Missing required features or target.")

    # Bin the continuous target using the new weekly bins
    df["Return_Bin"] = pd.cut(df[target], bins=bins, labels=bin_labels, right=False)

    df.dropna(inplace=True)
    X = df[features]
    y = df["Return_Bin"].astype(int)

    if X.empty or y.empty:
        raise ValueError("Training data is empty after processing.")

    # Using shuffle=False is crucial for time-series data to prevent lookahead bias.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    if show_report:
        y_pred = clf.predict(X_test)
        print("\n🎯 Classification Report (Weekly Buckets):")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(classification_report(y_test, y_pred, zero_division=0))

    return clf

# Optional test block (do not import get_data here)
if __name__ == "__main__":
    print("❗ This file is intended to be imported, not run standalone.")
