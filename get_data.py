import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# AAPL stock data for the last 10 years
ticker = yf.Ticker("AAPL")
df = ticker.history(period="10y")

# --- Technical indicators ---

df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
df['MACD'] = ta.trend.MACD(df['Close']).macd()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
df['Volume_MA'] = df['Volume'].rolling(window=20).mean()

# Remove rows with missing values
df.dropna(inplace=True)

print(df[['Close', 'RSI', 'MACD', 'SMA_50', 'SMA_200']].head())
print(f"\nTotal rows after adding indicators: {len(df)}")

df.to_csv("apple_data_with_indicators.csv")
print("\nSaved to apple_data_with_indicators.csv")

# --- AI INTEGRATION ---

# Did price rise more than 3% in the next 10 trading days?
future_return = df["Close"].shift(-10) / df["Close"] - 1

df["Target"] = (future_return > 0.03).astype(int)
df.dropna(inplace=True)

# --- FEATURES ---
features = ["RSI", "MACD", "SMA_50", "SMA_200", "Volume_MA"]
X = df[features]
y = df["Target"]

# --- AI TRAINING ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators=200, random_state=42)

model.fit(X_train, y_train)

# --- TEST ---
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(f"{accuracy:.2%}")

print("\nFeature Importance:")

for feature, importance in zip(features, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

latest_data = df[features].iloc[-1:]
prediction = model.predict(latest_data)[0]

probability = model.predict_proba(latest_data)[0][1]

print("\n========== LYNXAI REPORT ==========")
print(f"Ticker: AAPL")

if prediction == 1:
    print("Signal: BUY")
else:
    print("Signal: NO BUY")

print(
    f"Confidence: {probability:.2%}"
)
print("===================================")