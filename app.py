import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Financial Market Regime Detector", layout="wide")

st.title("📈 Financial Market Regime Detection using HMM")

# Market Options
market_options = {
    "NSE (India)": "^NSEI",
    "BSE (India)": "^BSESN",
    "S&P 500 (US)": "^GSPC",
    "NASDAQ (US)": "^IXIC"
}

user_choice = st.selectbox("Select Market", list(market_options.keys()))
selected_market = market_options[user_choice]

# Download Data
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, start="1990-01-01")
    df = df[['Close']]
    df.dropna(inplace=True)
    return df

df = load_data(selected_market)

# Feature Engineering
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
df['Rolling_Vol'] = df['Log_Return'].rolling(20).std()
df.dropna(inplace=True)

# Scaling
scaler = StandardScaler()
features = df[['Log_Return', 'Rolling_Vol']]
scaled_features = scaler.fit_transform(features)

# Train HMM
model = GaussianHMM(
    n_components=3,
    covariance_type="diag",
    n_iter=1000,
    random_state=42
)

model.fit(scaled_features)

df['Regime'] = model.predict(scaled_features)

# Identify Crisis Regime (highest variance in returns)
variances = model.covars_[:, 0, 0]
crisis_regime = np.argmax(variances)

# Detect Current Regime
current_regime = df['Regime'].iloc[-1]

st.subheader("📊 Current Market Status")

# Sort regimes based on variance
sorted_regimes = np.argsort(variances)

low_vol_regime = sorted_regimes[0]
medium_vol_regime = sorted_regimes[1]
high_vol_regime = sorted_regimes[2]

st.subheader("📊 Current Market Status")

if current_regime == low_vol_regime:
    st.success("🟢 Low Volatility Regime")
    st.write("Advisory: Market is relatively stable with controlled fluctuations.")

elif current_regime == medium_vol_regime:
    st.warning("🟡 Moderate Volatility Regime")
    st.write("Advisory: Market fluctuations increasing. Caution advised.")

else:
    st.error("🔴 High Volatility / Stress Regime")
    st.write("Advisory: Elevated market risk detected. Consider defensive positioning.")
# Plot Market with Regimes
st.subheader("📈 Market Regime Visualization")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(df['Close'], color='black', linewidth=1)

for regime in df['Regime'].unique():
    mask = df['Regime'] == regime
    ax.scatter(df.index[mask], df['Close'][mask], s=5, label=f"Regime {regime}")

ax.set_title(f"{user_choice} Regime Detection")
ax.legend()

st.pyplot(fig)