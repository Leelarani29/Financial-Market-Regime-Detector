# Financial Market Regime Detection Dashboard

This project detects financial market regimes using Hidden Markov Models (HMM) and time series analysis.

The system classifies market conditions into volatility regimes such as:

- Stable Market
- Moderate Volatility
- High Volatility

Users can select different market indices and analyze current market behavior.

Supported Markets:
- NSE (India)
- BSE (India)
- S&P 500 (US)
- NASDAQ (US)



## Technologies Used

- Python
- Hidden Markov Models (hmmlearn)
- Time Series Analysis
- Pandas, NumPy
- Streamlit
- Yahoo Finance API



## Features

- Detects hidden volatility regimes using Gaussian HMM
- Calculates log returns and rolling volatility
- Visualizes regime clusters in market price data
- Interactive dashboard built using Streamlit
- Supports multiple global indices

