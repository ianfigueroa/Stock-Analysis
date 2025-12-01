# Stock Analysis Dashboard

Interactive stock analysis tool with real-time data, technical indicators, and options pricing.

Built with Python, Streamlit, and Plotly.

## Features

### Stock Dashboard
- Real-time stock data via Yahoo Finance
- Candlestick charts with 50/200-day moving averages
- Volume analysis
- Live options chain with IV data

### Analysis Bot
Scores stocks on a 0-100% scale using:

| Category | Metrics |
|----------|---------|
| Valuation | P/E ratio, PEG ratio |
| Financials | Profit margins, debt-to-equity, revenue growth |
| Technicals | MA crossovers, RSI, MACD, 52-week range |
| Risk | Historical volatility |

### Black-Scholes Calculator
- European option pricing
- Full Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Interactive sensitivity charts
- P&L visualization

## Quick Start

```bash
git clone https://github.com/ianfigueroa/Stock-Analysis.git
cd Stock-Analysis
pip install -r requirements.txt
streamlit run Market_dashboard.py
```

**Docker:**
```bash
docker build -t stock-analysis .
docker run -p 8501:8501 stock-analysis
```

Open http://localhost:8501

## Tech Stack

| Tool | Purpose |
|------|---------|
| Streamlit | Web framework |
| yfinance | Market data |
| Plotly | Charts |
| NumPy/SciPy | Calculations |
| Pandas | Data handling |

## Files

```
Market_dashboard.py      # Main dashboard
pages/
  Black-Scholes_Calc.py  # Options calculator
requirements.txt
Dockerfile
```

## License

MIT
