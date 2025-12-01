# Stock Analysis Dashboard

A web-based stock analysis tool featuring real-time market data, technical indicators, an AI-powered analysis bot, and a Black-Scholes options calculator.

**[Live Demo](https://ianfigueroa-stock-analysis-market-dashboard-gcqasy.streamlit.app/)**

Built with Python, Streamlit, and Plotly.

## Features

### Market Dashboard
- Real-time stock data via Yahoo Finance
- Interactive candlestick charts with 50/200-day moving averages
- Volume analysis with color-coded bars
- Live options chain with implied volatility data
- Key financial metrics (market cap, P/E, dividend yield, etc.)

### AI Analysis Bot
Analyzes stocks and provides buy/hold/sell recommendations based on a 0-100% scoring system:

| Category | Metrics |
|----------|---------|
| Valuation | P/E ratio, PEG ratio |
| Financials | Profit margins, debt-to-equity, revenue growth |
| Technicals | MA crossovers, RSI, MACD, 52-week range |
| Risk | Historical volatility |

Scores are weighted and combined to generate an overall recommendation with detailed reasoning.

### Black-Scholes Options Calculator
- Price European call and put options
- Full Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Interactive heatmaps showing price sensitivity to spot and volatility
- Break-even price calculations
- P&L diagram at expiration
- Educational explanations of each Greek

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
