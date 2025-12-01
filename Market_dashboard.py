"""
Stock Analysis Dashboard
A Streamlit app for analyzing stocks with technical indicators and fundamental metrics.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stock Dashboard", page_icon=":chart_with_upwards_trend:", layout="wide")

# Sidebar title
st.sidebar.title("Stock Analysis")

# Page header
st.title(":chart_with_upwards_trend: Stock Analysis Dashboard")
st.markdown("Analyze stocks with technical indicators, fundamentals, and AI-powered recommendations.")


def calc_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI).
    RSI measures momentum by comparing recent gains vs losses.
    Values above 70 = overbought, below 30 = oversold.
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calc_macd(prices):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    Uses 12 and 26 day EMAs with a 9-day signal line.
    Bullish when MACD crosses above signal, bearish when below.
    """
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def analyze_stock(info, hist, price, vol):
    """
    Multi-factor stock analysis combining fundamental and technical indicators.
    Returns a rating, score, and detailed breakdown of each factor.
    
    Factors analyzed:
    - P/E ratio (valuation)
    - PEG ratio (growth-adjusted valuation)  
    - Profit margins
    - Debt levels
    - Revenue growth
    - Moving average trend
    - RSI momentum
    - MACD signals
    - 52-week price position
    - Volatility
    """
    score = 0
    max_score = 0
    analysis = []

    # Check P/E ratio - lower is generally better (value investing perspective)
    pe = info.get('trailingPE')
    max_score += 2
    if pe and pe > 0:
        if pe < 15:
            score += 2
            analysis.append("[+] Low P/E ({:.1f}) - potentially undervalued".format(pe))
        elif pe < 25:
            score += 1
            analysis.append("[o] Moderate P/E ({:.1f}) - fairly valued".format(pe))
        else:
            analysis.append("[-] High P/E ({:.1f}) - expensive valuation".format(pe))
    else:
        analysis.append("[o] P/E ratio unavailable")

    # PEG ratio accounts for growth - under 1 is ideal
    peg = info.get('pegRatio')
    max_score += 2
    if peg and peg > 0:
        if peg < 1:
            score += 2
            analysis.append("[+] PEG ratio ({:.2f}) suggests undervalued relative to growth".format(peg))
        elif peg < 2:
            score += 1
            analysis.append("[o] PEG ratio ({:.2f}) is reasonable".format(peg))
        else:
            analysis.append("[-] PEG ratio ({:.2f}) - paying premium for growth".format(peg))

    # Profit margins - higher margins = more efficient business
    margin = info.get('profitMargins')
    max_score += 1
    if margin:
        if margin > 0.15:
            score += 1
            analysis.append("[+] Strong profit margin ({:.1%})".format(margin))
        elif margin > 0.05:
            score += 0.5
            analysis.append("[o] Acceptable profit margin ({:.1%})".format(margin))
        else:
            analysis.append("[-] Thin profit margin ({:.1%})".format(margin))

    # Debt to equity - lower debt = less risky
    de = info.get('debtToEquity')
    max_score += 1
    if de is not None:
        de_ratio = de / 100  # yfinance returns this as a percentage
        if de_ratio < 0.5:
            score += 1
            analysis.append("[+] Low debt (D/E: {:.2f})".format(de_ratio))
        elif de_ratio < 1.5:
            score += 0.5
            analysis.append("[o] Moderate debt (D/E: {:.2f})".format(de_ratio))
        else:
            analysis.append("[-] High debt load (D/E: {:.2f})".format(de_ratio))

    # Revenue growth - want to see the business expanding
    rev_growth = info.get('revenueGrowth')
    max_score += 1
    if rev_growth:
        if rev_growth > 0.15:
            score += 1
            analysis.append("[+] Strong revenue growth ({:.1%})".format(rev_growth))
        elif rev_growth > 0:
            score += 0.5
            analysis.append("[o] Positive revenue growth ({:.1%})".format(rev_growth))
        else:
            analysis.append("[-] Revenue declining ({:.1%})".format(rev_growth))

    # Moving average crossover - classic trend indicator
    # Golden cross (50 > 200) is bullish, death cross is bearish
    if len(hist) >= 200:
        ma50 = hist['Close'].rolling(50).mean().iloc[-1]
        ma200 = hist['Close'].rolling(200).mean().iloc[-1]
        max_score += 2
        
        if price > ma50 > ma200:
            score += 2
            analysis.append("[+] Bullish trend (price > 50 MA > 200 MA)")
        elif price > ma200:
            score += 1
            analysis.append("[o] Above 200-day MA but mixed signals")
        else:
            analysis.append("[-] Below key moving averages - bearish")

    # RSI check - want to avoid overbought conditions
    rsi = calc_rsi(hist['Close'])
    current_rsi = rsi.iloc[-1]
    max_score += 1
    
    if 30 < current_rsi < 70:
        score += 1
        analysis.append("[+] RSI ({:.0f}) in healthy range".format(current_rsi))
    elif current_rsi <= 30:
        score += 0.5
        analysis.append("[o] RSI ({:.0f}) oversold - could bounce".format(current_rsi))
    else:
        analysis.append("[-] RSI ({:.0f}) overbought - caution".format(current_rsi))

    # MACD crossover signals
    macd, signal = calc_macd(hist['Close'])
    max_score += 1
    if macd.iloc[-1] > signal.iloc[-1]:
        if macd.iloc[-2] <= signal.iloc[-2]:
            score += 1
            analysis.append("[+] MACD bullish crossover")
        else:
            score += 0.5
            analysis.append("[o] MACD positive but no fresh signal")
    else:
        analysis.append("[-] MACD below signal line")

    # Where is price in the 52-week range?
    high_52w = hist['Close'][-252:].max() if len(hist) >= 252 else hist['Close'].max()
    low_52w = hist['Close'][-252:].min() if len(hist) >= 252 else hist['Close'].min()
    range_position = (price - low_52w) / (high_52w - low_52w) if high_52w != low_52w else 0.5
    
    max_score += 1
    if range_position > 0.8:
        score += 1
        analysis.append("[+] Near 52-week high ({:.0%} of range)".format(range_position))
    elif range_position > 0.4:
        score += 0.5
        analysis.append("[o] Mid-range in 52-week ({:.0%})".format(range_position))
    else:
        analysis.append("[-] Near 52-week low ({:.0%})".format(range_position))

    # Volatility - lower is generally safer for most investors
    max_score += 1
    if vol < 0.25:
        score += 1
        analysis.append("[+] Low volatility ({:.1%} annualized)".format(vol))
    elif vol < 0.40:
        score += 0.5
        analysis.append("[o] Moderate volatility ({:.1%})".format(vol))
    else:
        analysis.append("[-] High volatility ({:.1%}) - risky".format(vol))

    # Convert score to a rating
    pct = (score / max_score) * 100 if max_score > 0 else 0
    
    if pct >= 75:
        rating = "Strong Buy"
        color = "#00C853"
    elif pct >= 60:
        rating = "Buy"
        color = "#69F0AE"
    elif pct >= 45:
        rating = "Hold"
        color = "#FFB74D"
    elif pct >= 30:
        rating = "Weak"
        color = "#FF8A65"
    else:
        rating = "Avoid"
        color = "#EF5350"

    return rating, "{:.1f}/{:.0f}".format(score, max_score), pct, analysis, color


def load_stock(ticker):
    """
    Fetch stock data from Yahoo Finance.
    Returns the ticker object, company info dict, and 2 years of price history.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        info = stock.info
        if hist.empty:
            return None, None, None
        return stock, info, hist
    except:
        return None, None, None


# Sidebar for user input
st.sidebar.title("Controls")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper().strip()

st.title("Stock Analysis Dashboard")

# Load the data
stock, info, hist = load_stock(ticker)

if stock is None:
    st.error("Could not load data for '{}'. Check the symbol and try again.".format(ticker))
else:
    # Basic info
    name = info.get('longName', ticker)
    price = hist['Close'].iloc[-1]
    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else price
    change = price - prev_close
    change_pct = (change / prev_close) * 100
    
    # Calculate annualized volatility from daily returns
    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    volatility = returns.std() * np.sqrt(252)

    # Display header with key metrics
    st.subheader("{} ({})".format(name, ticker))
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", "${:.2f}".format(price), "{:+.2f} ({:+.1f}%)".format(change, change_pct))
    m2.metric("Volume", "{:,.0f}".format(hist['Volume'].iloc[-1]))
    m3.metric("52W High", "${:.2f}".format(hist['Close'][-252:].max()) if len(hist) >= 252 else "N/A")
    m4.metric("Volatility", "{:.1%}".format(volatility))

    # Main content area - chart on left, analysis on right
    chart_col, analysis_col = st.columns([1.8, 1.2])

    with chart_col:
        # Build candlestick chart with volume subplot
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist['Open'], high=hist['High'],
            low=hist['Low'], close=hist['Close'], name='Price'
        ), row=1, col=1)
        
        # Add moving averages if we have enough data
        if len(hist) >= 50:
            ma50 = hist['Close'].rolling(50).mean()
            fig.add_trace(go.Scatter(x=hist.index, y=ma50, name='50 MA', 
                                    line=dict(color='orange', width=1)), row=1, col=1)
        if len(hist) >= 200:
            ma200 = hist['Close'].rolling(200).mean()
            fig.add_trace(go.Scatter(x=hist.index, y=ma200, name='200 MA',
                                    line=dict(color='purple', width=1)), row=1, col=1)
        
        # Color volume bars based on price direction
        colors = ['#EF5350' if hist['Close'].iloc[i] < hist['Open'].iloc[i] 
                  else '#26A69A' for i in range(len(hist))]
        fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume',
                            marker_color=colors), row=2, col=1)
        
        fig.update_layout(
            template='plotly_dark',
            height=500,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        st.plotly_chart(fig, use_container_width=True)

    with analysis_col:
        st.markdown("### Analysis")
        
        # Run the analysis
        rating, score_str, pct, reasons, color = analyze_stock(info, hist, price, volatility)
        
        # Display rating in a colored box
        st.markdown("""
        <div style='background: linear-gradient(135deg, {}, {}dd); 
                    padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 15px;'>
            <h2 style='margin: 0; color: white;'>{}</h2>
            <p style='margin: 5px 0 0 0; color: rgba(255,255,255,0.9);'>Score: {} ({:.0f}%)</p>
        </div>
        """.format(color, color, rating, score_str, pct), unsafe_allow_html=True)
        
        # Show detailed breakdown
        with st.expander("View Detailed Breakdown", expanded=True):
            for reason in reasons:
                st.markdown("- {}".format(reason))

    # Options chain section
    st.markdown("---")
    st.subheader("Options Chain")
    
    try:
        exp_dates = stock.options
        if exp_dates:
            selected_exp = st.selectbox("Expiration", exp_dates)
            chain = stock.option_chain(selected_exp)
            
            opt_c1, opt_c2 = st.columns(2)
            
            with opt_c1:
                st.markdown("**Calls**")
                calls_display = chain.calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].copy()
                calls_display.columns = ['Strike', 'Last', 'Bid', 'Ask', 'Vol', 'OI', 'IV']
                calls_display['IV'] = calls_display['IV'].apply(lambda x: "{:.1%}".format(x) if pd.notna(x) else "N/A")
                st.dataframe(calls_display, height=400, use_container_width=True)
            
            with opt_c2:
                st.markdown("**Puts**")
                puts_display = chain.puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].copy()
                puts_display.columns = ['Strike', 'Last', 'Bid', 'Ask', 'Vol', 'OI', 'IV']
                puts_display['IV'] = puts_display['IV'].apply(lambda x: "{:.1%}".format(x) if pd.notna(x) else "N/A")
                st.dataframe(puts_display, height=400, use_container_width=True)
        else:
            st.info("No options available for this stock.")
    except Exception as e:
        st.warning("Could not load options data: {}".format(e))
