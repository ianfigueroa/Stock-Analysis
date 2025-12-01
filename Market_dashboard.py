import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stock Dashboard", page_icon="���", layout="wide")


def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calc_macd(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def analyze_stock(info, hist, price, vol):
    score = 0
    max_score = 0
    analysis = []

    # P/E Ratio Analysis
    pe = info.get('trailingPE')
    fwd_pe = info.get('forwardPE')
    sector = info.get('sector', 'Unknown')
    
    max_score += 2
    if pe and pe > 0:
        if pe < 15:
            score += 2
            analysis.append(f"✓ Low P/E ({pe:.1f}) - potentially undervalued")
        elif pe < 25:
            score += 1
            analysis.append(f"○ Moderate P/E ({pe:.1f}) - fairly valued")
        else:
            analysis.append(f"✗ High P/E ({pe:.1f}) - expensive valuation")
    else:
        analysis.append("○ P/E ratio unavailable")

    # PEG Ratio (growth-adjusted valuation)
    peg = info.get('pegRatio')
    max_score += 2
    if peg and peg > 0:
        if peg < 1:
            score += 2
            analysis.append(f"✓ PEG ratio ({peg:.2f}) suggests undervalued relative to growth")
        elif peg < 2:
            score += 1
            analysis.append(f"○ PEG ratio ({peg:.2f}) is reasonable")
        else:
            analysis.append(f"✗ PEG ratio ({peg:.2f}) - paying premium for growth")

    # Profit Margins
    margin = info.get('profitMargins')
    max_score += 1
    if margin:
        if margin > 0.15:
            score += 1
            analysis.append(f"✓ Strong profit margin ({margin:.1%})")
        elif margin > 0.05:
            score += 0.5
            analysis.append(f"○ Acceptable profit margin ({margin:.1%})")
        else:
            analysis.append(f"✗ Thin profit margin ({margin:.1%})")

    # Debt/Equity
    de = info.get('debtToEquity')
    max_score += 1
    if de is not None:
        de_ratio = de / 100  # yfinance reports as percentage
        if de_ratio < 0.5:
            score += 1
            analysis.append(f"✓ Low debt (D/E: {de_ratio:.2f})")
        elif de_ratio < 1.5:
            score += 0.5
            analysis.append(f"○ Moderate debt (D/E: {de_ratio:.2f})")
        else:
            analysis.append(f"✗ High debt load (D/E: {de_ratio:.2f})")

    # Revenue Growth
    rev_growth = info.get('revenueGrowth')
    max_score += 1
    if rev_growth:
        if rev_growth > 0.15:
            score += 1
            analysis.append(f"✓ Strong revenue growth ({rev_growth:.1%})")
        elif rev_growth > 0:
            score += 0.5
            analysis.append(f"○ Positive revenue growth ({rev_growth:.1%})")
        else:
            analysis.append(f"✗ Revenue declining ({rev_growth:.1%})")

    # Moving Average Trend
    if len(hist) >= 200:
        ma50 = hist['Close'].rolling(50).mean().iloc[-1]
        ma200 = hist['Close'].rolling(200).mean().iloc[-1]
        max_score += 2
        
        if price > ma50 > ma200:
            score += 2
            analysis.append("✓ Bullish trend (price > 50 MA > 200 MA)")
        elif price > ma200:
            score += 1
            analysis.append("○ Above 200-day MA but mixed signals")
        else:
            analysis.append("✗ Below key moving averages - bearish")

    # RSI
    rsi = calc_rsi(hist['Close'])
    current_rsi = rsi.iloc[-1]
    max_score += 1
    
    if 30 < current_rsi < 70:
        score += 1
        analysis.append(f"✓ RSI ({current_rsi:.0f}) in healthy range")
    elif current_rsi <= 30:
        score += 0.5
        analysis.append(f"○ RSI ({current_rsi:.0f}) oversold - could bounce")
    else:
        analysis.append(f"✗ RSI ({current_rsi:.0f}) overbought - caution")

    # MACD
    macd, signal = calc_macd(hist['Close'])
    max_score += 1
    if macd.iloc[-1] > signal.iloc[-1]:
        if macd.iloc[-2] <= signal.iloc[-2]:
            score += 1
            analysis.append("✓ MACD bullish crossover")
        else:
            score += 0.5
            analysis.append("○ MACD positive but no fresh signal")
    else:
        analysis.append("✗ MACD below signal line")

    # 52-week position
    high_52w = hist['Close'][-252:].max() if len(hist) >= 252 else hist['Close'].max()
    low_52w = hist['Close'][-252:].min() if len(hist) >= 252 else hist['Close'].min()
    range_position = (price - low_52w) / (high_52w - low_52w) if high_52w != low_52w else 0.5
    
    max_score += 1
    if range_position > 0.8:
        score += 1
        analysis.append(f"✓ Near 52-week high ({range_position:.0%} of range)")
    elif range_position > 0.4:
        score += 0.5
        analysis.append(f"○ Mid-range in 52-week ({range_position:.0%})")
    else:
        analysis.append(f"✗ Near 52-week low ({range_position:.0%})")

    # Volatility assessment
    max_score += 1
    if vol < 0.25:
        score += 1
        analysis.append(f"✓ Low volatility ({vol:.1%} annualized)")
    elif vol < 0.40:
        score += 0.5
        analysis.append(f"○ Moderate volatility ({vol:.1%})")
    else:
        analysis.append(f"✗ High volatility ({vol:.1%}) - risky")

    # Calculate final rating
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

    return rating, f"{score:.1f}/{max_score:.0f}", pct, analysis, color


@st.cache_data(ttl=300)
def load_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        info = stock.info
        if hist.empty:
            return None, None, None
        return stock, info, hist
    except:
        return None, None, None


# Sidebar
st.sidebar.title("Controls")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper().strip()

# Main content
st.title("��� Stock Analysis Dashboard")

stock, info, hist = load_stock(ticker)

if stock is None:
    st.error(f"Could not load data for '{ticker}'. Check the symbol and try again.")
else:
    name = info.get('longName', ticker)
    price = hist['Close'].iloc[-1]
    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else price
    change = price - prev_close
    change_pct = (change / prev_close) * 100
    
    # Calculate volatility
    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    volatility = returns.std() * np.sqrt(252)

    # Header metrics
    st.subheader(f"{name} ({ticker})")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", f"${price:.2f}", f"{change:+.2f} ({change_pct:+.1f}%)")
    m2.metric("Volume", f"{hist['Volume'].iloc[-1]:,.0f}")
    m3.metric("52W High", f"${hist['Close'][-252:].max():.2f}" if len(hist) >= 252 else "N/A")
    m4.metric("Volatility", f"{volatility:.1%}")

    # Two columns: chart and analysis
    chart_col, analysis_col = st.columns([1.8, 1.2])

    with chart_col:
        # Candlestick with volume
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist['Open'], high=hist['High'],
            low=hist['Low'], close=hist['Close'], name='Price'
        ), row=1, col=1)
        
        # Add moving averages
        if len(hist) >= 50:
            ma50 = hist['Close'].rolling(50).mean()
            fig.add_trace(go.Scatter(x=hist.index, y=ma50, name='50 MA', 
                                    line=dict(color='orange', width=1)), row=1, col=1)
        if len(hist) >= 200:
            ma200 = hist['Close'].rolling(200).mean()
            fig.add_trace(go.Scatter(x=hist.index, y=ma200, name='200 MA',
                                    line=dict(color='purple', width=1)), row=1, col=1)
        
        # Volume bars
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
        
        rating, score_str, pct, reasons, color = analyze_stock(info, hist, price, volatility)
        
        # Rating box
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {color}, {color}dd); 
                    padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 15px;'>
            <h2 style='margin: 0; color: white;'>{rating}</h2>
            <p style='margin: 5px 0 0 0; color: rgba(255,255,255,0.9);'>Score: {score_str} ({pct:.0f}%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Breakdown
        with st.expander("View Detailed Breakdown", expanded=True):
            for reason in reasons:
                st.markdown(f"- {reason}")

    # Options section
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
                calls_display['IV'] = calls_display['IV'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                st.dataframe(calls_display, height=400, use_container_width=True)
            
            with opt_c2:
                st.markdown("**Puts**")
                puts_display = chain.puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].copy()
                puts_display.columns = ['Strike', 'Last', 'Bid', 'Ask', 'Vol', 'OI', 'IV']
                puts_display['IV'] = puts_display['IV'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                st.dataframe(puts_display, height=400, use_container_width=True)
        else:
            st.info("No options available for this stock.")
    except Exception as e:
        st.warning(f"Could not load options data: {e}")
