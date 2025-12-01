import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Options Pricing", page_icon=":bar_chart:", layout="wide")

def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        intrinsic = max(0, S - K) if option_type == 'call' else max(0, K - S)
        return intrinsic, 0, 0, 0, 0, 0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    rho = K * T * np.exp(-r * T) * (norm.cdf(d2) if option_type == 'call' else -norm.cdf(-d2)) / 100
    
    return price, delta, gamma, theta, vega, rho


st.title("Black-Scholes Options Calculator")

st.sidebar.header("Parameters")
S = st.sidebar.number_input("Stock Price ($)", min_value=1.0, value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price ($)", min_value=1.0, value=100.0, step=1.0)
T = st.sidebar.slider("Days to Expiration", min_value=1, max_value=730, value=30) / 365
r = st.sidebar.slider("Risk-Free Rate (%)", min_value=0.0, max_value=15.0, value=5.0) / 100
sigma = st.sidebar.slider("Implied Volatility (%)", min_value=1.0, max_value=150.0, value=25.0) / 100

call_price, call_delta, call_gamma, call_theta, call_vega, call_rho = black_scholes(S, K, T, r, sigma, 'call')
put_price, put_delta, put_gamma, put_theta, put_vega, put_rho = black_scholes(S, K, T, r, sigma, 'put')

st.markdown("### Option Prices")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.metric("Call Price", "${:.2f}".format(call_price))
with col2:
    st.metric("Put Price", "${:.2f}".format(put_price))
with col3:
    parity_diff = call_price - put_price - S + K * np.exp(-r * T)
    st.metric("Put-Call Parity", "${:.4f}".format(abs(parity_diff)), 
              "Valid" if abs(parity_diff) < 0.01 else "Check inputs")

st.markdown("---")

st.markdown("### The Greeks")

greeks_col1, greeks_col2 = st.columns(2)

with greeks_col1:
    st.markdown("**Call Greeks**")
    gc1, gc2, gc3 = st.columns(3)
    gc1.metric("Delta", "{:.4f}".format(call_delta))
    gc2.metric("Gamma", "{:.4f}".format(call_gamma))
    gc3.metric("Theta", "{:.4f}".format(call_theta))
    gc4, gc5, _ = st.columns(3)
    gc4.metric("Vega", "{:.4f}".format(call_vega))
    gc5.metric("Rho", "{:.4f}".format(call_rho))

with greeks_col2:
    st.markdown("**Put Greeks**")
    gp1, gp2, gp3 = st.columns(3)
    gp1.metric("Delta", "{:.4f}".format(put_delta))
    gp2.metric("Gamma", "{:.4f}".format(put_gamma))
    gp3.metric("Theta", "{:.4f}".format(put_theta))
    gp4, gp5, _ = st.columns(3)
    gp4.metric("Vega", "{:.4f}".format(put_vega))
    gp5.metric("Rho", "{:.4f}".format(put_rho))

st.markdown("---")

st.markdown("### Price Sensitivity")

viz_tab1, viz_tab2 = st.tabs(["Price vs Stock Price", "Price vs Volatility"])

with viz_tab1:
    stock_range = np.linspace(S * 0.7, S * 1.3, 50)
    call_prices = [black_scholes(s, K, T, r, sigma, 'call')[0] for s in stock_range]
    put_prices = [black_scholes(s, K, T, r, sigma, 'put')[0] for s in stock_range]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_range, y=call_prices, name='Call', line=dict(color='#00C853', width=2)))
    fig.add_trace(go.Scatter(x=stock_range, y=put_prices, name='Put', line=dict(color='#EF5350', width=2)))
    fig.add_vline(x=K, line_dash="dash", line_color="gray", annotation_text="Strike")
    fig.add_vline(x=S, line_dash="dot", line_color="white", annotation_text="Current")
    
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Stock Price ($)',
        yaxis_title='Option Price ($)',
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)

with viz_tab2:
    vol_range = np.linspace(0.05, 1.0, 50)
    call_by_vol = [black_scholes(S, K, T, r, v, 'call')[0] for v in vol_range]
    put_by_vol = [black_scholes(S, K, T, r, v, 'put')[0] for v in vol_range]
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=vol_range * 100, y=call_by_vol, name='Call', line=dict(color='#00C853', width=2)))
    fig2.add_trace(go.Scatter(x=vol_range * 100, y=put_by_vol, name='Put', line=dict(color='#EF5350', width=2)))
    fig2.add_vline(x=sigma * 100, line_dash="dot", line_color="white", annotation_text="Current IV")
    
    fig2.update_layout(
        template='plotly_dark',
        xaxis_title='Implied Volatility (%)',
        yaxis_title='Option Price ($)',
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("### Profit/Loss at Expiration")

price_range = np.linspace(S * 0.5, S * 1.5, 100)
call_pnl = [max(0, p - K) - call_price for p in price_range]
put_pnl = [max(0, K - p) - put_price for p in price_range]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=price_range, y=call_pnl, name='Long Call', line=dict(color='#00C853', width=2)))
fig3.add_trace(go.Scatter(x=price_range, y=put_pnl, name='Long Put', line=dict(color='#EF5350', width=2)))
fig3.add_hline(y=0, line_dash="dash", line_color="gray")
fig3.add_vline(x=K, line_dash="dash", line_color="gray", annotation_text="Strike")

fig3.update_layout(
    template='plotly_dark',
    xaxis_title='Stock Price at Expiration ($)',
    yaxis_title='Profit/Loss ($)',
    height=350,
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
)
st.plotly_chart(fig3, use_container_width=True)

with st.expander("About the Greeks"):
    st.markdown("""
    - **Delta**: Rate of change of option price with respect to stock price
    - **Gamma**: Rate of change of delta with respect to stock price  
    - **Theta**: Daily time decay (how much value the option loses per day)
    - **Vega**: Sensitivity to 1% change in implied volatility
    - **Rho**: Sensitivity to 1% change in interest rate
    """)
