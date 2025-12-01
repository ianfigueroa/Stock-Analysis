"""
Black-Scholes Options Calculator
Interactive tool for pricing European options and visualizing sensitivities.
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="Options Pricing", page_icon=":bar_chart:", layout="wide")


def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option price and Greeks using Black-Scholes model.
    
    Parameters:
    - S: Current stock price
    - K: Strike price
    - T: Time to expiration (in years)
    - r: Risk-free interest rate (annualized)
    - sigma: Volatility (annualized)
    - option_type: 'call' or 'put'
    
    Returns tuple of (price, delta, gamma, theta, vega, rho)
    """
    # Handle edge cases
    if T <= 0 or sigma <= 0:
        intrinsic = max(0, S - K) if option_type == 'call' else max(0, K - S)
        return intrinsic, 0, 0, 0, 0, 0
    
    # Calculate d1 and d2 (core of Black-Scholes)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Price and delta depend on option type
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        # Theta: time decay per day
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    # Greeks that are the same for calls and puts
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))  # rate of change of delta
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # sensitivity to 1% vol change
    rho = K * T * np.exp(-r * T) * (norm.cdf(d2) if option_type == 'call' else -norm.cdf(-d2)) / 100
    
    return price, delta, gamma, theta, vega, rho


# Page title
st.title("Black-Scholes Options Calculator")

# Sidebar inputs
st.sidebar.header("Parameters")
S = st.sidebar.number_input("Stock Price ($)", min_value=1.0, value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price ($)", min_value=1.0, value=100.0, step=1.0)
T = st.sidebar.slider("Days to Expiration", min_value=1, max_value=730, value=30) / 365
r = st.sidebar.slider("Risk-Free Rate (%)", min_value=0.0, max_value=15.0, value=5.0) / 100
sigma = st.sidebar.slider("Implied Volatility (%)", min_value=1.0, max_value=150.0, value=25.0) / 100

# Calculate prices and Greeks for both calls and puts
call_price, call_delta, call_gamma, call_theta, call_vega, call_rho = black_scholes(S, K, T, r, sigma, 'call')
put_price, put_delta, put_gamma, put_theta, put_vega, put_rho = black_scholes(S, K, T, r, sigma, 'put')

# Display option prices
st.markdown("### Option Prices")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.metric("Call Price", "${:.2f}".format(call_price))
with col2:
    st.metric("Put Price", "${:.2f}".format(put_price))
with col3:
    # Put-call parity: C - P = S - K*e^(-rT)
    # If this doesn't hold, there's an arbitrage opportunity
    parity_diff = call_price - put_price - S + K * np.exp(-r * T)
    st.metric("Put-Call Parity", "${:.4f}".format(abs(parity_diff)), 
              "Valid" if abs(parity_diff) < 0.01 else "Check inputs")

st.markdown("---")

# Display Greeks
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

# Heatmaps showing how price changes with spot and vol
st.markdown("### Price Sensitivity Heatmaps")
st.write("See how option prices change with spot price and volatility.")

with st.expander("How to read the heatmaps"):
    st.markdown("""
    The heatmaps show option prices across different combinations of **spot price** (x-axis) 
    and **implied volatility** (y-axis).
    
    - **Brighter colors** = higher option prices
    - **Darker colors** = lower option prices
    - **Moving right** = stock price increases (calls get more valuable, puts less)
    - **Moving up** = volatility increases (both calls and puts get more valuable)
    
    Use these to visualize how your option's value might change if the stock moves 
    or if implied volatility shifts after earnings, news, etc.
    """)

# Create ranges for the heatmap axes (larger grid for more detail)
spot_range = np.linspace(S * 0.7, S * 1.3, 15)
vol_range = np.linspace(max(0.05, sigma * 0.4), min(1.5, sigma * 1.6), 15)

# Calculate option prices for each combination
call_matrix = np.zeros((len(vol_range), len(spot_range)))
put_matrix = np.zeros((len(vol_range), len(spot_range)))

for i, v in enumerate(vol_range):
    for j, s in enumerate(spot_range):
        call_matrix[i, j] = black_scholes(s, K, T, r, v, 'call')[0]
        put_matrix[i, j] = black_scholes(s, K, T, r, v, 'put')[0]

# Display heatmaps stacked (full width for bigger charts)
fig_call = go.Figure(data=go.Heatmap(
    z=call_matrix,
    x=["${:.0f}".format(s) for s in spot_range],
    y=["{:.0%}".format(v) for v in vol_range],
    colorscale='Viridis',
    text=np.round(call_matrix, 2),
    texttemplate="%{text}",
    textfont={"size": 10},
    hovertemplate="Spot: %{x}<br>Vol: %{y}<br>Price: $%{z:.2f}<extra></extra>"
))
fig_call.update_layout(
    title="Call Price Heatmap",
    xaxis_title="Spot Price",
    yaxis_title="Volatility",
    template='plotly_dark',
    height=500
)
st.plotly_chart(fig_call, use_container_width=True)

# Put heatmap below call (full width for bigger display)
fig_put = go.Figure(data=go.Heatmap(
    z=put_matrix,
    x=["${:.0f}".format(s) for s in spot_range],
    y=["{:.0%}".format(v) for v in vol_range],
    colorscale='Viridis',
    text=np.round(put_matrix, 2),
    texttemplate="%{text}",
    textfont={"size": 10},
    hovertemplate="Spot: %{x}<br>Vol: %{y}<br>Price: $%{z:.2f}<extra></extra>"
))
fig_put.update_layout(
    title="Put Price Heatmap",
    xaxis_title="Spot Price",
    yaxis_title="Volatility",
    template='plotly_dark',
    height=500
)
st.plotly_chart(fig_put, use_container_width=True)

st.markdown("---")

# P&L diagram at expiration
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

# Educational section
with st.expander("About the Greeks"):
    st.markdown("""
    **Delta** - How much the option price moves when the stock moves $1. 
    Calls have positive delta (0 to 1), puts have negative (-1 to 0).
    
    **Gamma** - How fast delta changes. High gamma means delta is unstable.
    Gamma is highest for at-the-money options near expiration.
    
    **Theta** - Time decay. How much value the option loses each day.
    Options are wasting assets - theta is almost always negative.
    
    **Vega** - Sensitivity to volatility. How much the price changes 
    when IV moves 1%. Higher for longer-dated options.
    
    **Rho** - Sensitivity to interest rates. Usually the least important 
    Greek for short-term options.
    """)
