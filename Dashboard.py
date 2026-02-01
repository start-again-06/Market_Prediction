import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import Market, ArbitrageDetector, BinomialPriceModel, KellyOptimizer
from backtesting import Backtester, HistoricalDataSimulator, simple_kelly_strategy
from visualizations import MarketVisualizer

st.set_page_config(
    page_title="Prediction Market Trading System",
    page_icon="chart",
    layout="wide"
)

st.title("Prediction Market Trading System")
st.markdown("*Quantitative trading with Bernoulli distributions and linear algebra*")

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Arbitrage Detector", "Price Simulator", "Kelly Optimizer"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This system combines:\n"
    "- Bernoulli/Binomial distributions\n"
    "- Linear algebra\n"
    "- Statistical testing\n"
    "- Portfolio optimization"
)

if page == "Home":
    st.header("Welcome to the Prediction Market Trading System")

    c1, c2, c3 = st.columns(3)
    c1.metric("Components", "5", "modules")
    c2.metric("Algorithms", "4", "strategies")
    c3.metric("Visualizations", "7", "charts")

    st.markdown("---")

    with st.expander("Arbitrage Detector"):
        st.markdown(
            "- Detects risk-free profit opportunities\n"
            "- Uses linear programming\n"
            "- Accounts for transaction costs\n"
            "- YES + NO < 1 implies guaranteed profit"
        )

    with st.expander("Binomial Price Model"):
        st.markdown(
            "- Simulates price evolution\n"
            "- Bernoulli random variables\n"
            "- Monte Carlo simulation\n"
            "- Uncertainty modeling"
        )

    with st.expander("Kelly Optimizer"):
        st.markdown(
            "- Optimal bet sizing\n"
            "- Maximizes long-run growth\n"
            "- Controls over-betting"
        )

elif page == "Arbitrage Detector":
    st.header("Arbitrage Detector")

    c1, c2, c3 = st.columns(3)
    with c1:
        market_name = st.text_input("Market Name", "Presidential Election")
    with c2:
        yes_price = st.number_input("YES Price", 0.0, 1.0, 0.52, 0.01)
    with c3:
        no_price = st.number_input("NO Price", 0.0, 1.0, 0.45, 0.01)

    transaction_cost = st.slider("Transaction Cost (%)", 0.0, 10.0, 2.0, 0.5) / 100

    if st.button("Detect Arbitrage", type="primary"):
        market = Market(market_name, yes_price, no_price)
        detector = ArbitrageDetector(transaction_cost)
        result = detector.detect_simple_arbitrage(market)

        if result:
            st.success("ARBITRAGE FOUND")

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Cost", f"${result['total_cost']:.3f}")
            c2.metric("Net Cost", f"${result['net_cost']:.3f}")
            c3.metric("Guaranteed Return", f"{result['guaranteed_return']:.2f}%")

            st.info(result["strategy"])
        else:
            st.error("No arbitrage opportunity")

elif page == "Price Simulator":
    st.header("Binomial Price Simulator")

    c1, c2, c3 = st.columns(3)
    with c1:
        initial_price = st.slider("Initial Price", 0.1, 0.9, 0.5, 0.05)
    with c2:
        volatility = st.slider("Volatility", 0.05, 0.30, 0.15, 0.01)
    with c3:
        periods = st.slider("Time Periods", 5, 50, 10, 5)

    n_simulations = st.slider("Simulations", 100, 5000, 1000, 100)

    if st.button("Run Simulation", type="primary"):
        model = BinomialPriceModel(initial_price, volatility, periods)
        paths = model.simulate_paths(n_simulations)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{paths[:, -1].mean():.3f}")
        c2.metric("Std", f"{paths[:, -1].std():.3f}")
        c3.metric("Min", f"{paths[:, -1].min():.3f}")
        c4.metric("Max", f"{paths[:, -1].max():.3f}")

        fig = MarketVisualizer.plot_price_paths(paths, "Price Evolution")
        st.pyplot(fig)

elif page == "Kelly Optimizer":
    st.header("Kelly Criterion Optimizer")

    bankroll = st.number_input("Total Bankroll", 1000, 100000, 10000, 1000)
    num_markets = st.number_input("Number of Markets", 1, 10, 3, 1)

    markets = []
    probs = []

    for i in range(num_markets):
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Name", f"Market_{i+1}", key=f"name_{i}")
        with c2:
            price = st.number_input("Price", 0.1, 0.9, 0.5, 0.01, key=f"price_{i}")
        with c3:
            prob = st.number_input("Your Probability", 0.1, 0.9, 0.6, 0.01, key=f"prob_{i}")

        markets.append(Market(name, price, price))
        probs.append(prob)

    if st.button("Optimize Portfolio", type="primary"):
        optimizer = KellyOptimizer(bankroll)
        allocations = optimizer.optimize_portfolio(markets, np.array(probs))

        for name, amount in allocations.items():
            if amount > 0:
                st.metric(name, f"${amount:,.2f}")
