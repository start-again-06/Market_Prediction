import numpy as np
from typing import List
from dataclasses import dataclass
import matplotlib.pyplot as plt

class Trade:
    market_name: str
    entry_time: int
    entry_price: float
    position_size: float
    exit_time: int = None
    outcome: int = None
    pnl: float = None

    def close_trade(self, outcome: int, exit_time: int):
        self.outcome = outcome
        self.exit_time = exit_time

        if outcome == 1:
            self.pnl = (self.position_size / self.entry_price) - self.position_size
        else:
            self.pnl = -self.position_size

class HistoricalMarket:
    name: str
    prices: np.ndarray
    outcome: int


class HistoricalDataSimulator:
    def __init__(self, n_markets: int = 10, n_periods: int = 30, seed: int = 42):
        self.n_markets = n_markets
        self.n_periods = n_periods
        np.random.seed(seed)

    def generate_market_data(self) -> List[HistoricalMarket]:
        markets = []

        for i in range(self.n_markets):
            start_price = np.random.uniform(0.3, 0.7)
            true_prob = np.clip(start_price + np.random.normal(0, 0.1), 0.2, 0.8)

            prices = np.zeros(self.n_periods)
            prices[0] = start_price

            for t in range(1, self.n_periods):
                drift = (true_prob - prices[t - 1]) * 0.1
                noise = np.random.normal(0, 0.03)
                prices[t] = np.clip(prices[t - 1] + drift + noise, 0.1, 0.9)

            outcome = np.random.binomial(1, true_prob)

            markets.append(
                HistoricalMarket(
                    name=f"Market_{i + 1}",
                    prices=prices,
                    outcome=outcome
                )
            )

        return markets


class Backtester:
    def __init__(self, initial_bankroll: float = 10000):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.trades: List[Trade] = []

    def run_strategy(self, markets: List[HistoricalMarket],
                     strategy_func, entry_time: int = 5):
        print(f"Starting Bankroll: ${self.bankroll:,.0f}")
        print(f"Markets: {len(markets)}")
        print("=" * 60)

        for market in markets:
            price = market.prices[entry_time]

            bet_size = strategy_func(
                market=market,
                time=entry_time,
                price=price,
                bankroll=self.bankroll
            )

            if bet_size > 0 and bet_size <= self.bankroll:
                trade = Trade(
                    market_name=market.name,
                    entry_time=entry_time,
                    entry_price=price,
                    position_size=bet_size
                )
                self.bankroll -= bet_size
                self.trades.append(trade)

        for trade in self.trades:
            market = next(m for m in markets if m.name == trade.market_name)
            trade.close_trade(market.outcome, len(market.prices) - 1)
            self.bankroll += trade.position_size + trade.pnl

        self.display_results()

    def display_results(self):
        total_return = self.bankroll - self.initial_bankroll
        return_pct = (total_return / self.initial_bankroll) * 100

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        print("\nRESULTS")
        print("=" * 60)
        print(f"Final Bankroll: ${self.bankroll:,.2f}")
        print(f"Total Return:   ${total_return:,.2f} ({return_pct:+.2f}%)")
        print(f"Trades:        {len(self.trades)}")
        print(f"Win Rate:      {(len(wins) / len(self.trades) * 100) if self.trades else 0:.1f}%")

    def plot_results(self):
        pnls = [t.pnl for t in self.trades]
        colors = ['green' if p > 0 else 'red' for p in pnls]

        plt.figure(figsize=(10, 4))
        plt.bar(range(len(pnls)), pnls, color=colors)
        plt.axhline(0)
        plt.title("Trade P&L Distribution")
        plt.xlabel("Trade #")
        plt.ylabel("P&L ($)")
        plt.tight_layout()
        return plt.gcf()


def simple_kelly_strategy(market, time, price, bankroll):
    future_prices = market.prices[time:]
    estimated_prob = np.mean(future_prices)

    edge = estimated_prob - price

    if edge <= 0.03:
        return 0

    kelly_fraction = edge / price
    kelly_fraction = np.clip(kelly_fraction, 0, 0.1)

    if price > 0.85:
        return 0

    return bankroll * kelly_fraction


def test_backtesting():
    simulator = HistoricalDataSimulator(n_markets=20, n_periods=30)
    markets = simulator.generate_market_data()

    backtester = Backtester(initial_bankroll=10000)
    backtester.run_strategy(
        markets,
        simple_kelly_strategy,
        entry_time=10
    )

    fig = backtester.plot_results()
    plt.savefig("backtest_results.png", dpi=150)
    print("Saved: backtest_results.png")

    return backtester


if __name__ == "__main__":
    test_backtesting()
