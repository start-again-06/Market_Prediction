import numpy as np
from dataclasses import dataclass
from scipy import stats
from visualizations import MarketVisualizer
import matplotlib.pyplot as plt


@dataclass
class Market:
    name: str
    yes_price: float
    no_price: float

    def __post_init__(self):
        if not (0 <= self.yes_price <= 1 and 0 <= self.no_price <= 1):
            raise ValueError("Market prices must be between 0 and 1")


class ArbitrageDetector:
    def __init__(self, transaction_cost: float = 0.02):
        self.transaction_cost = transaction_cost

    def detect_simple_arbitrage(self, market: Market):
        total_cost = market.yes_price + market.no_price
        net_cost = total_cost * (1 + self.transaction_cost)

        if net_cost < 1.0:
            profit_margin = (1.0 - net_cost) / net_cost
            return {
                "market": market.name,
                "type": "simple_arbitrage",
                "total_cost": total_cost,
                "net_cost": net_cost,
                "profit_margin": profit_margin,
                "guaranteed_return": profit_margin * 100,
            }

        return None


class BinomialPriceModel:
    def __init__(self, initial_price: float, volatility: float, periods: int):
        self.p0 = initial_price
        self.sigma = volatility
        self.periods = periods

    def simulate_paths(self, n_simulations: int = 1000):
        paths = np.zeros((n_simulations, self.periods + 1))
        paths[:, 0] = self.p0

        u = np.exp(self.sigma)
        d = np.exp(-self.sigma)

        for i in range(n_simulations):
            for t in range(1, self.periods + 1):
                step = u if np.random.binomial(1, 0.5) else d
                paths[i, t] = np.clip(paths[i, t - 1] * step, 0.01, 0.99)

        return paths


class KellyOptimizer:
    def __init__(
        self,
        bankroll: float = 10000.0,
        min_edge: float = 0.03,
        max_single_frac: float = 0.10,
        max_total_frac: float = 0.50,
        transaction_cost: float = 0.02,
    ):
        self.bankroll = bankroll
        self.min_edge = min_edge
        self.max_single_frac = max_single_frac
        self.max_total_frac = max_total_frac
        self.transaction_cost = transaction_cost

    def kelly_fraction(self, true_prob: float, market_price: float) -> float:
        if market_price > 0.85:
            return 0.0

        edge = true_prob - market_price
        if edge <= self.min_edge:
            return 0.0

        adj_price = market_price * (1 + self.transaction_cost)
        b = (1 / adj_price) - 1
        p = true_prob
        q = 1 - p

        kelly = (p * b - q) / b
        kelly *= 0.5

        return float(np.clip(kelly, 0, self.max_single_frac))

    def optimize_portfolio(self, markets: list, true_probs: np.ndarray) -> dict:
        allocations = {}
        allocated_frac = 0.0

        for i, market in enumerate(markets):
            if allocated_frac >= self.max_total_frac:
                allocations[market.name] = 0.0
                continue

            frac = self.kelly_fraction(true_probs[i], market.yes_price)
            frac = min(frac, self.max_total_frac - allocated_frac)

            bet = frac * self.bankroll
            allocations[market.name] = bet
            allocated_frac += frac

        return allocations


class MarketInefficiencyAnalyzer:
    def __init__(self):
        self.history = []

    def add_market_observation(self, market_price: float, outcome: int):
        self.history.append((market_price, outcome))

    def compute_brier_score(self):
        if not self.history:
            return None
        return np.mean([(p - o) ** 2 for p, o in self.history])

    def binomial_calibration_test(self, price_bins: int = 10):
        if len(self.history) < 10:
            return None

        import pandas as pd

        df = pd.DataFrame(self.history, columns=["price", "outcome"])
        df["price_bin"] = pd.cut(df["price"], bins=price_bins)

        results = []

        for bin_name, group in df.groupby("price_bin", observed=True):
            if len(group) < 3:
                continue

            avg_price = group["price"].mean()
            actual_rate = group["outcome"].mean()
            n = len(group)
            successes = group["outcome"].sum()

            p_value = stats.binomtest(successes, n, avg_price).pvalue

            results.append({
                "price_range": str(bin_name),
                "avg_market_price": avg_price,
                "actual_outcome_rate": actual_rate,
                "n_observations": n,
                "p_value": p_value,
                "calibrated": p_value > 0.05,
            })

        return pd.DataFrame(results)


if __name__ == "__main__":
    detector = ArbitrageDetector(transaction_cost=0.02)

    arb_market = Market("Election A", yes_price=0.52, no_price=0.45)
    no_arb_market = Market("Election B", yes_price=0.60, no_price=0.42)

    model = BinomialPriceModel(0.50, 0.15, 10)
    paths = model.simulate_paths(1000)

    optimizer = KellyOptimizer(bankroll=10000)

    test_markets = [
        Market("Pennsylvania", 0.52, 0.46),
        Market("Michigan", 0.54, 0.44),
        Market("Wisconsin", 0.51, 0.47),
        Market("Arizona", 0.48, 0.50),
    ]

    our_estimates = np.array([0.60, 0.58, 0.55, 0.45])
    allocations = optimizer.optimize_portfolio(test_markets, our_estimates)

    analyzer = MarketInefficiencyAnalyzer()
    np.random.seed(42)

    for _ in range(150):
        price = np.random.uniform(0.3, 0.8)
        true_prob = price * 0.95
        outcome = np.random.binomial(1, true_prob)
        analyzer.add_market_observation(price, outcome)

    calibration = analyzer.binomial_calibration_test(price_bins=5)

    fig1 = MarketVisualizer.plot_price_paths(paths, "Market Price Evolution")
    fig2 = MarketVisualizer.plot_kelly_allocations(allocations, optimizer.bankroll)
    fig3 = MarketVisualizer.plot_calibration(calibration)
    fig4 = MarketVisualizer.plot_arbitrage_opportunities(
        [arb_market, no_arb_market] + test_markets, detector
    )

    plt.show()
