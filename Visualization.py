import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class MarketVisualizer:

    @staticmethod
    def plot_price_paths(paths: np.ndarray, title: str = "Simulated Price Paths"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        n_show = min(50, len(paths))
        for i in range(n_show):
            ax1.plot(paths[i], alpha=0.3, linewidth=0.8, color='steelblue')

        mean_path = paths.mean(axis=0)
        ax1.plot(mean_path, 'r-', linewidth=3, label='Mean Path', alpha=0.8)

        upper = np.percentile(paths, 95, axis=0)
        lower = np.percentile(paths, 5, axis=0)
        ax1.fill_between(range(len(mean_path)), lower, upper,
                         alpha=0.2, color='red', label='90% Confidence')

        ax1.set_xlabel('Time Period', fontsize=12)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.set_title(f'{title} (showing {n_show} paths)', fontsize=13, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])

        final_prices = paths[:, -1]
        ax2.hist(final_prices, bins=30, edgecolor='black', alpha=0.7, color='steelblue')

        ax2.axvline(final_prices.mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {final_prices.mean():.3f}')
        ax2.axvline(np.median(final_prices), color='orange', linestyle='--',
                    linewidth=2, label=f'Median: {np.median(final_prices):.3f}')

        ax2.set_xlabel('Final Price', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Final Prices', fontsize=13, weight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_kelly_allocations(allocations: dict, bankroll: float):
        non_zero = {k: v for k, v in allocations.items() if v > 0}

        if not non_zero:
            print("No allocations to plot!")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        markets = list(non_zero.keys())
        amounts = list(non_zero.values())

        colors = plt.cm.viridis(np.linspace(0, 1, len(markets)))
        bars = ax1.bar(range(len(markets)), amounts, color=colors,
                       edgecolor='black', linewidth=1.5)

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'${height:,.0f}',
                     ha='center', va='bottom', fontsize=9)

        ax1.set_xticks(range(len(markets)))
        ax1.set_xticklabels(markets, rotation=45, ha='right')
        ax1.set_ylabel('Allocation ($)', fontsize=12)
        ax1.set_title('Kelly Criterion Allocations', fontsize=13, weight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        total_allocated = sum(amounts)
        cash_remaining = bankroll - total_allocated

        pie_data = list(amounts) + [cash_remaining]
        pie_labels = list(markets) + ['Cash Reserve']
        pie_colors = list(colors) + ['lightgray']

        wedges, texts, autotexts = ax2.pie(
            pie_data,
            labels=pie_labels,
            autopct='%1.1f%%',
            colors=pie_colors,
            startangle=90
        )

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
            autotext.set_fontsize(9)

        ax2.set_title(f'Portfolio Allocation\n(${bankroll:,.0f} Total)',
                      fontsize=13, weight='bold')

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_calibration(calibration_df):
        if calibration_df is None or calibration_df.empty:
            print("No calibration data to plot!")
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        x = calibration_df['avg_market_price'].values
        y = calibration_df['actual_outcome_rate'].values
        n_obs = calibration_df['n_observations'].values
        calibrated = calibration_df['calibrated'].values

        colors = ['green' if cal else 'red' for cal in calibrated]

        ax.scatter(x, y, s=n_obs * 20, c=colors, alpha=0.6,
                   edgecolors='black', linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5,
                label='Perfect Calibration')

        for i in range(len(x)):
            ax.annotate(f"n={int(n_obs[i])}",
                        (x[i], y[i]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9)

        ax.set_xlabel('Average Market Price', fontsize=12)
        ax.set_ylabel('Actual Outcome Rate', fontsize=12)
        ax.set_title(
            'Market Calibration Analysis\n(Green=Well Calibrated, Red=Miscalibrated)',
            fontsize=14,
            weight='bold'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        well_cal = sum(calibrated)
        total = len(calibrated)
        textstr = f'Well-Calibrated: {well_cal}/{total}\nSize = # Observations'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', bbox=props)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_arbitrage_opportunities(markets: list, detector):
        arb_results = []
        for market in markets:
            result = detector.detect_simple_arbitrage(market)
            if result:
                arb_results.append(result)

        if not arb_results:
            print("No arbitrage opportunities to plot!")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        names = [r['market'] for r in arb_results]
        returns = [r['guaranteed_return'] for r in arb_results]

        colors = ['green' if r > 1 else 'orange' for r in returns]
        bars = ax.barh(range(len(names)), returns, color=colors,
                       edgecolor='black', linewidth=1.5)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2.,
                    f'{width:.2f}%',
                    ha='left', va='center', fontsize=10, weight='bold')

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Guaranteed Return (%)', fontsize=12)
        ax.set_title('Arbitrage Opportunities', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig


def demo_visualizations():
    print("=" * 70)
    print("VISUALIZATION DEMO")
    print("=" * 70)
    print("\nVisualization module loaded successfully!")
    print("Import this into main.py to create charts.")


if __name__ == "__main__":
    demo_visualizations()
