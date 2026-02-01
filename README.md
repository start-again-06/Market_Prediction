# Prediction Market Visualization Toolkit

## Overview
This module provides a collection of visualization utilities for analyzing prediction markets, trading strategies, and probabilistic forecasts. It is designed to work alongside market simulation, Kelly betting, calibration analysis, and arbitrage detection systems.

The visualizations emphasize interpretability, risk analysis, and decision support, making them suitable for research, experimentation, and educational use.

## Features
- Visualization of simulated price paths with confidence bands  
- Distribution analysis of final market prices  
- Kelly Criterion allocation visualization using bar and pie charts  
- Market calibration analysis with observation-weighted scatter plots  
- Detection and visualization of arbitrage opportunities  
- Clean, publication-ready Matplotlib and Seaborn plots  

## Module Structure
- `visualization.py`

## Main Class
### MarketVisualizer
All visualization functionality is implemented as static methods, allowing easy integration without maintaining internal state.

## Visualizations Included

### Simulated Price Paths
**Method**  
`plot_price_paths(paths: np.ndarray, title: str)`

**Description**
- Plots multiple simulated price trajectories  
- Highlights the mean path  
- Displays 90% confidence bands using percentile ranges  
- Shows the distribution of final prices  

### Kelly Criterion Allocations
**Method**  
`plot_kelly_allocations(allocations: dict, bankroll: float)`

**Description**
- Bar chart of capital allocation per market  
- Pie chart showing portfolio composition  
- Automatically includes unallocated cash as a reserve  

### Market Calibration Analysis
**Method**  
`plot_calibration(calibration_df)`

**Description**
- Compares predicted market probabilities with actual outcomes  
- Marker size reflects number of observations  
- Color-coded calibration quality  
- Includes perfect calibration reference line  

**Required Columns**
- `avg_market_price`  
- `actual_outcome_rate`  
- `n_observations`  
- `calibrated`  

## Arbitrage Opportunity Visualization

### Method
`plot_arbitrage_opportunities(markets: list, detector)`

### Description
- Detects arbitrage opportunities across markets  
- Displays guaranteed returns using a horizontal bar chart  
- Color-coded by profitability  

## Dependencies
- Python 3.8 or higher  
- numpy  
- matplotlib  
- seaborn  

**Optional**
- pandas  

## Usage Example
```python
from visualization import MarketVisualizer

fig = MarketVisualizer.plot_price_paths(simulated_paths)
fig.show()
