# Monte Carlo Stock Price Simulator

This project simulates possible future stock prices using **Monte Carlo method**, based on historical Yahoo Finance data.  
It supports two models:
- **Geometric Brownian Motion (GBM)** – parametric stochastic simulation  
- **Bootstrap** – non-parametric block resampling of historical log-returns  

Both methods estimate short-term price distributions, generate fan charts and boxplots, and summarize probabilities of growth, stability, and decline.

---

## Features 

- Automatic historical data download from **Yahoo Finance**
- Simulation of **hundreds of thousands** of scenarios
- Two modes:  
  - **GBM** – based on drift (μ) and volatility (σ)  
  - **Bootstrap** – based on real resampled log-returns  
- Visual outputs:  
  - **Fan chart** of price quantiles  
  - **Boxplot** + histogram of final prices  
- Quantile summary (1%, 5%, 25%, 50%, 75%, 95%, 99%)  
- CSV export with quantiles  
- Scenario classification:  
  -  Growth (> +5%)  
  -  Flat (±5%)  
  -  Decline (< –5%)  


## Project Structure

Monte_Carlo/
│
├── main.py                 # Main simulation script
├── outputs/                # Results directory
│   ├── NVDA_quantiles.csv  # example
│   ├── NVDA_fan_chart.png
│   └── NVDA_boxplot.png
└── requirements.txt


## Requirements

All requirements you can download with requirements.txt

## Configuration 
You can also change parametrs as you want

TICKER = "NVDA"           # Stock symbol
HORIZON = 5               # Forecast horizon in business days
SIMS = 200000             # Number of Monte Carlo simulations
MODEL = "bootstrap"       # "gbm" or "bootstrap"
BOOTSTRAP_BLOCK = 5       # Block length for bootstrap sampling
LOOKBACK_YEARS = 2        # Historical window
FLAT_PCT = 0.05           # ±5% flat threshold
OUT_DIR = "outputs"       # Output folder
SEED = 42                 # Random seed

## Example output 

```bash

============================================================
Ticker: NVDA
Initial price: $188.32
Model: BOOTSTRAP
Simulations: 200,000
Horizon: 5 business days
============================================================

 Scenario distribution (after 5 days):
   Growth (>+5.0%):     42.13% (84,260 scenarios)
   Flat (±5.0%):        27.45% (54,900 scenarios)
   Decline (<-5.0%):    30.42% (60,840 scenarios)

 Final price quantiles:
  1%:  $175.21
  5%:  $181.92
  25%: $189.42
  50%: $195.80 (median)
  75%: $202.15
  95%: $213.52
  99%: $223.97

 Results saved:
  CSV: outputs/NVDA_quantiles.csv
  Fan Chart: outputs/NVDA_fan_chart.png
  Boxplot: outputs/NVDA_boxplot.png


