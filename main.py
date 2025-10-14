import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# =================== SETTINGS ===================

TICKER: str = "NVDA"     
HORIZON: int = 5           
SIMS: int = 200000         
MODEL: str = "bootstrap"   
BOOTSTRAP_BLOCK: int = 5
LOOKBACK_YEARS: int = 2     
FLAT_PCT: float = 0.05  # 5% flat zone     
OUT_DIR: str = "outputs"    
SEED: int = 42

# ===============================================================


def fetch_history(ticker: str, lookback_years: int = 3) -> pd.DataFrame:
    end = datetime.today().date()
    start = end - timedelta(days=int(365.25 * lookback_years))
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        progress=False,
        auto_adjust=True,  
    )
    if df.empty:
        raise RuntimeError(f"Cannot fetch any data for {ticker}")
    df = df[["Close"]].dropna()
    df["log_ret"] = np.log(df["Close"]).diff()
    df = df.dropna()
    return df


def estimate_params(log_returns: pd.Series) -> tuple[float, float]:
    mu = float(log_returns.mean())
    sigma = float(log_returns.std(ddof=1))
    return mu, sigma


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    horizon: int,
    sims: int,
    seed: Optional[int] = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=mu, scale=sigma, size=(horizon, sims))
    log_paths = shocks.cumsum(axis=0)
    paths = S0 * np.exp(log_paths)
    paths = np.vstack([np.full((1, sims), S0), paths])
    return paths


def simulate_bootstrap(
    S0: float,
    log_returns: np.ndarray,
    horizon: int,
    sims: int,
    block: int = 1,
    seed: Optional[int] = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(log_returns)
    if block <= 1:
        shocks = rng.choice(log_returns, size=(horizon, sims), replace=True)
    else:
        shocks = np.zeros((horizon, sims), dtype=float)
        for s in range(sims):
            pos = 0
            while pos < horizon:
                start_idx = rng.integers(0, max(1, n - block + 1))
                take = min(block, horizon - pos)
                shocks[pos:pos + take, s] = log_returns[start_idx:start_idx + take]
                pos += take
    log_paths = shocks.cumsum(axis=0)
    paths = S0 * np.exp(log_paths)
    paths = np.vstack([np.full((1, sims), S0), paths])
    return paths


def make_business_days(horizon: int) -> pd.DatetimeIndex:
    start = pd.Timestamp.today().normalize()
    return pd.bdate_range(start=start, periods=horizon + 1)


def summarize_quantiles(paths: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    qs = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    qvals = np.quantile(paths, q=qs, axis=1).T
    cols = [f"q{int(q*100):02d}" for q in qs]
    out = pd.DataFrame(qvals, index=dates, columns=cols)
    out.index.name = "date"
    return out


def plot_fan_chart(df_q: pd.DataFrame, ticker: str, out_path: str):
    plt.figure(figsize=(9, 5))
    x = df_q.index
    for lo, hi, a in [("q01", "q99", 0.15), ("q05", "q95", 0.25), ("q25", "q75", 0.35)]:
        plt.fill_between(x, df_q[lo], df_q[hi], alpha=a, step="pre")
    plt.plot(x, df_q["q50"], linewidth=2, label="Median")
    plt.title(f"{ticker} Monte Carlo Fan Chart")
    plt.xlabel("Business days")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_boxplot(paths: np.ndarray, S0: float, ticker: str, out_path: str):
    """Creates a boxplot of the final price distribution"""
    final_prices = paths[-1, :]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot of final prices
    bp = ax1.boxplot(final_prices, vert=True, patch_artist=True, 
                     widths=0.5, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8),
                     medianprops=dict(color='darkblue', linewidth=2),
                     boxprops=dict(facecolor='lightblue', alpha=0.7))
    
    ax1.axhline(y=S0, color='green', linestyle='--', linewidth=2, label=f'Initial price: ${S0:.2f}')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'{ticker} - Boxplot of Final Prices', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add annotations
    q1, median, q3 = np.percentile(final_prices, [25, 50, 75])
    mean_val = final_prices.mean()
    ax1.text(1.15, q1, f'Q1: ${q1:.2f}', fontsize=10)
    ax1.text(1.15, median, f'Median: ${median:.2f}', fontsize=10, color='darkblue', fontweight='bold')
    ax1.text(1.15, q3, f'Q3: ${q3:.2f}', fontsize=10)
    ax1.text(1.15, mean_val, f'Mean: ${mean_val:.2f}', fontsize=10, color='red', fontweight='bold')
    
    # Histogram of percentage changes
    pct_changes = ((final_prices - S0) / S0) * 100
    ax2.hist(pct_changes, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(x=0, color='green', linestyle='--', linewidth=2, label='No change')
    ax2.axvline(x=pct_changes.mean(), color='red', linestyle='-', linewidth=2, label=f'Mean: {pct_changes.mean():.2f}%')
    ax2.axvline(x=np.median(pct_changes), color='darkblue', linestyle='-', linewidth=2, label=f'Median: {np.median(pct_changes):.2f}%')
    
    ax2.set_xlabel('Price change (%)', fontsize=12)
    ax2.set_ylabel('Number of simulations', fontsize=12)
    ax2.set_title(f'{ticker} - Return Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    hist = fetch_history(TICKER, lookback_years=LOOKBACK_YEARS)
    S0 = hist["Close"].iloc[-1].item()
    mu, sigma = estimate_params(hist["log_ret"])

    if MODEL.lower() == "gbm":
        paths = simulate_gbm(S0, mu, sigma, horizon=HORIZON, sims=SIMS, seed=SEED)
        model_info = {"model": "gbm", "mu": mu, "sigma": sigma}
    elif MODEL.lower() == "bootstrap":
        paths = simulate_bootstrap(
            S0,
            hist["log_ret"].values,
            horizon=HORIZON,
            sims=SIMS,
            block=BOOTSTRAP_BLOCK,
            seed=SEED,
        )
        model_info = {"model": "bootstrap", "block": BOOTSTRAP_BLOCK}
    else:
        raise ValueError('MODEL must be "gbm" or "bootstrap"')

    # Generate dates and quantiles
    dates = make_business_days(HORIZON)
    df_q = summarize_quantiles(paths, dates)
    
    # Calculate percentage of growth / decline / flat
    final_prices = paths[-1, :]
    pct_changes = (final_prices - S0) / S0
    
    up_count = np.sum(pct_changes > FLAT_PCT)
    down_count = np.sum(pct_changes < -FLAT_PCT)
    flat_count = np.sum(np.abs(pct_changes) <= FLAT_PCT)
    
    pct_up = (up_count / SIMS) * 100
    pct_down = (down_count / SIMS) * 100
    pct_flat = (flat_count / SIMS) * 100
    
    # Save results
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"{TICKER}_quantiles.csv")
    df_q.to_csv(csv_path)
    
    # Create charts
    chart_path = os.path.join(OUT_DIR, f"{TICKER}_fan_chart.png")
    plot_fan_chart(df_q, TICKER, chart_path)
    
    boxplot_path = os.path.join(OUT_DIR, f"{TICKER}_boxplot.png")
    plot_boxplot(paths, S0, TICKER, boxplot_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Ticker: {TICKER}")
    print(f"Initial price: ${S0:.2f}")
    print(f"Model: {model_info['model'].upper()}")
    print(f"Simulations: {SIMS:,}")
    print(f"Horizon: {HORIZON} business days")
    print(f"{'='*60}")
    
    print(f"\n Scenario distribution (after {HORIZON} days):")
    print(f"   Growth (>{FLAT_PCT*100:+.1f}%):     {pct_up:6.2f}% ({up_count:,} scenarios)")
    print(f"   Flat (±{FLAT_PCT*100:.1f}%):      {pct_flat:6.2f}% ({flat_count:,} scenarios)")
    print(f"   Decline (<{-FLAT_PCT*100:+.1f}%): {pct_down:6.2f}% ({down_count:,} scenarios)")
    
    print(f"\n Final price quantiles:")
    final_row = df_q.iloc[-1]
    print(f"  1%:  ${final_row['q01']:.2f}")
    print(f"  5%:  ${final_row['q05']:.2f}")
    print(f"  25%: ${final_row['q25']:.2f}")
    print(f"  50%: ${final_row['q50']:.2f} (median)")
    print(f"  75%: ${final_row['q75']:.2f}")
    print(f"  95%: ${final_row['q95']:.2f}")
    print(f"  99%: ${final_row['q99']:.2f}")
    
    print(f"\n Results saved:")
    print(f"  CSV: {csv_path}")
    print(f"  Fan Chart: {chart_path}")
    print(f"  Boxplot: {boxplot_path}\n")


if __name__ == "__main__":
    main()
