
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# =================== SETTINGS ===================

TICKER: str = "NVD"     
HORIZON: int = 30           
SIMS: int = 200000         
MODEL: str = "bootstrap"   
BOOTSTRAP_BLOCK: int = 10 
LOOKBACK_YEARS: int = 1     
FLAT_PCT: float = 0.03     
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
        raise RuntimeError(f"Сannot get any data about {ticker}")
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


def main():
    hist = fetch_history(TICKER, lookback_years=LOOKBACK_YEARS)
    S0 = float(hist["Close"].iloc[-1])
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
        raise ValueError('MODEL должен быть "gbm" или "bootstrap"')

   
if __name__ == "__main__":
    main()
