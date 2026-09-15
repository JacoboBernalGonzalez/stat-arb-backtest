import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"


def fetch_prices(tickers, start, end, interval="1d", use_cache=True):
    """Download adjusted closes for a list of tickers into a single DataFrame."""
    tickers = list(dict.fromkeys(tickers))
    if len(tickers) < 2:
        raise ValueError("need at least two tickers")

    cache_file = CACHE_DIR / f"{'-'.join(sorted(tickers))}_{start}_{end}_{interval}.parquet"
    if use_cache and cache_file.exists():
        log.info("using cached prices (%s)", cache_file.name)
        df = pd.read_parquet(cache_file)
        return df.reindex(columns=tickers)

    log.info("downloading %s | %s -> %s", ", ".join(tickers), start, end)
    raw = yf.download(tickers, start=start, end=end, interval=interval,
                      auto_adjust=True, progress=False, group_by="column")

    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned nothing for {tickers}")

    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"].copy()
    else:
        px = raw[["Close"]].copy()
        px.columns = [tickers[0]]

    empty = [t for t in tickers if t not in px.columns or px[t].isna().all()]
    if empty:
        raise ValueError(f"no price history for {empty}")

    px = px.reindex(columns=tickers)
    idx = pd.to_datetime(px.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    px.index = idx
    px = px.sort_index()

    # holidays differ between exchanges, so patch small gaps before dropping the rest
    px = px.ffill(limit=3).dropna()
    if px.empty:
        raise ValueError("no overlapping history between the requested tickers")

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        px.to_parquet(cache_file)

    log.info("%d rows, %s -> %s", len(px), px.index[0].date(), px.index[-1].date())
    return px


def align_pair(prices, y, x):
    df = prices[[y, x]].dropna()
    if df.empty:
        raise ValueError(f"no overlapping data for {y}/{x}")
    return df
