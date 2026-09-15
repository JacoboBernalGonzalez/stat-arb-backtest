import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    capital: float = 100_000.0
    cost_bps: float = 1.0
    slippage_bps: float = 0.5
    ann_factor: int = 252


@dataclass
class BacktestResult:
    returns: pd.Series
    equity: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict


def _leg_weights(position, beta):
    """Dollar weights per leg, scaled so gross exposure is 1 while in a trade."""
    gross = 1.0 + beta.abs()
    w_y = position / gross
    w_x = -position * beta / gross
    return w_y.fillna(0.0), w_x.fillna(0.0)


def _extract_trades(active, net, z):
    arr = active.to_numpy()
    idx = active.index
    rows = []

    i, n = 0, len(arr)
    while i < n:
        if arr[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and arr[j + 1] == arr[i]:
            j += 1

        chunk = net.iloc[i:j + 1]
        rows.append({
            "entry_date": idx[i],
            "exit_date": idx[j],
            "side": "long" if arr[i] > 0 else "short",
            "days": j - i + 1,
            "entry_z": float(z.iloc[i - 1]) if i > 0 else np.nan,
            "exit_z": float(z.iloc[j]),
            "pnl": float((1.0 + chunk).prod() - 1.0),
        })
        i = j + 1

    cols = ["entry_date", "exit_date", "side", "days", "entry_z", "exit_z", "pnl"]
    return pd.DataFrame(rows, columns=cols)


def compute_metrics(rets, trades, ann=252):
    eq = (1.0 + rets).cumprod()
    sd = rets.std(ddof=1)
    downside = rets[rets < 0].std(ddof=1)
    years = len(rets) / ann

    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min()) if len(dd) else np.nan
    final = float(eq.iloc[-1]) if len(eq) else np.nan

    m = {
        "total_return": final - 1.0,
        "cagr": final ** (1 / years) - 1.0 if years > 0 and final > 0 else np.nan,
        "ann_vol": float(sd * np.sqrt(ann)) if sd > 0 else 0.0,
        "sharpe": float(rets.mean() / sd * np.sqrt(ann)) if sd > 0 else np.nan,
        "sortino": float(rets.mean() / downside * np.sqrt(ann)) if downside > 0 else np.nan,
        "max_drawdown": max_dd,
        "calmar": float(np.nan),
        "n_trades": int(len(trades)),
        "win_rate": np.nan,
        "profit_factor": np.nan,
        "avg_trade": np.nan,
        "avg_holding_days": np.nan,
        "exposure": float((rets != 0).mean()),
    }
    # calmar is meaningless without a drawdown to divide by
    if max_dd and max_dd < 0 and not np.isnan(m["cagr"]):
        m["calmar"] = m["cagr"] / abs(max_dd)

    if len(trades):
        wins = trades["pnl"] > 0
        gross_win = float(trades.loc[wins, "pnl"].sum())
        gross_loss = float(-trades.loc[~wins, "pnl"].sum())
        m["win_rate"] = float(wins.mean())
        m["profit_factor"] = gross_win / gross_loss if gross_loss > 0 else np.inf
        m["avg_trade"] = float(trades["pnl"].mean())
        m["avg_holding_days"] = float(trades["days"].mean())

    return m


def run_backtest(prices, signals, y, x, cfg=None):
    """Vectorised P&L for a pairs position, net of round-trip costs."""
    cfg = cfg or BacktestConfig()
    df = signals.join(prices[[y, x]], how="inner").dropna(subset=["position", "beta"])

    r_y = df[y].pct_change().fillna(0.0)
    r_x = df[x].pct_change().fillna(0.0)
    w_y, w_x = _leg_weights(df["position"], df["beta"])

    # signals are generated on the close of t, so they only earn the return of t+1
    gross = w_y.shift(1).fillna(0.0) * r_y + w_x.shift(1).fillna(0.0) * r_x

    turnover = w_y.diff().abs().fillna(w_y.abs()) + w_x.diff().abs().fillna(w_x.abs())
    costs = turnover * (cfg.cost_bps + cfg.slippage_bps) / 1e4
    net = (gross - costs).rename("returns")

    active = df["position"].shift(1).fillna(0.0)
    trades = _extract_trades(active, net, df["zscore"])
    metrics = compute_metrics(net, trades, ann=cfg.ann_factor)

    equity = (cfg.capital * (1.0 + net).cumprod()).rename("equity")
    log.info("backtest done: %d trades, sharpe %.2f, maxDD %.1f%%",
             metrics["n_trades"], metrics["sharpe"], 100 * metrics["max_drawdown"])

    weights = pd.DataFrame({f"w_{y}": w_y, f"w_{x}": w_x, "turnover": turnover,
                            "cost": costs, "gross_return": gross})
    return BacktestResult(returns=net, equity=equity, weights=weights,
                          trades=trades, metrics=metrics)


def format_metrics(metrics):
    pct = {"total_return", "cagr", "ann_vol", "max_drawdown", "win_rate",
           "avg_trade", "exposure"}
    lines = []
    for k, v in metrics.items():
        if isinstance(v, float) and np.isnan(v):
            val = "n/a"
        elif k in pct:
            val = f"{100 * v:8.2f}%"
        elif isinstance(v, float):
            val = f"{v:9.2f}"
        else:
            val = f"{v:9d}"
        lines.append(f"  {k:<18}{val}")
    return "\n".join(lines)
