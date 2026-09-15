import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class StrategyParams:
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    z_window: int = 60
    hedge_window: int = None
    hedge_min_periods: int = 120
    max_holding_days: int = None
    use_log_prices: bool = True

    def validate(self):
        if not 0 <= self.exit_z < self.entry_z:
            raise ValueError("need 0 <= exit_z < entry_z")
        if self.stop_z <= self.entry_z:
            raise ValueError("stop_z must sit above entry_z")
        if self.z_window < 5:
            raise ValueError("z_window is too short to estimate a std")


def _rolling_hedge(y, x, window, min_periods):
    if window is None:
        # expanding instead of a single full-sample OLS: a static beta fitted on the
        # whole history leaks the future into every signal
        cov = y.expanding(min_periods).cov(x)
        var = x.expanding(min_periods).var()
        mu_y, mu_x = y.expanding(min_periods).mean(), x.expanding(min_periods).mean()
    else:
        cov = y.rolling(window).cov(x)
        var = x.rolling(window).var()
        mu_y, mu_x = y.rolling(window).mean(), x.rolling(window).mean()

    beta = cov / var.replace(0.0, np.nan)
    return beta, mu_y - beta * mu_x


def build_spread(prices, y, x, params):
    """Spread, hedge ratio and rolling z-score for the pair (y, x)."""
    px = prices[[y, x]].dropna()
    ly, lx = (np.log(px[y]), np.log(px[x])) if params.use_log_prices else (px[y], px[x])

    beta, alpha = _rolling_hedge(ly, lx, params.hedge_window, params.hedge_min_periods)
    spread = ly - (alpha + beta * lx)

    mu = spread.rolling(params.z_window).mean()
    sd = spread.rolling(params.z_window).std(ddof=1)
    z = (spread - mu) / sd.replace(0.0, np.nan)

    out = pd.DataFrame({"spread": spread, "beta": beta, "zscore": z,
                        "mu": mu, "sigma": sd}, index=px.index)
    valid = out["zscore"].notna()
    log.info("spread ready: %d usable bars of %d", int(valid.sum()), len(out))
    return out


def generate_positions(z, params):
    """Map the z-score into a -1/0/+1 position on the spread."""
    params.validate()
    zv = z.to_numpy(dtype=float)
    pos = np.zeros(len(zv))

    state = 0
    held = 0
    for i in range(len(zv)):
        zi = zv[i]
        if np.isnan(zi):
            state, held = 0, 0
            continue

        expired = False
        if state != 0:
            held += 1
            expired = params.max_holding_days is not None and held >= params.max_holding_days
            if expired or abs(zi) >= params.stop_z or abs(zi) <= params.exit_z:
                state, held = 0, 0

        # no new position outside the stop band, it would be closed on the next bar
        if state == 0 and not expired and abs(zi) < params.stop_z:
            if zi >= params.entry_z:
                state = -1
            elif zi <= -params.entry_z:
                state = 1

        pos[i] = state

    return pd.Series(pos, index=z.index, name="position")


def run_strategy(prices, y, x, params):
    df = build_spread(prices, y, x, params)
    df["position"] = generate_positions(df["zscore"], params)

    flips = int((df["position"].diff().fillna(0) != 0).sum())
    log.info("%d position changes, %.1f%% of days in market",
             flips, 100 * (df["position"] != 0).mean())
    return df
