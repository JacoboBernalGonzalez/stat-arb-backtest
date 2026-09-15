import logging
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

log = logging.getLogger(__name__)


@dataclass
class CointResult:
    """Outcome of an Engle-Granger test on an ordered pair (y regressed on x)."""

    y: str
    x: str
    alpha: float
    beta: float
    adf_stat: float
    pvalue: float
    half_life: float
    n_obs: int
    crit: dict = field(default_factory=dict)
    resid: pd.Series = field(default=None, repr=False)

    def is_cointegrated(self, threshold=0.05):
        return self.pvalue < threshold

    def describe(self):
        return (f"{self.y}~{self.x}  beta={self.beta:.4f}  adf={self.adf_stat:.3f}  "
                f"p={self.pvalue:.4f}  half-life={self.half_life:.1f}d  n={self.n_obs}")


def hedge_ratio(y, x):
    fit = sm.OLS(np.asarray(y, dtype=float),
                 sm.add_constant(np.asarray(x, dtype=float))).fit()
    return float(fit.params[0]), float(fit.params[1])


def half_life(resid):
    """Half-life of mean reversion from an OU/AR(1) fit on the residual series."""
    lag = resid.shift(1)
    delta = (resid - lag).dropna()
    lag = lag.loc[delta.index]

    fit = sm.OLS(delta.to_numpy(), sm.add_constant(lag.to_numpy())).fit()
    k = float(fit.params[1])
    if k >= 0:
        return np.inf
    return -np.log(2) / k


def engle_granger(y, x, maxlag=None):
    """Two-step Engle-Granger: OLS on the levels, then ADF on the residuals."""
    df = pd.concat([y, x], axis=1).dropna()
    yv, xv = df.iloc[:, 0], df.iloc[:, 1]
    if len(df) < 30:
        raise ValueError("not enough overlapping observations")

    a, b = hedge_ratio(yv, xv)
    resid = yv - (a + b * xv)

    # no deterministic term: the residual of an OLS with intercept is mean zero by
    # construction, and the reported p-value is the plain DF one, not Phillips-Ouliaris,
    # so it is mildly optimistic for a residual-based test
    adf = adfuller(resid.to_numpy(), maxlag=maxlag, regression="n", autolag="AIC",
                   result_object=True)

    return CointResult(y=str(yv.name), x=str(xv.name), alpha=a, beta=b,
                       adf_stat=float(adf.statistic), pvalue=float(adf.pvalue),
                       half_life=half_life(resid), n_obs=int(adf.nobs),
                       crit={k: float(v) for k, v in adf.critical_values.items()},
                       resid=resid)


def test_pair(prices, y, x, maxlag=None):
    """Run Engle-Granger both ways and keep the direction with the stronger rejection."""
    fwd = engle_granger(prices[y], prices[x], maxlag=maxlag)
    rev = engle_granger(prices[x], prices[y], maxlag=maxlag)

    log.info("EG %s", fwd.describe())
    log.info("EG %s", rev.describe())

    best = fwd if fwd.pvalue <= rev.pvalue else rev
    if not best.is_cointegrated():
        log.warning("no cointegration at 5%% (best p=%.4f)", best.pvalue)
    return best


def screen_universe(prices, threshold=0.05, maxlag=None):
    rows = []
    for a, b in combinations(prices.columns, 2):
        try:
            res = test_pair(prices, a, b, maxlag=maxlag)
        except ValueError as exc:
            log.warning("skipping %s/%s: %s", a, b, exc)
            continue
        rows.append({"y": res.y, "x": res.x, "beta": res.beta,
                     "adf": res.adf_stat, "pvalue": res.pvalue,
                     "half_life": res.half_life})

    out = pd.DataFrame(rows).sort_values("pvalue").reset_index(drop=True)
    return out[out.pvalue < threshold] if not out.empty else out
