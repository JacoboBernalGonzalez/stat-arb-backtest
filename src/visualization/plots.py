import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

log = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
PLANE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
RED = "#e34948"
GREEN = "#008300"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})


def _clean(ax, ygrid=True):
    ax.grid(axis="y" if ygrid else "both", alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8, labelsize=8)
    return ax


def _pct(v, nd=1):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.{nd}f}%"


def _num(v, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{nd}f}" if np.isfinite(v) else "inf"


def plot_equity(result, ax=None):
    """Equity curve with the drawdown underneath."""
    if ax is None:
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig, axes = ax.figure, [ax, None]

    _equity_panel(axes[0], result)
    if axes[1] is not None:
        _drawdown_panel(axes[1], result)
    fig.tight_layout()
    return fig


def _equity_panel(ax, result, label_end=True):
    eq = result.equity
    ax.plot(eq.index, eq.values, color=BLUE, lw=2.0, solid_capstyle="round")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    ax.set_ylabel("equity")
    _clean(ax)

    if label_end:
        ax.annotate(f"{eq.iloc[-1]:,.0f}", xy=(eq.index[-1], eq.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, fontweight="bold", color=INK)
    return ax


def _drawdown_panel(ax, result):
    dd = 100 * (result.equity / result.equity.cummax() - 1.0)
    ax.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.16, lw=0)
    ax.plot(dd.index, dd.values, color=RED, lw=1.1)
    ax.set_ylabel("drawdown %")
    _clean(ax)

    trough = dd.idxmin()
    ax.annotate(f"{dd.min():.1f}%", xy=(trough, dd.min()), xytext=(5, 3),
                textcoords="offset points", fontsize=8, color=INK_2, va="bottom")
    return ax


def plot_spread_signals(signals, ax=None):
    """Spread and z-score with entry/exit markers."""
    if ax is None:
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    else:
        fig, axes = ax.figure, [ax, None]

    axes[0].plot(signals.index, signals["spread"], color=INK_2, lw=1.0, label="spread")
    axes[0].plot(signals.index, signals["mu"], color=MUTED, lw=1.0, ls="--",
                 label="rolling mean")
    axes[0].set_ylabel("spread")
    axes[0].legend(loc="upper left", fontsize=8)
    _clean(axes[0])

    if axes[1] is not None:
        _zscore_panel(axes[1], signals)
    fig.tight_layout()
    return fig


def _zscore_panel(ax, signals, entry_z=2.0, stop_z=4.0):
    z = signals["zscore"]
    pos = signals["position"]

    ax.plot(z.index, z.values, color=INK_2, lw=0.9)
    ax.axhline(0, color=AXIS, lw=0.8)
    for level in (entry_z, -entry_z):
        ax.axhline(level, color=MUTED, lw=0.8, ls="--")
    for level in (stop_z, -stop_z):
        ax.axhline(level, color=MUTED, lw=0.7, ls=":")

    changes = pos.diff().fillna(pos)
    longs = changes[(changes > 0) & (pos > 0)].index
    shorts = changes[(changes < 0) & (pos < 0)].index
    exits = changes[(changes != 0) & (pos == 0)].index

    ax.scatter(longs, z.reindex(longs), marker="^", s=46, color=GREEN,
               edgecolor=SURFACE, lw=0.8, zorder=3, label=f"long ({len(longs)})")
    ax.scatter(shorts, z.reindex(shorts), marker="v", s=46, color=RED,
               edgecolor=SURFACE, lw=0.8, zorder=3, label=f"short ({len(shorts)})")
    ax.scatter(exits, z.reindex(exits), marker="o", s=22, color=MUTED,
               edgecolor=SURFACE, lw=0.6, zorder=3, label="exit")

    ax.set_ylabel("z-score")
    ax.legend(loc="upper left", fontsize=8, ncol=3, handletextpad=0.3, columnspacing=1.2)
    _clean(ax)
    return ax


def plot_return_distribution(returns, ax=None):
    fig, ax = plt.subplots(figsize=(6, 4)) if ax is None else (ax.figure, ax)

    r = 100 * returns[returns != 0]
    ax.hist(r, bins=60, color=BLUE, alpha=0.85, edgecolor=SURFACE, lw=0.4)
    ax.axvline(float(r.mean()), color=INK, lw=1.2)
    ax.annotate(f"mean {r.mean():.3f}%", xy=(float(r.mean()), 0.94), xycoords=("data", "axes fraction"),
                xytext=(5, 0), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("daily return (%)")
    ax.set_ylabel("days")
    ax.set_title(f"Daily returns in market  (skew {r.skew():.2f}, kurtosis {r.kurt():.2f})",
                 fontsize=10, loc="left", color=INK)
    _clean(ax)
    fig.tight_layout()
    return fig


def _trade_histogram(ax, trades):
    pnl = 100 * trades["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    bins = np.histogram_bin_edges(pnl, bins=24)

    ax.hist(losses, bins=bins, color=RED, alpha=0.85, edgecolor=SURFACE, lw=0.5,
            label=f"{len(losses)} losers")
    ax.hist(wins, bins=bins, color=GREEN, alpha=0.85, edgecolor=SURFACE, lw=0.5,
            label=f"{len(wins)} winners")
    ax.axvline(0, color=AXIS, lw=0.9)
    ax.set_xlabel("trade P&L (%)")
    ax.set_ylabel("trades")
    ax.legend(loc="upper left", fontsize=8)
    _clean(ax)
    return ax


def _kpi_band(ax, metrics):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0, 0.86, "SHARPE RATIO", fontsize=8, color=MUTED)
    ax.text(0, 0.24, _num(metrics["sharpe"]), fontsize=38, color=INK,
            fontweight="bold", va="baseline")
    ax.plot([0.175, 0.175], [0.1, 0.95], color=GRID, lw=1.0)

    tiles = [("TOTAL RETURN", _pct(metrics["total_return"])),
             ("CAGR", _pct(metrics["cagr"])),
             ("MAX DRAWDOWN", _pct(metrics["max_drawdown"])),
             ("WIN RATE", _pct(metrics["win_rate"], 0)),
             ("PROFIT FACTOR", _num(metrics["profit_factor"])),
             ("TRADES", f"{metrics['n_trades']}")]

    for i, (label, value) in enumerate(tiles):
        x = 0.22 + i * 0.132
        ax.text(x, 0.86, label, fontsize=8, color=MUTED)
        ax.text(x, 0.34, value, fontsize=17, color=INK, fontweight="bold", va="baseline")
    return ax


def tearsheet(result, signals, meta):
    """One-page summary figure: headline metrics, equity, drawdown, signals and trades."""
    fig = plt.figure(figsize=(14, 10.5))
    gs = fig.add_gridspec(5, 3, height_ratios=[0.55, 0.62, 2.1, 0.85, 1.7],
                          hspace=0.45, wspace=0.22,
                          left=0.06, right=0.965, top=0.965, bottom=0.085)

    head = fig.add_subplot(gs[0, :])
    head.set_axis_off()
    head.text(0, 0.42, f"{meta['y']} / {meta['x']}", fontsize=23, fontweight="bold", color=INK)
    head.text(0, 0.0, meta["subtitle"], fontsize=10, color=INK_2)

    _kpi_band(fig.add_subplot(gs[1, :]), result.metrics)

    ax_eq = fig.add_subplot(gs[2, :])
    _equity_panel(ax_eq, result)
    ax_eq.set_title("Equity curve", fontsize=11, loc="left", color=INK, pad=8)

    ax_dd = fig.add_subplot(gs[3, :], sharex=ax_eq)
    _drawdown_panel(ax_dd, result)
    ax_eq.tick_params(labelbottom=False)
    ax_dd.xaxis.set_major_locator(mdates.YearLocator())
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax_z = fig.add_subplot(gs[4, :2])
    _zscore_panel(ax_z, signals, meta["entry_z"], meta["stop_z"])
    ax_z.set_title("Spread z-score and trades", fontsize=11, loc="left", color=INK, pad=8)
    ax_z.xaxis.set_major_locator(mdates.YearLocator())
    ax_z.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax_h = fig.add_subplot(gs[4, 2])
    _trade_histogram(ax_h, result.trades)
    ax_h.set_title("Trade P&L", fontsize=11, loc="left", color=INK, pad=8)

    fig.text(0.06, 0.022, meta["footer"], fontsize=8, color=MUTED)
    return fig


def save_report(result, signals, outdir, meta=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    figures = {
        "equity_curve.png": plot_equity(result),
        "spread_signals.png": plot_spread_signals(signals),
        "return_distribution.png": plot_return_distribution(result.returns),
    }
    if meta is not None and not result.trades.empty:
        figures["tearsheet.png"] = tearsheet(result, signals, meta)

    for name, fig in figures.items():
        path = outdir / name
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        log.info("wrote %s", path)

    return list(figures)
