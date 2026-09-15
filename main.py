import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.analysis.cointegration import test_pair
from src.backtest.engine import BacktestConfig, format_metrics, run_backtest
from src.data.fetcher import fetch_prices
from src.strategy.pairs import StrategyParams, run_strategy
from src.visualization.plots import save_report

log = logging.getLogger("statarb")

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Statistical arbitrage pairs backtester")
    p.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("-t", "--tickers", nargs=2, metavar=("Y", "X"),
                   help="override the pair from the config file")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--entry-z", type=float)
    p.add_argument("--exit-z", type=float)
    p.add_argument("--stop-z", type=float)
    p.add_argument("--z-window", type=int)
    p.add_argument("--cost-bps", type=float)
    p.add_argument("--outdir", type=Path)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="backtest even if the pair fails the cointegration test")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def load_config(path):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    for section in ("data", "strategy", "backtest", "output"):
        cfg.setdefault(section, {})
    return cfg


def apply_overrides(cfg, args):
    data, strat, bt, out = cfg["data"], cfg["strategy"], cfg["backtest"], cfg["output"]
    if args.tickers:
        data["tickers"] = list(args.tickers)
    if args.start:
        data["start"] = args.start
    if args.end:
        data["end"] = args.end
    if args.no_cache:
        data["use_cache"] = False

    for name, value in [("entry_z", args.entry_z), ("exit_z", args.exit_z),
                        ("stop_z", args.stop_z), ("z_window", args.z_window)]:
        if value is not None:
            strat[name] = value

    if args.cost_bps is not None:
        bt["cost_bps"] = args.cost_bps
    if args.outdir:
        out["dir"] = str(args.outdir)
    if args.no_plots:
        out["save_plots"] = False
    return cfg


def _report_meta(coint, params, bt_cfg, prices):
    hedge = "expanding OLS" if params.hedge_window is None else f"{params.hedge_window}d rolling OLS"
    footer = (f"entry |z|>{params.entry_z}  ·  exit |z|<{params.exit_z}  ·  stop |z|>{params.stop_z}"
              f"  ·  z-window {params.z_window}d  ·  max holding {params.max_holding_days}d"
              f"  ·  hedge ratio {hedge}  ·  costs {bt_cfg.cost_bps}bps + {bt_cfg.slippage_bps}bps "
              f"slippage per side  ·  backtest on adjusted closes, not a live track record")
    return {
        "y": coint.y,
        "x": coint.x,
        "subtitle": (f"Engle-Granger pairs trading  ·  {prices.index[0]:%b %Y} - {prices.index[-1]:%b %Y}"
                     f"  ·  ADF p={coint.pvalue:.4f}  ·  half-life {coint.half_life:.0f}d"),
        "entry_z": params.entry_z,
        "stop_z": params.stop_z,
        "footer": footer,
    }


def setup_logging(level):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    logging.getLogger("yfinance").setLevel(logging.WARNING)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.log_level)

    cfg = apply_overrides(load_config(args.config), args)
    data_cfg, out_cfg = cfg["data"], cfg["output"]

    tickers = data_cfg["tickers"]
    if len(tickers) != 2:
        log.error("this pipeline runs one pair at a time, got %s", tickers)
        return 2

    prices = fetch_prices(tickers, data_cfg["start"], data_cfg["end"],
                          interval=data_cfg.get("interval", "1d"),
                          use_cache=data_cfg.get("use_cache", True))

    coint = test_pair(prices, tickers[0], tickers[1],
                      maxlag=cfg.get("cointegration", {}).get("maxlag"))
    threshold = cfg.get("cointegration", {}).get("pvalue_threshold", 0.05)

    if not coint.is_cointegrated(threshold) and not args.force:
        log.error("pair rejected (p=%.4f >= %.3f); rerun with --force to backtest anyway",
                  coint.pvalue, threshold)
        return 1

    # the EG regression picks the direction, the strategy re-estimates beta out of sample
    y, x = coint.y, coint.x
    params = StrategyParams(**cfg["strategy"])
    signals = run_strategy(prices, y, x, params)

    bt_cfg = BacktestConfig(**cfg["backtest"])
    result = run_backtest(prices, signals, y, x, bt_cfg)

    print(f"\n{y} / {x}   {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"  {'hedge beta (EG)':<18}{coint.beta:9.4f}")
    print(f"  {'adf pvalue':<18}{coint.pvalue:9.4f}")
    print(f"  {'half life (days)':<18}{coint.half_life:9.1f}")
    print(format_metrics(result.metrics))
    print()

    outdir = Path(out_cfg.get("dir", "output"))
    if out_cfg.get("save_trades", True) and not result.trades.empty:
        outdir.mkdir(parents=True, exist_ok=True)
        result.trades.to_csv(outdir / "trades.csv", index=False)
        result.equity.to_csv(outdir / "equity.csv")
        log.info("trade log written to %s", outdir / "trades.csv")

    if out_cfg.get("save_plots", True):
        save_report(result, signals, outdir, _report_meta(coint, params, bt_cfg, prices))

    return 0


if __name__ == "__main__":
    sys.exit(main())
