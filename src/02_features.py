"""
02_features.py — Estimación de parámetros de entrada para el optimizador QP
Actinver · Estrategia Investor US Equities

Genera:
  μ  — rendimientos esperados anualizados  (historical / EWMA / James-Stein + walk-forward CV)
  Σ  — covarianza Ledoit-Wolf rolling 36 meses (anualizada)
  σ  — volatilidad individual GARCH(1,1) anualizada para señales stop-loss / take-profit
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.covariance import LedoitWolf

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

LOOKBACK_MONTHS: int   = 36     # rolling window for μ and Σ
TRADING_DAYS:    int   = 252
EWMA_LAMBDA:     float = 0.70   # RiskMetrics decay; pandas alpha = 1 − λ = 0.30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output container
# ─────────────────────────────────────────────────────────────────────────────
class FeatureSet(NamedTuple):
    mu:           pd.Series    # annualized expected returns  (N,)
    cov:          pd.DataFrame # annualized covariance matrix (N×N)
    sigma_garch:  pd.Series    # annualized GARCH conditional vol  (N,)
    mu_method:    str          # 'historical' | 'ewma' | 'james_stein'
    as_of_date:   pd.Timestamp


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_prices(
    tickers:        list[str],
    start:          str       = "2013-01-01",
    end:            str | None = None,
    force_download: bool      = False,
    save:           bool      = True,
) -> pd.DataFrame:
    """
    Loads daily adjusted-close prices.
    Uses data/processed/prices.parquet when available and fresh (≤3 days old);
    otherwise downloads from yfinance and updates the cache.

    Returns
    -------
    pd.DataFrame  rows = trading days, columns = tickers
    """
    import yfinance as yf

    parquet_path = DATA_PROCESSED / "prices.parquet"
    end_dt = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()

    if not force_download and parquet_path.exists():
        cached = pd.read_parquet(parquet_path)
        missing = [t for t in tickers if t not in cached.columns]
        stale   = (end_dt - cached.index[-1]).days > 3
        if not missing and not stale:
            logger.info(
                f"Precios desde caché: {len(cached)} días, "
                f"{len(cached.columns)} tickers"
            )
            return cached[tickers].loc[start:]

    logger.info(f"Descargando precios: {len(tickers)} tickers desde {start}")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]

    # Forward-fill ≤5 trading days; drop tickers with >10 % missing
    prices = prices.ffill(limit=5)
    prices = prices.dropna(axis=1, thresh=int(0.90 * len(prices)))

    available = [t for t in tickers if t in prices.columns]
    prices    = prices[available]

    dropped = set(tickers) - set(available)
    if dropped:
        logger.warning(f"Descartados por datos insuficientes: {sorted(dropped)}")

    logger.info(
        f"Precios limpios: {len(available)} tickers, {len(prices)} días "
        f"({prices.index[0].date()} → {prices.index[-1].date()})"
    )

    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(parquet_path)
        logger.info(f"Guardado → {parquet_path}")

    return prices


# ─────────────────────────────────────────────────────────────────────────────
# 2. Expected returns (μ) — tres métodos
# ─────────────────────────────────────────────────────────────────────────────
def _mu_historical(returns_window: pd.DataFrame) -> pd.Series:
    """Arithmetic mean over the window, annualized (×12)."""
    return returns_window.mean() * 12


def _mu_ewma(returns_window: pd.DataFrame, lam: float = EWMA_LAMBDA) -> pd.Series:
    """
    EWMA mean with RiskMetrics λ=0.94 (alpha=0.06).
    More weight on recent months; uses the full window for stable initialization.
    """
    return returns_window.ewm(alpha=1.0 - lam, adjust=True).mean().iloc[-1] * 12


def _mu_james_stein(returns_window: pd.DataFrame) -> pd.Series:
    """
    Positive-part James-Stein estimator: shrinks each asset's mean toward the
    cross-sectional grand mean.

        B = min(1, (n−2) × σ̄²/T / ‖μ − μ̄‖²)
        μ_JS = μ̄ + (1 − B)(μ_hist − μ̄)

    where σ̄² is the average per-ticker sample variance.
    B→0 when means are dispersed (little shrinkage);
    B→1 when all means cluster near the grand mean (full shrinkage to μ̄).
    """
    T, n = returns_window.shape
    mu_hist  = returns_window.mean()
    mu_grand = float(mu_hist.mean())

    sigma2_avg = float((returns_window.var(ddof=1) / T).mean())
    d          = mu_hist - mu_grand
    ss         = float((d ** 2).sum())

    if ss < 1e-14 or n <= 2:
        return mu_hist * 12  # degenerate: all means identical

    B     = min(1.0, max(0.0, (n - 2) * sigma2_avg / ss))
    mu_js = mu_grand + (1.0 - B) * d
    logger.debug(f"James-Stein: B={B:.3f}, grand mean={mu_grand * 12:.2%}")
    return mu_js * 12


def compute_expected_returns(
    returns_window: pd.DataFrame,
    method: Literal["historical", "ewma", "james_stein"] = "ewma",
) -> pd.Series:
    """
    Dispatches to the chosen μ estimator.

    Parameters
    ----------
    returns_window : monthly returns already sliced to the estimation window
    method         : estimation method

    Returns
    -------
    pd.Series of annualized expected returns, indexed by ticker
    """
    dispatch = {
        "historical":  _mu_historical,
        "ewma":        _mu_ewma,
        "james_stein": _mu_james_stein,
    }
    if method not in dispatch:
        raise ValueError(f"method must be one of {list(dispatch)}; got '{method}'")
    return dispatch[method](returns_window).rename(method)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Covariance matrix (Σ) — Ledoit-Wolf shrinkage
# ─────────────────────────────────────────────────────────────────────────────
def compute_cov_matrix(returns_window: pd.DataFrame) -> pd.DataFrame:
    """
    Ledoit-Wolf shrinkage covariance matrix estimated on the given monthly-return
    window, then annualized (×12).

    Tickers with any NaN in the window are excluded so the estimator never
    sees an incomplete column (LedoitWolf requires full-rank input).
    """
    ret = returns_window.dropna(axis=1, how="any")
    lw  = LedoitWolf().fit(ret.values)
    cov_monthly = pd.DataFrame(
        lw.covariance_, index=ret.columns, columns=ret.columns
    )
    return cov_monthly * 12


# ─────────────────────────────────────────────────────────────────────────────
# 4. Individual volatility (σ) — GARCH(1,1)
# ─────────────────────────────────────────────────────────────────────────────
def _fit_garch_series(daily_returns: pd.Series) -> pd.Series:
    """
    Fits GARCH(1,1) on a single ticker and returns the in-sample conditional-
    volatility time series, annualized (×√252).

    Returns are scaled to percentage before fitting for numerical stability,
    then converted back to decimal.

    Note: parameters (ω, α, β) are estimated from the full available sample.
    The conditional-volatility series is causal (σ_t² = ω + α·ε²_{t-1} + β·σ²_{t-1}),
    so slicing vol[: as_of_date] in the backtester introduces no lookahead.
    """
    r      = daily_returns.dropna() * 100
    model  = arch_model(r, vol="GARCH", p=1, q=1, rescale=False)
    result = model.fit(disp="off", show_warning=False)
    return (result.conditional_volatility / 100) * np.sqrt(TRADING_DAYS)


def precompute_garch_vols(
    prices_daily: pd.DataFrame,
    save:         bool = True,
) -> pd.DataFrame:
    """
    Fits GARCH(1,1) once per ticker over the full price history and saves the
    daily conditional-volatility series to data/processed/garch_vols.parquet.

    Run this once before executing the backtest (05_backtest.py). The backtester
    loads the parquet and performs a simple date lookup — no re-fitting per month.

    Tickers that fail GARCH convergence fall back to 30-day EWMA volatility
    so the backtester always receives a complete σ vector.
    """
    ret_d = prices_daily.pct_change().dropna(how="all")
    vols:  dict[str, pd.Series] = {}
    n      = len(ret_d.columns)

    for i, ticker in enumerate(ret_d.columns, 1):
        r = ret_d[ticker].dropna()
        if len(r) < TRADING_DAYS:
            logger.warning(f"[{ticker}] Datos insuficientes ({len(r)} días) — EWMA fallback")
            vols[ticker] = r.ewm(span=30).std() * np.sqrt(TRADING_DAYS)
            continue
        try:
            vols[ticker] = _fit_garch_series(r)
            logger.info(f"  GARCH ({i}/{n}) {ticker} ✓")
        except Exception as exc:
            logger.warning(f"[{ticker}] GARCH falló: {exc} — EWMA fallback")
            vols[ticker] = r.ewm(span=30).std() * np.sqrt(TRADING_DAYS)

    vol_df = pd.DataFrame(vols).sort_index()

    if save:
        path = DATA_PROCESSED / "garch_vols.parquet"
        vol_df.to_parquet(path)
        logger.info(f"GARCH vols guardadas → {path}")

    return vol_df


def load_garch_vols() -> pd.DataFrame:
    """Loads precomputed GARCH conditional vols from parquet."""
    path = DATA_PROCESSED / "garch_vols.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Ejecuta precompute_garch_vols() primero."
        )
    return pd.read_parquet(path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Walk-forward cross-validation
# ─────────────────────────────────────────────────────────────────────────────
def walk_forward_cv(
    returns_monthly: pd.DataFrame,
    train_months:    int = LOOKBACK_MONTHS,
    test_months:     int = 12,
) -> dict:
    """
    Non-overlapping walk-forward CV to compare the three μ methods.

    Each fold:
      - Train on `train_months` months → compute μ with each method
      - Test on next `test_months` months → MSE(μ_monthly_pred, r_actual)

    Returns
    -------
    dict with keys:
      'best_method' : str            — method with lowest mean MSE
      'summary'     : pd.Series      — mean MSE per method across folds
      'results'     : pd.DataFrame   — MSE per fold × method
    """
    methods = ["historical", "ewma", "james_stein"]
    n       = len(returns_monthly)

    if n < train_months + test_months:
        logger.warning(
            f"Datos insuficientes para CV ({n} meses < {train_months + test_months}), "
            "usando 'ewma' por defecto"
        )
        return {"best_method": "ewma", "summary": None, "results": None}

    folds: list[dict] = []
    start = 0
    while start + train_months + test_months <= n:
        train = returns_monthly.iloc[start : start + train_months]
        test  = returns_monthly.iloc[
            start + train_months : start + train_months + test_months
        ]

        fold_row: dict = {"fold_start": returns_monthly.index[start]}
        for method in methods:
            try:
                mu_ann     = compute_expected_returns(train, method=method)
                mu_monthly = mu_ann / 12
                common     = mu_monthly.index.intersection(test.columns)
                errors     = test[common].sub(mu_monthly[common])
                fold_row[method] = float((errors ** 2).values.mean())
            except Exception as exc:
                logger.debug(f"CV fold {start} / {method}: {exc}")
                fold_row[method] = np.nan

        folds.append(fold_row)
        start += test_months

    results_df  = pd.DataFrame(folds).set_index("fold_start")
    summary     = results_df[methods].mean()
    best_method = str(summary.idxmin())

    logger.info(
        f"Walk-forward CV: {len(folds)} folds\n"
        f"  MSE medio  → {summary.round(7).to_dict()}\n"
        f"  Mejor método: {best_method}"
    )
    return {"best_method": best_method, "summary": summary, "results": results_df}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main API — llamada por el backtester en cada rebalanceo
# ─────────────────────────────────────────────────────────────────────────────
def compute_features(
    prices_daily:  pd.DataFrame,
    as_of_date:    pd.Timestamp | str,
    mu_method:     str | None   = None,
    window_months: int          = LOOKBACK_MONTHS,
    garch_vols:    pd.DataFrame | None = None,
) -> FeatureSet:
    """
    Computes μ, Σ, σ using only data available on or before `as_of_date`.

    Parameters
    ----------
    prices_daily  : daily adjusted-close prices for the current universe
    as_of_date    : rebalance date (data beyond this date is not used)
    mu_method     : 'historical' | 'ewma' | 'james_stein' | None (auto via CV)
    window_months : rolling estimation window in months  (default 36)
    garch_vols    : precomputed daily GARCH conditional vols from
                    precompute_garch_vols().  If None, falls back to EWMA vol.

    Returns
    -------
    FeatureSet(mu, cov, sigma_garch, mu_method, as_of_date)
    """
    as_of = pd.Timestamp(as_of_date)

    # Strict slice — no lookahead
    prices_hist = prices_daily.loc[:as_of]

    # Monthly prices (last trading day of each month) → monthly returns
    prices_m = prices_hist.resample("BME").last()
    ret_m    = prices_m.pct_change().dropna(how="all")

    if len(ret_m) < window_months:
        raise ValueError(
            f"Historial insuficiente en {as_of.date()}: "
            f"{len(ret_m)} meses disponibles, se requieren {window_months}"
        )

    ret_window = ret_m.iloc[-window_months:]

    # Auto-select μ method via walk-forward CV when not specified.
    # CV runs on ret_m, which is already sliced to as_of_date — no lookahead.
    # The backtester should call compute_features(mu_method=None) each monthly
    # rebalance so the estimator selection tracks market-regime changes.
    if mu_method is None:
        cv        = walk_forward_cv(ret_m, train_months=window_months, test_months=12)
        mu_method = cv["best_method"]

    mu  = compute_expected_returns(ret_window, method=mu_method)
    cov = compute_cov_matrix(ret_window)

    # GARCH σ: look up precomputed series; fall back to EWMA if not available
    if garch_vols is not None:
        valid_idx = garch_vols.index[garch_vols.index <= as_of]
        sigma = (
            garch_vols.loc[valid_idx[-1]]
            if len(valid_idx) > 0
            else _ewma_vol_fallback(prices_hist)
        )
    else:
        sigma = _ewma_vol_fallback(prices_hist)

    # Align: keep only tickers present in all three outputs
    common = mu.index.intersection(cov.index).intersection(sigma.index)
    mu     = mu.loc[common]
    cov    = cov.loc[common, common]
    sigma  = sigma.loc[common]

    return FeatureSet(
        mu=mu,
        cov=cov,
        sigma_garch=sigma,
        mu_method=mu_method,
        as_of_date=as_of,
    )


def _ewma_vol_fallback(prices_daily: pd.DataFrame, span: int = 30) -> pd.Series:
    """30-day EWMA vol fallback when GARCH vols are not available."""
    ret_d = prices_daily.pct_change().dropna(how="all")
    return ret_d.ewm(span=span).std().iloc[-1] * np.sqrt(TRADING_DAYS)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Standalone run
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Standalone execution:
      1. Loads latest universe from data/raw/universe_*.csv
      2. Downloads / refreshes daily prices (parquet cache)
      3. Runs walk-forward CV to pick the best mu method
      4. Precomputes GARCH(1,1) vols for all tickers  (~3-5 min, once)
      5. Computes today's FeatureSet and prints a summary table
      6. Saves mu, cov, sigma and CV results to data/processed/
    """
    import glob as _glob
    import sys

    # Windows cp1252 consoles reject Greek letters — force UTF-8 output
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ── Universe ──────────────────────────────────────────────────────────────
    universe_files = sorted(_glob.glob(str(DATA_RAW / "universe_*.csv")))
    if not universe_files:
        raise FileNotFoundError(
            "No se encontró ningún archivo de universo. Ejecuta 01_universe.py primero."
        )
    universe = pd.read_csv(universe_files[-1])
    tickers  = universe["ticker"].tolist()
    logger.info(f"Universo: {len(tickers)} tickers  ←  {universe_files[-1]}")

    # ── Prices ────────────────────────────────────────────────────────────────
    prices = load_prices(tickers, start="2013-01-01")

    # ── Walk-forward CV (MSE display only) ───────────────────────────────────
    # compute_features(mu_method=None) runs its own CV internally on the
    # as_of_date slice.  This separate call exists only to print the MSE table.
    prices_m   = prices.resample("BME").last()
    ret_m      = prices_m.pct_change().dropna(how="all")
    cv_results = walk_forward_cv(ret_m)

    # ── GARCH vols (cached) ───────────────────────────────────────────────────
    garch_path = DATA_PROCESSED / "garch_vols.parquet"
    if garch_path.exists():
        logger.info(f"GARCH vols desde caché ({garch_path})")
        garch_vols = load_garch_vols()
    else:
        logger.info("Calculando GARCH(1,1) para todos los tickers (~3-5 min)…")
        garch_vols = precompute_garch_vols(prices)

    # ── Today's features ──────────────────────────────────────────────────────
    as_of    = prices.index[-1]
    features = compute_features(
        prices, as_of_date=as_of, mu_method=None, garch_vols=garch_vols
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    diag_vol = pd.Series(
        np.sqrt(np.diag(features.cov.values)),
        index=features.cov.index,
    )
    summary = pd.DataFrame({
        "μ_anual (%)":       (features.mu * 100).round(2),
        "σ_GARCH (%)":       (features.sigma_garch * 100).round(2),
        "σ_LW_diag (%)":     (diag_vol * 100).round(2),
    }).sort_values("μ_anual (%)", ascending=False)

    print("\n" + "=" * 70)
    print(f"FEATURES  {as_of.date()}   μ método: {features.mu_method}")
    print("=" * 70)
    print(summary.to_string())
    print("─" * 70)
    print(f"Tickers          : {len(features.mu)}")
    print(f"μ promedio anual : {features.mu.mean():.2%}")
    print(f"σ GARCH promedio : {features.sigma_garch.mean():.2%}")
    print(f"Cov matrix shape : {features.cov.shape}")

    if cv_results["summary"] is not None:
        print("\nWalk-forward CV — MSE medio por método:")
        print(cv_results["summary"].to_string())

    # ── Persist ───────────────────────────────────────────────────────────────
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    features.mu.to_frame("mu").to_parquet(DATA_PROCESSED / "mu_latest.parquet")
    features.cov.to_parquet(DATA_PROCESSED / "cov_latest.parquet")
    features.sigma_garch.to_frame("sigma").to_parquet(
        DATA_PROCESSED / "sigma_latest.parquet"
    )
    if cv_results["results"] is not None:
        cv_results["results"].to_parquet(DATA_PROCESSED / "cv_results.parquet")

    logger.info("Features guardadas en data/processed/")


if __name__ == "__main__":
    main()
