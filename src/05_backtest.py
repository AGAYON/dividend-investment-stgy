"""
05_backtest.py — Loop mensual completo + métricas
Actinver · Estrategia Investor US Equities

Para cada mes t desde ene 2016:
  1. Mark-to-market del portafolio del mes anterior
  2. Re-screen universo con datos al cierre de t-1  (sin lookahead)
  3. Calcular features: μ_BL, Σ_LW  (lookback 36m ending t-1)
  4. Evaluar señales del portafolio actual al cierre t-1
  5. Optimizar pesos para el mes t
  6. Ejecutar rebalanceo
  7. Registrar NAV, retornos, yield realizado, señales

Benchmark: SPY (total return, dividendos reinvertidos vía auto_adjust).

Output:
    data/processed/backtest_results.parquet
    data/processed/backtest_metrics.json
"""

import importlib.util
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
SRC = Path(__file__).parent

# ---------------------------------------------------------------------------
# Parámetros globales
# ---------------------------------------------------------------------------
BACKTEST_START   = "2016-01-01"
HISTORY_START    = "2013-01-01"
LOOKBACK_MONTHS  = 36
MIN_OBS_MONTHS   = 24

MIN_MKTCAP       = 100e9
MIN_YIELD_SCREEN = 0.005
MIN_ADV          = 50e6
HISTORY_GRACE_DAYS = 45

MIN_PORTFOLIO_YIELD = 0.03
MAX_WEIGHT          = 0.05
GAMMA               = 1.0
LAMBDA_BL           = 3.0
TAU                 = 0.05

EWMA_SPAN    = 12
SL_THRESHOLD = 1.0
TP_THRESHOLD = 1.0

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Importar funciones matemáticas desde módulos 02 y 03
# (nombres de archivo con prefijo numérico requieren importlib)
# ---------------------------------------------------------------------------
def _import_src(alias: str, fname: str):
    spec = importlib.util.spec_from_file_location(alias, SRC / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_feat = _import_src("features", "02_features.py")
_opt  = _import_src("optimizer", "03_optimizer.py")

_ledoit_wolf          = _feat._ledoit_wolf
_black_litterman      = _feat._black_litterman
optimize_portfolio    = _opt.optimize_portfolio
_max_achievable_yield = _opt._max_achievable_yield
_log_infeasibility    = _opt._log_infeasibility


# ---------------------------------------------------------------------------
# A. Carga de datos
# ---------------------------------------------------------------------------

def fetch_sp500_tickers() -> list[str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(SP500_WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    logger.info(f"S&P 500: {len(tickers)} tickers de Wikipedia")
    return tickers


def load_price_data(
    tickers: list[str],
    start: str = HISTORY_START,
    end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descarga precios ajustados + volúmenes diarios. Retorna (close, volume)."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    logger.info(f"Descargando precios: {len(tickers)} tickers | {start} → {end}")
    raw = yf.download(
        tickers=tickers, start=start, end=end,
        auto_adjust=True, progress=True,
    )
    close = raw["Close"]
    volume = raw["Volume"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    if isinstance(volume, pd.Series):
        volume = volume.to_frame(name=tickers[0])
    logger.info(f"Precios diarios cargados: {close.shape}")
    return close, volume


def load_dividends(tickers: list[str], max_workers: int = 16) -> dict[str, pd.Series]:
    """Descarga historial de dividendos en paralelo. Retorna {ticker: Series}."""
    logger.info(f"Descargando dividendos: {len(tickers)} tickers...")

    def _fetch(tk: str) -> tuple[str, pd.Series]:
        try:
            return tk, yf.Ticker(tk).dividends
        except Exception:
            return tk, pd.Series(dtype=float)

    result = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for tk, divs in pool.map(_fetch, tickers):
            result[tk] = divs
    n_with_divs = sum(1 for v in result.values() if not v.empty)
    logger.info(f"Dividendos cargados: {n_with_divs}/{len(tickers)} tickers con historial")
    return result


def load_current_mktcap(tickers: list[str], max_workers: int = 16) -> pd.Series:
    """Descarga market cap actual para usar como proxy histórico."""
    logger.info(f"Descargando market cap: {len(tickers)} tickers...")

    def _fetch(tk: str) -> tuple[str, float]:
        try:
            return tk, getattr(yf.Ticker(tk).fast_info, "market_cap", 0) or 0
        except Exception:
            return tk, 0.0

    mc = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for tk, cap in pool.map(_fetch, tickers):
            mc[tk] = cap
    return pd.Series(mc)


# ---------------------------------------------------------------------------
# B. Universe screening histórico (sin lookahead)
# ---------------------------------------------------------------------------

def screen_universe_at(
    date: pd.Timestamp,
    tickers: list[str],
    daily_close: pd.DataFrame,
    daily_volume: pd.DataFrame,
    dividends: dict[str, pd.Series],
    current_mktcap: pd.Series,
    current_price: pd.Series,
    min_mktcap: float = MIN_MKTCAP,
    min_yield: float = MIN_YIELD_SCREEN,
    min_adv: float = MIN_ADV,
) -> pd.DataFrame:
    """Universo elegible en `date` usando solo datos disponibles hasta esa fecha."""
    close_at = daily_close.loc[:date]
    if close_at.empty:
        return pd.DataFrame()

    prices_t = close_at.iloc[-1]
    adv_daily = (daily_close.loc[:date] * daily_volume.loc[:date]).iloc[-90:]
    history_cutoff = pd.Timestamp(HISTORY_START) + pd.Timedelta(days=HISTORY_GRACE_DAYS)

    passed = []
    for tk in tickers:
        if tk not in daily_close.columns:
            continue

        price = prices_t.get(tk, np.nan)
        if pd.isna(price) or price <= 0:
            continue

        # Market cap proxy: escala current_mktcap por price ratio
        cp = current_price.get(tk, np.nan)
        mc = current_mktcap.get(tk, 0.0)
        if pd.isna(cp) or cp <= 0 or mc <= 0:
            continue
        mc_proxy = mc * (price / cp)
        if mc_proxy < min_mktcap:
            continue

        # Dividend yield TTM
        divs = dividends.get(tk, pd.Series(dtype=float))
        dy = 0.0
        if not divs.empty:
            try:
                tz = divs.index.tz
                d_end = date.tz_localize(tz) if tz else date
                d_start = d_end - pd.DateOffset(years=1)
                dy = float(divs[(divs.index >= d_start) & (divs.index <= d_end)].sum()) / price
            except Exception:
                dy = 0.0
        if dy < min_yield:
            continue

        # ADV 90 días
        if tk not in adv_daily.columns:
            continue
        adv = adv_daily[tk].dropna().mean()
        if pd.isna(adv) or adv < min_adv:
            continue

        # Historial desde 2013
        hist = daily_close[tk].dropna()
        if hist.empty:
            continue
        first = hist.index[0]
        first = first.tz_localize(None) if first.tzinfo else first
        if first > history_cutoff:
            continue

        passed.append({
            "ticker": tk,
            "market_cap": mc_proxy,
            "dividend_yield": round(dy, 4),
            "adv_90d": round(adv, 0),
            "price": round(float(price), 2),
        })

    if not passed:
        return pd.DataFrame()
    return (
        pd.DataFrame(passed)
        .sort_values("market_cap", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# C. Features desde datos cacheados (sin descarga por mes)
# ---------------------------------------------------------------------------

def compute_features_cached(
    universe_df: pd.DataFrame,
    monthly_ret: pd.DataFrame,
    window_end: pd.Timestamp,
    lookback_months: int = LOOKBACK_MONTHS,
    min_obs_months: int = MIN_OBS_MONTHS,
    lambda_bl: float = LAMBDA_BL,
    tau: float = TAU,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """B-L + Ledoit-Wolf usando monthly_ret pre-cargado."""
    tickers = universe_df["ticker"].tolist()
    window_start = window_end - pd.DateOffset(months=lookback_months + 1)
    ret_w = monthly_ret.loc[window_start:window_end]

    available = [t for t in tickers if t in ret_w.columns]
    if not available:
        return None, None

    obs = ret_w[available].notna().sum()
    valid = obs[obs >= min_obs_months].index.tolist()
    if len(valid) < 2:
        return None, None

    returns_clean = ret_w[valid].dropna()
    if returns_clean.shape[0] < min_obs_months:
        return None, None

    sigma = _ledoit_wolf(returns_clean)

    univ = universe_df.set_index("ticker")
    mktcap   = np.array([univ.loc[t, "market_cap"]    for t in valid], dtype=float)
    q_views  = np.array([univ.loc[t, "dividend_yield"] for t in valid], dtype=float)
    w_mkt    = mktcap / mktcap.sum()
    mu_bl    = _black_litterman(sigma, w_mkt, q_views, lambda_bl, tau)

    features_df = pd.DataFrame({
        "ticker":             valid,
        "mu_bl":              mu_bl,
        "sigma_diag":         np.diag(sigma),
        "dividend_yield_ttm": q_views,
    })
    cov_df = pd.DataFrame(sigma, index=valid, columns=valid)
    return features_df, cov_df


# ---------------------------------------------------------------------------
# D. Señales desde datos cacheados
# ---------------------------------------------------------------------------

def evaluate_signals_cached(
    holdings: dict,
    monthly_close: pd.DataFrame,
    eval_date: pd.Timestamp,
    sl_threshold: float = SL_THRESHOLD,
    tp_threshold: float = TP_THRESHOLD,
    ewma_span: int = EWMA_SPAN,
    lookback_months: int = LOOKBACK_MONTHS,
) -> dict[str, str]:
    """Retorna {ticker: 'hold'|'stop'|'take'} usando monthly_close hasta eval_date."""
    signals = {}
    for tk, info in holdings.items():
        if tk not in monthly_close.columns:
            signals[tk] = "hold"
            continue

        prices = monthly_close[tk].dropna().loc[:eval_date]
        if prices.empty:
            signals[tk] = "hold"
            continue

        current_price = float(prices.iloc[-1])
        entry_price   = float(info.get("entry_price", 0))
        if entry_price <= 0:
            signals[tk] = "hold"
            continue

        r_acum = (current_price / entry_price) - 1.0

        hist = prices.loc[prices.index >= eval_date - pd.DateOffset(months=lookback_months + 1)]
        log_ret = np.log(hist / hist.shift(1)).dropna()
        if len(log_ret) < 2:
            signals[tk] = "hold"
            continue

        sigma_ewma = float(log_ret.ewm(span=ewma_span).std().iloc[-1])
        if sigma_ewma <= 0:
            signals[tk] = "hold"
        elif r_acum > tp_threshold * sigma_ewma:
            signals[tk] = "take"
        elif r_acum < -sl_threshold * sigma_ewma:
            signals[tk] = "stop"
        else:
            signals[tk] = "hold"

    return signals


# ---------------------------------------------------------------------------
# E. Yield realizado (para reporting; NAV ya lo incluye vía adj prices)
# ---------------------------------------------------------------------------

def realized_yield_for_month(
    holdings: dict,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    dividends: dict[str, pd.Series],
    monthly_close: pd.DataFrame,
) -> float:
    """
    Yield mensual realizado = Σ weight_i × (divs_en_mes / precio_inicio_mes).
    """
    total = 0.0
    for tk, info in holdings.items():
        divs = dividends.get(tk, pd.Series(dtype=float))
        if divs.empty:
            continue
        try:
            tz = divs.index.tz
            ms = month_start.tz_localize(tz) if tz else month_start
            me = month_end.tz_localize(tz)   if tz else month_end
            month_divs = float(divs[(divs.index > ms) & (divs.index <= me)].sum())
            if month_divs <= 0:
                continue
            # precio al inicio del mes
            p0_series = monthly_close[tk].dropna().loc[:month_start] if tk in monthly_close.columns else pd.Series()
            if p0_series.empty:
                continue
            p0 = float(p0_series.iloc[-1])
            if p0 <= 0:
                continue
            total += info["weight"] * (month_divs / p0)
        except Exception:
            continue
    return total


# ---------------------------------------------------------------------------
# F. Métricas de desempeño
# ---------------------------------------------------------------------------

def compute_metrics(results: pd.DataFrame, rf_annual: float = 0.04) -> dict:
    """Todas las métricas de la sección 6 del README."""
    ret_p = results["portfolio_return"].dropna()
    ret_b = results["benchmark_return"].dropna()
    # Alinear
    idx = ret_p.index.intersection(ret_b.index)
    ret_p, ret_b = ret_p.loc[idx], ret_b.loc[idx]

    rf_m   = rf_annual / 12
    n      = len(ret_p)
    excess = ret_p - rf_m

    cagr_p = (1 + ret_p).prod() ** (12 / n) - 1
    cagr_b = (1 + ret_b).prod() ** (12 / n) - 1
    vol_p  = ret_p.std() * np.sqrt(12)

    # Drawdown
    nav    = results.loc[idx, "nav"]
    dd     = (nav / nav.cummax()) - 1
    max_dd = float(dd.min())

    # Ratios
    sharpe = float((excess.mean() / ret_p.std()) * np.sqrt(12)) if ret_p.std() > 0 else np.nan
    down   = excess[ret_p < rf_m]
    sdenom = np.sqrt((down**2).mean() * 12) if len(down) > 0 else np.nan
    sortino = float((ret_p.mean() - rf_m) * 12 / sdenom) if sdenom else np.nan
    calmar  = float(cagr_p / abs(max_dd)) if max_dd != 0 else np.nan

    # VaR / CVaR
    var95 = float(ret_p.quantile(0.05))
    var99 = float(ret_p.quantile(0.01))
    cvar95 = float(ret_p[ret_p <= var95].mean())
    cvar99 = float(ret_p[ret_p <= var99].mean())

    # Alpha / TE / IR
    active  = ret_p - ret_b
    te      = float(active.std() * np.sqrt(12))
    alpha_a = float(active.mean() * 12)
    ir      = alpha_a / te if te > 0 else np.nan

    # Beta
    cov_mat = np.cov(ret_p, ret_b)
    beta    = float(cov_mat[0, 1] / cov_mat[1, 1]) if cov_mat[1, 1] > 0 else np.nan

    # UPM (umbral 3%/12 mensual)
    thr = 0.03 / 12
    upm = float(((np.maximum(ret_p - thr, 0))**2).mean())

    # Pain index
    pain  = float(abs(dd.mean()))
    pain_gain = float(pain / ret_p.mean()) if ret_p.mean() > 0 else np.nan

    # Yield stats
    yields  = results.loc[idx, "realized_yield"]
    y_ann   = float(yields.mean() * 12)
    y_pct   = float((yields >= 0.03 / 12).mean())

    # Turnover + stops/takes
    turn    = float(results["turnover"].mean()) if "turnover" in results.columns else np.nan
    n_sl    = int(results["n_stops"].sum()) if "n_stops" in results.columns else 0
    n_tp    = int(results["n_takes"].sum()) if "n_takes" in results.columns else 0

    return {
        "CAGR_portfolio":            round(cagr_p,  4),
        "CAGR_benchmark":            round(cagr_b,  4),
        "Vol_annual_portfolio":      round(vol_p,   4),
        "Sharpe_Ratio":              round(sharpe,  4),
        "Sortino_Ratio":             round(sortino, 4) if not np.isnan(sortino) else None,
        "Calmar_Ratio":              round(calmar,  4) if not np.isnan(calmar) else None,
        "Max_Drawdown":              round(max_dd,  4),
        "VaR_95_monthly":            round(var95,   4),
        "VaR_99_monthly":            round(var99,   4),
        "CVaR_95_monthly":           round(cvar95,  4),
        "CVaR_99_monthly":           round(cvar99,  4),
        "Upper_Partial_Moment":      round(upm,     6),
        "Pain_Index":                round(pain,    4),
        "Pain_Gain_Ratio":           round(pain_gain, 4) if not np.isnan(pain_gain) else None,
        "Alpha_annual":              round(alpha_a, 4),
        "Tracking_Error":            round(te,      4),
        "Information_Ratio":         round(ir,      4) if not np.isnan(ir) else None,
        "Beta":                      round(beta,    4) if not np.isnan(beta) else None,
        "Avg_Realized_Yield_annual": round(y_ann,   4),
        "Pct_months_yield_met":      round(y_pct,   4),
        "Avg_Monthly_Turnover":      round(turn,    4) if not np.isnan(turn) else None,
        "N_StopLoss_activations":    n_sl,
        "N_TakeProfit_activations":  n_tp,
        "N_months":                  n,
    }


# ---------------------------------------------------------------------------
# G. Loop principal
# ---------------------------------------------------------------------------

def run_backtest(
    tickers: list[str] | None = None,
    backtest_start: str = BACKTEST_START,
    min_mktcap: float = MIN_MKTCAP,
    min_yield_screen: float = MIN_YIELD_SCREEN,
    min_adv: float = MIN_ADV,
    min_portfolio_yield: float = MIN_PORTFOLIO_YIELD,
    max_weight: float = MAX_WEIGHT,
    gamma: float = GAMMA,
    lambda_bl: float = LAMBDA_BL,
    tau: float = TAU,
    ewma_span: int = EWMA_SPAN,
    sl_threshold: float = SL_THRESHOLD,
    tp_threshold: float = TP_THRESHOLD,
    lookback_months: int = LOOKBACK_MONTHS,
    min_obs_months: int = MIN_OBS_MONTHS,
    save: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Corre el backtest completo. Retorna (results_df, metrics_dict).

    results_df index: date (último día hábil de cada mes)
    Columnas: nav | portfolio_return | benchmark_return | realized_yield |
              n_stocks | n_universe | cash_pct | rebalance_flag |
              infeasibility_flag | n_stops | n_takes | turnover
    """
    # ------------------------------------------------------------------
    # 1. Cargar datos
    # ------------------------------------------------------------------
    if tickers is None:
        tickers = fetch_sp500_tickers()

    daily_close, daily_volume = load_price_data(tickers, start=HISTORY_START)
    dividends     = load_dividends(tickers)
    current_mktcap = load_current_mktcap(tickers)
    current_price  = pd.Series(
        {tk: float(daily_close[tk].dropna().iloc[-1])
         for tk in tickers if tk in daily_close.columns and not daily_close[tk].dropna().empty}
    )

    # Benchmark: SPY (total return)
    spy_raw  = yf.download("SPY", start=HISTORY_START, auto_adjust=True, progress=False)
    spy_close = spy_raw["Close"]
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    spy_monthly = spy_close.resample("ME").last()
    spy_ret     = (spy_monthly / spy_monthly.shift(1) - 1).dropna()

    # Precios mensuales + retornos mensuales (para features)
    monthly_close = daily_close.resample("ME").last()
    monthly_ret   = np.log(monthly_close / monthly_close.shift(1))

    # Fechas de rebalanceo: últimos días de mes a partir de backtest_start
    rebalance_dates = monthly_close.loc[backtest_start:].index.tolist()
    logger.info(
        f"Backtest: {len(rebalance_dates)} meses | "
        f"{rebalance_dates[0].strftime('%Y-%m')} → {rebalance_dates[-1].strftime('%Y-%m')}"
    )

    # prev_date: último cierre mensual ANTES del backtest_start
    pre = monthly_close.loc[:pd.Timestamp(backtest_start) - pd.Timedelta(days=1)]
    prev_date = pre.index[-1] if not pre.empty else rebalance_dates[0] - pd.offsets.MonthEnd(1)

    # ------------------------------------------------------------------
    # 2. Estado inicial
    # ------------------------------------------------------------------
    holdings: dict[str, dict] = {}   # {ticker: {weight, entry_date, entry_price}}
    nav  = 100.0
    records = []

    # ------------------------------------------------------------------
    # 3. Loop mensual
    # ------------------------------------------------------------------
    for i, date in enumerate(rebalance_dates):

        # ---- Mark-to-market ----
        if holdings and prev_date is not None:
            port_return = 0.0
            for tk, info in holdings.items():
                if tk not in monthly_close.columns:
                    continue
                p_series = monthly_close[tk].dropna()
                p_prev = p_series.loc[:prev_date]
                p_curr = p_series.loc[:date]
                if p_prev.empty or p_curr.empty:
                    continue
                r = float(p_curr.iloc[-1]) / float(p_prev.iloc[-1]) - 1.0
                port_return += info["weight"] * r

            # Yield realizado (solo reporting; adj returns ya incluyen dividendos en NAV)
            realized_yield = realized_yield_for_month(
                holdings, prev_date, date, dividends, monthly_close
            )
            nav = nav * (1.0 + port_return)

            # Benchmark
            spy_candidates = spy_ret.loc[spy_ret.index <= date]
            benchmark_return = float(spy_candidates.iloc[-1]) if not spy_candidates.empty else 0.0
        else:
            port_return = benchmark_return = realized_yield = 0.0

        # ---- Universe (datos al cierre de prev_date, sin lookahead) ----
        universe = screen_universe_at(
            date=prev_date,
            tickers=tickers,
            daily_close=daily_close,
            daily_volume=daily_volume,
            dividends=dividends,
            current_mktcap=current_mktcap,
            current_price=current_price,
            min_mktcap=min_mktcap,
            min_yield=min_yield_screen,
            min_adv=min_adv,
        )
        n_universe = len(universe)

        # ---- Señales al cierre de prev_date ----
        signals: dict[str, str] = {}
        if holdings:
            signals = evaluate_signals_cached(
                holdings=holdings,
                monthly_close=monthly_close,
                eval_date=prev_date,
                sl_threshold=sl_threshold,
                tp_threshold=tp_threshold,
                ewma_span=ewma_span,
                lookback_months=lookback_months,
            )
        n_stops = sum(1 for s in signals.values() if s == "stop")
        n_takes = sum(1 for s in signals.values() if s == "take")

        # Tickers a liquidar
        universe_set = set(universe["ticker"].tolist()) if n_universe > 0 else set()
        to_liquidate = {
            tk for tk, sig in signals.items() if sig in ("stop", "take")
        } | {
            tk for tk in holdings if tk not in universe_set
        }
        cash_freed = sum(holdings[tk]["weight"] for tk in to_liquidate if tk in holdings)

        # ---- Features + Optimización ----
        infeasible_flag = False
        new_weights_df  = None

        if n_universe >= 2:
            features_df, cov_df = compute_features_cached(
                universe_df=universe,
                monthly_ret=monthly_ret,
                window_end=prev_date,
                lookback_months=lookback_months,
                min_obs_months=min_obs_months,
                lambda_bl=lambda_bl,
                tau=tau,
            )
            if features_df is not None:
                new_weights_df = optimize_portfolio(
                    features_df=features_df,
                    cov_df=cov_df,
                    gamma=gamma,
                    min_yield=min_portfolio_yield,
                    max_weight=max_weight,
                    rebalance_date=date,
                    save=False,
                )
                if new_weights_df is None:
                    infeasible_flag = True

        # ---- Rebalanceo ----
        prev_w = {tk: info["weight"] for tk, info in holdings.items()}

        if new_weights_df is not None:
            prices_now = daily_close.loc[:prev_date].iloc[-1]
            new_holdings: dict[str, dict] = {}
            for _, row in new_weights_df.iterrows():
                tk = row["ticker"]
                w  = float(row["weight"])
                if w < 1e-4:
                    continue
                if tk in holdings and tk not in to_liquidate:
                    new_holdings[tk] = {**holdings[tk], "weight": w}
                else:
                    ep = float(prices_now.get(tk, 0)) if tk in prices_now.index else 0.0
                    new_holdings[tk] = {
                        "weight":      w,
                        "entry_date":  prev_date,
                        "entry_price": ep if ep > 0 else 1.0,
                    }
            holdings = new_holdings
        # Si infeasible o universo vacío: mantener portafolio actual sin cambios

        new_w = {tk: info["weight"] for tk, info in holdings.items()}
        all_tks = set(prev_w) | set(new_w)
        turnover = sum(abs(new_w.get(tk, 0) - prev_w.get(tk, 0)) for tk in all_tks) / 2

        # ---- Registrar ----
        records.append({
            "date":               date,
            "nav":                round(nav, 4),
            "portfolio_return":   round(port_return, 6),
            "benchmark_return":   round(benchmark_return, 6),
            "realized_yield":     round(realized_yield, 6),
            "n_stocks":           len(holdings),
            "n_universe":         n_universe,
            "cash_pct":           round(cash_freed, 4),
            "rebalance_flag":     new_weights_df is not None,
            "infeasibility_flag": infeasible_flag,
            "n_stops":            n_stops,
            "n_takes":            n_takes,
            "turnover":           round(turnover, 4),
        })

        if i % 12 == 0 or i == len(rebalance_dates) - 1:
            logger.info(
                f"  {date.strftime('%Y-%m')}  NAV={nav:>8.2f}  "
                f"r={port_return:+.2%}  yield={realized_yield:.3%}  "
                f"n={len(holdings)}  univ={n_universe}"
                + ("  [INFEASIBLE]" if infeasible_flag else "")
            )

        prev_date = date

    # ------------------------------------------------------------------
    # 4. Métricas y guardado
    # ------------------------------------------------------------------
    results = pd.DataFrame(records).set_index("date")

    metrics = compute_metrics(results)

    logger.info("\n" + "=" * 60)
    logger.info("MÉTRICAS DE DESEMPEÑO")
    logger.info("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:<38} {v:>10.4f}")
        else:
            logger.info(f"  {k:<38} {str(v):>10}")

    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        res_path = DATA_PROCESSED / "backtest_results.parquet"
        results.to_parquet(res_path)
        logger.info(f"Resultados → {res_path}")

        met_path = DATA_PROCESSED / "backtest_metrics.json"
        with open(met_path, "w") as f:
            json.dump(
                {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                 for k, v in metrics.items()},
                f, indent=2,
            )
        logger.info(f"Métricas   → {met_path}")

    return results, metrics


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Backtest completo Actinver US Equity")
    p.add_argument("--start",     default=BACKTEST_START, help="Fecha inicio (YYYY-MM-DD)")
    p.add_argument("--gamma",     type=float, default=GAMMA)
    p.add_argument("--min-yield", type=float, default=MIN_PORTFOLIO_YIELD)
    args = p.parse_args()

    results, metrics = run_backtest(
        backtest_start=args.start,
        gamma=args.gamma,
        min_portfolio_yield=args.min_yield,
    )

    print("\n" + "=" * 70)
    print(f"BACKTEST — {args.start} → {results.index[-1].strftime('%Y-%m')}")
    print("=" * 70)
    print(f"NAV final:                {results['nav'].iloc[-1]:>10.2f}  (base 100)")
    print(f"CAGR portafolio:          {metrics['CAGR_portfolio']:>10.2%}")
    print(f"CAGR benchmark (SPY):     {metrics['CAGR_benchmark']:>10.2%}")
    print(f"Sharpe Ratio:             {metrics['Sharpe_Ratio']:>10.3f}")
    print(f"Sortino Ratio:            {str(metrics['Sortino_Ratio']):>10}")
    print(f"Calmar Ratio:             {str(metrics['Calmar_Ratio']):>10}")
    print(f"Max Drawdown:             {metrics['Max_Drawdown']:>10.2%}")
    print(f"VaR 95% (mensual):        {metrics['VaR_95_monthly']:>10.2%}")
    print(f"CVaR 95% (mensual):       {metrics['CVaR_95_monthly']:>10.2%}")
    print(f"Information Ratio:        {str(metrics['Information_Ratio']):>10}")
    print(f"Beta vs SPY:              {str(metrics['Beta']):>10}")
    print(f"Yield realizado anual:    {metrics['Avg_Realized_Yield_annual']:>10.2%}")
    print(f"% meses yield ≥ 3%:       {metrics['Pct_months_yield_met']:>10.1%}")
    print(f"Activaciones stop-loss:   {metrics['N_StopLoss_activations']:>10}")
    print(f"Activaciones take-profit: {metrics['N_TakeProfit_activations']:>10}")
    print("=" * 70)
