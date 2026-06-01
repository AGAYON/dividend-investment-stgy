"""
05_backtest.py — Motor de backtesting mensual
Actinver · Estrategia Investor US Equities

Período : ene 2016 → hoy
Lookback : ene 2013 – dic 2015  (36 meses antes del inicio)
Rebalanceo: último día hábil de cada mes

Flujo por mes t:
  1. Compute features con datos hasta t  (μ, Σ, σ sin lookahead)
  2. Obtener yields TTM históricos en t
  3. Aplicar señales stop-loss / take-profit sobre posiciones abiertas
  4. Optimizar con universo = elegibles − stop-loss,
     max_weight/2 para take-profit tickers
  5. Registrar retorno total del mes con pesos previos × retornos realizados
  6. Actualizar libro de posiciones
  7. Re-entrenar CV de μ cada 12 meses

Nota de sesgo: el universo es estático (tickers del CSV más reciente de 01_universe.py).
Introduce survivorship bias leve para mega-caps del S&P 500 — documentado en el informe.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS        = ROOT / "outputs"

BACKTEST_START:   str   = "2016-01-01"
LOOKBACK_MONTHS:  int   = 36      # meses de historia para μ / Σ
CV_RETRAIN_FREQ:  int   = 12      # re-entrenar CV cada 12 meses
DEFAULT_MU_METHOD: str  = "james_stein"   # método hasta que haya datos para CV
MAX_WEIGHT:       float = 0.05
MIN_YIELD:        float = 0.03
LAMBDA_RISK:      float = 0.5
WEIGHT_TOL:       float = 1e-6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Carga de módulos numerados via importlib
# ─────────────────────────────────────────────────────────────────────────────
def _load(name: str, fname: str):
    """Carga un módulo desde src/ aunque su nombre empiece con dígito."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / fname)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_feat = _load("features",  "02_features.py")
_opt  = _load("optimizer", "03_optimizer.py")
_sig  = _load("signals",   "04_signals.py")

compute_features     = _feat.compute_features
compute_cov_matrix   = _feat.compute_cov_matrix
compute_expected_returns = _feat.compute_expected_returns
walk_forward_cv      = _feat.walk_forward_cv

optimize_portfolio   = _opt.optimize_portfolio

apply_signals        = _sig.apply_signals
update_position_book = _sig.update_position_book
Position             = _sig.Position


# ─────────────────────────────────────────────────────────────────────────────
# Contenedor de resultado mensual
# ─────────────────────────────────────────────────────────────────────────────
class MonthlyRecord(NamedTuple):
    date:           pd.Timestamp
    port_return:    float    # retorno total del mes (pesos previos × retornos adj.)
    proj_yield:     float    # yield proyectado por el optimizador (Σwᵢyᵢ/Σwᵢ)
    port_vol:       float    # volatilidad anualizada proyectada (√(w'Σw))
    port_mu:        float    # retorno esperado anualizado (w'μ)
    sum_weights:    float    # Σwᵢ — fracción del capital en equidades
    n_positions:    int      # acciones con w > 0
    mu_method:      str
    feasible:       bool     # False si el optimizador relajó el yield constraint
    n_stop_loss:    int
    n_take_profit:  int


# ─────────────────────────────────────────────────────────────────────────────
# Yields históricos
# ─────────────────────────────────────────────────────────────────────────────
def build_historical_yields(
    tickers: list[str],
    prices:  pd.DataFrame,
    save:    bool = True,
    force:   bool = False,
) -> pd.DataFrame:
    """
    Para cada fin de mes y cada ticker calcula el TTM dividend yield:
        yield_t = Σ(dividendos en (t−1año, t]) / precio_t

    Descarga dividendos de yfinance una sola vez y cachea el resultado.
    Si ya existe data/processed/yields_historical.parquet (y force=False),
    lo carga y retorna.

    Nota: yfinance devuelve dividendos con timestamps tz-aware en America/New_York
    (e.g. "2025-02-12 09:30:00-05:00"). La normalización los convierte a fechas
    tz-naive a medianoche antes de la búsqueda por rango.

    Returns
    -------
    pd.DataFrame  index = month-end dates, columns = tickers, values = TTM yield
    """
    import yfinance as yf

    cache = DATA_PROCESSED / "yields_historical.parquet"
    if not force and cache.exists():
        logger.info(f"Yields históricos desde caché ({cache})")
        return pd.read_parquet(cache)

    logger.info(f"Descargando dividendos de yfinance para {len(tickers)} tickers…")
    div_map: dict[str, pd.Series] = {}
    for tk in tickers:
        try:
            raw = yf.Ticker(tk).dividends
            if raw.empty:
                continue
            # Normalizar: quitar timezone y hora → fecha tz-naive a medianoche
            if raw.index.tz is not None:
                raw.index = raw.index.tz_convert(None)
            raw.index = raw.index.normalize()
            div_map[tk] = raw
            logger.debug(f"[{tk}] {len(raw)} dividendos")
        except Exception as exc:
            logger.debug(f"[{tk}] dividendos no disponibles: {exc}")

    logger.info(f"Dividendos descargados: {len(div_map)} de {len(tickers)} tickers")

    # Para cada ticker: TTM yield en cada fin de mes vía búsqueda por rango de fechas
    month_ends     = prices.resample("ME").last().index
    prices_monthly = prices.resample("ME").last()
    result: dict[str, pd.Series] = {}

    for tk in tickers:
        if tk not in prices_monthly.columns:
            continue

        px_series = prices_monthly[tk]

        if tk not in div_map:
            result[tk] = pd.Series(0.0, index=month_ends)
            continue

        divs = div_map[tk]
        tk_yields: dict[pd.Timestamp, float] = {}

        for date in month_ends:
            px = float(px_series.get(date, np.nan))
            if np.isnan(px) or px <= 0.0:
                tk_yields[date] = np.nan
                continue
            cutoff   = date - pd.DateOffset(years=1)
            ttm_sum  = float(divs[(divs.index > cutoff) & (divs.index <= date)].sum())
            tk_yields[date] = ttm_sum / px

        result[tk] = pd.Series(tk_yields)

    df = pd.DataFrame(result).sort_index().fillna(0.0).clip(lower=0.0)

    if save:
        df.to_parquet(cache)
        logger.info(f"Yields históricos guardados → {cache}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Motor principal
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(
    start:       str   = BACKTEST_START,
    end:         str  | None = None,
    lambda_risk: float = LAMBDA_RISK,
    verbose:     bool  = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el backtest mensual desde `start` hasta `end` (o hoy).

    Returns
    -------
    records_df   : pd.DataFrame — una fila por mes, métricas del portafolio
    weights_df   : pd.DataFrame — pesos mensuales (index=fecha, columns=tickers)
    """
    t0 = time.time()

    # ── 1. Cargar datos ───────────────────────────────────────────────────────
    import glob as _glob

    universe_files = sorted(_glob.glob(str(DATA_RAW / "universe_*.csv")))
    if not universe_files:
        raise FileNotFoundError("Ejecuta 01_universe.py primero.")
    universe  = pd.read_csv(universe_files[-1]).set_index("ticker")
    tickers   = universe.index.tolist()

    prices     = pd.read_parquet(DATA_PROCESSED / "prices.parquet")
    garch_vols = pd.read_parquet(DATA_PROCESSED / "garch_vols.parquet")

    hist_yields = build_historical_yields(tickers, prices)

    # Precios y retornos mensuales (total return via adjusted prices)
    prices_monthly = prices.resample("ME").last()
    returns_monthly = prices_monthly.pct_change()

    # Fechas de rebalanceo: fin de mes desde start hasta end
    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end) if end else prices.index[-1]
    all_month_ends = prices_monthly.index
    rebal_dates    = all_month_ends[
        (all_month_ends >= start_dt) & (all_month_ends <= end_dt)
    ]

    logger.info(
        f"Backtest: {rebal_dates[0].date()} → {rebal_dates[-1].date()} "
        f"({len(rebal_dates)} meses)"
    )

    # ── 2. Estado inicial ────────────────────────────────────────────────────
    position_book: dict[str, Position] = {}
    prev_weights:  pd.Series | None    = None
    mu_method      = DEFAULT_MU_METHOD
    months_since_cv = 0

    records:      list[MonthlyRecord] = []
    weights_rows: dict[pd.Timestamp, pd.Series] = {}

    # ── 3. Bucle mensual ─────────────────────────────────────────────────────
    for i, t in enumerate(rebal_dates):

        # ── 3a. Re-entrenar CV cada CV_RETRAIN_FREQ meses ─────────────────
        if months_since_cv == 0 or months_since_cv >= CV_RETRAIN_FREQ:
            ret_so_far = returns_monthly.loc[:t].dropna(how="all")
            if len(ret_so_far) >= LOOKBACK_MONTHS + 12:
                cv = walk_forward_cv(ret_so_far)
                mu_method = cv["best_method"]
                logger.info(f"[{t.date()}] CV → método μ: {mu_method}")
            months_since_cv = 0

        # ── 3b. Compute features (sin lookahead) ──────────────────────────
        try:
            features = compute_features(
                prices_daily  = prices,
                as_of_date    = t,
                mu_method     = mu_method,
                window_months = LOOKBACK_MONTHS,
                garch_vols    = garch_vols,
            )
        except ValueError as exc:
            logger.warning(f"[{t.date()}] features insuficientes: {exc} — skip")
            months_since_cv += 1
            continue

        # ── 3c. Yields históricos en t ────────────────────────────────────
        valid_yield_dates = hist_yields.index[hist_yields.index <= t]
        if len(valid_yield_dates) == 0:
            logger.warning(f"[{t.date()}] Sin yields históricos — skip")
            months_since_cv += 1
            continue
        yields_t = hist_yields.loc[valid_yield_dates[-1]]

        # Tickers con datos completos (μ, Σ, yield) en este mes
        eligible = (
            features.mu.index
            .intersection(features.cov.index)
            .intersection(yields_t.dropna().index)
        )
        if len(eligible) == 0:
            logger.warning(f"[{t.date()}] Sin tickers elegibles — skip")
            months_since_cv += 1
            continue

        # ── 3d. Retorno del mes t (con pesos de t−1) ──────────────────────
        if prev_weights is not None and t in returns_monthly.index:
            ret_t   = returns_monthly.loc[t]
            common  = prev_weights.index.intersection(ret_t.dropna().index)
            port_ret = float((prev_weights[common] * ret_t[common]).sum())
            # Cash no invertida: retorno = 0 (conservador; Sharpe usa T-bill en 06)
        else:
            port_ret = 0.0

        # ── 3e. Señales sobre posiciones abiertas ─────────────────────────
        sigma_t = garch_vols.loc[
            garch_vols.index[garch_vols.index <= t][-1]
        ]
        curr_px = prices.loc[t] if t in prices.index else prices.iloc[-1]

        if position_book:
            signals = apply_signals(position_book, curr_px, sigma_t)
            sl_tickers = signals.stop_loss_tickers
            tp_tickers = signals.take_profit_tickers
        else:
            sl_tickers = []
            tp_tickers = []

        # ── 3f. Filtrar universo y configurar overrides ───────────────────
        # Stop-loss: excluidos este mes
        # Take-profit: max_weight/2 vía override
        opt_eligible = [tk for tk in eligible if tk not in sl_tickers]
        tp_overrides = {tk: MAX_WEIGHT / 2.0 for tk in tp_tickers
                        if tk in opt_eligible}

        if len(opt_eligible) == 0:
            logger.warning(f"[{t.date()}] Todos los tickers en stop-loss — skip")
            months_since_cv += 1
            continue

        mu_t   = features.mu.loc[opt_eligible]
        cov_t  = features.cov.loc[opt_eligible, opt_eligible]
        y_t    = yields_t.loc[opt_eligible]

        # ── 3g. Optimizador QP ────────────────────────────────────────────
        try:
            opt = optimize_portfolio(
                mu                    = mu_t,
                cov                   = cov_t,
                yields                = y_t,
                min_yield             = MIN_YIELD,
                max_weight            = MAX_WEIGHT,
                max_weight_per_ticker = tp_overrides if tp_overrides else None,
                lambda_risk           = lambda_risk,
                as_of_date            = t,
            )
        except RuntimeError as exc:
            logger.error(f"[{t.date()}] Optimizador falló: {exc} — pesos anteriores")
            months_since_cv += 1
            continue

        # ── 3h. Actualizar posición y registrar ───────────────────────────
        position_book = update_position_book(
            position_book, opt.weights, curr_px, t
        )
        prev_weights = opt.weights[opt.weights > WEIGHT_TOL]

        record = MonthlyRecord(
            date          = t,
            port_return   = port_ret,
            proj_yield    = opt.port_yield,
            port_vol      = opt.port_vol,
            port_mu       = opt.port_mu,
            sum_weights   = opt.sum_weights,
            n_positions   = int((opt.weights > WEIGHT_TOL).sum()),
            mu_method     = mu_method,
            feasible      = opt.feasible,
            n_stop_loss   = len(sl_tickers),
            n_take_profit = len(tp_tickers),
        )
        records.append(record)
        weights_rows[t] = opt.weights.rename(t)

        if verbose:
            logger.info(
                f"[{t.date()}] ret={port_ret:+.2%}  yield={opt.port_yield:.2%}  "
                f"vol={opt.port_vol:.2%}  n={record.n_positions}  "
                f"SL={len(sl_tickers)}  TP={len(tp_tickers)}  "
                f"{'INFEASIBLE' if not opt.feasible else ''}"
            )

        months_since_cv += 1

    elapsed = time.time() - t0
    logger.info(f"Backtest completado en {elapsed:.1f}s — {len(records)} meses")

    # ── 4. Construir DataFrames de salida ────────────────────────────────────
    records_df = pd.DataFrame(records).set_index("date")

    weights_df = (
        pd.DataFrame(weights_rows).T
        .sort_index()
        .fillna(0.0)
    )

    return records_df, weights_df


# ─────────────────────────────────────────────────────────────────────────────
# Persistencia
# ─────────────────────────────────────────────────────────────────────────────
def save_results(
    records_df: pd.DataFrame,
    weights_df: pd.DataFrame,
) -> None:
    """
    Guarda los resultados del backtest en outputs/ y data/processed/.

    Archivos generados:
      outputs/portfolio_history.csv    — métricas mensuales
      outputs/weights_history.parquet  — pesos (fecha × ticker)
    """
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    records_df.to_csv(OUTPUTS / "portfolio_history.csv")
    weights_df.to_parquet(OUTPUTS / "weights_history.parquet")

    logger.info(f"Resultados guardados → {OUTPUTS}")


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución directa
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ── Correr backtest ───────────────────────────────────────────────────────
    records_df, weights_df = run_backtest()

    # ── Descargar benchmark SPY ───────────────────────────────────────────────
    try:
        import yfinance as yf
        spy_raw = yf.download("SPY", start="2016-01-01", auto_adjust=True, progress=False)
        spy_m   = spy_raw["Close"].resample("ME").last().pct_change().dropna()
        spy_m.name = "SPY"
        spy_m.to_frame().to_parquet(DATA_PROCESSED / "spy_returns.parquet")
    except Exception as exc:
        logger.warning(f"No se pudo descargar SPY: {exc}")
        spy_m = None

    # ── Guardar ───────────────────────────────────────────────────────────────
    save_results(records_df, weights_df)

    # ── Resumen de resultados ─────────────────────────────────────────────────
    r       = records_df["port_return"]
    n       = len(r)
    cagr    = float((1 + r).prod() ** (12 / n) - 1) if n > 0 else 0.0
    vol_ann = float(r.std() * np.sqrt(12))
    sharpe  = cagr / vol_ann if vol_ann > 0 else 0.0

    cum_ret = (1 + r).cumprod()
    peak    = cum_ret.cummax()
    max_dd  = float((cum_ret / peak - 1).min())

    yield_ok = (records_df["proj_yield"] >= MIN_YIELD - 1e-4).mean()

    print("\n" + "=" * 70)
    print(f"BACKTEST  {records_df.index[0].date()} → {records_df.index[-1].date()}")
    print("=" * 70)
    print(f"  Meses totales         : {n}")
    print(f"  CAGR                  : {cagr:.2%}")
    print(f"  Vol anualizada        : {vol_ann:.2%}")
    print(f"  Sharpe (sin Rf)       : {sharpe:.2f}")
    print(f"  Max Drawdown          : {max_dd:.2%}")
    print(f"  Yield ≥ 3% cumplido   : {yield_ok:.1%} de los meses")
    print(f"  Infeasible (yield)    : {(~records_df['feasible']).sum()} meses")
    print(f"  Stop-loss total       : {records_df['n_stop_loss'].sum()}")
    print(f"  Take-profit total     : {records_df['n_take_profit'].sum()}")
    print(f"  Posiciones promedio   : {records_df['n_positions'].mean():.1f}")
    print(f"  Capital invertido prom: {records_df['sum_weights'].mean():.1%}")

    if spy_m is not None:
        common_idx = r.index.intersection(spy_m.index)
        spy_r      = spy_m.loc[common_idx]
        port_r     = r.loc[common_idx]
        spy_cagr   = float((1 + spy_r).prod() ** (12 / len(spy_r)) - 1)
        spy_vol    = float(spy_r.std() * np.sqrt(12))
        print(f"\n  SPY CAGR              : {spy_cagr:.2%}")
        print(f"  SPY Vol               : {spy_vol:.2%}")
        print(f"  Exceso CAGR           : {cagr - spy_cagr:+.2%}")

    # Últimas 6 filas
    print("\n  Últimos 6 meses:")
    tail = records_df[["port_return", "proj_yield", "port_vol",
                        "n_positions", "feasible"]].tail(6)
    tail_display = tail.copy()
    for col in ["port_return", "proj_yield", "port_vol"]:
        tail_display[col] = (tail[col] * 100).round(2).astype(str) + "%"
    print(tail_display.to_string())


if __name__ == "__main__":
    main()
