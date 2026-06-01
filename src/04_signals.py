"""
04_signals.py — Stop-loss / Take-profit por acción (EWMA)
Actinver · Estrategia Investor US Equities

Para cada posición abierta en el mes t:
  1. Retorno acumulado desde entrada: r_acum = (P_t / P_t0) - 1
  2. Volatilidad EWMA mensual span=12: σ = ewm(span=12).std().iloc[-1]
  3. Señal al cierre del mes:
       r_acum >  +tp_threshold · σ  →  take-profit → liquidar
       r_acum <  -sl_threshold · σ  →  stop-loss   → liquidar  (sl_threshold positivo)
       en rango                      →  hold

El cash generado por liquidaciones se mantiene hasta el siguiente rebalanceo.

Output:
    data/processed/signals_YYYY_MM.parquet
        ticker | entry_month | r_acum | sigma_ewma | signal (hold/stop/take)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Parámetros por defecto
# ---------------------------------------------------------------------------
EWMA_SPAN = 12
SL_THRESHOLD = 1.0   # positivo: se aplica como r_acum < -SL_THRESHOLD * σ
TP_THRESHOLD = 1.0   # positivo: se aplica como r_acum > +TP_THRESHOLD * σ
LOOKBACK_MONTHS = 36

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_monthly_closes(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Descarga precios ajustados diarios y resamples a cierre de mes.
    Devuelve DataFrame (months × tickers).
    """
    raw = yf.download(
        tickers=tickers,
        start=(start - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise ValueError("yfinance no devolvió datos de precios")

    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    return close.resample("ME").last().loc[:end]


def _ewma_vol(monthly_prices: pd.Series, span: int) -> float:
    """
    Volatilidad EWMA mensual para una serie de precios mensuales.
    σ = ewm(span).std() sobre log-retornos mensuales → último valor.
    """
    log_ret = np.log(monthly_prices / monthly_prices.shift(1)).dropna()
    if len(log_ret) < 2:
        return np.nan
    return float(log_ret.ewm(span=span).std().iloc[-1])


def _price_at_date(monthly_closes: pd.Series, target: pd.Timestamp) -> float:
    """
    Precio de cierre mensual más cercano (hacia adelante) a target.
    Usado para obtener el precio de entrada a partir de la fecha del rebalanceo.
    """
    candidates = monthly_closes.loc[monthly_closes.index >= target]
    if candidates.empty:
        candidates = monthly_closes
    return float(candidates.iloc[0])


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def evaluate_signals(
    positions_df: pd.DataFrame,
    eval_date: str | pd.Timestamp,
    sl_threshold: float = SL_THRESHOLD,
    tp_threshold: float = TP_THRESHOLD,
    ewma_span: int = EWMA_SPAN,
    lookback_months: int = LOOKBACK_MONTHS,
    save: bool = True,
) -> pd.DataFrame:
    """
    Evalúa señales stop-loss / take-profit para cada posición abierta.

    Parámetros
    ----------
    positions_df  : DataFrame con columnas [ticker, entry_date, entry_price]
                    entry_date  : pd.Timestamp del mes de entrada (último día hábil)
                    entry_price : float, precio ajustado al cierre de entrada
    eval_date     : fecha de evaluación (último día hábil del mes t)
    sl_threshold  : multiplicador σ para stop-loss (default 1.0, positivo)
    tp_threshold  : multiplicador σ para take-profit (default 1.0, positivo)
    ewma_span     : span mensual para EWMA (default 12)
    lookback_months: meses de historia para estimar σ_EWMA (default 36)
    save          : guarda signals_YYYY_MM.parquet en data/processed/

    Retorna
    -------
    pd.DataFrame:
        ticker | entry_month | r_acum | sigma_ewma | signal (hold/stop/take)
    """
    eval_date = pd.Timestamp(eval_date)
    tickers = positions_df["ticker"].tolist()

    logger.info(
        f"evaluate_signals | {eval_date.strftime('%Y-%m')} | "
        f"{len(tickers)} posiciones | SL=±{sl_threshold}σ | TP=±{tp_threshold}σ"
    )

    # Fecha más antigua de entrada → define cuánto historial necesitamos
    min_entry = positions_df["entry_date"].min()
    hist_start = min(
        pd.Timestamp(min_entry),
        eval_date - pd.DateOffset(months=lookback_months + 1),
    )

    # Descargar precios mensuales
    monthly = _fetch_monthly_closes(tickers, hist_start, eval_date)

    records = []
    for _, row in positions_df.iterrows():
        ticker = row["ticker"]
        entry_date = pd.Timestamp(row["entry_date"])
        entry_price = float(row["entry_price"])

        if ticker not in monthly.columns:
            logger.warning(f"[{ticker}] sin datos de precio — señal: hold (por defecto)")
            records.append({
                "ticker": ticker,
                "entry_month": entry_date.strftime("%Y-%m"),
                "r_acum": np.nan,
                "sigma_ewma": np.nan,
                "signal": "hold",
            })
            continue

        prices = monthly[ticker].dropna()

        # Precio actual (último cierre del mes de evaluación)
        if prices.empty:
            logger.warning(f"[{ticker}] precios vacíos — señal: hold")
            records.append({
                "ticker": ticker,
                "entry_month": entry_date.strftime("%Y-%m"),
                "r_acum": np.nan,
                "sigma_ewma": np.nan,
                "signal": "hold",
            })
            continue

        current_price = float(prices.iloc[-1])

        # Si entry_price no fue provisto (0 o NaN), inferir del historial
        if entry_price <= 0 or np.isnan(entry_price):
            entry_price = _price_at_date(prices, entry_date)

        # Retorno acumulado desde entrada
        r_acum = (current_price / entry_price) - 1.0

        # Volatilidad EWMA sobre el historial disponible para este ticker
        # Usamos los últimos lookback_months de historia
        hist_window = prices.loc[
            prices.index >= (eval_date - pd.DateOffset(months=lookback_months + 1))
        ]
        sigma_ewma = _ewma_vol(hist_window, ewma_span)

        # Evaluar señal
        if np.isnan(sigma_ewma) or sigma_ewma <= 0:
            signal = "hold"
        elif r_acum > tp_threshold * sigma_ewma:
            signal = "take"
        elif r_acum < -sl_threshold * sigma_ewma:
            signal = "stop"
        else:
            signal = "hold"

        logger.info(
            f"  {ticker:<6}  entrada={entry_date.strftime('%Y-%m')}  "
            f"r_acum={r_acum:+.2%}  σ_ewma={sigma_ewma:.2%}  "
            f"banda=[{-sl_threshold*sigma_ewma:+.2%}, {+tp_threshold*sigma_ewma:+.2%}]  "
            f"→ {signal.upper()}"
        )

        records.append({
            "ticker": ticker,
            "entry_month": entry_date.strftime("%Y-%m"),
            "r_acum": round(r_acum, 6),
            "sigma_ewma": round(sigma_ewma, 6),
            "signal": signal,
        })

    signals_df = pd.DataFrame(records)

    stops = (signals_df["signal"] == "stop").sum()
    takes = (signals_df["signal"] == "take").sum()
    holds = (signals_df["signal"] == "hold").sum()
    logger.info(
        f"Resumen: {holds} hold | {stops} stop-loss | {takes} take-profit"
    )

    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        suffix = eval_date.strftime("%Y_%m")
        path = DATA_PROCESSED / f"signals_{suffix}.parquet"
        signals_df.to_parquet(path, index=False)
        logger.info(f"Guardado → {path}")

    return signals_df


# ---------------------------------------------------------------------------
# Ejecución directa: carga el weights más reciente y simula un mes después
# ---------------------------------------------------------------------------

def _load_latest_weights() -> tuple[pd.DataFrame, pd.Timestamp]:
    files = sorted(DATA_PROCESSED.glob("weights_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No se encontró ningún weights_*.parquet en {DATA_PROCESSED}\n"
            "Ejecuta primero: python src/03_optimizer.py"
        )
    path = files[-1]
    logger.info(f"Cargando: {path.name}")
    df = pd.read_parquet(path)
    # Inferir fecha del nombre (weights_YYYY_MM.parquet)
    parts = path.stem.split("_")
    entry_dt = pd.Timestamp(f"{parts[1]}-{parts[2]}-01") + pd.offsets.MonthEnd(0)
    return df, entry_dt


if __name__ == "__main__":
    weights, entry_dt = _load_latest_weights()

    # Solo posiciones con peso real (> 0.01%)
    active = weights[weights["weight"] > 1e-4].copy()

    # Necesitamos precios de entrada — los descargamos del mes de la optimización
    logger.info(f"Descargando precios de entrada para {len(active)} posiciones...")
    monthly_entry = _fetch_monthly_closes(
        active["ticker"].tolist(),
        entry_dt - pd.DateOffset(months=1),
        entry_dt,
    )

    positions = []
    for _, row in active.iterrows():
        tk = row["ticker"]
        if tk in monthly_entry.columns:
            ep = float(monthly_entry[tk].dropna().iloc[-1])
        else:
            ep = 0.0
        positions.append({
            "ticker": tk,
            "entry_date": entry_dt,
            "entry_price": ep,
        })
    positions_df = pd.DataFrame(positions)

    # Evaluar señales un mes después de la entrada
    eval_dt = entry_dt + pd.offsets.MonthEnd(1)
    # Si eval_dt es futuro, usar hoy
    eval_dt = min(eval_dt, pd.Timestamp.today().normalize())

    signals = evaluate_signals(
        positions_df=positions_df,
        eval_date=eval_dt,
    )

    print("\n" + "=" * 70)
    print(f"SEÑALES — evaluadas al {eval_dt.strftime('%Y-%m-%d')}")
    print("=" * 70)
    print(signals.to_string(index=False))
    print("=" * 70)
    stops = (signals["signal"] == "stop").sum()
    takes = (signals["signal"] == "take").sum()
    holds = (signals["signal"] == "hold").sum()
    print(f"\nHold: {holds}  |  Stop-loss: {stops}  |  Take-profit: {takes}")
    if stops + takes > 0:
        triggered = signals[signals["signal"] != "hold"]
        cash_pct = active.set_index("ticker").loc[
            triggered["ticker"], "weight"
        ].sum()
        print(f"Capital liberado a cash: {cash_pct:.1%} del portafolio")
