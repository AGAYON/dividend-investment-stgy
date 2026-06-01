"""
01_universe.py — Construcción del universo elegible de acciones
Actinver · Estrategia Investor US Equities

Filtros aplicados sobre el S&P 500:
  1. Market cap >= $100B
  2. Dividend yield TTM >= 0.5%
  3. ADV 90 días > $50M  (volumen x precio promedio)
  4. Historial de precios disponible desde 2013-01-01
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from io import StringIO

import requests
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Constantes del universo
# ---------------------------------------------------------------------------
HISTORY_START = "2013-01-01"        # 3 años antes del backtest (ene 2016)
HISTORY_GRACE_DAYS = 45             # tolerancia: acción puede empezar hasta 45d tarde
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paso 1: obtener lista de tickers del S&P 500
# ---------------------------------------------------------------------------

def fetch_sp500_tickers() -> list[str]:
    """
    Descarga los constituyentes actuales del S&P 500 desde Wikipedia.
    Reemplaza '.' por '-' para compatibilidad con yfinance (e.g. BRK.B → BRK-B).
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    resp = requests.get(SP500_WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text), header=0)
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    logger.info(f"S&P 500: {len(tickers)} tickers descargados de Wikipedia")
    return tickers


# ---------------------------------------------------------------------------
# Paso 2: helpers de filtro por ticker
# ---------------------------------------------------------------------------

def _ttm_dividend_yield(t: yf.Ticker, current_price: float) -> float:
    """
    Calcula el dividend yield TTM (últimos 12 meses) como:
        sum(dividendos últimos 12m) / precio_actual
    Más confiable que el campo 'dividendYield' de yfinance, que a veces
    usa datos desactualizados.
    """
    try:
        divs = t.dividends
        if divs.empty or current_price <= 0:
            return 0.0
        tz = divs.index.tz
        cutoff = pd.Timestamp.now(tz=tz) - pd.DateOffset(years=1)
        ttm_total = float(divs[divs.index >= cutoff].sum())
        return ttm_total / current_price
    except Exception:
        return 0.0


def _screen_ticker(
    ticker: str,
    min_mktcap: float,
    min_yield: float,
    min_adv: float,
) -> dict | None:
    """
    Evalúa un solo ticker contra los cuatro filtros del universo.
    Devuelve un dict con sus fundamentales si pasa todos; None si falla alguno.
    """
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info  # llamada ligera, sin descargar todo el prospecto

        # --- Filtro 1: market cap ---
        mktcap = getattr(fi, "market_cap", None)
        if not mktcap or mktcap < min_mktcap:
            return None

        current_price = getattr(fi, "last_price", None)
        if not current_price or current_price <= 0:
            return None

        # --- Filtro 2: dividend yield TTM ---
        dy = _ttm_dividend_yield(t, current_price)
        if dy < min_yield:
            return None

        # --- Filtro 3: liquidez (ADV 90 días en USD) ---
        hist_90 = t.history(period="90d", auto_adjust=True)
        if hist_90.empty:
            return None
        adv = float((hist_90["Volume"] * hist_90["Close"]).mean())
        if adv < min_adv:
            return None

        # --- Filtro 4: historial desde 2013 ---
        hist_full = t.history(start=HISTORY_START, auto_adjust=True)
        if hist_full.empty:
            return None
        first_date = hist_full.index[0].tz_localize(None)
        deadline = pd.Timestamp(HISTORY_START) + pd.Timedelta(days=HISTORY_GRACE_DAYS)
        if first_date > deadline:
            return None

        return {
            "ticker": ticker,
            "market_cap": mktcap,
            "dividend_yield": round(dy, 4),
            "adv_90d": round(adv, 0),
            "history_start": first_date.strftime("%Y-%m-%d"),
            "price": round(current_price, 2),
        }

    except Exception as e:
        logger.debug(f"[{ticker}] excepción: {e}")
        return None


# ---------------------------------------------------------------------------
# Paso 3: función principal
# ---------------------------------------------------------------------------

def get_universe(
    min_mktcap: float = 100e9,
    min_yield: float = 0.005,
    min_adv: float = 50e6,
    max_workers: int = 12,
    save: bool = True,
) -> pd.DataFrame:
    """
    Construye el universo elegible aplicando los cuatro filtros del blueprint.

    Parámetros
    ----------
    min_mktcap  : capitalización mínima en USD (default 100B)
    min_yield   : dividend yield TTM mínimo  (default 3%)
    min_adv     : volumen promedio diario 90d en USD (default 50M)
    max_workers : hilos paralelos para las llamadas a yfinance
    save        : guarda CSV en data/raw/universe_YYYYMMDD.csv

    Retorna
    -------
    pd.DataFrame con columnas:
        ticker, market_cap, dividend_yield, adv_90d, history_start, price
    ordenado por market_cap descendente.
    """
    tickers = fetch_sp500_tickers()
    total = len(tickers)
    passed: list[dict] = []
    failed_reasons = {"mktcap": 0, "yield": 0, "adv": 0, "history": 0, "error": 0}

    logger.info(
        f"Iniciando screening: {total} tickers | "
        f"mktcap≥${min_mktcap/1e9:.0f}B | yield≥{min_yield:.0%} | "
        f"adv≥${min_adv/1e6:.0f}M | workers={max_workers}"
    )
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_screen_ticker, tk, min_mktcap, min_yield, min_adv): tk
            for tk in tickers
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            tk = futures[fut]
            result = fut.result()
            if result:
                passed.append(result)
                logger.info(
                    f"  ✓ {tk:<6}  mktcap=${result['market_cap']/1e9:>6.0f}B  "
                    f"yield={result['dividend_yield']:>5.2%}  "
                    f"adv=${result['adv_90d']/1e6:>5.0f}M  "
                    f"[{done}/{total}]"
                )
            else:
                if done % 50 == 0:
                    logger.info(f"  ... {done}/{total} evaluados, {len(passed)} pasaron hasta ahora")

    elapsed = time.time() - t0
    logger.info(
        f"\nScreening completado en {elapsed:.1f}s — "
        f"{len(passed)} de {total} tickers pasaron los filtros"
    )

    if not passed:
        logger.warning("Universo vacío — revisa los filtros o la conexión a yfinance")
        return pd.DataFrame()

    universe = (
        pd.DataFrame(passed)
        .sort_values("market_cap", ascending=False)
        .reset_index(drop=True)
    )

    if save:
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        date_str = datetime.today().strftime("%Y%m%d")
        path = DATA_RAW / f"universe_{date_str}.csv"
        universe.to_csv(path, index=False)
        logger.info(f"Universo guardado → {path}")

    return universe


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = get_universe()

    if df.empty:
        print("No se encontraron tickers que pasen los filtros.")
    else:
        print("\n" + "=" * 70)
        print(f"UNIVERSO ELEGIBLE — {datetime.today().strftime('%Y-%m-%d')}")
        print("=" * 70)
        print(df.to_string(index=False))
        print("=" * 70)
        print(f"\nTotal: {len(df)} acciones")
        print(f"Market cap promedio:  ${df['market_cap'].mean()/1e9:.0f}B")
        print(f"Dividend yield medio: {df['dividend_yield'].mean():.2%}")
        print(f"ADV medio 90d:        ${df['adv_90d'].mean()/1e6:.0f}M")
