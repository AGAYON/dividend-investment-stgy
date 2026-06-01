"""
02_features.py — Black-Litterman + Ledoit-Wolf
Actinver · Estrategia Investor US Equities

Para cada mes de rebalanceo:
  1. Descarga retornos mensuales de los últimos 36 meses (ventana rolling).
  2. Estima la matriz de covarianza con Ledoit-Wolf shrinkage.
  3. Calcula implied returns CAPM (prior) y posterior Black-Litterman
     usando el dividend yield TTM como view cuantitativa por acción.
     λ=3.0, τ=0.05, P=identidad, Ω=τ·diag(Σ).

Output:
  data/processed/features_YYYY_MM.parquet  — ticker | mu_bl | sigma_diag | dividend_yield_ttm
  data/processed/cov_YYYY_MM.parquet       — Sigma (n×n), índice y columnas = tickers
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Parámetros por defecto
# ---------------------------------------------------------------------------
LOOKBACK_MONTHS = 36
MIN_OBS_MONTHS = 24
LAMBDA_BL = 3.0
TAU = 0.05

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_monthly_returns(
    tickers: list[str],
    end_date: pd.Timestamp,
    lookback_months: int,
) -> pd.DataFrame:
    """
    Descarga precios ajustados diarios, resamples a cierre de mes y calcula
    log-retornos. Devuelve DataFrame shape (n_months, n_tickers).
    """
    # +2 meses de buffer para asegurar lookback_months retornos completos
    start_date = end_date - pd.DateOffset(months=lookback_months + 2)

    logger.info(
        f"Descargando precios: {len(tickers)} tickers | "
        f"{start_date.date()} → {end_date.date()}"
    )

    raw = yf.download(
        tickers=tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError("yfinance no devolvió datos de precios")

    # Extraer Close — yfinance con múltiples tickers devuelve MultiIndex
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    # Resamplear a cierre de mes, solo hasta end_date
    monthly = close.resample("ME").last().loc[:end_date]

    # Log-retornos mensuales
    log_ret = np.log(monthly / monthly.shift(1)).dropna(how="all")

    logger.info(f"Retornos mensuales calculados: {log_ret.shape[0]} meses × {log_ret.shape[1]} tickers")
    return log_ret


def _ledoit_wolf(returns: pd.DataFrame) -> np.ndarray:
    """Ajusta Ledoit-Wolf y devuelve la matriz de covarianza Sigma (n×n)."""
    lw = LedoitWolf()
    lw.fit(returns.values)
    logger.info(f"Ledoit-Wolf — shrinkage coeff: {lw.shrinkage_:.4f}")
    return lw.covariance_


def _black_litterman(
    sigma: np.ndarray,
    w_mkt: np.ndarray,
    q_views: np.ndarray,
    lambda_bl: float,
    tau: float,
) -> np.ndarray:
    """
    Black-Litterman posterior con P = identidad (una view por activo).

    Prior:     Π = λ · Σ · w_mkt
    Views:     Q = dividend yield TTM por acción
    Omega:     Ω = τ · diag(Σ)   (incertidumbre proporcional a varianza propia)
    Posterior: μ_BL = [(τΣ)⁻¹ + Ω⁻¹]⁻¹ · [(τΣ)⁻¹Π + Ω⁻¹Q]
    """
    # Implied equilibrium returns
    pi = lambda_bl * sigma @ w_mkt

    # Ω diagonal — incertidumbre proporcional a varianza de cada activo
    omega_diag = tau * np.diag(sigma)
    omega_inv = np.diag(1.0 / omega_diag)

    # (τΣ)⁻¹
    tau_sigma_inv = np.linalg.inv(tau * sigma)

    # Posterior
    M = tau_sigma_inv + omega_inv
    rhs = tau_sigma_inv @ pi + omega_inv @ q_views
    mu_bl = np.linalg.solve(M, rhs)

    return mu_bl


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def get_features(
    rebalance_date: str | pd.Timestamp,
    universe_df: pd.DataFrame,
    lookback_months: int = LOOKBACK_MONTHS,
    min_obs_months: int = MIN_OBS_MONTHS,
    lambda_bl: float = LAMBDA_BL,
    tau: float = TAU,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula Black-Litterman (μ_BL) y Ledoit-Wolf (Σ) para el mes de rebalanceo.

    Parámetros
    ----------
    rebalance_date : fecha de rebalanceo (último día hábil del mes)
    universe_df    : universo elegible — columnas [ticker, market_cap, dividend_yield, ...]
    lookback_months: ventana rolling para Ledoit-Wolf y B-L (default 36)
    min_obs_months : mínimo de meses de historia para incluir un stock (default 24)
    lambda_bl      : coeficiente de aversión al riesgo λ (default 3.0)
    tau            : escalar de incertidumbre del prior τ (default 0.05)
    save           : guarda parquets en data/processed/

    Retorna
    -------
    features_df : pd.DataFrame — ticker | mu_bl | sigma_diag | dividend_yield_ttm
    cov_df      : pd.DataFrame — Sigma (n×n), índice y columnas = tickers
    """
    rebalance_date = pd.Timestamp(rebalance_date)
    tickers = universe_df["ticker"].tolist()

    logger.info(
        f"get_features | {rebalance_date.strftime('%Y-%m')} | "
        f"{len(tickers)} tickers | λ={lambda_bl} | τ={tau} | lookback={lookback_months}m"
    )

    # 1. Retornos mensuales
    log_ret = _download_monthly_returns(tickers, rebalance_date, lookback_months)

    # 2. Filtrar tickers con suficiente historia
    obs_count = log_ret.notna().sum()
    valid_cols = obs_count[obs_count >= min_obs_months].index.tolist()
    # Mantener solo tickers que están en el universo (por si yfinance renombró alguno)
    valid_cols = [t for t in valid_cols if t in tickers]

    dropped = set(tickers) - set(valid_cols)
    if dropped:
        logger.info(
            f"Excluidos por historia insuficiente (<{min_obs_months}m): {sorted(dropped)}"
        )

    if len(valid_cols) < 2:
        raise ValueError(
            f"Solo {len(valid_cols)} tickers con historia suficiente — "
            "no se puede estimar covarianza"
        )

    # Usar solo filas completas (sin NaN en ningún ticker válido)
    returns_clean = log_ret[valid_cols].dropna()
    logger.info(
        f"Matriz de retornos final: {returns_clean.shape[0]} meses × {len(valid_cols)} tickers"
    )

    if returns_clean.shape[0] < min_obs_months:
        raise ValueError(
            f"Solo {returns_clean.shape[0]} meses de retornos completos — "
            f"mínimo requerido: {min_obs_months}"
        )

    # 3. Ledoit-Wolf
    sigma = _ledoit_wolf(returns_clean)

    # 4. Pesos de mercado dentro del universo elegible (relativo)
    univ_idx = universe_df.set_index("ticker")
    mktcap = np.array([univ_idx.loc[t, "market_cap"] for t in valid_cols], dtype=float)
    w_mkt = mktcap / mktcap.sum()

    # 5. Views: dividend yield TTM por acción
    q_views = np.array([univ_idx.loc[t, "dividend_yield"] for t in valid_cols], dtype=float)

    # 6. Black-Litterman posterior
    mu_bl = _black_litterman(sigma, w_mkt, q_views, lambda_bl, tau)

    logger.info(
        f"μ_BL — media: {mu_bl.mean():.4f}  "
        f"min: {mu_bl.min():.4f}  max: {mu_bl.max():.4f}"
    )

    # 7. Construir DataFrames de salida
    features_df = pd.DataFrame({
        "ticker": valid_cols,
        "mu_bl": mu_bl,
        "sigma_diag": np.diag(sigma),
        "dividend_yield_ttm": q_views,
    })

    cov_df = pd.DataFrame(sigma, index=valid_cols, columns=valid_cols)

    # 8. Guardar
    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        suffix = rebalance_date.strftime("%Y_%m")
        feat_path = DATA_PROCESSED / f"features_{suffix}.parquet"
        cov_path = DATA_PROCESSED / f"cov_{suffix}.parquet"
        features_df.to_parquet(feat_path, index=False)
        cov_df.to_parquet(cov_path)
        logger.info(f"Guardado → {feat_path}")
        logger.info(f"Guardado → {cov_path}")

    return features_df, cov_df


# ---------------------------------------------------------------------------
# Ejecución directa: usa el universo CSV más reciente de data/raw/
# ---------------------------------------------------------------------------

def _load_latest_universe() -> pd.DataFrame:
    csvs = sorted(DATA_RAW.glob("universe_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No se encontró ningún universe_*.csv en {DATA_RAW}\n"
            "Ejecuta primero: python src/01_universe.py"
        )
    path = csvs[-1]
    logger.info(f"Universo cargado: {path.name}")
    return pd.read_csv(path)


if __name__ == "__main__":
    universe = _load_latest_universe()
    rebalance_dt = pd.Timestamp.today().normalize()

    features, cov = get_features(
        rebalance_date=rebalance_dt,
        universe_df=universe,
    )

    print("\n" + "=" * 70)
    print(f"FEATURES — {rebalance_dt.strftime('%Y-%m')}")
    print("=" * 70)
    print(features.to_string(index=False))
    print("=" * 70)
    print(f"\nMatriz de covarianza: {cov.shape}")
    print(f"Tickers incluidos:   {len(features)}")
    print(f"Yield ponderado implícito: {(features['dividend_yield_ttm'] * (universe.set_index('ticker').loc[features['ticker'], 'market_cap'].values / universe['market_cap'].sum())).sum():.2%}")
