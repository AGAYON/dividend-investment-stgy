"""
03_optimizer.py — Optimización cuadrática media-varianza
Actinver · Estrategia Investor US Equities

Formulación del QP (quadprog.solve_qp):
    min  (1/2) w' Σ w  -  γ · μ_BL' · w
    s.t.
        Σᵢ wᵢ = 1                     fully invested (igualdad)
        wᵢ ≥ 0          ∀i            long-only
        wᵢ ≤ MAX_WEIGHT  ∀i           concentración máxima 5%
        Σᵢ wᵢ · dyᵢ ≥ MIN_YIELD       yield portafolio ≥ 3%

Protocolo de infeasibility (sección 5 del README):
    Si solve_qp lanza excepción → calcular yield máximo alcanzable
    (greedy sobre restricciones de peso sin constraint de yield),
    loguear en infeasibility_log.csv y devolver None.

Output:
    data/processed/weights_YYYY_MM.parquet
        ticker | weight | expected_return | contribution_yield
"""

import csv
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import quadprog

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
INFEASIBILITY_LOG = DATA_PROCESSED / "infeasibility_log.csv"

# ---------------------------------------------------------------------------
# Parámetros por defecto
# ---------------------------------------------------------------------------
MIN_YIELD = 0.03
MAX_WEIGHT = 0.05
GAMMA = 1.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_qp_inputs(
    sigma: np.ndarray,
    mu: np.ndarray,
    dy: np.ndarray,
    gamma: float,
    min_yield: float,
    max_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Construye G, a, C, b, meq para quadprog.solve_qp.

    solve_qp resuelve: min (1/2) x'Gx - a'x  s.t. C'x >= b
    Las primeras `meq` restricciones son igualdades.

    Restricciones (en orden):
      [0]      igualdad:  Σwᵢ = 1
      [1..n]   lower:     wᵢ ≥ 0
      [n+1..2n] upper:    -wᵢ ≥ -max_weight  (≡ wᵢ ≤ max_weight)
      [2n+1]   yield:     Σwᵢ·dyᵢ ≥ min_yield
    """
    n = len(mu)

    # G: pequeño jitter para garantizar positive-definiteness numérica
    G = sigma + 1e-8 * np.eye(n)
    G = (G + G.T) / 2  # forzar simetría exacta

    # a: coeficientes de retorno
    a = gamma * mu

    # C: (n, 2n+2) — cada columna es una restricción
    C = np.zeros((n, 2 * n + 2))

    # Col 0: igualdad suma = 1
    C[:, 0] = 1.0

    # Cols 1..n: lower bounds wᵢ ≥ 0
    C[:, 1 : n + 1] = np.eye(n)

    # Cols n+1..2n: upper bounds -wᵢ ≥ -max_weight
    C[:, n + 1 : 2 * n + 1] = -np.eye(n)

    # Col 2n+1: yield constraint
    C[:, 2 * n + 1] = dy

    # b: RHS
    b = np.zeros(2 * n + 2)
    b[0] = 1.0                      # suma = 1
    # b[1..n] = 0                   # lower bounds (ya en cero)
    b[n + 1 : 2 * n + 1] = -max_weight  # upper bounds
    b[2 * n + 1] = min_yield            # yield mínimo

    meq = 1  # solo la primera restricción es igualdad

    return G, a, C, b, meq


def _max_achievable_yield(dy: np.ndarray, max_weight: float) -> tuple[float, np.ndarray]:
    """
    Calcula el yield máximo posible con las restricciones de peso
    (sum=1, long-only, max_weight), sin el constraint de yield.

    Solución greedy: asignar max_weight a los stocks de mayor yield
    hasta completar 100%. Válido porque el problema es separable y lineal.

    Retorna (max_yield, weights).
    """
    n = len(dy)
    w = np.zeros(n)
    sorted_idx = np.argsort(-dy)
    remaining = 1.0
    for i in sorted_idx:
        alloc = min(max_weight, remaining)
        w[i] = alloc
        remaining -= alloc
        if remaining <= 1e-10:
            break
    return float(np.dot(w, dy)), w


def _log_infeasibility(
    rebalance_date: pd.Timestamp,
    max_yield: float,
    n_eligible: int,
    reason: str,
) -> None:
    """Agrega una fila al log de infeasibility (CSV append)."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    write_header = not INFEASIBILITY_LOG.exists()
    with open(INFEASIBILITY_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["date", "max_achievable_yield", "n_eligible_stocks", "reason"])
        writer.writerow([
            rebalance_date.strftime("%Y-%m-%d"),
            round(max_yield, 6),
            n_eligible,
            reason,
        ])
    logger.warning(f"Infeasibility logueada → {INFEASIBILITY_LOG.name}")


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def optimize_portfolio(
    features_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    gamma: float = GAMMA,
    min_yield: float = MIN_YIELD,
    max_weight: float = MAX_WEIGHT,
    rebalance_date: str | pd.Timestamp | None = None,
    save: bool = True,
) -> pd.DataFrame | None:
    """
    Resuelve el QP media-varianza con las restricciones del mandato.

    Parámetros
    ----------
    features_df   : DataFrame con columnas [ticker, mu_bl, dividend_yield_ttm, ...]
    cov_df        : DataFrame (n×n) con Sigma, índice y columnas = tickers
    gamma         : aversión al riesgo en la función objetivo (default 1.0)
    min_yield     : yield mínimo del portafolio como fracción (default 0.03)
    max_weight    : peso máximo por acción (default 0.05)
    rebalance_date: fecha del mes; si None usa hoy
    save          : guarda weights_YYYY_MM.parquet en data/processed/

    Retorna
    -------
    pd.DataFrame con columnas:
        ticker | weight | expected_return | contribution_yield
    None si el QP es infeasible (ver protocolo de infeasibility).
    """
    if rebalance_date is None:
        rebalance_date = pd.Timestamp.today().normalize()
    rebalance_date = pd.Timestamp(rebalance_date)

    # Alinear tickers: solo los presentes en ambos inputs
    tickers = [t for t in features_df["ticker"].tolist() if t in cov_df.index]
    features = features_df.set_index("ticker").loc[tickers]
    sigma = cov_df.loc[tickers, tickers].values.astype(float)
    mu = features["mu_bl"].values.astype(float)
    dy = features["dividend_yield_ttm"].values.astype(float)
    n = len(tickers)

    logger.info(
        f"optimize_portfolio | {rebalance_date.strftime('%Y-%m')} | "
        f"n={n} | γ={gamma} | min_yield={min_yield:.1%} | max_w={max_weight:.1%}"
    )

    # Construir inputs del QP
    G, a, C, b, meq = _build_qp_inputs(sigma, mu, dy, gamma, min_yield, max_weight)

    # Resolver
    try:
        sol = quadprog.solve_qp(G, a, C, b, meq)
        w = sol[0]

        # Limpiar ruido numérico: clamp a [0, max_weight] y renormalizar
        w = np.clip(w, 0.0, max_weight)
        w /= w.sum()

        port_yield = float(np.dot(w, dy))
        port_var = float(w @ sigma @ w)
        port_ret = float(np.dot(w, mu))

        logger.info(
            f"  QP resuelto ✓  "
            f"yield={port_yield:.2%}  "
            f"σ_anual={np.sqrt(port_var * 12):.2%}  "
            f"μ_BL_anual={port_ret * 12:.2%}  "
            f"n_activos={np.sum(w > 1e-4)}"
        )

        weights_df = pd.DataFrame({
            "ticker": tickers,
            "weight": w,
            "expected_return": mu,
            "contribution_yield": w * dy,
        })
        weights_df = weights_df.sort_values("weight", ascending=False).reset_index(drop=True)

    except (ValueError, np.linalg.LinAlgError) as exc:
        # ----------------------------------------------------------------
        # Protocolo de infeasibility
        # ----------------------------------------------------------------
        max_yield, _ = _max_achievable_yield(dy, max_weight)
        logger.warning(
            f"  QP INFEASIBLE ({exc.__class__.__name__}: {exc})\n"
            f"  Yield máximo alcanzable sin constraint: {max_yield:.2%}\n"
            f"  N° stocks elegibles: {n}"
        )
        _log_infeasibility(
            rebalance_date=rebalance_date,
            max_yield=max_yield,
            n_eligible=n,
            reason=str(exc),
        )
        return None

    # Guardar
    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        suffix = rebalance_date.strftime("%Y_%m")
        path = DATA_PROCESSED / f"weights_{suffix}.parquet"
        weights_df.to_parquet(path, index=False)
        logger.info(f"Guardado → {path}")

    return weights_df


# ---------------------------------------------------------------------------
# Ejecución directa: carga features/cov del mes más reciente
# ---------------------------------------------------------------------------

def _load_latest(prefix: str) -> pd.DataFrame:
    files = sorted(DATA_PROCESSED.glob(f"{prefix}_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No se encontró ningún {prefix}_*.parquet en {DATA_PROCESSED}\n"
            f"Ejecuta primero: python src/02_features.py"
        )
    path = files[-1]
    logger.info(f"Cargando: {path.name}")
    return pd.read_parquet(path)


if __name__ == "__main__":
    features = _load_latest("features")
    # cov parquet tiene tickers como índice
    cov_files = sorted(DATA_PROCESSED.glob("cov_*.parquet"))
    if not cov_files:
        raise FileNotFoundError("No se encontró cov_*.parquet en data/processed/")
    cov = pd.read_parquet(cov_files[-1])
    logger.info(f"Cargando: {cov_files[-1].name}")

    # Inferir fecha del nombre del archivo (cov_YYYY_MM.parquet)
    stem = cov_files[-1].stem  # e.g. "cov_2026_05"
    parts = stem.split("_")
    rebalance_dt = pd.Timestamp(f"{parts[1]}-{parts[2]}-01") + pd.offsets.MonthEnd(0)

    weights = optimize_portfolio(
        features_df=features,
        cov_df=cov,
        rebalance_date=rebalance_dt,
    )

    if weights is None:
        print("\nQP infeasible — ver infeasibility_log.csv para detalles.")
    else:
        port_yield = weights["contribution_yield"].sum()
        port_var = 0.0  # no recalculamos aquí; ya logueado arriba

        print("\n" + "=" * 70)
        print(f"PESOS ÓPTIMOS — {rebalance_dt.strftime('%Y-%m')}")
        print("=" * 70)
        print(weights.to_string(index=False))
        print("=" * 70)
        print(f"\nYield portafolio:          {port_yield:.2%}")
        print(f"Acciones con peso > 0.01%: {(weights['weight'] > 0.0001).sum()}")
        print(f"Peso máximo:               {weights['weight'].max():.2%}  ({weights.iloc[0]['ticker']})")
        print(f"Peso mínimo activo:        {weights[weights['weight'] > 0.0001]['weight'].min():.2%}")
