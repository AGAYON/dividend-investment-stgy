"""
03_optimizer.py — Optimizador cuadrático (QP)
Actinver · Estrategia Investor US Equities

Problema:
    min  ½ w' Σ w − λ μ' w
    s.a. 0 ≤ wᵢ ≤ max_weight              (i = 1 … N)
         Σwᵢ ≤ 1
         Σwᵢ(yᵢ − min_yield) ≥ 0          yield sobre capital invertido ≥ 3%

La restricción de yield es la forma linealizada de:
    Σ(wᵢ yᵢ) / Σwᵢ ≥ min_yield
    ↔  Σwᵢ(yᵢ − min_yield) ≥ 0      [válida para Σwᵢ > 0]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import quadprog

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

NUGGET:      float = 1e-8   # diagonal perturbation → G strictly positive definite
MAX_WEIGHT:  float = 0.05   # máximo 5 % por acción (requerimiento explícito)
MIN_YIELD:   float = 0.03   # yield mínimo sobre capital invertido (requerimiento)
LAMBDA_RISK: float = 0.5    # aversión al riesgo λ en el objetivo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output container
# ─────────────────────────────────────────────────────────────────────────────
class OptimizationResult(NamedTuple):
    weights:      pd.Series          # w*, indexed by ticker (incluye ceros)
    port_yield:   float              # Σ(wᵢ yᵢ) / Σwᵢ sobre capital invertido
    port_var:     float              # w' Σ w  (anualizado)
    port_vol:     float              # sqrt(port_var)
    port_mu:      float              # w' μ  (retorno total esperado anualizado)
    sum_weights:  float              # Σwᵢ  — fracción del capital desplegado
    lambda_risk:  float              # λ usado en el objetivo
    feasible:     bool               # False si se relajó la restricción de yield
    as_of_date:   pd.Timestamp | None


# ─────────────────────────────────────────────────────────────────────────────
# Construcción de matrices de restricción
# ─────────────────────────────────────────────────────────────────────────────
def _build_constraints(
    N:             int,
    yields_arr:    np.ndarray,
    ub_per_ticker: np.ndarray,   # shape (N,) — upper bound per ticker
    min_yield:     float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye las matrices C y vectores b para quadprog.solve_qp.

    quadprog convention:  C.T @ w >= b  (todas desigualdades, meq=0)

    Devuelve dos pares (C_full, b_full) y (C_no_yield, b_no_yield) para
    poder degradar al problema sin restricción de yield si el full es infactible.
    """
    # 1. Cotas inferiores:  wᵢ ≥ 0
    C_lb = np.eye(N)
    b_lb = np.zeros(N)

    # 2. Cotas superiores: wᵢ ≤ ub_per_ticker[i]  →  −wᵢ ≥ −ub_per_ticker[i]
    C_ub = -np.eye(N)
    b_ub = -ub_per_ticker

    # 3. Budget: Σwᵢ ≤ 1  →  −Σwᵢ ≥ −1
    C_bgt = -np.ones((N, 1))
    b_bgt = np.array([-1.0])

    # 4. Yield: Σwᵢ(yᵢ − min_yield) ≥ 0
    C_yld = (yields_arr - min_yield).reshape(N, 1)
    b_yld = np.array([0.0])

    C_full     = np.column_stack([C_lb, C_ub, C_bgt, C_yld])
    b_full     = np.concatenate([b_lb, b_ub, b_bgt, b_yld])

    C_no_yield = np.column_stack([C_lb, C_ub, C_bgt])
    b_no_yield = np.concatenate([b_lb, b_ub, b_bgt])

    return C_full, b_full, C_no_yield, b_no_yield


# ─────────────────────────────────────────────────────────────────────────────
# Core: solve + wrap
# ─────────────────────────────────────────────────────────────────────────────
def _solve_and_wrap(
    G:           np.ndarray,
    a:           np.ndarray,
    C:           np.ndarray,
    b:           np.ndarray,
    tickers:     pd.Index,
    yields_a:    pd.Series,
    mu_a:        pd.Series,
    cov_a:       pd.DataFrame,
    lambda_risk: float,
    feasible:    bool,
    as_of_date:  pd.Timestamp | None,
) -> OptimizationResult:
    """
    Llama a quadprog.solve_qp y empaqueta el resultado en OptimizationResult.
    Levanta la excepción de quadprog sin atraparla — el caller decide cómo manejarla.
    """
    sol = quadprog.solve_qp(G, a, C, b, meq=0)
    w   = sol[0]

    # Los solvers QP devuelven ruido numérico pequeño negativo; se proyecta a [0, ∞)
    w = np.clip(w, 0.0, None)
    w[w < 1e-6] = 0.0

    w_series  = pd.Series(w, index=tickers)
    sum_w     = float(w_series.sum())

    if sum_w < 1e-9:
        logger.warning("Solución degenerada: todos los pesos ≈ 0.")
        port_yield = 0.0
    else:
        port_yield = float((w_series * yields_a).sum() / sum_w)

    port_var = float(w_series.values @ cov_a.values @ w_series.values)
    port_mu  = float((w_series * mu_a).sum())

    return OptimizationResult(
        weights     = w_series,
        port_yield  = port_yield,
        port_var    = port_var,
        port_vol    = float(np.sqrt(max(port_var, 0.0))),
        port_mu     = port_mu,
        sum_weights = sum_w,
        lambda_risk = lambda_risk,
        feasible    = feasible,
        as_of_date  = as_of_date,
    )


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────
def optimize_portfolio(
    mu:                    pd.Series,
    cov:                   pd.DataFrame,
    yields:                pd.Series,
    min_yield:             float                     = MIN_YIELD,
    max_weight:            float                     = MAX_WEIGHT,
    max_weight_per_ticker: dict[str, float] | None   = None,
    lambda_risk:           float                     = LAMBDA_RISK,
    as_of_date:            pd.Timestamp | None       = None,
) -> OptimizationResult:
    """
    Resuelve el QP de la estrategia Actinver via quadprog.solve_qp.

    Flujo de fallback ante infactibilidad:
      1. Verificación analítica previa: si la mejor combinación posible de activos
         no alcanza el 3 %, se resuelve sin restricción de yield (feasible=False).
      2. Si el QP con yield falla numéricamente, reintento sin yield.
      3. Si el reintento también falla, se lanza RuntimeError.

    Parameters
    ----------
    mu                    : rendimientos esperados anualizados
    cov                   : covarianza anualizada
    yields                : dividend yield TTM por ticker
    min_yield             : piso de yield sobre capital invertido  (default 3 %)
    max_weight            : peso máximo global por acción         (default 5 %)
    max_weight_per_ticker : overrides por ticker  {ticker: cap}  — usado por el
                            backtester para aplicar take-profit (w/2 ese mes)
    lambda_risk           : aversión al riesgo λ en ½w'Σw − λμ'w (default 0.5)
    as_of_date  : fecha del rebalanceo (solo para trazabilidad en el resultado)

    Returns
    -------
    OptimizationResult
    """
    # ── Alineación de índices ─────────────────────────────────────────────────
    common = mu.index.intersection(cov.index).intersection(yields.index)
    if len(common) == 0:
        raise ValueError("Sin tickers en común entre mu, cov y yields.")

    mu_a  = mu.loc[common]
    cov_a = cov.loc[common, common]
    y_a   = yields.loc[common]
    N     = len(common)

    # ── Matrices del QP ──────────────────────────────────────────────────────
    G = cov_a.values.astype(float) + np.eye(N) * NUGGET
    a = lambda_risk * mu_a.values.astype(float)

    # Per-ticker upper bounds (scalar default, overridden for take-profit tickers)
    ub_arr = np.full(N, max_weight, dtype=float)
    if max_weight_per_ticker:
        for i, tk in enumerate(common):
            if tk in max_weight_per_ticker:
                ub_arr[i] = float(max_weight_per_ticker[tk])

    C_full, b_full, C_noy, b_noy = _build_constraints(
        N, y_a.values.astype(float), ub_arr, min_yield
    )

    # ── Pre-check de factibilidad del yield ──────────────────────────────────
    # Greedy upper bound: allocate budget greedily to highest-yield tickers
    sorted_idx    = y_a.argsort()[::-1]
    budget        = 1.0
    wtd_yield_sum = 0.0
    for idx in sorted_idx:
        alloc          = min(float(ub_arr[idx]), budget)
        wtd_yield_sum += alloc * float(y_a.iloc[idx])
        budget        -= alloc
        if budget < 1e-9:
            break
    invested       = 1.0 - budget
    max_achievable = wtd_yield_sum / invested if invested > 1e-9 else 0.0

    if max_achievable < min_yield:
        logger.warning(
            f"[{as_of_date}] Restricción de yield INFACTIBLE: "
            f"max yield posible = {max_achievable:.2%} < {min_yield:.2%}. "
            "Optimizando varianza mínima sin restricción de yield."
        )
        try:
            return _solve_and_wrap(
                G, a, C_noy, b_noy, common, y_a, mu_a, cov_a,
                lambda_risk, feasible=False, as_of_date=as_of_date,
            )
        except Exception as exc:
            raise RuntimeError(
                f"QP sin restricción de yield también falló: {exc}"
            ) from exc

    # ── QP con restricción de yield ───────────────────────────────────────────
    try:
        result = _solve_and_wrap(
            G, a, C_full, b_full, common, y_a, mu_a, cov_a,
            lambda_risk, feasible=True, as_of_date=as_of_date,
        )
        logger.info(
            f"[{as_of_date}] QP OK — "
            f"{int((result.weights > 1e-4).sum())} activos, "
            f"Σw={result.sum_weights:.1%}, "
            f"yield={result.port_yield:.2%}, "
            f"vol={result.port_vol:.2%}"
        )
        return result

    except Exception as exc:
        logger.warning(
            f"[{as_of_date}] QP con yield falló ({exc}). "
            "Reintentando sin restricción de yield."
        )

    # ── Fallback: mínima varianza sin yield ───────────────────────────────────
    try:
        result = _solve_and_wrap(
            G, a, C_noy, b_noy, common, y_a, mu_a, cov_a,
            lambda_risk, feasible=False, as_of_date=as_of_date,
        )
        logger.warning(
            f"[{as_of_date}] Fallback — "
            f"yield real={result.port_yield:.2%} (objetivo {min_yield:.2%} no garantizado)"
        )
        return result

    except Exception as exc2:
        raise RuntimeError(
            f"[{as_of_date}] El optimizador no pudo encontrar solución: {exc2}"
        ) from exc2


def optimize_from_features(
    features:    Any,
    yields:      pd.Series,
    **kwargs,
) -> OptimizationResult:
    """
    Wrapper que acepta un FeatureSet de 02_features.py directamente.

    Ejemplo de uso en el backtester (05_backtest.py):
        result = optimize_from_features(features, universe["dividend_yield"])
    """
    return optimize_portfolio(
        mu         = features.mu,
        cov        = features.cov,
        yields     = yields,
        as_of_date = getattr(features, "as_of_date", None),
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución directa
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Test standalone: carga las features más recientes de data/processed/,
    corre el optimizador y muestra el portafolio resultante.
    """
    import glob as _glob
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ── Yields del universo ───────────────────────────────────────────────────
    universe_files = sorted(_glob.glob(str(DATA_RAW / "universe_*.csv")))
    if not universe_files:
        raise FileNotFoundError("Sin archivo de universo. Ejecuta 01_universe.py primero.")
    universe = pd.read_csv(universe_files[-1]).set_index("ticker")
    yields   = universe["dividend_yield"]

    # ── Features ──────────────────────────────────────────────────────────────
    mu_path  = DATA_PROCESSED / "mu_latest.parquet"
    cov_path = DATA_PROCESSED / "cov_latest.parquet"
    if not (mu_path.exists() and cov_path.exists()):
        raise FileNotFoundError("Features no encontradas. Ejecuta 02_features.py primero.")

    mu_ser  = pd.read_parquet(mu_path)["mu"]
    cov_mat = pd.read_parquet(cov_path)

    # ── Optimizar ─────────────────────────────────────────────────────────────
    result = optimize_portfolio(
        mu         = mu_ser,
        cov        = cov_mat,
        yields     = yields,
        as_of_date = pd.Timestamp.today().normalize(),
    )

    # ── Resumen ───────────────────────────────────────────────────────────────
    active = result.weights[result.weights > 1e-4].sort_values(ascending=False)

    print("\n" + "=" * 70)
    print(
        f"PORTFOLIO  {result.as_of_date.date() if result.as_of_date else 'hoy'}   "
        f"feasible={result.feasible}"
    )
    print("=" * 70)
    print(f"  Activos en cartera  : {len(active)}")
    print(f"  Capital invertido   : {result.sum_weights:.1%}  (cash restante: {1-result.sum_weights:.1%})")
    print(f"  Yield dividendos    : {result.port_yield:.2%}  (piso: {MIN_YIELD:.0%})")
    print(f"  Volatilidad anual   : {result.port_vol:.2%}")
    print(f"  Retorno esperado    : {result.port_mu:.2%}")
    print(f"  Varianza anual      : {result.port_var:.6f}")
    print(f"  lambda (risk aver.) : {result.lambda_risk}")
    print()

    detail = pd.DataFrame({
        "peso (%)":  (active * 100).round(2),
        "yield (%)": (yields.reindex(active.index).fillna(0) * 100).round(2),
        "mu (%)":    (mu_ser.reindex(active.index).fillna(0) * 100).round(2),
    })
    print(detail.to_string())
    print("─" * 70)

    # Verificación explícita de la restricción de yield
    w       = result.weights
    y_check = yields.reindex(w.index).fillna(0)
    num     = float((w * y_check).sum())
    den     = float(w.sum())
    print(
        f"\nVerificacion yield: "
        f"sum(w*y)={num:.4f}  /  sum(w)={den:.4f}  =  {num/den if den>0 else 0:.4f}"
        f"  {'OK' if num/den >= MIN_YIELD - 1e-6 else 'INCUMPLE'}"
    )


if __name__ == "__main__":
    main()
