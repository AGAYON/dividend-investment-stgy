"""
04_signals.py — Stop-loss y take-profit a nivel de acción individual
Actinver · Estrategia Investor US Equities

Reglas (por acción, no por portafolio):
  Stop-loss  : retorno desde entrada < −1σ_anual  → peso → 0, capital redistribuido
  Take-profit: retorno desde entrada > +1σ_anual  → peso → w/2, exceso redistribuido

σ_anual proviene de GARCH(1,1) precomputado en 02_features.precompute_garch_vols().

Uso en el backtester (05_backtest.py):
  1. apply_signals()     → identifica tickers stop-loss / take-profit
  2. El optimizador corre excluyendo los tickers en stop-loss del universo
  3. update_position_book() → actualiza entradas y precios al cerrar/abrir posiciones
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"

SL_MULTIPLE:  float = 1.0   # stop-loss  a −1σ anual
TP_MULTIPLE:  float = 1.0   # take-profit a +1σ anual
MAX_WEIGHT:   float = 0.05  # cap en la redistribución (consistente con 03_optimizer)
WEIGHT_TOL:   float = 1e-6  # umbral para considerar un peso activo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Contenedores de datos
# ─────────────────────────────────────────────────────────────────────────────
class Position(NamedTuple):
    ticker:      str
    entry_date:  pd.Timestamp  # fecha en que se abrió (o reabrió) la posición
    entry_price: float         # precio ajustado en entry_date
    weight:      float         # peso vigente en el portafolio


class SignalResult(NamedTuple):
    stop_loss_tickers:   list[str]      # triggers stop-loss → excluir del optimizador
    take_profit_tickers: list[str]      # triggers take-profit → reducir max_weight a la mitad
    adjusted_weights:    pd.Series      # pesos después de redistribución (uso standalone)
    detail:              pd.DataFrame   # diagnóstico por ticker


# ─────────────────────────────────────────────────────────────────────────────
# Predicados puros
# ─────────────────────────────────────────────────────────────────────────────
def check_stop_loss(
    entry_price:   float,
    current_price: float,
    sigma_annual:  float,
    sl_multiple:   float = SL_MULTIPLE,
) -> bool:
    """
    Devuelve True si el retorno desde la entrada cruza el umbral de stop-loss.

        retorno = (current_price − entry_price) / entry_price
        trigger si retorno < −sl_multiple × sigma_annual
    """
    if entry_price <= 0 or sigma_annual <= 0:
        return False
    return (current_price - entry_price) / entry_price < -sl_multiple * sigma_annual


def check_take_profit(
    entry_price:   float,
    current_price: float,
    sigma_annual:  float,
    tp_multiple:   float = TP_MULTIPLE,
) -> bool:
    """
    Devuelve True si el retorno desde la entrada cruza el umbral de take-profit.

        trigger si retorno > +tp_multiple × sigma_annual
    """
    if entry_price <= 0 or sigma_annual <= 0:
        return False
    return (current_price - entry_price) / entry_price > tp_multiple * sigma_annual


# ─────────────────────────────────────────────────────────────────────────────
# Redistribución de pesos
# ─────────────────────────────────────────────────────────────────────────────
def redistribute_weights(
    current_weights:     pd.Series,
    stop_loss_tickers:   list[str],
    take_profit_tickers: list[str],
    max_weight:          float = MAX_WEIGHT,
) -> pd.Series:
    """
    Aplica las reglas de redistribución al vector de pesos después de señales:

      - Stop-loss  : wᵢ → 0  (todo el peso liberado al pool)
      - Take-profit: wᵢ → wᵢ/2  (la mitad liberada al pool)

    El pool se redistribuye proporcionalmente entre las posiciones restantes,
    respetando el cap de max_weight. Cualquier exceso que no pueda absorberse
    queda como cash (Σwᵢ < 1). El optimizador lo reasigna en el siguiente
    rebalanceo.

    Nota: esta función se usa principalmente en 07_update.py (producción) o
    en escenarios intra-mes. En el backtester, el optimizador corre
    inmediatamente después de las señales y rehace los pesos desde cero.
    """
    w         = current_weights.copy().astype(float)
    sl_set    = set(stop_loss_tickers)
    tp_set    = set(take_profit_tickers) - sl_set  # stop-loss tiene precedencia

    freed = 0.0

    for t in sl_set:
        if t in w.index:
            freed += float(w[t])
            w[t]   = 0.0

    for t in tp_set:
        if t in w.index:
            half   = float(w[t]) / 2.0
            freed += half
            w[t]  -= half

    if freed < WEIGHT_TOL:
        return w

    # Tickers elegibles para recibir el capital liberado
    eligible = [
        t for t in w.index
        if t not in sl_set and t not in tp_set and w[t] > WEIGHT_TOL
    ]

    if not eligible:
        return w  # no hay receptores, freed queda en cash

    # Redistribución proporcional con cap iterativo (máx 20 pasos)
    remaining = freed
    for _ in range(20):
        if remaining < WEIGHT_TOL or not eligible:
            break

        sum_el = float(w[eligible].sum())
        if sum_el < WEIGHT_TOL:
            break

        overflow = 0.0
        still_eligible = []
        for t in eligible:
            share  = float(w[t]) / sum_el
            add    = remaining * share
            new_w  = float(w[t]) + add
            if new_w > max_weight:
                overflow    += new_w - max_weight
                w[t]         = max_weight
            else:
                w[t]         = new_w
                still_eligible.append(t)

        eligible  = still_eligible
        remaining = overflow

    return w


# ─────────────────────────────────────────────────────────────────────────────
# Procesador principal de señales
# ─────────────────────────────────────────────────────────────────────────────
def apply_signals(
    position_book:  dict[str, Position],
    current_prices: pd.Series,
    sigma_garch:    pd.Series,
    sl_multiple:    float = SL_MULTIPLE,
    tp_multiple:    float = TP_MULTIPLE,
    max_weight:     float = MAX_WEIGHT,
) -> SignalResult:
    """
    Evalúa todas las posiciones abiertas contra stop-loss y take-profit.

    Parameters
    ----------
    position_book  : dict ticker → Position (de update_position_book)
    current_prices : precios actuales por ticker
    sigma_garch    : volatilidad GARCH anualizada por ticker (de 02_features)
    sl_multiple    : múltiplo σ para stop-loss  (default 1.0)
    tp_multiple    : múltiplo σ para take-profit (default 1.0)
    max_weight     : cap en redistribución

    Returns
    -------
    SignalResult
      .stop_loss_tickers   → pasar como exclusión al optimizador este mes
      .take_profit_tickers → reducir max_weight a max_weight/2 en el optimizador
      .adjusted_weights    → pesos redistribuidos (útil sin re-optimización)
      .detail              → DataFrame para logging / diagnóstico
    """
    sl_tickers: list[str] = []
    tp_tickers: list[str] = []
    rows: list[dict] = []

    current_weights = pd.Series(
        {t: p.weight for t, p in position_book.items()},
        dtype=float,
    )

    for ticker, pos in position_book.items():
        curr_px  = float(current_prices.get(ticker, np.nan))
        sigma    = float(sigma_garch.get(ticker, np.nan))

        if np.isnan(curr_px) or np.isnan(sigma):
            rows.append(_detail_row(pos, curr_px, sigma, ret=np.nan, signal="NO_DATA"))
            continue

        ret = (curr_px - pos.entry_price) / pos.entry_price

        if check_stop_loss(pos.entry_price, curr_px, sigma, sl_multiple):
            signal = "STOP_LOSS"
            sl_tickers.append(ticker)
        elif check_take_profit(pos.entry_price, curr_px, sigma, tp_multiple):
            signal = "TAKE_PROFIT"
            tp_tickers.append(ticker)
        else:
            signal = "HOLD"

        rows.append(_detail_row(pos, curr_px, sigma, ret, signal))

    detail = (
        pd.DataFrame(rows)
        .set_index("ticker")
        .sort_values("ret_desde_entrada", ascending=True)
        if rows else pd.DataFrame()
    )

    adj_weights = redistribute_weights(
        current_weights, sl_tickers, tp_tickers, max_weight
    )

    if sl_tickers:
        logger.info(f"STOP-LOSS  ({len(sl_tickers)}): {sl_tickers}")
    if tp_tickers:
        logger.info(f"TAKE-PROFIT ({len(tp_tickers)}): {tp_tickers}")
    if not sl_tickers and not tp_tickers:
        logger.info("Señales: ninguna posición cruza el umbral (HOLD)")

    return SignalResult(
        stop_loss_tickers   = sl_tickers,
        take_profit_tickers = tp_tickers,
        adjusted_weights    = adj_weights,
        detail              = detail,
    )


def _detail_row(
    pos:     Position,
    curr_px: float,
    sigma:   float,
    ret:     float,
    signal:  str,
) -> dict:
    """Construye una fila del DataFrame de diagnóstico."""
    return {
        "ticker":           pos.ticker,
        "entry_date":       pos.entry_date,
        "entry_price":      round(pos.entry_price, 4),
        "current_price":    round(curr_px, 4) if not np.isnan(curr_px) else np.nan,
        "ret_desde_entrada": round(ret, 4) if not np.isnan(ret) else np.nan,
        "sigma_anual":      round(sigma, 4) if not np.isnan(sigma) else np.nan,
        "sl_threshold":     round(-SL_MULTIPLE * sigma, 4) if not np.isnan(sigma) else np.nan,
        "tp_threshold":     round(+TP_MULTIPLE * sigma, 4) if not np.isnan(sigma) else np.nan,
        "weight":           round(pos.weight, 4),
        "signal":           signal,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mantenimiento del libro de posiciones
# ─────────────────────────────────────────────────────────────────────────────
def update_position_book(
    prev_book:       dict[str, Position],
    new_weights:     pd.Series,
    current_prices:  pd.Series,
    rebalance_date:  pd.Timestamp,
) -> dict[str, Position]:
    """
    Actualiza el libro de posiciones después de un rebalanceo.

    Reglas de actualización:
      - Nuevo  (prev_weight = 0 → new_weight > 0): entry_price = current_price
      - Continúa (prev_weight > 0 → new_weight > 0): conserva entry_price y entry_date
      - Cerrado (new_weight = 0): se elimina del libro

    Llamar DESPUÉS de apply_signals y DESPUÉS del optimizador:
        signals = apply_signals(prev_book, ...)
        opt     = optimize_portfolio(...)          # con stop_loss excluidos
        book    = update_position_book(prev_book, opt.weights, prices, date)
    """
    book: dict[str, Position] = {}

    for ticker, new_w in new_weights.items():
        if new_w < WEIGHT_TOL:
            continue  # posición cerrada → no entra al nuevo libro

        price = float(current_prices.get(ticker, np.nan))
        if np.isnan(price) or price <= 0:
            logger.warning(f"[{ticker}] Precio no disponible en {rebalance_date.date()}, omitido.")
            continue

        prev = prev_book.get(ticker)
        if prev is not None and prev.weight > WEIGHT_TOL:
            # Posición que continúa: conserva precio de entrada original
            book[ticker] = Position(
                ticker      = ticker,
                entry_date  = prev.entry_date,
                entry_price = prev.entry_price,
                weight      = float(new_w),
            )
        else:
            # Nueva posición
            book[ticker] = Position(
                ticker      = ticker,
                entry_date  = rebalance_date,
                entry_price = price,
                weight      = float(new_w),
            )

    return book


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución directa
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Test standalone:
      1. Carga precios y GARCH vols desde data/processed/
      2. Simula un libro de posiciones con entrada hace ~1 mes
      3. Aplica señales a fecha de hoy
      4. Muestra el detalle por ticker con umbrales y señales
    """
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ── Cargar datos ──────────────────────────────────────────────────────────
    prices_path = DATA_PROCESSED / "prices.parquet"
    garch_path  = DATA_PROCESSED / "garch_vols.parquet"

    if not prices_path.exists() or not garch_path.exists():
        raise FileNotFoundError("Ejecuta 02_features.py primero para generar los parquets.")

    prices     = pd.read_parquet(prices_path)
    garch_vols = pd.read_parquet(garch_path)

    today      = prices.index[-1]
    entry_date = prices.index[-22]  # ~1 mes atrás (22 días hábiles)

    logger.info(f"Fecha actual: {today.date()}  |  Fecha de entrada: {entry_date.date()}")

    # ── Cargar pesos del optimizador (últimos guardados) o usar top-20 uniform ─
    sigma_today = garch_vols.loc[
        garch_vols.index[garch_vols.index <= today][-1]
    ]

    # Tickers disponibles en ambas series
    common_tickers = sorted(set(prices.columns) & set(garch_vols.columns))

    # Construir posiciones uniformes en los top-20 por μ esperado
    # (equivalente al portafolio que saldría del optimizador)
    n_pos = 20
    port_tickers = common_tickers[:n_pos]  # primeros 20 del universo ordenado
    uniform_w    = MAX_WEIGHT              # 5 % cada uno → 100 % invertido

    position_book: dict[str, Position] = {}
    for tk in port_tickers:
        ep = float(prices.loc[entry_date, tk]) if entry_date in prices.index else np.nan
        if np.isnan(ep) or ep <= 0:
            continue
        position_book[tk] = Position(
            ticker      = tk,
            entry_date  = entry_date,
            entry_price = ep,
            weight      = uniform_w,
        )

    current_prices = prices.loc[today]

    # ── Aplicar señales ───────────────────────────────────────────────────────
    signals = apply_signals(
        position_book  = position_book,
        current_prices = current_prices,
        sigma_garch    = sigma_today,
    )

    # ── Imprimir resultados ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"SIGNALS   entrada: {entry_date.date()}  →  hoy: {today.date()}")
    print(f"          umbral: ±{SL_MULTIPLE:.0f}σ anual por acción")
    print("=" * 80)

    if not signals.detail.empty:
        cols = ["entry_price", "current_price", "ret_desde_entrada",
                "sigma_anual", "sl_threshold", "tp_threshold", "weight", "signal"]
        disp = signals.detail[cols].copy()
        disp["ret_desde_entrada"] = (disp["ret_desde_entrada"] * 100).round(2).astype(str) + " %"
        disp["sigma_anual"]       = (disp["sigma_anual"]       * 100).round(1).astype(str) + " %"
        disp["sl_threshold"]      = (disp["sl_threshold"]      * 100).round(1).astype(str) + " %"
        disp["tp_threshold"]      = (disp["tp_threshold"]      * 100).round(1).astype(str) + " %"
        print(disp.to_string())

    print("─" * 80)
    print(f"HOLD       : {signals.detail['signal'].eq('HOLD').sum()}")
    print(f"STOP-LOSS  : {len(signals.stop_loss_tickers)}  {signals.stop_loss_tickers}")
    print(f"TAKE-PROFIT: {len(signals.take_profit_tickers)}  {signals.take_profit_tickers}")
    print()
    print("Pesos ajustados post-señales:")
    adj = signals.adjusted_weights[signals.adjusted_weights > WEIGHT_TOL].sort_values(ascending=False)
    print(f"  {len(adj)} posiciones  |  capital invertido: {adj.sum():.1%}")

    # ── Verificar update_position_book ───────────────────────────────────────
    opt_weights = signals.adjusted_weights.copy()
    new_book    = update_position_book(position_book, opt_weights, current_prices, today)
    new_entries = [t for t in new_book if t not in position_book]
    same_entry  = [
        t for t in new_book
        if t in position_book
        and new_book[t].entry_date == position_book[t].entry_date
    ]
    print(f"\nLibro de posiciones actualizado: {len(new_book)} posiciones")
    print(f"  Entradas nuevas     : {len(new_entries)}  {new_entries}")
    print(f"  Entradas conservadas: {len(same_entry)}")


if __name__ == "__main__":
    main()
