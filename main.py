"""
main.py — Orquestador central de la estrategia Actinver US Equity Dividend
===========================================================================
Uso desde terminal:
    python main.py [--mode backtest|live] [--start YYYY-MM-DD]
                   [--gamma G] [--min-yield Y] [--max-weight W]

Uso desde notebook:
    from main import (
        ROOT, DATA_RAW, DATA_PROCESSED,
        read_last_run, run_universe, run_features, run_optimizer,
        run_signals, run_backtest, diagnose_infeasibility,
        run_walk_forward, export_excel, build_metrics_table, run_pipeline,
    )
"""

import importlib.util
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT           = Path(__file__).parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
SRC            = ROOT / "src"
LAST_RUN_FILE  = DATA_PROCESSED / "last_run.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cargar módulos src (nombres con prefijo numérico requieren importlib)
# ---------------------------------------------------------------------------
def _import_src(alias: str, fname: str):
    spec = importlib.util.spec_from_file_location(alias, SRC / fname)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_u = _import_src("universe",  "01_universe.py")
_f = _import_src("features",  "02_features.py")
_o = _import_src("optimizer", "03_optimizer.py")
_s = _import_src("signals",   "04_signals.py")
_b = _import_src("backtest",  "05_backtest.py")


# ---------------------------------------------------------------------------
# Metadata del último run
# ---------------------------------------------------------------------------

def read_last_run() -> dict | None:
    """Lee last_run.json. Retorna dict o None si no existe."""
    if not LAST_RUN_FILE.exists():
        return None
    with open(LAST_RUN_FILE) as f:
        return json.load(f)


def write_last_run(metadata: dict) -> None:
    """Escribe last_run.json de forma atómica (write → rename)."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    tmp = LAST_RUN_FILE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    os.replace(tmp, LAST_RUN_FILE)
    logger.info(f"last_run.json actualizado → {LAST_RUN_FILE}")


# ---------------------------------------------------------------------------
# Wrappers delgados sobre los scripts src
# ---------------------------------------------------------------------------

def run_universe(
    min_mktcap: float = 100e9,
    min_yield_screen: float = 0.005,
    min_adv: float = 50e6,
    save: bool = True,
) -> pd.DataFrame:
    """Wrapper de 01_universe.get_universe()."""
    return _u.get_universe(
        min_mktcap=min_mktcap,
        min_yield=min_yield_screen,
        min_adv=min_adv,
        save=save,
    )


def run_features(
    rebalance_date: "str | pd.Timestamp",
    universe_df: pd.DataFrame,
    lookback_months: int = 36,
    min_obs_months: int = 24,
    lambda_bl: float = 3.0,
    tau: float = 0.05,
    save: bool = True,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Wrapper de 02_features.get_features()."""
    return _f.get_features(
        rebalance_date=rebalance_date,
        universe_df=universe_df,
        lookback_months=lookback_months,
        min_obs_months=min_obs_months,
        lambda_bl=lambda_bl,
        tau=tau,
        save=save,
    )


def run_optimizer(
    features_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    gamma: float = 1.0,
    min_yield: float = 0.03,
    max_weight: float = 0.05,
    rebalance_date: "str | pd.Timestamp | None" = None,
    save: bool = True,
) -> "pd.DataFrame | None":
    """Wrapper de 03_optimizer.optimize_portfolio()."""
    return _o.optimize_portfolio(
        features_df=features_df,
        cov_df=cov_df,
        gamma=gamma,
        min_yield=min_yield,
        max_weight=max_weight,
        rebalance_date=rebalance_date,
        save=save,
    )


def run_signals(
    positions_df: pd.DataFrame,
    eval_date: "str | pd.Timestamp",
    sl_threshold: float = 1.0,
    tp_threshold: float = 1.0,
    ewma_span: int = 12,
    save: bool = True,
) -> pd.DataFrame:
    """Wrapper de 04_signals.evaluate_signals()."""
    return _s.evaluate_signals(
        positions_df=positions_df,
        eval_date=eval_date,
        sl_threshold=sl_threshold,
        tp_threshold=tp_threshold,
        ewma_span=ewma_span,
        save=save,
    )


def run_backtest(
    backtest_start: str = "2016-01-01",
    backtest_end: "str | None" = None,
    min_portfolio_yield: float = 0.03,
    max_weight: float = 0.05,
    gamma: float = 1.0,
    min_mktcap: float = 100e9,
    min_yield_screen: float = 0.005,
    min_adv: float = 50e6,
    tau: float = 0.05,
    lambda_bl: float = 3.0,
    lookback_months: int = 36,
    min_obs_months: int = 24,
    ewma_span: int = 12,
    sl_threshold: float = 1.0,
    tp_threshold: float = 1.0,
    save: bool = True,
) -> "tuple[pd.DataFrame, dict]":
    """Wrapper de 05_backtest.run_backtest()."""
    return _b.run_backtest(
        backtest_start=backtest_start,
        min_portfolio_yield=min_portfolio_yield,
        max_weight=max_weight,
        gamma=gamma,
        min_mktcap=min_mktcap,
        min_yield_screen=min_yield_screen,
        min_adv=min_adv,
        tau=tau,
        lambda_bl=lambda_bl,
        lookback_months=lookback_months,
        min_obs_months=min_obs_months,
        ewma_span=ewma_span,
        sl_threshold=sl_threshold,
        tp_threshold=tp_threshold,
        save=save,
    )


# ---------------------------------------------------------------------------
# Protocolo de infeasibility — diagnóstico de 4 escenarios
# ---------------------------------------------------------------------------

def diagnose_infeasibility(
    features_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    rebalance_dt: pd.Timestamp,
    params: dict,
) -> dict:
    """
    Ejecuta los 4 escenarios del protocolo de infeasibility y retorna un dict.

    params debe contener:
        min_yield, max_weight, gamma,
        relaxed_yield, relaxed_max_weight

    Retorna:
        max_achievable_yield : float
        n_eligible           : int
        scenarios            : {'A': {...}, 'B': {...}, 'C': {...}, 'D': None}
            Cada escenario (excepto D): {feasible, yield, n_stocks, weights_df}
    """
    dy = features_df.set_index("ticker")["dividend_yield_ttm"].values
    max_yield, _ = _o._max_achievable_yield(dy, params["max_weight"])

    def _try(min_y: float, max_w: float) -> dict:
        sol = _o.optimize_portfolio(
            features_df=features_df, cov_df=cov_df,
            gamma=params["gamma"], min_yield=min_y, max_weight=max_w,
            rebalance_date=rebalance_dt, save=False,
        )
        if sol is None:
            return {"feasible": False, "yield": None, "n_stocks": 0, "weights_df": None}
        sol = sol[sol["weight"] > 1e-4].copy()
        return {
            "feasible": True,
            "yield": float(sol["contribution_yield"].sum()),
            "n_stocks": len(sol),
            "weights_df": sol,
        }

    return {
        "max_achievable_yield": max_yield,
        "n_eligible": len(features_df),
        "scenarios": {
            "A": _try(params["relaxed_yield"],  params["max_weight"]),
            "B": _try(params["min_yield"],       params["relaxed_max_weight"]),
            "C": _try(0.0,                       params["max_weight"]),
            "D": None,
        },
    }


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def run_walk_forward(
    universe_df: pd.DataFrame,
    results: pd.DataFrame,
    lookback_months: int = 36,
    min_obs_months: int = 24,
    lambda_bl: float = 3.0,
    tau: float = 0.05,
    save: bool = True,
) -> pd.DataFrame:
    """
    Validación walk-forward: calidad predictiva de μ_BL vs retornos realizados.

    Para cada par de meses consecutivos en results.index:
      - Predice μ_BL con datos hasta el mes anterior
      - Compara contra retorno realizado del mes siguiente
      - Calcula MAE, RMSE y hit ratio (top tercil)

    Si walk_forward_results.parquet ya existe, lo carga directamente.
    """
    wf_file = DATA_PROCESSED / "walk_forward_results.parquet"
    if wf_file.exists():
        logger.info(f"Walk-forward cargado desde caché: {wf_file.name}")
        return pd.read_parquet(wf_file)

    logger.info("Computando walk-forward — descargando precios...")
    tks_all = universe_df["ticker"].tolist()
    raw_wf  = yf.download(tks_all, start="2013-01-01", auto_adjust=True, progress=True)
    cl_wf   = raw_wf["Close"] if isinstance(raw_wf["Close"], pd.DataFrame) else raw_wf[["Close"]]
    mo_cl   = cl_wf.resample("ME").last()
    mo_ret  = np.log(mo_cl / mo_cl.shift(1))

    wf_records = []
    months = results.index.tolist()

    for i in range(1, len(months)):
        prev_dt = months[i - 1]
        curr_dt = months[i]
        wend    = prev_dt
        wstart  = wend - pd.DateOffset(months=lookback_months + 1)
        ret_w   = mo_ret.loc[wstart:wend]

        available = [
            t for t in tks_all
            if t in ret_w.columns and ret_w[t].notna().sum() >= min_obs_months
        ]
        if len(available) < 5:
            continue

        returns_clean = ret_w[available].dropna()
        if returns_clean.shape[0] < min_obs_months:
            continue

        try:
            # Un solo fit — covariance_ y shrinkage_ en una sola llamada
            lw_fit    = LedoitWolf().fit(returns_clean.values)
            sigma_all = lw_fit.covariance_
            shrinkage = lw_fit.shrinkage_

            # Subconjunto con datos de universe_df
            univ_sub  = universe_df[universe_df["ticker"].isin(available)].set_index("ticker")
            available2 = [t for t in available if t in univ_sub.index]
            if len(available2) < 3:
                continue

            # Indexar sigma correctamente por posición
            idx    = [available.index(t) for t in available2]
            sigma2 = sigma_all[np.ix_(idx, idx)]
            mc     = np.array([univ_sub.loc[t, "market_cap"]    for t in available2], dtype=float)
            dy     = np.array([univ_sub.loc[t, "dividend_yield"] for t in available2], dtype=float)
            w_mkt  = mc / mc.sum()
            mu_bl  = _f._black_litterman(sigma2, w_mkt, dy, lambda_bl, tau)

            # Retorno realizado
            actual = (
                mo_ret.loc[curr_dt, available2]
                if curr_dt in mo_ret.index
                else pd.Series(dtype=float)
            )
            if actual.empty:
                continue

            pred   = pd.Series(mu_bl, index=available2)
            common = pred.index.intersection(actual.dropna().index)
            if len(common) < 3:
                continue

            mae  = float(np.abs(pred[common] - actual[common]).mean())
            rmse = float(np.sqrt(((pred[common] - actual[common]) ** 2).mean()))

            n3         = max(1, len(common) // 3)
            top_pred   = set(pred[common].nlargest(n3).index)
            top_actual = set(actual[common].nlargest(n3).index)
            hit        = len(top_pred & top_actual) / n3

            wf_records.append({
                "date": curr_dt, "mae": mae, "rmse": rmse,
                "hit_ratio": hit, "shrinkage": shrinkage, "n_stocks": len(common),
            })

        except Exception:
            continue

    wf = pd.DataFrame(wf_records).set_index("date")
    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        wf.to_parquet(wf_file)
        logger.info(f"Walk-forward guardado → {wf_file}")
    return wf


# ---------------------------------------------------------------------------
# Exportar reporte Excel
# ---------------------------------------------------------------------------

def export_excel(
    metrics: dict,
    weights_df: pd.DataFrame,
    results: "pd.DataFrame | None" = None,
    wf: "pd.DataFrame | None" = None,
    output_dir: "Path | str | None" = None,
) -> Path:
    """
    Genera el reporte Excel con hasta 6 hojas:
    Metricas | Pesos | NAV | Dividends | Infeasibility | WalkForward

    Retorna la ruta del archivo generado.
    """
    out_dir = Path(output_dir) if output_dir else ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.today().strftime("%Y%m%d")
    out_path  = out_dir / f"reporte_{timestamp}.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

        # Hoja 1: Métricas
        met_df = pd.DataFrame(list(metrics.items()), columns=["Métrica", "Valor"])
        met_df.to_excel(writer, sheet_name="Metricas", index=False)

        # Hoja 2: Pesos actuales
        weights_df[["ticker", "weight", "contribution_yield"]].to_excel(
            writer, sheet_name="Pesos", index=False
        )

        # Hoja 3: NAV
        if results is not None:
            results[["nav", "portfolio_return", "benchmark_return"]].to_excel(
                writer, sheet_name="NAV"
            )

        # Hoja 4: Dividends
        if results is not None:
            results[["realized_yield"]].assign(
                realized_yield_annual=results["realized_yield"] * 12
            ).to_excel(writer, sheet_name="Dividends")

        # Hoja 5: Infeasibility log (si existe)
        inf_log = DATA_PROCESSED / "infeasibility_log.csv"
        if inf_log.exists():
            pd.read_csv(inf_log).to_excel(writer, sheet_name="Infeasibility", index=False)

        # Hoja 6: Walk-forward
        if wf is not None:
            wf.to_excel(writer, sheet_name="WalkForward")

    logger.info(f"Reporte Excel guardado → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Helper: tabla de métricas formateada para el notebook
# ---------------------------------------------------------------------------

def build_metrics_table(metrics: dict, min_portfolio_yield: float = 0.03) -> pd.DataFrame:
    """
    Construye el DataFrame de métricas con formato para mostrar en el notebook.
    Retorna pd.DataFrame con columnas: Métrica | Portafolio | Benchmark / Referencia.
    """
    def _pct(k):  return f"{metrics[k]:.2%}"   if metrics.get(k) is not None else "—"
    def _f3(k):   return f"{metrics[k]:.3f}"   if metrics.get(k) is not None else "—"
    def _f4(k):   return f"{metrics[k]:.4f}"   if metrics.get(k) is not None else "—"
    def _int(k):  return str(metrics.get(k, "—"))

    rows = [
        ("── RENDIMIENTO ──",                "",                                 ""),
        ("CAGR portafolio",                  _pct("CAGR_portfolio"),              _pct("CAGR_benchmark")),
        ("Volatilidad anualizada",            _pct("Vol_annual_portfolio"),        "—"),
        ("Sharpe Ratio",                      _f3("Sharpe_Ratio"),                 "—"),
        ("Sortino Ratio",                     _f3("Sortino_Ratio"),                "—"),
        ("Calmar Ratio",                      _f3("Calmar_Ratio"),                 "—"),
        ("── RIESGO ──",                      "",                                  ""),
        ("Max Drawdown",                      _pct("Max_Drawdown"),                "—"),
        ("VaR 95% mensual",                   _pct("VaR_95_monthly"),              "—"),
        ("VaR 99% mensual",                   _pct("VaR_99_monthly"),              "—"),
        ("CVaR 95% mensual",                  _pct("CVaR_95_monthly"),             "—"),
        ("CVaR 99% mensual",                  _pct("CVaR_99_monthly"),             "—"),
        ("Upper Partial Moment",              _f4("Upper_Partial_Moment"),         "—"),
        ("Pain Index",                        _f4("Pain_Index"),                   "—"),
        ("Pain-Gain Ratio",                   _f3("Pain_Gain_Ratio"),              "—"),
        ("── ALPHA ──",                       "",                                  ""),
        ("Alpha anual vs SPY",                _pct("Alpha_annual"),                "—"),
        ("Tracking Error",                    _pct("Tracking_Error"),              "—"),
        ("Information Ratio",                 _f3("Information_Ratio"),            "—"),
        ("Beta vs SPY",                       _f3("Beta"),                         "—"),
        ("── YIELD ──",                       "",                                  ""),
        ("Yield realizado anual (promedio)",  _pct("Avg_Realized_Yield_annual"),   f"Objetivo: {min_portfolio_yield:.0%}"),
        ("% meses con yield ≥ 3%",            f"{metrics.get('Pct_months_yield_met', 0):.1%}", "—"),
        ("── OPERACIONES ──",                 "",                                  ""),
        ("Turnover mensual promedio",         _pct("Avg_Monthly_Turnover"),        "—"),
        ("Activaciones stop-loss",            _int("N_StopLoss_activations"),      "—"),
        ("Activaciones take-profit",          _int("N_TakeProfit_activations"),    "—"),
        ("N meses backtest",                  _int("N_months"),                    "—"),
    ]
    return pd.DataFrame(rows, columns=["Métrica", "Portafolio", "Benchmark / Referencia"])


# ---------------------------------------------------------------------------
# Orquestador completo — para CLI y uso desde notebook
# ---------------------------------------------------------------------------

def run_pipeline(
    run_mode: str = "backtest",
    backtest_start: str = "2016-01-01",
    backtest_end: "str | None" = None,
    min_portfolio_yield: float = 0.03,
    max_weight: float = 0.05,
    risk_aversion: float = 1.0,
    min_mktcap: float = 100e9,
    min_yield_screen: float = 0.005,
    min_adv: float = 50e6,
    tau: float = 0.05,
    lambda_bl: float = 3.0,
    lookback_months: int = 36,
    min_obs_months: int = 24,
    ewma_span: int = 12,
    sl_threshold: float = 1.0,
    tp_threshold: float = 1.0,
    save: bool = True,
) -> dict:
    """
    Orquesta el pipeline completo: 01 → 02 → 03 → 04 → 05.
    Escribe last_run.json al finalizar (incluso si falla algún paso).

    En modo "live" corre 01-04 para el mes actual.
    En modo "backtest" corre 01-04 + backtest histórico completo.

    Retorna dict con universe_df, features_df, cov_df, weights_df, signals_df,
    results, metrics, infeasible, last_run.
    """
    run_ts   = datetime.now()
    run_date = run_ts.strftime("%Y-%m-%d")
    params_meta = dict(
        min_portfolio_yield=min_portfolio_yield, max_weight=max_weight,
        risk_aversion=risk_aversion, min_mktcap=min_mktcap,
        min_yield_screen=min_yield_screen, min_adv=min_adv,
        tau=tau, lambda_bl=lambda_bl, lookback_months=lookback_months,
        min_obs_months=min_obs_months, ewma_span=ewma_span,
        sl_threshold=sl_threshold, tp_threshold=tp_threshold,
        backtest_start=backtest_start, backtest_end=backtest_end,
    )
    result = {
        "universe_df": None, "features_df": None, "cov_df": None,
        "weights_df": None, "signals_df": None,
        "results": None, "metrics": None,
        "infeasible": False, "last_run": {},
    }
    outcome     = "failed"
    failed_step = None
    error_msg   = None

    try:
        # ── 1. Universe ──────────────────────────────────────────────────────
        logger.info("=== PASO 1: Universe ===")
        failed_step = "universe"
        universe_df = run_universe(min_mktcap, min_yield_screen, min_adv, save=save)
        result["universe_df"] = universe_df

        # ── 2. Features ──────────────────────────────────────────────────────
        logger.info("=== PASO 2: Features ===")
        failed_step = "features"
        rebalance_dt = pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0)
        features_df, cov_df = run_features(
            rebalance_date=rebalance_dt, universe_df=universe_df,
            lookback_months=lookback_months, min_obs_months=min_obs_months,
            lambda_bl=lambda_bl, tau=tau, save=save,
        )
        result["features_df"] = features_df
        result["cov_df"]      = cov_df

        # ── 3. Optimizer ─────────────────────────────────────────────────────
        logger.info("=== PASO 3: Optimizer ===")
        failed_step = "optimizer"
        weights_df = run_optimizer(
            features_df=features_df, cov_df=cov_df,
            gamma=risk_aversion, min_yield=min_portfolio_yield,
            max_weight=max_weight, rebalance_date=rebalance_dt, save=save,
        )
        result["weights_df"] = weights_df
        if weights_df is None:
            result["infeasible"] = True
            logger.warning("QP infeasible — portafolio previo se mantiene (sin rebalanceo automático)")

        # ── 4. Signals (solo si hay pesos) ───────────────────────────────────
        logger.info("=== PASO 4: Signals ===")
        failed_step = "signals"
        if weights_df is not None:
            active = weights_df[weights_df["weight"] > 1e-4].copy()
            tks    = active["ticker"].tolist()
            raw_p  = yf.download(tks, period="2d", auto_adjust=True, progress=False)
            close  = raw_p["Close"]
            prices_entry = close.iloc[-1] if isinstance(close, pd.DataFrame) else close
            positions_df = pd.DataFrame([
                {
                    "ticker":      tk,
                    "entry_date":  rebalance_dt,
                    "entry_price": float(prices_entry.get(tk, 0)) if hasattr(prices_entry, "get") else 0.0,
                }
                for tk in tks
            ])
            signals_df = run_signals(
                positions_df=positions_df, eval_date=rebalance_dt,
                sl_threshold=sl_threshold, tp_threshold=tp_threshold,
                ewma_span=ewma_span, save=save,
            )
            result["signals_df"] = signals_df

        # ── 5. Backtest (modo backtest) ───────────────────────────────────────
        if run_mode == "backtest":
            logger.info("=== PASO 5: Backtest ===")
            failed_step = "backtest"
            results, metrics = run_backtest(
                backtest_start=backtest_start, backtest_end=backtest_end,
                min_portfolio_yield=min_portfolio_yield, max_weight=max_weight,
                gamma=risk_aversion, min_mktcap=min_mktcap,
                min_yield_screen=min_yield_screen, min_adv=min_adv,
                tau=tau, lambda_bl=lambda_bl,
                lookback_months=lookback_months, min_obs_months=min_obs_months,
                ewma_span=ewma_span, sl_threshold=sl_threshold,
                tp_threshold=tp_threshold, save=save,
            )
            result["results"] = results
            result["metrics"] = metrics

        outcome     = "partial" if result["infeasible"] else "success"
        failed_step = None

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Pipeline falló en paso '{failed_step}': {e}")

    # ── Construir outputs para last_run.json ──────────────────────────────────
    outputs: dict = {}
    if result["universe_df"] is not None:
        outputs["universe_n_stocks"] = len(result["universe_df"])
    if result["features_df"] is not None:
        outputs["features_n_stocks"] = len(result["features_df"])
    if result["weights_df"] is not None:
        aw = result["weights_df"][result["weights_df"]["weight"] > 1e-4]
        outputs["optimizer_feasible"]  = True
        outputs["portfolio_yield"]     = round(float(aw["contribution_yield"].sum()), 4)
        outputs["portfolio_n_stocks"]  = len(aw)
    else:
        outputs["optimizer_feasible"] = False
    if result["signals_df"] is not None:
        sdf = result["signals_df"]
        outputs["signals_n_stops"] = int((sdf["signal"] == "stop").sum())
        outputs["signals_n_takes"] = int((sdf["signal"] == "take").sum())
    if result["metrics"] is not None:
        outputs["backtest_cagr"]     = result["metrics"].get("CAGR_portfolio")
        outputs["backtest_sharpe"]   = result["metrics"].get("Sharpe_Ratio")
        outputs["backtest_n_months"] = result["metrics"].get("N_months")

    metadata = {
        "schema_version": 1,
        "run_date":        run_date,
        "run_timestamp":   run_ts.isoformat(timespec="seconds"),
        "run_mode":        run_mode,
        "outcome":         outcome,
        "failed_step":     failed_step,
        "error_message":   error_msg,
        "params":          params_meta,
        "outputs":         outputs,
    }
    write_last_run(metadata)
    result["last_run"] = metadata
    return result


# ---------------------------------------------------------------------------
# Entry point CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Actinver US Equity Dividend Strategy — Pipeline completo"
    )
    p.add_argument("--mode",       default="backtest", choices=["backtest", "live"],
                   help="Modo de ejecución (default: backtest)")
    p.add_argument("--start",      default="2016-01-01",
                   help="Fecha inicio backtest (YYYY-MM-DD)")
    p.add_argument("--gamma",      type=float, default=1.0,
                   help="Aversión al riesgo γ (default: 1.0)")
    p.add_argument("--min-yield",  type=float, default=0.03,
                   help="Yield mínimo del portafolio (default: 0.03)")
    p.add_argument("--max-weight", type=float, default=0.05,
                   help="Peso máximo por acción (default: 0.05)")
    args = p.parse_args()

    r = run_pipeline(
        run_mode=args.mode,
        backtest_start=args.start,
        risk_aversion=args.gamma,
        min_portfolio_yield=args.min_yield,
        max_weight=args.max_weight,
    )

    last = r["last_run"]
    out  = last.get("outputs", {})
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETADO — {last['run_date']}")
    print("=" * 70)
    print(f"Modo:     {last['run_mode']}")
    print(f"Outcome:  {last['outcome']}")
    if last.get("error_message"):
        print(f"Error:    {last['error_message']}")
    if out.get("universe_n_stocks"):
        print(f"Universo: {out['universe_n_stocks']} stocks elegibles")
    if out.get("optimizer_feasible"):
        print(f"Portafolio: {out.get('portfolio_n_stocks')} stocks  yield={out.get('portfolio_yield', 0):.2%}")
    else:
        print("Portafolio: QP infeasible — sin rebalanceo automático")
    if out.get("backtest_cagr") is not None:
        print(
            f"Backtest:  CAGR={out['backtest_cagr']:.2%}  "
            f"Sharpe={out['backtest_sharpe']:.3f}  "
            f"({out['backtest_n_months']} meses)"
        )
    print("=" * 70)
