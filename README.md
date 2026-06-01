# Actinver — Estrategia de Renta Variable EE.UU. con Dividendo Mínimo

Documento técnico de referencia del repositorio. Describe qué contiene el código, qué hace la estrategia propuesta y cómo se implementa, con base en el mandato definido en `Caso para Posición Investor.docx`.

---

## Tabla de contenidos

1. [Resumen](#1-resumen)
2. [Mandato y restricciones](#2-mandato-y-restricciones)
3. [Modelo matemático](#3-modelo-matemático)
4. [Pipeline, módulos y artefactos](#4-pipeline-módulos-y-artefactos)
5. [Loop de backtest y protocolo anti-lookahead](#5-loop-de-backtest-y-protocolo-anti-lookahead)
6. [Métricas de desempeño](#6-métricas-de-desempeño)
7. [Validación walk-forward](#7-validación-walk-forward)
8. [Protocolo de infeasibility](#8-protocolo-de-infeasibility)
9. [Fuentes de datos](#9-fuentes-de-datos)
10. [Setup, ejecución y parámetros](#10-setup-ejecución-y-parámetros)
11. [Estructura del repositorio](#11-estructura-del-repositorio)

---

## 1. Resumen

Este repositorio implementa una estrategia cuantitativa de inversión en acciones estadounidenses del S&P 500 orientada a cumplir, en cada rebalanceo mensual, un **dividend yield ponderado del portafolio ≥ 3% anual**, sujeto a límites de concentración y elegibilidad del universo.

La construcción del portafolio combina:

- **Estimación de inputs:** retornos esperados vía **Black-Litterman** (views = dividend yield TTM) y matriz de covarianzas vía **Ledoit-Wolf** sobre retornos logarítmicos mensuales.
- **Optimización:** programación cuadrática media-varianza con **`quadprog.solve_qp`**, incorporando el yield mínimo como restricción lineal.
- **Gestión táctica:** **stop-loss** y **take-profit** por acción con umbrales en múltiplos de volatilidad **EWMA** mensual; las liquidaciones se reinvierten en el **mismo ciclo de rebalanceo mensual**.
- **Evaluación:** backtest histórico enero 2016–presente, métricas de riesgo-rendimiento vs. **SPY**, y validación **walk-forward** de la calidad predictiva de $\mu_{\mathrm{BL}}$.

**Puntos de entrada:**

| Componente | Rol |
|---|---|
| `main.py` | Orquestador por línea de comandos y API Python (`run_pipeline`, exportación Excel, walk-forward, diagnóstico de infeasibility). |
| `strategy.ipynb` | Centro de control interactivo para revisión mensual del portafolio, visualización y generación de reportes. |

---

## 2. Mandato y restricciones

### 2.1 Objetivo del mandato

Construir y evaluar una estrategia de inversión en acciones de EE.UU. que, a nivel de portafolio ponderado, mantenga un **rendimiento de dividendos anual ≥ 3%** en todo momento, optimizando el trade-off riesgo-rendimiento bajo las restricciones operativas del cliente.

### 2.2 Restricciones duras del portafolio (QP)

Estas restricciones se imponen en el optimizador (`03_optimizer.py`) en cada rebalanceo:

| Restricción | Valor | Notación |
|---|---|---|
| Peso máximo por acción | 5% | $0 \le w_i \le 0{,}05$ |
| Long-only | 0% mínimo | $w_i \ge 0$ |
| Fully invested | 100% | $\sum_i w_i = 1$ |
| Dividend yield ponderado | ≥ 3% anual | $\sum_i w_i \,\mathrm{dy}_i \ge 0{,}03$ |

donde `dyᵢ` es el dividend yield TTM de la acción *i* al cierre del mes de información.

### 2.3 Restricciones de elegibilidad del universo

Filtros mensuales aplicados antes de la optimización (`01_universe.py` / `screen_universe_at` en backtest):

| Filtro | Valor | Fundamento |
|---|---|---|
| Índice | S&P 500 | Mandato de renta variable EE.UU. large cap |
| Market cap | ≥ USD 100 B | Mandato de tamaño mínimo |
| Dividend yield TTM | ≥ 0,5% | Pre-filtro operativo (ver §2.4) |
| ADV 90 días | ≥ USD 50 M | Liquidez mínima para rebalanceo |
| Historial de precios | Desde 2013-01-01 (±45 días) | Lookback de calibración (36 meses antes de ene 2016) |

### 2.4 Decisión de negocio: pre-filtro de yield al 0,5%

El mandato exige **3% a nivel portafolio**, no por acción. El pre-filtro al **0,5%** no relaja el mandato; garantiza un **universo factible** compuesto exclusivamente por emisoras con dividendos positivos y yield observable, evitando incluir acciones sin componente de dividendo que consumirían peso en el QP sin aportar al constraint de yield.

La restricción efectiva del 3% permanece en el optimizador. Un pre-filtro al 3% por acción reduciría el universo a pocas emisoras de alto yield, dificultando la diversificación bajo el límite del 5% por acción y elevando el riesgo de **infeasibility** del QP.

### 2.5 Calendario de evaluación

| Concepto | Período |
|---|---|
| Backtest | Enero 2016 → presente |
| Calibración inicial | Enero 2013 – diciembre 2015 (36 meses) |
| Frecuencia de rebalanceo | Mensual, al último día hábil del mes |
| Revisión del portafolio | Mensual (mandato operativo) |

---

## 3. Modelo matemático

Las ecuaciones de esta sección usan **LaTeX** (bloques `$$…$$`); GitHub, GitLab y la mayoría de visores Markdown las renderizan como en un paper.

### 3.1 Notación

| Símbolo | Descripción |
|---|---|
| $n$ | Número de acciones elegibles en el mes |
| $w \in \mathbb{R}^n$ | Vector de pesos del portafolio |
| $\Sigma \in \mathbb{R}^{n \times n}$ | Matriz de covarianzas (Ledoit-Wolf) |
| $\mu_{\mathrm{BL}} \in \mathbb{R}^n$ | Retornos esperados mensuales (Black-Litterman) |
| $\mathrm{dy} \in \mathbb{R}^n$ | Dividend yield TTM anual por acción |
| $\lambda$ | Aversión al riesgo del mercado (prior BL), default 3,0 |
| $\gamma$ | Aversión al riesgo en el objetivo QP, default 1,0 |
| $\tau$ | Escalar de incertidumbre del prior BL, default 0,05 |

### 3.2 Universo — dividend yield TTM

Para cada acción $i$ al cierre del mes de información $t$:

$$
\mathrm{dy}_i = \frac{\sum_{d \in \mathcal{D}_{12m}(t)} D_d}{P_{i,t}}
$$

donde $\mathcal{D}_{12m}(t)$ es el conjunto de pagos de dividendo en los últimos 12 meses, $D_d$ el monto del dividendo en la fecha $d$, y $P_{i,t}$ el precio ajustado al cierre. Se calcula sobre el historial de `yfinance` (`t.dividends`), no sobre el campo `dividendYield` del proveedor.

### 3.3 Matriz de covarianzas — Ledoit-Wolf

Sobre una ventana rolling de **36 meses** de **log-retornos mensuales**:

$$
r_{i,t} = \ln\!\left(\frac{P_{i,t}}{P_{i,t-1}}\right), \qquad
\Sigma = \mathrm{LedoitWolf}(R), \quad R \in \mathbb{R}^{T \times n}
$$

Se requiere un mínimo de **24 observaciones** mensuales por acción; de lo contrario, la acción se excluye del universo ese mes.

Ledoit-Wolf aplica shrinkage óptimo de la covarianza muestral hacia un objetivo estructurado, reduciendo el error de estimación en muestras pequeñas (n ≈ 20–60).

### 3.4 Retornos esperados — Black-Litterman

**Prior (retornos de equilibrio implícitos):**

$$
\Pi = \lambda \, \Sigma \, w_{\mathrm{mkt}}
$$

donde $w_{\mathrm{mkt}}$ son pesos de capitalización relativa dentro del universo elegible (proxy de pesos de mercado).

**Views cuantitativas:**

Se impone una view por activo ($P = I_n$):

$$
Q_i = \mathrm{dy}_i
$$

La hipótesis de negocio: el dividend yield es la componente observable y predecible del retorno total; el residual de precio queda absorbido en la estructura del modelo.

**Matriz de incertidumbre de las views:**

$$
\Omega = \tau \,\mathrm{diag}(\Sigma)
$$

**Posterior (fórmula estándar de Black-Litterman):**

$$
\mu_{\mathrm{BL}}
= \underbrace{\left[(\tau\Sigma)^{-1} + \Omega^{-1}\right]^{-1}}_{\text{precisión posterior}}
\;\underbrace{\left[(\tau\Sigma)^{-1}\Pi + \Omega^{-1} Q\right]}_{\text{señal combinada}}
$$

Implementación: `src/02_features.py` → `_black_litterman()`.

### 3.5 Optimización media-varianza — formulación QP

**Problema primal:**

$$
\begin{aligned}
\min_{w} \quad & \tfrac{1}{2}\, w^\top \Sigma w - \gamma\, \mu_{\mathrm{BL}}^\top w \\
\text{s.a.} \quad
& \sum_i w_i = 1 \\
& 0 \le w_i \le 0{,}05 && \forall i \\
& \sum_i w_i \,\mathrm{dy}_i \ge 0{,}03
\end{aligned}
$$

**Mapeo a `quadprog.solve_qp`:**

La librería resuelve:

$$
\min_{x} \;\tfrac{1}{2}\, x^\top G x - a^\top x
\qquad \text{s.a.} \qquad C^\top x \ge b
$$

con las primeras `meq` restricciones como igualdades. La construcción en `_build_qp_inputs()` es:

| Elemento | Definición |
|---|---|
| $G$ | $\Sigma + \varepsilon I_n$ (simetrizada; $\varepsilon = 10^{-8}$ por estabilidad numérica) |
| $a$ | $\gamma \,\mu_{\mathrm{BL}}$ |
| `meq` | 1 (solo la restricción de suma = 1) |

**Restricciones en $C^\top x \ge b$ (orden):**

| Índice | Tipo | Restricción |
|---|---|---|
| 0 | Igualdad | $\sum_i w_i = 1$ |
| 1…$n$ | Desigualdad | $w_i \ge 0$ |
| $n{+}1$…$2n$ | Desigualdad | $w_i \le 0{,}05$ (equivalente a $-w_i \ge -0{,}05$) |
| $2n{+}1$ | Desigualdad | $\sum_i w_i \,\mathrm{dy}_i \ge 0{,}03$ |

Tras la solución, los pesos se recortan al intervalo [0, 0.05] y se renormalizan a suma 1.

**Dos parámetros de aversión al riesgo:**

- $\lambda = 3.0$: calibra el prior de equilibrio en Black-Litterman.
- $\gamma = 1.0$: pondera el retorno esperado en la función objetivo del QP.

### 3.6 Stop-loss y take-profit — volatilidad EWMA

Para cada posición abierta con precio de entrada `P_{i,t₀}` al mes t₀:

**Retorno acumulado al cierre del mes de evaluación $t$:**

$$
r_{\mathrm{acum}}(i,t) = \frac{P_{i,t}}{P_{i,t_0}} - 1
$$

**Volatilidad EWMA mensual** sobre log-retornos con `span = 12`:

$$
\sigma_{\mathrm{EWMA}}(i,t)
= \mathrm{StdEWMA}\!\left(\ln\frac{P_{i,s}}{P_{i,s-1}},\; \mathrm{span}{=}12\right)_t
$$

**Reglas de señal** (umbrales default ±1σ):

| Condición | Señal | Acción en rebalanceo |
|---|---|---|
| $r_{\mathrm{acum}} > +\mathrm{TP}\cdot\sigma_{\mathrm{EWMA}}$ | Take-profit | Liquidar posición |
| $r_{\mathrm{acum}} < -\mathrm{SL}\cdot\sigma_{\mathrm{EWMA}}$ | Stop-loss | Liquidar posición |
| En rango | Hold | Mantener hasta siguiente evaluación |

Default: SL = TP = 1.0.

**Reinversión:** las acciones liquidadas por stop-loss o take-profit se sustituyen en el **mismo rebalanceo mensual**: el optimizador reasigna el **100% del capital** al nuevo portafolio óptimo (restricción $\sum_i w_i = 1$). No existe fase intermedia de cash intra-mes.

### 3.7 Evolución del NAV

El NAV base 100 se actualiza mensualmente con retornos totales sobre precios ajustados (`auto_adjust=True`, dividendos reinvertidos en la serie de precios):

$$
\mathrm{NAV}_t = \mathrm{NAV}_{t-1}\left(1 + \sum_{i \in \mathcal{H}} w_i \, r_{i,t}^{\mathrm{tot}}\right)
$$

donde $\mathcal{H}$ es el conjunto de posiciones vigentes y $r_{i,t}^{\mathrm{tot}}$ es el retorno simple mensual del precio ajustado entre cierres consecutivos.

El **yield realizado** se calcula adicionalmente para reporte como flujo de dividendos explícitos sobre valor de posición; no duplica el efecto en NAV (ya capturado vía precios ajustados).

---

## 4. Pipeline, módulos y artefactos

### 4.1 Flujo de datos

```
01_universe.py          →  universo elegible (CSV)
02_features.py          →  μ_BL, Σ (parquet)
03_optimizer.py         →  pesos óptimos (parquet)
04_signals.py           →  señales stop/take (parquet)
05_backtest.py          →  simulación histórica + métricas (parquet, JSON)
main.py                 →  orquestación, walk-forward, Excel, last_run.json
strategy.ipynb          →  ejecución interactiva y reportes visuales
```

Cada módulo persiste outputs en `data/raw/` o `data/processed/`. El backtest invoca la lógica de los módulos 01–04 en loop mensual sin lookahead.

### 4.2 Módulos `src/`

#### `01_universe.py`

Construye el universo elegible para el mes actual (modo live) o alimenta la lógica de screening del backtest.

- **Input:** constituyentes S&P 500, datos `yfinance`.
- **Output:** `data/raw/universe_YYYYMMDD.csv` con columnas `ticker | market_cap | dividend_yield | adv_90d | history_start | price`.
- **Ejecución:** paralela con `ThreadPoolExecutor`.

#### `02_features.py`

Estima Σ (Ledoit-Wolf) y μ_BL para la fecha de rebalanceo.

- **Outputs:** `features_YYYY_MM.parquet`, `cov_YYYY_MM.parquet`.

#### `03_optimizer.py`

Resuelve el QP con `quadprog`. Si el problema es infactible, ejecuta el protocolo de §8.

- **Output:** `weights_YYYY_MM.parquet` (`ticker | weight | expected_return | contribution_yield`).

#### `04_signals.py`

Evalúa stop-loss / take-profit sobre posiciones abiertas.

- **Output:** `signals_YYYY_MM.parquet` (`ticker | entry_month | r_acum | sigma_ewma | signal`).

#### `05_backtest.py`

Orquesta el loop mensual histórico, calcula métricas y persiste resultados.

- **Outputs:** `backtest_results.parquet`, `backtest_metrics.json`.

### 4.3 Orquestador `main.py`

API principal exportada para CLI y notebook:

| Función | Descripción |
|---|---|
| `run_pipeline()` | Ejecuta pasos 01→05 según modo `backtest` o `live`; escribe `last_run.json`. |
| `run_universe()` / `run_features()` / `run_optimizer()` / `run_signals()` / `run_backtest()` | Wrappers delgados sobre módulos `src/`. |
| `diagnose_infeasibility()` | Cuatro escenarios de diagnóstico (§8). |
| `run_walk_forward()` | Validación out-of-sample (§7). |
| `export_excel()` | Reporte Excel en `reports/reporte_YYYYMMDD.xlsx`. |
| `build_metrics_table()` | Tabla formateada para visualización. |

**CLI:**

```bash
python main.py --mode backtest --start 2016-01-01 --gamma 1.0 --min-yield 0.03 --max-weight 0.05
```

### 4.4 Notebook `strategy.ipynb`

Centro de control para la **revisión mensual del portafolio** (mandato operativo). Importa funciones de `main.py`, expone parámetros en celda `[PARAMS]`, genera visualizaciones (NAV, yield, drawdown, heatmap de pesos) y exporta el reporte Excel.

No sustituye a `main.py`: el notebook es la interfaz analítica; `main.py` es la API reproducible y el entry point por terminal.

### 4.5 Artefactos generados

| Archivo | Contenido |
|---|---|
| `data/raw/universe_*.csv` | Universo elegible |
| `data/processed/features_*.parquet` | $\mu_{\mathrm{BL}}$ por acción |
| `data/processed/cov_*.parquet` | Matriz $\Sigma$ |
| `data/processed/weights_*.parquet` | Pesos óptimos |
| `data/processed/signals_*.parquet` | Señales stop/take |
| `data/processed/backtest_results.parquet` | Serie temporal del backtest |
| `data/processed/backtest_metrics.json` | Métricas consolidadas |
| `data/processed/infeasibility_log.csv` | Registro de meses infactibles |
| `data/processed/walk_forward_results.parquet` | Resultados walk-forward |
| `data/processed/last_run.json` | Metadata del último run (parámetros, outcome) |
| `reports/reporte_*.xlsx` | Reporte Excel (Metricas, Pesos, NAV, Dividends, Infeasibility, WalkForward) |

---

## 5. Loop de backtest y protocolo anti-lookahead

### 5.1 Convención temporal

Sea $\{t_1, t_2, \ldots, t_T\}$ la secuencia de fechas de cierre mensual desde enero 2016. En la iteración del mes $t_k$:

- $t_{k-1}$ = **fecha de información** (último dato observable para decisiones).
- $[t_{k-1}, t_k]$ = **intervalo de retorno** aplicado al portafolio vigente.

El portafolio rebalanceado al final de $t_k$ con información de $t_{k-1}$ rige durante el mes $(t_k, t_{k+1}]$.

### 5.2 Secuencia mensual (implementación en `05_backtest.py`)

Para cada mes t_k:

```
PASO 1 — Mark-to-market
         Aplicar retornos de t_{k-1} → t_k al portafolio vigente.
         Actualizar NAV. Registrar yield realizado del mes.

PASO 2 — Universo (información ≤ t_{k-1})
         screen_universe_at(date = t_{k-1})
         Precios, dividendos, ADV e historial truncados con .loc[:t_{k-1}].

PASO 3 — Señales (información ≤ t_{k-1})
         evaluate_signals_cached(eval_date = t_{k-1})
         Identificar posiciones stop-loss, take-profit o fuera de universo.

PASO 4 — Features (información ≤ t_{k-1})
         Ventana [t_{k-1} - 36m, t_{k-1}] sobre retornos mensuales.
         Estimar Σ (Ledoit-Wolf) y μ_BL.

PASO 5 — Optimización
         Resolver QP con dy, μ_BL, Σ del universo elegible.
         Si infactible → protocolo §8 (mantener portafolio previo).

PASO 6 — Rebalanceo
         Liquidar posiciones señaladas / fuera de universo.
         Asignar pesos óptimos; capital 100% invertido.
         Precio de entrada de posiciones nuevas = cierre en t_{k-1}.

PASO 7 — Registro
         NAV, retornos, turnover, flags de infeasibility, conteo de señales.
```

### 5.3 Reglas anti-lookahead

| Componente | Regla | Mecanismo en código |
|---|---|---|
| Screening de universo | Solo datos hasta tₖ₋₁ | `daily_close.loc[:date]`, dividendos con ventana [tₖ₋₁−12m, tₖ₋₁] |
| Covarianza / μ_BL | Ventana ending tₖ₋₁ | `monthly_ret.loc[window_start:window_end]` con `window_end = t_{k-1}` |
| Señales EWMA | Precios hasta tₖ₋₁ | `monthly_close[tk].loc[:eval_date]` |
| Retorno del mes | Portafolio pre-decisión | Mark-to-market sobre [tₖ₋₁, tₖ] **antes** de rebalancear |
| Benchmark | SPY mismo mes | Retorno mensual de SPY en tₖ |

**Principio:** ninguna variable observada después de $t_{k-1}$ entra en el universo, las features, las señales ni el QP del rebalanceo en $t_k$.

### 5.4 Benchmark

**SPY** (`yfinance`, `auto_adjust=True`) como proxy de renta total del mercado EE.UU. large cap. Los retornos mensuales del benchmark se calculan sobre cierres mensuales:

$$
r_{\mathrm{SPY},t} = \frac{P_{\mathrm{SPY},t}}{P_{\mathrm{SPY},t-1}} - 1
$$

---

## 6. Métricas de desempeño

Todas las métricas se calculan en `compute_metrics()` sobre la serie de retornos mensuales del portafolio. Tasa libre de riesgo default: $r_f = 4\%$ anual ($r_{f,m} = r_f / 12$ mensual).

Sea $\{r_{p,t}\}_{t=1}^{T}$ la serie de retornos mensuales del portafolio y $\{r_{b,t}\}$ los retornos de SPY, alineados en el mismo índice temporal. $\mathrm{NAV}_t$ es el valor del portafolio (base 100).

### 6.1 Rendimiento

**CAGR (Compound Annual Growth Rate):**

$$
\mathrm{CAGR}_p = \left[\prod_{t=1}^{T}(1 + r_{p,t})\right]^{12/T} - 1
$$

Análogo para SPY → `CAGR_benchmark`.

**Volatilidad anualizada:**

$$
\sigma_p = \mathrm{std}(r_p)\,\sqrt{12}
$$

### 6.2 Ratios ajustados por riesgo

**Sharpe Ratio:**

$$
\mathrm{Sharpe} = \frac{\overline{r_p} - r_{f,m}}{\mathrm{std}(r_p)}\,\sqrt{12}
$$

**Sortino Ratio** (volatilidad a la baja respecto a $r_{f,m}$):

$$
\sigma_{\downarrow} = \sqrt{\frac{1}{T}\sum_{t:\, r_{p,t} < r_{f,m}} (r_{p,t} - r_{f,m})^2 \cdot 12}, \qquad
\mathrm{Sortino} = \frac{(\overline{r_p} - r_{f,m}) \cdot 12}{\sigma_{\downarrow}}
$$

**Calmar Ratio:**

$$
\mathrm{Calmar} = \frac{\mathrm{CAGR}_p}{|\mathrm{MDD}|}
$$

Donde $\mathrm{MDD}$ es el máximo drawdown.

### 6.3 Riesgo de cola y drawdown

**Max Drawdown:**

$$
\mathrm{DD}_t = \frac{\mathrm{NAV}_t}{\max_{s \le t} \mathrm{NAV}_s} - 1, \qquad
\mathrm{MDD} = \min_t \mathrm{DD}_t
$$

**Value at Risk (VaR) histórico** al nivel $\alpha$:

$$
\mathrm{VaR}_\alpha = Q_{1-\alpha}(r_p)
$$

Donde $Q_{1-\alpha}$ es el cuantil $(1-\alpha)$ de la distribución empírica de retornos.

Implementado: VaR 95% (`quantile(0.05)`) y VaR 99% (`quantile(0.01)`).

**Conditional VaR (CVaR / Expected Shortfall):**

$$
\mathrm{CVaR}_\alpha = \mathbb{E}\!\left[r_p \;\middle|\; r_p \le \mathrm{VaR}_\alpha\right]
$$

### 6.4 Métricas de upside y dolor

**Upper Partial Moment (UPM)** con umbral mensual equivalente al 3% anual ($\theta = 0{,}03/12$):

$$
\mathrm{UPM} = \frac{1}{T}\sum_{t=1}^{T} \max(r_{p,t} - \theta,\, 0)^2
$$

**Pain Index:**

$$
\mathrm{Pain} = \frac{1}{T}\sum_{t=1}^{T} |\mathrm{DD}_t|
$$

**Pain-Gain Ratio:**

$$
\mathrm{PainGain} = \frac{\mathrm{Pain}}{\overline{r_p}}
$$

### 6.5 Alpha, tracking e información

**Retorno activo mensual:** $a_t = r_{p,t} - r_{b,t}$

**Alpha anualizado:**

$$
\alpha = \overline{a} \cdot 12
$$

**Tracking Error:**

$$
\mathrm{TE} = \mathrm{std}(a)\,\sqrt{12}
$$

**Information Ratio:**

$$
\mathrm{IR} = \frac{\alpha}{\mathrm{TE}}
$$

**Beta** (regresión de mínimos cuadrados implícita vía covarianza):

$$
\beta = \frac{\mathrm{Cov}(r_p, r_b)}{\mathrm{Var}(r_b)}
$$

### 6.6 Yield y operaciones

**Yield realizado mensual** (reporte de flujos):

$$
y_t = \sum_i w_{i,t-1}\,\frac{D_{i,t}}{V_{i,t-1}}
$$

Donde $D_{i,t}$ son los dividendos percibidos en el mes y $V_{i,t-1}$ el valor de la posición al inicio del mes.

**Yield realizado anual promedio:** $\overline{y} \cdot 12$

**% meses con yield ≥ objetivo:**

$$
\frac{1}{T}\sum_{t=1}^{T} \mathbf{1}\{y_t \ge 0{,}03/12\}
$$

**Turnover mensual:**

$$
\mathrm{Turnover}_t = \frac{1}{2}\sum_i \left|w_{i,t} - w_{i,t-1}\right|
$$

**Activaciones stop-loss / take-profit:** conteo acumulado de señales `stop` y `take` por mes.

---

## 7. Validación walk-forward

Implementada en `main.run_walk_forward()`. Objetivo: evaluar si los inputs del QP (especialmente $\mu_{\mathrm{BL}}$) tienen poder predictivo out-of-sample, sin reutilizar información futura.

### 7.1 Esquema

Ventana rolling de entrenamiento de **36 meses**, horizonte de predicción de **1 mes**:

```
|--- 36 meses calibración ---|-- mes test --|
                              |--- 36 meses ---|-- mes test --|
                                                  ...
```

Para cada par consecutivo de meses $(t_{k-1}, t_k)$ en el backtest:

1. **Calibrar** con retornos en $[t_{k-1} - 36\,\mathrm{m},\, t_{k-1}]$:
   - Estimar $\Sigma$ con Ledoit-Wolf (registrar coeficiente de shrinkage).
   - Calcular $\mu_{\mathrm{BL}}$ con el universo vigente.

2. **Predecir** $\hat{r}_{i,t_k} = \mu_{\mathrm{BL},i}$ (retorno mensual esperado).

3. **Observar** $r_{i,t_k}$ (log-retorno mensual realizado).

4. **Medir error** por mes y agregar.

Requisito mínimo: ≥ 5 acciones con ≥ 24 observaciones en la ventana.

### 7.2 Métricas de validación

Para cada mes t_k, sobre el conjunto de acciones S_k con predicción y realización:

**Error absoluto medio (MAE):**

$$
\mathrm{MAE}_k = \frac{1}{|S_k|}\sum_{i \in S_k} \left|\hat{r}_{i,t_k} - r_{i,t_k}\right|
$$

**Error cuadrático medio (RMSE):**

$$
\mathrm{RMSE}_k = \sqrt{\frac{1}{|S_k|}\sum_{i \in S_k} \left(\hat{r}_{i,t_k} - r_{i,t_k}\right)^2}
$$

**Hit ratio de ranking (top tercil):**

Sea $n_3 = \max(1, \lfloor |S_k|/3 \rfloor)$. Sean $\mathcal{T}_k^{\mathrm{pred}}$ y $\mathcal{T}_k^{\mathrm{act}}$ los conjuntos de $n_3$ acciones con mayor $\hat{r}$ y mayor $r$ realizado, respectivamente:

$$
\mathrm{Hit}_k = \frac{\left|\mathcal{T}_k^{\mathrm{pred}} \cap \mathcal{T}_k^{\mathrm{act}}\right|}{n_3}
$$

**Estabilidad de Ledoit-Wolf:** serie temporal del coeficiente de shrinkage mensual.

### 7.3 Output

`data/processed/walk_forward_results.parquet`:

```
date | mae | rmse | hit_ratio | shrinkage | n_stocks
```

Resumen agregado (promedios de MAE, RMSE, hit ratio) disponible en el notebook y en la hoja `WalkForward` del Excel.

### 7.4 Validación del constraint de yield

Complementariamente, el backtest reporta `% meses con yield ≥ 3%` sobre el yield realizado mensual (§6.6), verificando el cumplimiento operativo del mandato en simulación histórica.

---

## 8. Protocolo de infeasibility

Cuando el QP no admite solución factible con las restricciones del mandato (yield ≥ 3%, max 5%, long-only, fully invested), el sistema **no relaja restricciones de forma silenciosa**.

### 8.1 Detección

`quadprog.solve_qp` lanza excepción → se captura en `optimize_portfolio()`.

### 8.2 Yield máximo alcanzable

Se resuelve un subproblema auxiliar: **maximizar** $\sum_i w_i \,\mathrm{dy}_i$ sujeto a $\sum_i w_i = 1$, $0 \le w_i \le 0{,}05$, **sin** el constraint de yield. Solución greedy: asignar 5% iterativamente a las acciones de mayor yield hasta completar el 100%.

### 8.3 Registro

Fila en `data/processed/infeasibility_log.csv`:

```
date | max_achievable_yield | n_eligible_stocks | reason
```

### 8.4 Comportamiento default

**Mantener el portafolio del mes anterior** sin rebalancear hasta que el gestor tome una decisión.

### 8.5 Opciones para decisión del cliente

Cuando el yield de 3% no es alcanzable dadas las restricciones, el gestor dispone de las siguientes alternativas (evaluables vía `diagnose_infeasibility()` en `main.py` o `INFEASIBILITY_MODE` en el notebook):

| Escenario | Acción | Parámetro ilustrativo |
|---|---|---|
| **A** | Relajar yield objetivo del QP | `MIN_PORTFOLIO_YIELD = 2,5%` |
| **B** | Relajar límite de concentración | `MAX_WEIGHT = 8%` |
| **C** | Ampliar universo (relajar filtros operativos) | Reducir `MIN_ADV`, `MIN_MKTCAP` o pre-filtro de yield |
| **D** | No rebalancear ese mes | Mantener portafolio previo (cash implícito solo si el portafolio previo lo contiene) |

Los escenarios A, B y C se recalculan automáticamente en modo diagnóstico; el gestor selecciona la opción, ajusta parámetros en `[PARAMS]` y documenta la decisión en el log.

---

## 9. Fuentes de datos

| Dato | Fuente | Uso |
|---|---|---|
| Constituyentes S&P 500 | [Wikipedia — List of S&P 500 companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) | Universo base de tickers |
| Precios ajustados, volumen, dividendos, market cap | [yfinance](https://pypi.org/project/yfinance/) | Screening, features, señales, backtest |
| Benchmark | SPY vía yfinance | Comparación de desempeño |
| Tasa libre de riesgo | Constante r_f = 4% anual en métricas | Sharpe, Sortino |

**Cálculos propios sobre datos crudos:**

- Dividend yield TTM: suma de dividendos últimos 12 meses / precio actual.
- ADV 90 días: media de `Volume × Close` sobre 90 días hábiles.
- Retornos mensuales: log-retornos sobre cierres mensuales de precios ajustados.
- Market cap histórico en backtest: proxy por escalamiento de market cap actual según ratio de precios.

---

## 10. Setup, ejecución y parámetros

### 10.1 Instalación

```bash
python -m venv .ACTINVER
source .ACTINVER/bin/activate      # Linux / macOS
# .ACTINVER\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 10.2 Dependencias principales

| Librería | Uso |
|---|---|
| `yfinance >= 0.2.40` | Precios, dividendos, market cap |
| `quadprog >= 0.1.11` | Optimización cuadrática |
| `scikit-learn >= 1.4` | Ledoit-Wolf |
| `pandas >= 2.0`, `numpy >= 1.26` | Datos y álgebra lineal |
| `openpyxl` | Exportación Excel |

### 10.3 Ejecución

**Pipeline completo (CLI):**

```bash
python main.py --mode backtest --start 2016-01-01
python main.py --mode live
```

**Scripts individuales** (modo live paso a paso):

```bash
python src/01_universe.py
python src/02_features.py
python src/03_optimizer.py
python src/04_signals.py
python src/05_backtest.py
```

**Notebook interactivo:**

```bash
jupyter notebook strategy.ipynb
```

Modificar parámetros en la celda `[PARAMS]` antes de ejecutar.

### 10.4 Parámetros globales

Los defaults están definidos en cada módulo `src/` y se sobreescriben desde `main.run_pipeline()` o el notebook.

| Parámetro | Default | Módulo / función |
|---|---|---|
| `MIN_MKTCAP` | 100 B USD | `01_universe`, backtest |
| `MIN_YIELD_SCREEN` | 0,5% | Pre-filtro universo |
| `MIN_ADV` | 50 M USD | Pre-filtro universo |
| `HISTORY_START` | 2013-01-01 | Calibración |
| `BACKTEST_START` | 2016-01-01 | Backtest |
| `LOOKBACK_MONTHS` | 36 | Ledoit-Wolf, BL, walk-forward |
| `MIN_OBS_MONTHS` | 24 | Mínimo historia por acción |
| `MIN_PORTFOLIO_YIELD` | 3% | Constraint QP |
| `MAX_WEIGHT` | 5% | Constraint QP |
| `LAMBDA_BL` (λ) | 3,0 | Prior Black-Litterman |
| `GAMMA` (γ) | 1,0 | Objetivo QP |
| `TAU` (τ) | 0,05 | Incertidumbre views BL |
| `EWMA_SPAN` | 12 | Volatilidad señales |
| `SL_THRESHOLD` / `TP_THRESHOLD` | 1,0 | Multiplicadores σ |
| `REBALANCE_FREQ` | Mensual | Último día hábil del mes |

---

## 11. Estructura del repositorio

```
dividend-investment-stgy/
├── Caso para Posición Investor.docx   # Requerimiento original del mandato
├── main.py                            # Orquestador CLI / API Python
├── strategy.ipynb                     # Centro de control mensual interactivo
├── src/
│   ├── 01_universe.py                 # Construcción y filtrado del universo
│   ├── 02_features.py                 # Black-Litterman + Ledoit-Wolf
│   ├── 03_optimizer.py                # QP con quadprog + infeasibility
│   ├── 04_signals.py                # Stop-loss / take-profit EWMA
│   └── 05_backtest.py                 # Loop mensual + métricas
├── data/
│   ├── raw/                           # universe_*.csv
│   └── processed/                     # features, pesos, señales, resultados
├── reports/                           # reporte_*.xlsx (generados)
├── requirements.txt
└── README.md                          # Este documento
```

---

*Documento alineado al código en `main.py` y `src/`. Ante discrepancia entre este README y la implementación, prevalece el código fuente.*
