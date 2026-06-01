# Actinver — US Equity Dividend Strategy


Estrategia cuantitativa de renta variable estadounidense diseñada para garantizar un dividend yield de portafolio ≥ 3% anual, optimización media-varianza vía programación cuadrática y rebalanceo mensual con stop-loss/take-profit por acción.

---

## Tabla de contenidos

1. [Objetivo y restricciones del mandato](#1-objetivo-y-restricciones-del-mandato)
2. [Arquitectura del pipeline](#2-arquitectura-del-pipeline)
3. [Diseño de cada módulo](#3-diseño-de-cada-módulo)
   - [01 Universe](#01_universepy)
   - [02 Features](#02_featurespy)
   - [03 Optimizer](#03_optimizerpy)
   - [04 Signals](#04_signalspy)
   - [05 Backtest](#05_backtestpy)
4. [Metodología detallada](#4-metodología-detallada)
5. [Protocolo de infeasibility](#5-protocolo-de-infeasibility)
6. [Métricas de desempeño](#6-métricas-de-desempeño)
7. [Validación cruzada walk-forward](#7-validación-cruzada-walk-forward)
8. [Estructura de directorios](#8-estructura-de-directorios)
9. [Setup y ejecución](#9-setup-y-ejecución)
10. [Entregables](#10-entregables)
11. [Parámetros globales de referencia](#11-parámetros-globales-de-referencia)
12. [Notebook `strategy.ipynb` — Centro de control mensual](#12-notebook-strategyipynb--centro-de-control-mensual)

---

## 1. Objetivo y restricciones del mandato

### Objetivo
Construir y evaluar una estrategia de inversión en acciones del S&P 500 que garantice, a nivel de portafolio ponderado, un **dividend yield anual ≥ 3%** en todo momento.

### Restricciones duras del portafolio (hard constraints para el QP)

| Restricción | Valor | Nivel |
|---|---|---|
| Peso máximo por acción | 5% | Por acción |
| Peso mínimo por acción | 0% (long-only) | Por acción |
| Suma de pesos | 100% (siempre invertidos) | Portafolio |
| Dividend yield ponderado | ≥ 3% anual | Portafolio |
| Market cap mínima | ≥ USD 100 B | Por acción (filtro universo) |

### Restricciones de universo (pre-filtros, aplicados mensualmente)

| Filtro | Valor | Razón |
|---|---|---|
| Market cap | ≥ USD 100 B | Mandato |
| Dividend yield TTM | ≥ 0.5% | Pre-filtro holgado; la restricción real es 3% en el QP |
| ADV 90 días | ≥ USD 50 M | Liquidez mínima operativa |
| Historial de precios | Disponible desde 2013-01-01 (±45 días de gracia) | Suficiente lookback para calibración |

### Período de evaluación

- **Backtest:** enero 2016 – presente
- **Lookback de calibración inicial:** enero 2013 – diciembre 2015 (36 meses)
- El universo se re-evalúa con los filtros completos **cada mes**; stocks pueden entrar y salir.

---

## 2. Arquitectura del pipeline

```
01_universe.py
      │  universo elegible del mes (CSV)
      ▼
02_features.py
      │  retornos esperados (Black-Litterman)
      │  matriz de covarianza (Ledoit-Wolf)
      ▼
03_optimizer.py
      │  pesos óptimos del portafolio (quadprog QP)
      │  protocolo de infeasibility si aplica
      ▼
04_signals.py
      │  señales de stop-loss / take-profit por acción (EWMA)
      ▼
05_backtest.py
      │  simulación histórica completa
      │  NAV, retornos, dividend yield realizado
      ▼
    métricas + reporte + presentación
```

Cada módulo lee el output del anterior desde `data/processed/`. El backtest orquesta el loop mensual completo: universe → features → optimize → apply signals → mark-to-market.

---

## 3. Diseño de cada módulo

### `01_universe.py`

**Responsabilidad:** Construir el universo elegible para un mes dado.

**Inputs:**
- Lista de constituyentes actuales del S&P 500 (Wikipedia scraping)
- yfinance: `fast_info`, `dividends`, `history`

**Proceso:**
1. Descargar tickers del S&P 500 desde Wikipedia.
2. Para cada ticker, aplicar secuencialmente los 4 filtros de universo. Ejecución paralela con `ThreadPoolExecutor`.
3. Calcular dividend yield TTM = Σ(dividendos últimos 12m) / precio actual. Se usa cálculo propio sobre `t.dividends` (más confiable que el campo `dividendYield` de yfinance).

**Output:** `data/raw/universe_YYYYMMDD.csv`

```
ticker | market_cap | dividend_yield | adv_90d | history_start | price
```

**Nota:** El filtro `min_yield=0.5%` es intencional y holgado. El universo pre-filtrado alimenta al optimizador, que impone la restricción real de 3% como constraint del QP. Un universo con solo stocks ≥3% podría hacer infeasible el QP por falta de diversificación.

---

### `02_features.py`

**Responsabilidad:** Estimar retornos esperados y matriz de covarianzas para el mes de rebalanceo.

**Inputs:**
- `data/raw/universe_YYYYMMDD.csv` (universo elegible del mes)
- Precios mensuales ajustados desde yfinance (ventana rolling 36 meses)
- Pesos de mercado del S&P 500 para los stocks elegibles (proxy: market cap relativa dentro del universo elegible)

**Proceso:**

#### 2a. Matriz de covarianzas — Ledoit-Wolf shrinkage

```python
from sklearn.covariance import LedoitWolf

lw = LedoitWolf()
lw.fit(returns_matrix)   # retornos mensuales, ventana 36m
Sigma = lw.covariance_   # shape: (n, n)
```

Se usa Ledoit-Wolf porque el universo elegible es típicamente pequeño (20–60 stocks) y la covarianza muestral es inestable en muestras cortas. Ledoit-Wolf reduce el error de estimación via shrinkage hacia la identidad escalada.

#### 2b. Retornos esperados — Black-Litterman

El modelo Black-Litterman parte de los **implied equilibrium returns** del mercado como prior y los ajusta con **views cuantitativas** del gestor.

**Prior (implied returns):**
```
Π = λ · Σ · w_mkt
```
- `λ`: coeficiente de aversión al riesgo del mercado (calibrado típicamente entre 2.5 y 3.5; default 3.0)
- `Σ`: matriz de covarianzas Ledoit-Wolf
- `w_mkt`: pesos de capitalización de mercado relativa dentro del universo elegible

**Views cuantitativas:**

Se usa el **dividend yield esperado** de cada acción como view sobre su retorno total esperado. La lógica: el dividend yield es la componente observable y predecible del retorno; el precio esperado captura el residual.

```
Q = vector de views (dividend yield TTM por acción)
P = matriz de selección de views (identity en este caso, una view por acción)
Ω = τ · (P · Σ · P')  # incertidumbre proporcional a la varianza del activo
τ = 0.05              # escalar de incertidumbre del prior (convención estándar)
```

**Posterior (retornos esperados Black-Litterman):**
```
μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ · [(τΣ)⁻¹Π + P'Ω⁻¹Q]
```

**Output:** `data/processed/features_YYYY_MM.parquet`

```
ticker | mu_bl | sigma_diag | dividend_yield_ttm
```

Y la matriz Sigma completa: `data/processed/cov_YYYY_MM.parquet`

---

### `03_optimizer.py`

**Responsabilidad:** Resolver el problema de optimización cuadrática media-varianza con las restricciones del mandato.

**Formulación del problema:**

```
min  (1/2) · w' · Σ · w  -  γ · μ_BL' · w

s.t.
    Σᵢ wᵢ = 1                    (fully invested)
    0 ≤ wᵢ ≤ 0.05  ∀i            (long-only, max 5%)
    Σᵢ wᵢ · dyᵢ ≥ 0.03           (dividend yield portafolio ≥ 3%)
```

donde:
- `w`: vector de pesos (n,)
- `Σ`: matriz de covarianzas Ledoit-Wolf (n×n)
- `μ_BL`: retornos esperados Black-Litterman (n,)
- `γ`: parámetro de aversión al riesgo (sweepable; default 1.0)
- `dyᵢ`: dividend yield TTM del stock i

**Librería:** `quadprog.solve_qp`

La función `solve_qp(G, a, C, b, meq)` resuelve:
```
min  (1/2) x'Gx - a'x
s.t. C'x >= b  (las primeras meq restricciones son igualdades)
```

La restricción de igualdad `Σwᵢ = 1` se pasa como `meq=1`. Las restricciones de desigualdad incluyen bounds por acción y el constraint de yield.

**Manejo de infeasibility:** Ver sección 5.

**Output:** `data/processed/weights_YYYY_MM.parquet`

```
ticker | weight | expected_return | contribution_yield
```

---

### `04_signals.py`

**Responsabilidad:** Calcular señales de stop-loss y take-profit para cada posición abierta al cierre de cada mes.

**Lógica:**

Para cada acción `i` con posición abierta desde el mes de entrada `t₀`:

1. Calcular el retorno acumulado desde entrada:
   ```
   r_acum(i, t) = (P_t / P_t0) - 1
   ```

2. Estimar la volatilidad EWMA de la acción con span=12 mensual:
   ```
   σ_ewma(i, t) = EWMA(retornos_mensuales_i, span=12).iloc[-1] ** 0.5  # si es varianza
   ```
   Usando `pandas.DataFrame.ewm(span=12).std()` sobre la serie de retornos mensuales históricos.

3. Evaluar triggers al **cierre del mes**:

   | Condición | Señal | Acción |
   |---|---|---|
   | `r_acum(i,t) > +1 · σ_ewma(i,t)` | Take-profit | Liquidar posición |
   | `r_acum(i,t) < -1 · σ_ewma(i,t)` | Stop-loss | Liquidar posición |
   | En rango | Hold | Mantener hasta siguiente rebalanceo |

4. El cash generado por liquidaciones **se mantiene en cash** hasta el siguiente rebalanceo mensual. En ese momento el optimizador re-asigna el capital total disponible (incluyendo cash) al nuevo portafolio óptimo.

**Importante:** El stop-loss/take-profit se verifica **después** del mark-to-market mensual y **antes** de que el optimizador calcule los nuevos pesos. El capital en cash en `t` entra como capital disponible al optimizador en `t+1`.

**Output:** `data/processed/signals_YYYY_MM.parquet`

```
ticker | entry_month | r_acum | sigma_ewma | signal (hold/stop/take)
```

---

### `05_backtest.py`

**Responsabilidad:** Orquestar el loop mensual completo y calcular todas las métricas de desempeño.

**Loop mensual (para cada mes t desde ene 2016):**

```
1. Re-screen universo (01_universe logic)
2. Calcular features: μ_BL, Σ_LW (02_features logic)
3. Evaluar señales del mes anterior → determinar cash disponible (04_signals logic)
4. Optimizar pesos sobre capital invertible (capital total - cash reservado) (03_optimizer logic)
5. Ejecutar rebalanceo:
   a. Stocks con señal stop/take → liquidar al cierre del mes t-1
   b. Stocks que salieron del universo → liquidar
   c. Reasignar pesos según optimizador
6. Mark-to-market al cierre del mes t
7. Registrar: NAV, pesos, dividend yield realizado, señales activas
```

**Dividend yield realizado:** Se calcula sumando los dividendos ex-date que cayeron en el mes para cada posición ponderada. Se compara contra el 3% objetivo (anualizado) para el reporting.

**Output principal:** `data/processed/backtest_results.parquet`

```
date | nav | portfolio_return | benchmark_return | realized_yield | 
n_stocks | cash_pct | rebalance_flag | infeasibility_flag
```

---

## 4. Metodología detallada

### Black-Litterman — detalles de implementación

| Parámetro | Valor | Fuente |
|---|---|---|
| `λ` (risk aversion) | 3.0 | Convención literature (He & Litterman 1999) |
| `τ` (prior uncertainty) | 0.05 | Convención estándar |
| `w_mkt` | market cap relativa dentro del universo elegible | yfinance fast_info |
| Views `Q` | dividend yield TTM por acción | Calculado en 01_universe |
| `Ω` | `τ · P · Σ · P'` (diagonal) | Proporcional a varianza del activo |

### Ledoit-Wolf — ventana de estimación

- **Ventana rolling:** 36 meses de retornos mensuales
- **Retornos:** log-retornos mensuales sobre precios ajustados
- **Mínimo de observaciones:** 24 meses (si un stock entró al universo hace menos de 36m, se usa su historial disponible; si < 24m, se excluye del universo ese mes)

### EWMA — volatilidad para stop-loss/take-profit

- **Frecuencia:** mensual
- **Span:** 12 (equivale a λ = 1 - 2/(12+1) ≈ 0.846)
- **Serie:** retornos mensuales simples del stock
- **σ en t:** `returns.ewm(span=12).std().iloc[-1]`
- El umbral ±1σ se recalcula cada mes con la volatilidad EWMA actualizada al cierre del mes anterior.

---

## 5. Protocolo de infeasibility

Cuando el optimizador no puede satisfacer simultáneamente todas las restricciones duras (yield ≥ 3%, max 5% por acción, sum = 1, long-only), el sistema **no relaja restricciones silenciosamente**.

**Pasos:**

1. **Detectar infeasibility:** `quadprog.solve_qp` lanza excepción → capturar.

2. **Calcular yield máximo alcanzable:** Resolver un problema secundario: maximizar `Σ wᵢ · dyᵢ` sujeto a todas las restricciones *excepto* el constraint de yield. Reportar el yield máximo obtenible ese mes.

3. **Documentar la excepción** en `data/processed/infeasibility_log.csv`:
   ```
   date | max_achievable_yield | n_eligible_stocks | reason
   ```

4. **Mantener portafolio del mes anterior** como posición por defecto mientras se resuelve.

5. **Presentar opciones al cliente** (en el informe):
   - Aceptar el yield inferior ese mes (máximo alcanzable)
   - Relajar el filtro de ADV o yield mínimo del universo
   - Mantener cash ese mes

---

## 6. Métricas de desempeño

Calculadas sobre la serie de retornos mensuales del portafolio vs. benchmark (S&P 500 total return).

| Métrica | Definición |
|---|---|
| **Sharpe Ratio** | `(μ_p - r_f) / σ_p` anualizado; `r_f` = T-Bill 3m |
| **Sortino Ratio** | `(μ_p - r_f) / σ_downside` anualizado |
| **Calmar Ratio** | `CAGR / |Max Drawdown|` |
| **Max Drawdown** | Máxima caída pico-a-valle en NAV |
| **VaR 95% / 99%** | Percentil 5% / 1% de retornos mensuales (histórico) |
| **CVaR 95% / 99%** | Media de retornos por debajo del VaR (Expected Shortfall) |
| **Upper Partial Moment** | `E[max(r - r_umbral, 0)^n]`; mide upside vs. umbral (3%) |
| **Pain-Gain Ratio** | `Pain Index / retorno medio`; Pain Index = media de drawdowns durante el período |
| **Dividend Yield Realizado** | Yield mensual realizado vs. objetivo 3% anual (25 bps/mes); reportado mes a mes |
| **Tracking Error** | Desviación estándar de (retorno portafolio - retorno benchmark) |
| **Information Ratio** | `Alpha / Tracking Error` |
| **Beta** | Regresión de retornos portafolio vs. S&P 500 |
| **Turnover mensual** | Suma de cambios absolutos en pesos; proxy de costo de rebalanceo |

---

## 7. Validación cruzada walk-forward

Objetivo: validar que las estimaciones de retornos esperados, volatilidad y correlación (los inputs al QP) son estadísticamente confiables y no sobreajustadas.

**Esquema:**

```
|--- 36 meses train ---|-- 1 mes test --|
                        |--- 36 meses train ---|-- 1 mes test --|
                                                ...
```

- **Ventana de entrenamiento:** 36 meses rolling (no expanding)
- **Horizonte de predicción:** 1 mes
- **Métrica de validación:** MAE y RMSE entre retorno predicho (μ_BL) y retorno realizado

**Outputs de validación:**

1. Distribución del error de predicción de retornos (μ_BL vs. realizado)
2. Estabilidad de los pesos de Ledoit-Wolf en el tiempo (shrinkage coefficient mensual)
3. Hit ratio del modelo B-L: % de meses donde el ranking de retornos predichos coincide con el realizado (top tercil)
4. Backtesting del constraint de yield: % de meses donde el yield realizado superó el 3% objetivo

---

## 8. Estructura de directorios

```
Actinver/
├── strategy.ipynb          # Centro de control mensual — orquesta todo el pipeline
├── src/
│   ├── 01_universe.py      # Construcción y filtrado del universo (mensual)
│   ├── 02_features.py      # Black-Litterman + Ledoit-Wolf
│   ├── 03_optimizer.py     # QP con quadprog, protocolo de infeasibility
│   ├── 04_signals.py       # Stop-loss / take-profit EWMA
│   └── 05_backtest.py      # Loop mensual, métricas, reporting
├── data/
│   ├── raw/                # universe_YYYYMMDD.csv, precios descargados
│   └── processed/          # features, pesos, señales, resultados backtest
├── reports/                # Informe escrito (Word/PDF)
├── presentation/           # Archivo PowerPoint
├── requirements.txt
└── README.md               # Este archivo — fuente de verdad del proyecto
```

---

## 9. Setup y ejecución

### Instalación

```bash
# Crear y activar entorno virtual
python -m venv .ACTINVER
.ACTINVER\Scripts\activate        # Windows
source .ACTINVER/bin/activate      # Mac/Linux

pip install -r requirements.txt
```

### Dependencias principales

| Librería | Uso |
|---|---|
| `yfinance >= 0.2.40` | Precios, dividendos, market cap |
| `quadprog >= 0.1.11` | Optimización cuadrática (QP) |
| `scikit-learn >= 1.4` | Ledoit-Wolf shrinkage |
| `arch >= 6.3` | EWMA (alternativa robusta a pandas ewm) |
| `pandas >= 2.0` | Manipulación de datos |
| `numpy >= 1.26` | Álgebra lineal |
| `python-pptx >= 0.6.23` | Generación de presentación PowerPoint |

### Ejecución del pipeline completo

```bash
# Paso a paso
python src/01_universe.py       # construye universo actual
python src/02_features.py       # calcula μ_BL y Σ_LW
python src/03_optimizer.py      # optimiza pesos
python src/04_signals.py        # evalúa stop-loss/take-profit
python src/05_backtest.py       # corre backtest completo 2016-presente
```

El backtest (`05_backtest.py`) orquesta internamente los pasos 1–4 para cada mes del período histórico. Los scripts individuales (1–4) sirven también para modo "live": calcular el portafolio óptimo del mes actual.

---

## 10. Entregables

| Entregable | Formato | Descripción |
|---|---|---|
| Código Python | `.py` (src/) | Pipeline completo, documentado, ejecutable |
| Informe escrito | Word / PDF | Estrategia, metodología, métricas, excepciones |
| Presentación | PowerPoint | Resumen ejecutivo para equipo de inversión y clientes |

### Contenido mínimo del informe

1. Descripción de la estrategia y sus restricciones
2. Metodología: Black-Litterman, Ledoit-Wolf, QP, EWMA
3. Resultados del backtest (2016–presente): métricas de desempeño
4. Dividend yield realizado mes a mes vs. objetivo 3%
5. Análisis de stop-loss/take-profit activados (frecuencia, impacto)
6. Resultados de validación cruzada walk-forward
7. Log de excepciones de infeasibility (si ocurrieron)
8. Conclusiones y recomendaciones para el cliente

---

## 11. Parámetros globales de referencia

Estos valores se centralizan en una sección de constantes dentro de un `config.py`. Son la fuente de verdad para reproducibilidad.

```python
# Universo
MIN_MKTCAP       = 100e9          # USD 100B
MIN_YIELD_SCREEN = 0.005          # 0.5% — pre-filtro holgado del universo
MIN_ADV          = 50e6           # USD 50M ADV 90 días
HISTORY_START    = "2013-01-01"   # 3 años antes del backtest
GRACE_DAYS       = 45             # tolerancia historial

# Backtest
BACKTEST_START   = "2016-01-01"
LOOKBACK_MONTHS  = 36             # ventana de calibración (Ledoit-Wolf, B-L)
MIN_OBS_MONTHS   = 24             # mínimo histórico para incluir stock

# Optimización QP
MIN_PORTFOLIO_YIELD = 0.03        # 3% anual — constraint duro del QP
MAX_WEIGHT          = 0.05        # 5% por acción
RISK_AVERSION       = 3.0         # λ Black-Litterman
TAU                 = 0.05        # τ incertidumbre del prior B-L

# Stop-loss / Take-profit
EWMA_SPAN      = 12               # span mensual para volatilidad EWMA
SL_THRESHOLD   = -1.0             # multiplicador σ para stop-loss
TP_THRESHOLD   = +1.0             # multiplicador σ para take-profit

# Rebalanceo
REBALANCE_FREQ = "M"              # mensual, al último día hábil del mes
```

---

## 12. Notebook `strategy.ipynb` — Centro de control mensual

### Propósito

`strategy.ipynb` es el único punto de entrada para el uso operativo mensual. Orquesta el pipeline completo (módulos 01–05), genera el reporte de métricas y provee controles explícitos para manejar el protocolo de infeasibility sin tocar código fuente.

### Estructura de celdas

#### Celda 1 — `[PARAMS]` Parámetros operativos (única celda a modificar en uso normal)

```python
# ============================================================
# PARAMS — ajustar aquí antes de correr el notebook cada mes
# ============================================================

# Modo de ejecución
RUN_MODE = "backtest"   # "backtest" | "live"
                        # "backtest": corre el loop histórico completo (2016-presente)
                        # "live":     calcula solo el portafolio del mes actual

# Período
BACKTEST_START   = "2016-01-01"
BACKTEST_END     = None           # None = hasta hoy

# --- Restricciones del portafolio (QP) ---
MIN_PORTFOLIO_YIELD = 0.03        # 3% objetivo del mandato
MAX_WEIGHT          = 0.05        # 5% por acción
RISK_AVERSION       = 3.0         # λ Black-Litterman

# --- Filtros del universo ---
MIN_MKTCAP          = 100e9       # USD 100B
MIN_YIELD_SCREEN    = 0.005       # pre-filtro holgado (0.5%)
MIN_ADV             = 50e6        # USD 50M ADV 90d

# --- Protocolo de infeasibility ---
# Si el QP es infeasible con los parámetros de arriba, cambiar aquí:
INFEASIBILITY_MODE  = False       # True: activa modo de diagnóstico
RELAXED_YIELD       = 0.025       # yield objetivo relajado (ej. 2.5%)
RELAXED_MAX_WEIGHT  = 0.08        # peso máximo relajado (ej. 8%)
RELAXED_MIN_ADV     = 25e6        # ADV mínimo relajado (ej. 25M)
RELAXED_MIN_MKTCAP  = 50e9        # market cap mínima relajada (ej. 50B)

# --- Black-Litterman ---
TAU              = 0.05
LOOKBACK_MONTHS  = 36
MIN_OBS_MONTHS   = 24

# --- Stop-loss / Take-profit ---
EWMA_SPAN        = 12
SL_THRESHOLD     = -1.0
TP_THRESHOLD     = +1.0
```

Toda la lógica del notebook lee de estas variables. Cambiar un parámetro aquí se propaga a todos los pasos sin modificar `src/`.

#### Celda 2 — Imports y setup

Carga de librerías, importación de módulos `src/`, configuración de paths.

#### Celda 3 — Construcción del universo

Llama a `get_universe()` de `01_universe.py` con los parámetros de `[PARAMS]`. Muestra tabla del universo elegible con market cap, yield TTM y ADV. En modo `"live"`, muestra el universo del mes actual.

#### Celda 4 — Features: Black-Litterman + Ledoit-Wolf

Ejecuta `02_features.py`. Muestra:
- Heatmap de la matriz de correlación Ledoit-Wolf
- Bar chart de retornos esperados μ_BL por acción
- Tabla comparativa: implied returns (prior) vs. μ_BL (posterior) vs. dividend yield TTM (view)

#### Celda 5 — Optimización QP

Ejecuta `03_optimizer.py`. Muestra:
- Pesos óptimos (bar chart, ordenados por peso descendente)
- Yield esperado del portafolio vs. umbral 3%
- Contribución al yield por acción (`wᵢ · dyᵢ`)
- Volatilidad esperada del portafolio anualizada
- **Si `INFEASIBILITY_MODE = True`:** diagnóstico completo (ver sección 12a)

#### Celda 6 — Señales stop-loss / take-profit

Ejecuta `04_signals.py` sobre las posiciones activas. Muestra tabla de posiciones con retorno acumulado, σ_EWMA y señal activa.

#### Celda 7 — Backtest completo *(solo en `RUN_MODE = "backtest"`)*

Ejecuta `05_backtest.py`. Genera:
- Gráfica de NAV del portafolio vs. S&P 500 Total Return (2016–presente)
- Gráfica de dividend yield realizado mensual con línea de corte en 3%
- Gráfica de drawdown acumulado
- Tabla de pesos mensuales (heatmap)

#### Celda 8 — Reporte de métricas

Tabla consolidada de todas las métricas del mandato más métricas complementarias:

```
MÉTRICAS DEL MANDATO          MÉTRICAS COMPLEMENTARIAS
──────────────────────        ─────────────────────────────
Sharpe Ratio (anual)          Sortino Ratio
Max Drawdown                  Calmar Ratio
VaR 95% (mensual)             CVaR 95% / 99%
VaR 99% (mensual)             Information Ratio
Upper Partial Moment          Tracking Error vs S&P 500
Pain-Gain Ratio               Beta
                              CAGR (portafolio vs benchmark)
                              Volatilidad anualizada
                              Turnover mensual promedio
                              % meses con yield ≥ 3%
                              Yield realizado promedio anual
                              # activaciones stop-loss
                              # activaciones take-profit
```

Todas las métricas se muestran también en comparación con el benchmark (S&P 500 Total Return).

#### Celda 9 — Validación cruzada walk-forward

Resultados resumidos: MAE de predicción de retornos, hit ratio del ranking B-L, estabilidad del shrinkage coefficient de Ledoit-Wolf a través del tiempo.

#### Celda 10 — Exportar reporte

Genera automáticamente `reports/reporte_YYYYMMDD.xlsx` con:
- Hoja `Metricas`: tabla completa de desempeño
- Hoja `Pesos`: histórico mensual de asignación por acción
- Hoja `NAV`: serie temporal del portafolio
- Hoja `Dividends`: yield realizado mes a mes vs. objetivo
- Hoja `Infeasibility`: log de excepciones (si las hubo)

---

### 12a. Modo de infeasibility (`INFEASIBILITY_MODE = True`)

Cuando el QP no encuentra solución con los parámetros base, el notebook activa una sección de diagnóstico interactivo en lugar de fallar silenciosamente.

**Flujo en celda 5 cuando `INFEASIBILITY_MODE = True`:**

```
┌─ DIAGNÓSTICO DE INFEASIBILITY ──────────────────────────────────────┐
│  Fecha:                  2024-03  (ejemplo)                          │
│  Yield máx. alcanzable:  2.71%   (con restricciones base intactas)  │
│  N° stocks elegibles:    18                                          │
│                                                                      │
│  Escenario A — Relajar yield objetivo:                               │
│    MIN_PORTFOLIO_YIELD = RELAXED_YIELD (2.5%) → ¿factible? Sí       │
│    Portafolio resultante + métricas de riesgo                        │
│                                                                      │
│  Escenario B — Relajar peso máximo:                                  │
│    MAX_WEIGHT = RELAXED_MAX_WEIGHT (8%) → ¿factible? Sí             │
│    Portafolio resultante + métricas de riesgo                        │
│                                                                      │
│  Escenario C — Relajar filtros de universo:                          │
│    MIN_ADV = RELAXED_MIN_ADV (25M) → N° stocks elegibles: 27        │
│    Optimizar con universo ampliado → ¿factible? Sí                  │
│                                                                      │
│  Escenario D — Cash ese mes:                                         │
│    Mantener portafolio del mes anterior sin rebalancear              │
│                                                                      │
│  → Todos los escenarios se muestran lado a lado para decisión        │
│    del gestor. El log se exporta automáticamente al reporte.         │
└─────────────────────────────────────────────────────────────────────┘
```

Los escenarios A, B, C y D se calculan automáticamente al activar `INFEASIBILITY_MODE = True`. El gestor elige el escenario ajustando los parámetros `RELAXED_*` en `[PARAMS]` y re-ejecutando la celda 5. La decisión y el escenario elegido quedan registrados en `data/processed/infeasibility_log.csv`.
