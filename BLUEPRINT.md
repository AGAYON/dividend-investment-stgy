# BLUEPRINT — Estrategia de Inversión en Acciones EE.UU.
**Posición: Investor — Actinver**  
**Plazo de entrega: 1 semana**

---

## 0. Visión general

El objetivo es construir un **pipeline de inversión cuantitativa** que:

1. Selecciona acciones de alta capitalización (≥ USD 100B) con dividend yield ≥ 3% anual.
2. Optimiza pesos vía programación cuadrática (`qp.solve_qp`) con restricción de peso máximo 5% por acción.
3. Se rebalancea mensualmente con lógica de stop-loss y take-profit basada en desviación estándar.
4. Hace backtesting desde enero 2016 al presente (usando al menos 3 años previos como lookback inicial).
5. Produce métricas de desempeño, informe escrito y presentación PowerPoint.

**Entregables finales:** `strategy.py` / notebook · `Informe.docx` · `Presentacion.pptx`

---

## 1. Estructura del repositorio

```
actinver-investor/
│
├── BLUEPRINT.md               ← este archivo
├── README.md
│
├── data/
│   ├── raw/                   ← precios, dividendos, market cap descargados
│   └── processed/             ← series limpias listas para el modelo
│
├── src/
│   ├── 01_universe.py         ← construcción y filtrado del universo de acciones
│   ├── 02_features.py         ← cálculo de rendimientos esperados, vol, correlación
│   ├── 03_optimizer.py        ← optimización QP con qp.solve_qp
│   ├── 04_signals.py          ← lógica stop-loss / take-profit
│   ├── 05_backtest.py         ← motor de backtesting mensual
│   ├── 06_metrics.py          ← Sharpe, Drawdown, VaR, UPM, Pain-Gain, etc.
│   └── 07_update.py           ← script de actualización mensual (≤ 5 min de trabajo)
│
├── notebooks/
│   └── full_pipeline.ipynb    ← orquesta los 7 módulos, reproducible de punta a punta
│
├── outputs/
│   ├── portfolio_history.csv  ← pesos mensuales históricos
│   ├── metrics_summary.csv    ← tabla de métricas
│   ├── Informe.docx
│   └── Presentacion.pptx
│
└── requirements.txt
```

---

## 2. Fase 1 — Extracción y preparación de datos

### 2.1 Universo inicial

**Fuente:** `yfinance` (libre, sin API key). Punto de partida: índice S&P 500 constituents (lista pública de Wikipedia o archivo local).

**Filtros de entrada al universo:**

| Criterio | Valor | Justificación |
|---|---|---|
| Market cap | ≥ USD 100B | Requerimiento explícito del caso |
| Dividend yield TTM | ≥ 0.5% | Solo garantiza que la empresa tenga historial de dividendos; el 3% se exige a nivel de **portafolio**, no por acción individual (ver §3.2) |
| Liquidez mínima | ADV 90d > USD 50M | Garantiza ejecución sin impacto de mercado relevante |
| Historial | Disponible desde ene 2013 | Cubre el lookback de 3 años antes del inicio del backtest (ene 2016) |

> **Decisión de diseño — el 3% es una restricción del portafolio, no del screening:**
> El objetivo del cliente es *"un rendimiento de dividendos anual de al menos el 3% en todo momento"* — esto se refiere al portafolio agregado. Aplicar el 3% como filtro individual en el screening sería una sobre-restricción que reduce innecesariamente el universo elegible (de ~50-60 acciones a ~13 con datos actuales), haciendo infactible la restricción de peso máximo del 5% por acción. La restricción correcta vive en el optimizador como `Σ(wᵢ × yieldᵢ) ≥ 0.03` (ver §3.2).

```python
# src/01_universe.py — esquema
import yfinance as yf

def get_universe(min_mktcap=100e9, min_yield=0.005) -> list[str]:
    """Devuelve lista de tickers que pasan los filtros en la fecha dada."""
    ...
```

> **Nota:** El universo se recalcula cada mes; una acción puede entrar o salir del portafolio elegible.

### 2.2 Series de precios y dividendos

- **Precios ajustados por splits y dividendos:** `yf.download(tickers, start, end, auto_adjust=True)`
- **Dividendos brutos:** `Ticker.dividends` — para validar que el yield proyectado se pagó históricamente.
- **Market cap mensual:** `Ticker.fast_info.market_cap` (snapshot actual) + estimación histórica con `shares_outstanding × price`.

### 2.3 Limpieza

- Forward-fill máximo 5 días hábiles para precios faltantes.
- Drop de tickers con > 10% de datos faltantes en el período.
- Almacenar en `data/processed/prices.parquet` y `dividends.parquet`.

---

## 3. Fase 2 — Modelo y pipeline

### 3.1 Estimación de parámetros de entrada (features)

Los tres insumos del optimizador QP son **rendimientos esperados (μ)**, **volatilidad esperada (σ)** y **correlación esperada (ρ)**. El caso pide proponer métodos y hacer validación cruzada.

#### Rendimientos esperados (μ)

| Método | Descripción |
|---|---|
| **Historical mean** | Promedio rolling 36 meses — baseline simple |
| **Shrinkage (James-Stein)** | Encoge μ_i hacia la media global — reduce estimation error |
| **EWMA** | Media exponencial ponderada, más peso a datos recientes (λ = 0.94) |

**Selección por validación cruzada:** Walk-forward CV con ventanas de 12 meses de test; se elige el método con menor MSE en rendimientos futuros. El CV se ejecuta en **cada rebalanceo mensual** — `compute_features(mu_method=None)` lo corre automáticamente sobre el slice de retornos disponibles hasta `as_of_date`, sin lookahead, de modo que la selección del estimador refleja el régimen de mercado vigente con la misma cadencia que el portafolio.

#### Volatilidad esperada (σ)

- **GARCH(1,1)** via `arch` library — captura clustering de volatilidad.
- **EWMA** como alternativa computacionalmente barata; también usado como fallback si GARCH no converge para algún ticker.
- Validación: comparar vol pronosticada vs vol realizada en la ventana test.
- **Implementación:** GARCH se precomputa una sola vez sobre el historial completo y se guarda en `data/processed/garch_vols.parquet`. La serie condicional es causal (σ²_t solo depende de ε_{t-1} y σ²_{t-1}), por lo que slicear `vol[:as_of]` en el backtester es correcto sin lookahead. Los parámetros α, β, ω sí se estiman con datos futuros — limitación estándar y aceptada documentada en el informe.

#### Rendimientos esperados — retorno total

- μ debe calcularse sobre **retorno total** (precio + dividendo), no solo retorno de precio. En un portafolio de dividendos, usar solo precio crea una contradicción: la restricción `Σ(wᵢ × yieldᵢ)/Σwᵢ ≥ 3%` fuerza acciones de alto yield, pero μ basado en precio las penalizaría si su precio no ha subido. Retorno total alinea el estimador de μ con el objetivo del cliente.

#### Correlación esperada (ρ)

- **Ledoit-Wolf shrinkage** (implementado en `sklearn.covariance.LedoitWolf`) — evita matrices singulares con universos grandes.
- Lookback rolling: 36 meses.

```python
# src/02_features.py — esquema
def compute_expected_returns(prices: pd.DataFrame, method='ewma') -> pd.Series: ...
def compute_cov_matrix(returns: pd.DataFrame) -> pd.DataFrame: ...  # Ledoit-Wolf
```

### 3.2 Optimizador cuadrático (`qp.solve_qp`)

**Problema:** Maximizar dividend yield del portafolio sujeto a:

- `Σ wᵢ ≤ 1` (pesos suman como máximo 1; capital no desplegado queda en T-bills)
- `0 ≤ wᵢ ≤ 0.05` (máximo 5% por acción)
- `Σ(wᵢ × yieldᵢ) / Σwᵢ ≥ 0.03` (yield sobre el capital **efectivamente invertido en acciones** ≥ 3%)

> **Decisión de diseño — definición del denominador del yield:**
> El requerimiento del cliente es *"rendimiento de dividendos anual de al menos el 3% en todo momento"* — hace referencia explícita a dividendos, no al retorno total del portafolio. Por lo tanto, el 3% se mide sobre el capital invertido en acciones (`Σwᵢ`), no sobre el capital total incluyendo cash. Esto permite que en meses donde el mercado no ofrece suficientes opciones dentro del cap de 5%, el sistema reduzca la exposición a equidad y mantenga el remanente en T-bills, preservando la garantía de yield sobre la porción invertida. Esta interpretación debe declararse explícitamente en el **Informe.docx**.

> **Nota sobre factibilidad:** Con `Σwᵢ ≤ 1` el QP siempre tiene solución factible: en el caso extremo puede asignar pesos mínimos a las acciones de mayor yield del universo y cubrir el 3% sin necesidad de estar 100% invertido.

Alternativamente: minimizar varianza sujeto a yield ≥ 3% y retorno esperado objetivo (Markowitz con restricción de dividendo).

```python
# src/03_optimizer.py — esquema
import quadprog  # qp.solve_qp

def optimize_portfolio(mu, cov, yields, min_yield=0.03, max_weight=0.05) -> np.ndarray:
    """
    Minimiza: 0.5 * w' Σ w - λ * μ' w
    s.a.: Aw = b, Cw ≥ d
    """
    ...
```

> **Tip de implementación:** `quadprog.solve_qp` requiere que la matriz de covarianza sea positiva definida. Añadir un nugget: `Σ + ε·I` con `ε = 1e-8`.

### 3.3 Señales de stop-loss y take-profit

Las señales operan **a nivel de acción individual**, no a nivel de portafolio.

- **Stop-loss:** Si el precio de la acción `i` cae más de `1σ` (rolling 252d) desde el precio de compra → vender posición, asignar peso 0 y redistribuir proporcionalmente entre el resto.
- **Take-profit:** Si el precio sube más de `1σ` desde el precio de compra → realizar ganancia, reducir peso a la mitad, redistribuir el exceso.

```python
# src/04_signals.py — esquema
def check_stop_loss(position_price, current_price, sigma_1y) -> bool: ...
def check_take_profit(position_price, current_price, sigma_1y) -> bool: ...
```

> **Nota metodológica:** Usar `σ` de los **últimos 252 días hábiles** del precio de la acción, no del portafolio.

### 3.4 Motor de backtesting mensual

```
Para cada mes t desde ene-2016 hasta hoy:
  1. Recalcular universo elegible (market cap, yield, historial).
  2. Calcular μ, σ, ρ con datos hasta t-1.
  3. Aplicar stop-loss/take-profit sobre posiciones actuales.
  4. Correr optimizador QP → nuevos pesos w*.
  5. Registrar retorno del portafolio en el mes t.
  6. Calcular dividendo efectivamente pagado y comparar vs proyectado.
```

```python
# src/05_backtest.py — esquema
def run_backtest(start='2016-01', end=None) -> pd.DataFrame:
    """Devuelve DataFrame con columnas: date, weights, port_return, div_yield_actual."""
    ...
```

---

## 4. Fase 3 — Métricas y evaluación

Todas las métricas en `src/06_metrics.py`:

| Métrica | Fórmula / método |
|---|---|
| **Sharpe Ratio** | `(R̄ - Rf) / σ_port` · √252; Rf = T-Bill 3m |
| **Sortino Ratio** | Como Sharpe pero σ solo de retornos negativos |
| **Max Drawdown** | Caída peak-to-trough máxima en el período |
| **Calmar Ratio** | `CAGR / Max Drawdown` |
| **VaR (95%, 99%)** | Histórico y paramétrico normal |
| **CVaR / ES** | Pérdida esperada más allá del VaR |
| **Upside Partial Moment** | `E[max(R - τ, 0)^n]` con τ = benchmark (S&P 500) |
| **Pain-Gain Ratio** | `Pain Index / CAGR` |
| **Beta vs S&P 500** | Regresión OLS |
| **Alpha (Jensen's α)** | Intercepto del CAPM |
| **Dividend Yield realizado** | Dividendos cobrados / valor portafolio |
| **Dividend forecast accuracy** | RMSE entre yield proyectado y pagado |

Comparar todo contra **benchmark: SPY (S&P 500 ETR)**.

---

## 5. Script de actualización mensual (`07_update.py`)

Este es el script que se usará **en producción** (o en el rol real en Actinver). Debe correr en ≤ 5 minutos:

```
python src/07_update.py --month 2025-06
```

Pasos internos:
1. Descargar datos nuevos de `yfinance` para el mes indicado.
2. Recalcular universo.
3. Verificar señales stop-loss / take-profit.
4. Correr optimizador → nuevos pesos.
5. Append a `portfolio_history.csv`.
6. Recalcular todas las métricas year-to-date.
7. Imprimir resumen en consola + exportar `metrics_summary.csv` actualizado.

---

## 6. Validación cruzada del pipeline

**Walk-forward validation:**

```
Entrenar: 2013-2015 (36 meses)
Test:      2016-2016 (12 meses)
→ desplazar 12 meses
Entrenar: 2013-2016
Test:      2017
→ ... hasta el presente
```

Métricas a reportar por fold:
- MSE de retornos esperados vs realizados.
- Accuracy del yield proyectado (MAE en puntos base).
- Sharpe Ratio en período test.

> **Frecuencia de ejecución:** El CV de selección de μ se ejecuta en **cada rebalanceo mensual** (no anualmente). `compute_features(mu_method=None)` lo dispara automáticamente con los datos hasta `as_of_date`, eliminando el lag que introduciría una selección estática o actualizada una vez al año.

---

## 7. Entregables y cronograma sugerido (7 días)

| Día | Tarea |
|---|---|
| 1 | Configurar repo, instalar dependencias, `01_universe.py` funcional, descarga de datos |
| 2 | `02_features.py`: μ, σ, ρ con los tres métodos + validación cruzada setup |
| 3 | `03_optimizer.py` + `04_signals.py` — optimizador QP y señales |
| 4 | `05_backtest.py` — motor completo, primera corrida histórica 2016-2025 |
| 5 | `06_metrics.py` — todas las métricas, tablas y gráficas |
| 6 | `Informe.docx` — metodología, resultados, validación |
| 7 | `Presentacion.pptx` — slide deck para equipo de inversión + clientes |

---

## 8. Dependencias (`requirements.txt`)

```
yfinance>=0.2.40
pandas>=2.0
numpy>=1.26
quadprog>=0.1.11          # qp.solve_qp
arch>=6.3                 # GARCH
scikit-learn>=1.4         # LedoitWolf, cross-validation
matplotlib>=3.8
seaborn>=0.13
python-docx>=1.1          # informe
python-pptx>=0.6.23       # presentación
openpyxl>=3.1
pyarrow>=15.0             # parquet
```

---

## 9. Puntos críticos y decisiones de diseño

| Decisión | Elección | Justificación |
|---|---|---|
| Fuente de datos | `yfinance` | Libre, amplia cobertura, fácil actualización mensual |
| Estimación de μ | EWMA + shrinkage, seleccionado por CV | Robusto a outliers, bien documentado en literatura |
| Estimación de Σ | Ledoit-Wolf | Evita matrices singulares, estándar en gestión de portafolios |
| Optimización | `quadprog.solve_qp` | Requerimiento explícito del caso |
| Rebalanceo | Mensual, primer día hábil | Requerimiento explícito + equilibrio costo/beneficio |
| Benchmark | SPY | Referencia estándar del mercado americano |
| Stop-loss/TP | 1σ individual por acción | Requerimiento explícito; σ rolling 252d |
| Lookback inicial | 36 meses (ene 2013) | Cumple el requisito "al menos 3 años antes de ene 2016" |
| Frecuencia del CV de μ | Mensual (cada rebalanceo) | El régimen de mercado puede cambiar mes a mes; recalcular el CV con cada rebalanceo elimina el lag que introduciría una actualización anual, y el costo computacional es aceptable dado que el rebalanceo ya es mensual |
| Estimación de GARCH | Una sola vez sobre el historial completo | Trade-off explícito de costo computacional vs precisión: los parámetros α, β, ω de GARCH son más estables entre meses que μ, y su recómputo mensual incrementaría significativamente el tiempo de cada rebalanceo sin justificación en el modelo. Limitación aceptada y documentada: los parámetros se estiman con datos futuros respecto al inicio del backtest. |
| Filtro yield en screening | ≥ 0.5% por acción | El 3% aplica al portafolio agregado (`Σwᵢyᵢ ≥ 3%`), no por acción; filtrar al 3% individual reduciría el universo a ~13 acciones, haciendo infactible la restricción de 5% máximo por posición |
| Restricción de suma de pesos | `Σwᵢ ≤ 1` | El requerimiento garantiza yield de dividendos ≥ 3%, no plena inversión del capital; `≤ 1` permite mantener cash cuando el mercado no ofrece condiciones para cumplir el yield dentro del cap de 5% |
| Denominador del yield | Capital invertido en acciones (`Σwᵢ`) | El cliente pide rendimiento de dividendos, no retorno total; cash no genera dividendos y no debe diluir la métrica de cumplimiento |

---

## 10. Narrativa para la presentación

La presentación debe contar esta historia en ese orden:

1. **El problema del cliente** — necesita yield ≥ 3% con riesgo controlado.
2. **Nuestra solución** — universo filtrado + optimización cuadrática.
3. **Cómo estimamos el futuro** — μ, σ, ρ con validación cruzada.
4. **Resultados del backtesting** — tabla de métricas vs SPY.
5. **Gestión de riesgo** — stop-loss, take-profit, VaR.
6. **¿Realmente pagó los dividendos?** — validación vs dividendos históricos.
7. **Cómo funciona la actualización mensual** — proceso simple, reproducible.
8. **Próximos pasos** — posibles mejoras (factores ESG, ML en predicción de μ).
