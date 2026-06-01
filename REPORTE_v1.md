# AUDITORÍA TÉCNICA — PIPELINE ACTINVER INVESTOR US EQUITIES

**Módulos analizados:** `01_universe.py`, `02_features.py`, `03_optimizer.py`, `04_signals.py`, `05_backtest.py` + `BLUEPRINT.md`
**Fecha:** 2026-05-30
**Auditor:** Diagnóstico técnico senior (revisor de código · científico de datos cuantitativo · investment banker)

---

## SOMBRERO 1 — REVISOR DE CÓDIGO Y ARQUITECTURA

---

### A. Alineación con el requerimiento

**Restricción de yield 3%:** El BLUEPRINT decide correctamente que aplica a nivel de portafolio. `03_optimizer.py` la implementa como `Σwᵢ(yᵢ − min_yield) ≥ 0` — la linealización correcta de `Σwᵢyᵢ/Σwᵢ ≥ 3%`. ✓

**Cap 5% por posición:** `MAX_WEIGHT = 0.05` en `03_optimizer.py:34` y `04_signals.py:34`. ✓

**Stop-loss / take-profit a 1σ:** `SL_MULTIPLE = 1.0`, `TP_MULTIPLE = 1.0` en `04_signals.py:32-33`. ✓

**Rebalanceo mensual:** Implementado en `05_backtest.py`. ✓

**Backtesting desde ene 2016, lookback 3 años:** `BACKTEST_START = "2016-01-01"`, `LOOKBACK_MONTHS = 36`, datos desde `2013-01-01`. ✓

**`quadprog.solve_qp`:** Utilizado en `03_optimizer.py:123`. ✓

**HALLAZGO — Docstring incorrecto en `01_universe.py:8`:**
El encabezado del módulo dice:
```
  2. Dividend yield TTM >= 3%
```
Pero la implementación usa `min_yield: float = 0.005` (0.5%), que es lo correcto conforme al BLUEPRINT §2.1. El docstring contradice el código y puede inducir a error a cualquier reviewer. **→ Menor.**

---

### B. Consistencia con el BLUEPRINT

**Denominador del yield:** BLUEPRINT §3.2 especifica `Σwᵢyᵢ/Σwᵢ ≥ 0.03`. El código usa la linealización `Σwᵢ(yᵢ − min_yield) ≥ 0`, válida cuando `Σwᵢ > 0`. Correctamente implementada. ✓

**Filtro de universo 0.5%:** Correcto. ✓

**EWMA λ — CRÍTICO:** BLUEPRINT §3.1 especifica λ = 0.94 (alpha = 0.06). El código en `02_features.py:31`:
```python
EWMA_LAMBDA: float = 0.70   # RiskMetrics decay; pandas alpha = 1 − λ = 0.30
```
Con λ = 0.70, la vida media efectiva es `ln(0.5)/ln(0.70) ≈ 1.94 meses`. La EWMA ignora casi todo el historial de 36 meses y se comporta como un promedio de ~2-3 meses. Incluso el docstring de `_mu_ewma` reconoce el valor correcto: "RiskMetrics λ=0.94 (alpha=0.06)" — la constante fue cambiada sin actualizar el docstring. Es un bug de copia-pega que produce estimadores de μ extremadamente inestables y ruidosos. **→ Crítico.**

**GARCH precomputado:** Implementado conforme a BLUEPRINT. ✓

**σ para señales — discrepancia moderada:** BLUEPRINT §3.3 y §9 especifican explícitamente "σ rolling 252d" para stop-loss/take-profit. El código usa GARCH(1,1) condicional. GARCH es metodológicamente superior pero diverge de la especificación literal del documento, lo que puede causar problemas al justificar la metodología ante el cliente o en la presentación. **→ Moderado.**

---

### C. Contratos entre módulos

**`compute_features()` → `05_backtest.py`:**
`FeatureSet` devuelve `(mu, cov, sigma_garch, mu_method, as_of_date)`. El backtester accede `features.mu`, `features.cov` — compatible. Sin embargo, `features.sigma_garch` es devuelto pero **nunca usado** en el bucle del backtester:

```python
# 05_backtest.py:305-306 — sigma_t se calcula por separado, duplicando trabajo
sigma_t = garch_vols.loc[
    garch_vols.index[garch_vols.index <= t][-1]
]
# features.sigma_garch nunca se referencia en el loop
```
El backtester calcula `sigma_t` directamente de `garch_vols` sin usar el `sigma_garch` del FeatureSet. Redundancia inofensiva pero confusa. **→ Menor.**

**`optimize_portfolio()` → `05_backtest.py`:**
Firma del optimizador completamente compatible con la llamada en `05_backtest.py:336-345`. ✓

**`apply_signals()` → `05_backtest.py`:**
Firma y uso de `SignalResult.stop_loss_tickers`, `SignalResult.take_profit_tickers` — compatible. El backtester ignora `adjusted_weights` (correcto: el optimizador rehace los pesos desde cero). ✓

**`update_position_book()` → `05_backtest.py`:**
Llamada en `05_backtest.py:352` compatible con la firma. ✓

---

### D. Consistencia temporal (BME vs ME)

**`02_features.py:384`:**
```python
prices_m = prices_hist.resample("BME").last()   # Business Month End
```

**`05_backtest.py:225-226`:**
```python
prices_monthly = prices.resample("ME").last()    # Calendar Month End
returns_monthly = prices_monthly.pct_change()
```

**`05_backtest.py:154` (build_historical_yields):**
```python
month_ends = prices.resample("ME").last().index  # Calendar Month End
```

La inconsistencia tiene impacto REAL en los índices: `resample("ME")` indexa por el último día CALENDARIO del mes (e.g., 2016-01-31, que puede ser domingo), mientras que `resample("BME")` indexa por el último día HÁBIL (e.g., 2016-01-29 si el 31 es domingo). En la práctica, como yfinance solo entrega datos de días hábiles, el precio subyacente es idéntico — solo difieren los índices. El impacto en valores del backtest es mínimo, pero genera el bug crítico descrito en la sección E. **→ Moderado a Crítico (ver C1).**

---

### E. Lookahead Bias — Bug Crítico en `curr_px`

**`05_backtest.py:308`:**
```python
curr_px = prices.loc[t] if t in prices.index else prices.iloc[-1]
```

`t` es una fecha ME (último día calendario del mes). `prices.index` contiene solo días hábiles. Cuando `t` cae en fin de semana (e.g., 2016-01-31 es domingo), `t not in prices.index` → el fallback es `prices.iloc[-1]` que devuelve **el precio más reciente de todo el dataset** (2025 o 2026).

**Frecuencia del bug:** Aproximadamente 28-35% de los meses tienen su último día calendario en fin de semana. En un backtest 2016-2026 (120 meses), esto afecta ~35-42 meses.

**Impacto:**
1. **Stop-loss:** `curr_px` (precio 2025) vs `entry_price` (precio 2016-2022) → retorno enorme positivo → stop-loss NUNCA dispara en esos meses → estrategia sin stop-loss real el 30% del tiempo.
2. **Take-profit:** Retorno >>1σ → take-profit dispara para casi TODAS las posiciones en esos meses.
3. **Libro de posiciones:** `update_position_book` asigna `entry_price = precio_2025` para nuevas posiciones abiertas en esos meses → los umbrales de stop-loss de esas posiciones en meses futuros son completamente incorrectos.
4. **Efecto cascada:** Los pesos corruptos de esos meses contaminan retornos y señales de los meses subsecuentes.

**→ Crítico. Rompe el backtest silenciosamente para ~30% de las iteraciones del bucle.**

**Corrección:**
```python
valid_px_idx = prices.index[prices.index <= t]
curr_px = prices.loc[valid_px_idx[-1]] if len(valid_px_idx) > 0 else pd.Series(dtype=float)
```

---

### F. Lógica de señales en el backtester — Orden de operaciones

El BLUEPRINT §3.4 especifica el orden: (1) retorno del mes, (2) señales, (3) optimizar, (4) actualizar libro. El código en `05_backtest.py` sigue el orden: (3d) retorno → (3e) señales → (3g) optimizador → (3h) actualizar. Correcto económicamente. ✓

---

### G. Frecuencia del CV — Divergencia crítica del BLUEPRINT

**`05_backtest.py:44`:**
```python
CV_RETRAIN_FREQ: int = 12   # re-entrenar CV cada 12 meses
```

El BLUEPRINT es explícito en dos lugares:
> *"El CV de selección de μ se ejecuta en cada rebalanceo mensual"* (§3.1)
> *"Frecuencia del CV de μ: Mensual (cada rebalanceo)"* (§9)

El código diverge de esto en **dos niveles**:

1. **`CV_RETRAIN_FREQ = 12`** ejecuta el CV cada 12 meses, no mensualmente.
2. **`compute_features` se llama con `mu_method=mu_method` explícito** (`05_backtest.py:268`), no con `mu_method=None`. Esto hace que el auto-CV de `compute_features` nunca se ejecute en el backtest, independientemente de la frecuencia.

El efecto: si el mercado cambia de régimen (ej: COVID marzo 2020), el estimador de μ podría estar subóptimamente fijado por hasta 12 meses. **→ Crítico (contradice decisión de diseño explícita del BLUEPRINT).**

**Corrección:** Cambiar `CV_RETRAIN_FREQ = 1` y llamar `compute_features(mu_method=None)`, dejando que el CV interno seleccione el método mes a mes.

---

## SOMBRERO 2 — CIENTÍFICO DE DATOS CUANTITATIVO

---

### H. Calidad de los estimadores

**Historical mean:** `returns_window.mean() * 12`. Correcto. ✓

**EWMA:** Ya cubierto en B — lambda incorrecto (0.70 vs 0.94). **→ Crítico (ver C2).**

**James-Stein:** La fórmula implementada en `02_features.py:139-165` es:
```python
sigma2_avg = float((returns_window.var(ddof=1) / T).mean())
B = min(1.0, max(0.0, (n - 2) * sigma2_avg / ss))
mu_js = mu_grand + (1.0 - B) * d
```
Implementa correctamente el *positive-part James-Stein estimator* para medias muestrales con distribución N(μ, σ²/T). `sigma2_avg = var/T` es la varianza de la media muestral y `ss = Σ(μᵢ - μ̄)²` es la norma cuadrada de las desviaciones. La aproximación de usar σ̄² única (homoscedástica) es estándar. ✓ El acotamiento `B ∈ [0,1]` es correcto (positive-part). ✓

**Dirección del EWMA:** `ewm(alpha=1-lam, adjust=True).mean().iloc[-1]` — más peso a datos recientes con alpha mayor. Dirección correcta. ✓

---

### I. Matriz de covarianza

**Ledoit-Wolf sobre retornos mensuales, anualización ×12:** Correcto para varianza (σ²_anual = σ²_mensual × 12 bajo i.i.d.). ✓

**Definitud positiva:** Ledoit-Wolf garantiza definitud positiva. El nugget `ε·I` con `NUGGET = 1e-8` en `03_optimizer.py:204` añade seguridad adicional. Sin embargo, con covarianzas anualizadas del orden O(0.01-0.10), un nugget de 1e-8 es de hecho insignificante si la matriz Ledoit-Wolf tiene eigenvalores cercanos a cero. En la práctica con universos de 20-60 acciones y 36 meses de datos el condicionamiento es aceptable. **→ Menor.**

**`dropna(axis=1, how="any")` en `compute_cov_matrix`:** Si algún ticker tiene UN valor NaN en la ventana de 36 meses, se excluye completamente de la covarianza. El `intersection(cov.index)` en `compute_features:418` alinea correctamente los outputs. ✓

---

### J. GARCH y su uso

**Lookahead en parámetros:** Documentado en BLUEPRINT §9 y en código (`02_features.py:224-228`). Limitación aceptada. ✓

**Fallback EWMA — inconsistencia de ventana:**
GARCH se estima sobre TODO el historial desde 2013 (~3000 días).
El fallback en `precompute_garch_vols:263` usa `ewm(span=30)` — solo 30 días de memoria efectiva. Esta asimetría produce estimadores de vol de mucho peor calidad para los tickers con GARCH fallido. Debe usarse al menos `span=252`. **→ Moderado.**

**Anualización:** `(conditional_volatility / 100) * np.sqrt(252)`. Correcto. ✓

**GARCH vols en el optimizador vs. señales:** El optimizador usa la covarianza Ledoit-Wolf, NO el GARCH σ directamente. El GARCH solo alimenta las señales (stop-loss/take-profit). Diseño coherente con el BLUEPRINT, aunque significa que la covarianza del optimizador no captura clustering de volatilidad. Compromiso de diseño aceptable. **→ Menor.**

---

### K. Walk-forward CV

**Solapamiento de folds:** `start += test_months` avanza de 12 en 12. Folds de entrenamiento se solapan (rolling window). Los períodos de TEST son no solapados: `[36:48], [48:60], [60:72]...` Sin lookahead dentro del CV. ✓

**Suficiencia de folds:** Con `train_months=36, test_months=12`, el primer fold requiere 48 meses de historia. En el primer rebalanceo del backtest (enero 2016, ~36 meses de datos disponibles), la condición `len(ret_so_far) >= 48` falla y se usa `DEFAULT_MU_METHOD = "james_stein"` hasta ~febrero 2017. En 2026 hay ~11 folds disponibles — estadísticamente razonable. **→ Menor.**

**Métrica MSE:** Es estándar para pronóstico de punto pero subóptima para selección de estimador en contexto de portafolios. Sharpe realizado en el período test o directional accuracy estarían más alineados con el objetivo final. MSE es defensible pero no óptimo. **→ Moderado (mejora metodológica).**

---

### L. Retorno total vs. retorno de precio

`auto_adjust=True` en yfinance retroajusta los precios históricos por dividendos y splits. `pct_change()` sobre series ajustadas = retorno total (precio + dividendo incorporado). Correcto según BLUEPRINT. ✓

**Sin embargo — bug en yields históricos:** `build_historical_yields` en `05_backtest.py:154-155` calcula yields usando `prices.resample("ME").last()` — los precios AUTO-AJUSTADOS:

```python
prices_monthly = prices.resample("ME").last()  # precios ajustados hacia atrás (INCORRECTO para yield)
# ...
px = float(px_series.get(date, np.nan))
tk_yields[date] = ttm_sum / px                 # yield inflado sistemáticamente
```

`auto_adjust=True` reduce los precios históricos cuando se pagan dividendos futuros. Para un stock con ~3% yield anual y 9 años de historia (2013-2022), el precio ajustado de 2016 es aproximadamente 24% menor que el precio real de 2016. Esto **infla los yields históricos en ~30%**.

Consecuencia: el backtest reporta que la restricción de yield ≥3% se cumple con más frecuencia de lo que ocurriría en producción real. La métrica principal de la estrategia está sistemáticamente sesgada hacia arriba. **→ Crítico (sesgo sistemático en la métrica clave de la estrategia).**

**Corrección:** Descargar precios sin ajustar (`auto_adjust=False`) solo para el cálculo del denominador del yield histórico. Los precios ajustados se siguen usando exclusivamente para retornos totales (μ).

---

### M. Survivorship Bias

El universo usa constituents ACTUALES del S&P 500. Para mega-caps (≥$100B), los cambios de membership desde 2016 son mínimos. Las empresas que eran ≥$100B en 2016 y ya no están en S&P 500 en su forma actual son escasas en el segmento dividend payer de alta capitalización. El sesgo existe pero es leve comparado con estrategias mid/small cap. Documentado en el código (`05_backtest.py:18`) y en BLUEPRINT. **→ Menor (documentado y aceptado).**

---

## SOMBRERO 3 — INVESTMENT BANKER / PORTFOLIO MANAGER

---

### N. Viabilidad de la restricción de yield

El pre-check greedy del optimizador maneja correctamente la infactibilidad y el bucle registra `feasible=False` por mes. Sin embargo:

- En el bull run 2019-2021, yields comprimidos (AAPL 0.6%, MSFT 0.9%, GOOGL 0%) reducen el yield promedio ponderado del universo elegible.
- La restricción podría relajarse durante 12-18 meses consecutivos sin alerta explícita al cliente.

El BLUEPRINT afirma "Con `Σwᵢ ≤ 1` el QP siempre tiene solución factible" — esto es **incorrecto**: si el yield máximo de cualquier acción disponible es < 3%, ni concentrando el 100% en ese stock se cumple la restricción. El pre-check del optimizador maneja esto correctamente (fallback a mínima varianza), pero el BLUEPRINT induce a error al afirmar factibilidad garantizada. **→ Moderado.**

---

### O. Calidad crediticia del yield

El modelo filtra solo por yield TTM. No existe filtro de sostenibilidad del dividendo. Casos problemáticos en el universo probable:

- **AT&T (T):** Recortó su dividendo 47% en mayo 2022 post-spin-off de WarnerMedia. El modelo usaría el yield pre-corte para optimizar enero-abril 2022 con un dividendo que luego se reduce a la mitad.
- **MO (Altria):** Alto yield (~6%) pero en declive secular. Payout ratio > 80%.
- **PFE (Pfizer):** Yield 6.6% con pipeline de I+D incierto y desaceleración del crecimiento del dividendo.
- **VZ (Verizon):** Yield 5.8% con deuda elevada y capex intensivo en 5G.

El pipeline no verifica payout ratio, años consecutivos de dividendos estables, ni cobertura por flujo de caja libre. Para un cliente que asume "rendimiento de dividendos garantizado de 3%", ignorar la sostenibilidad del dividendo es un riesgo de presentación significativo. **→ Moderado.**

---

### P. Concentración de riesgo sectorial

El QP tiene cap de 5% por acción pero **ningún límite por sector**. Con la restricción de yield ≥3%, el optimizador gravita naturalmente hacia Utilities, Telecom y Energy — sectores de mayor yield y alta correlación en shocks de tasas. En el ciclo de subida de tasas 2022-2023:

- Utilities: −4.5% en el año
- Telecom (T, VZ): −20% a −25%
- REITs: −25%

Un portafolio concentrado en estos sectores experimenta drawdowns sincronizados precisamente cuando las tasas suben — el escenario de estrés más probable para esta estrategia. **→ Moderado.**

---

### Q. Costo de transacción y rotación

El backtest es retorno bruto sin costos de transacción. Con rebalanceo mensual más señales de stop-loss y take-profit, la rotación puede ser sustancial. Para mega-caps con ADV > $50M el impacto de mercado es bajo, pero comisiones + bid-ask spread acumulados podrían reducir el CAGR en 20-50bp. No crítico dado la liquidez del universo, pero debería estimarse en la presentación. **→ Menor.**

---

### R. Comparación con el benchmark

El único benchmark es SPY. Alpha vs. SPY puede reflejar simplemente el factor valor/yield (beta al factor dividendo) y no skill del modelo cuantitativo. La comparación natural para una estrategia dividend equity es contra:
- **SCHD** (Schwab U.S. Dividend Equity): ~3.8% yield histórico, bajo costo, selección por fundamentales
- **VYM** (Vanguard High Dividend Yield): filtro yield > promedio del mercado
- **DVY** (iShares Select Dividend): concentrado en high yield

Sin esta comparación, la pregunta del cliente "¿por qué no simplemente comprar SCHD?" no tiene respuesta cuantificada. **→ Moderado.**

---

### S. Interpretabilidad para el cliente

**Outputs actuales vs. necesarios:**

| Output | Disponible | Necesario |
|--------|-----------|-----------|
| Pesos mensuales | ✓ `weights_df` | ✓ |
| Yield proyectado | ✓ `proj_yield` en MonthlyRecord | ✓ |
| Yield realizado (dividendos cobrados) | ✗ No implementado | ✓ Métrica clave |
| Atribución de retorno (precio vs. dividendo) | ✗ No implementado | Para cliente |
| Razón de exclusión por SL/TP (ticker, fecha, precio) | ✗ Solo count agregado | Para auditoría |
| Método μ seleccionado por mes | ✓ `mu_method` en records | ✓ |

El modelo produce `proj_yield` pero no registra dividendos realmente cobrados. Esto impide calcular la métrica "Dividend forecast accuracy: RMSE entre yield proyectado y pagado" del BLUEPRINT §4. **→ Moderado.**

---

## RESUMEN ORDENADO POR SEVERIDAD

---

### CRÍTICOS — Rompen ejecución, producen resultados incorrectos silenciosamente, o invalidan las conclusiones del backtest

---

**[C1] Bug de lookahead en `curr_px` — `05_backtest.py:308`**

Cuando el último día del mes cae en fin de semana (~30% de los meses), `t` no está en el índice de días hábiles de `prices`, y el fallback usa `prices.iloc[-1]` — el precio más reciente de todo el dataset (2025/2026). Esto hace que los stop-loss casi nunca disparen esos meses (retorno desde entrada aparece enorme y positivo), los take-profit disparen para prácticamente todas las posiciones, y el libro de posiciones asigne `entry_price = precio_futuro`, corrompiendo umbrales de señales en todos los meses subsecuentes.

**Corrección:**
```python
valid_px_idx = prices.index[prices.index <= t]
curr_px = prices.loc[valid_px_idx[-1]] if len(valid_px_idx) > 0 else pd.Series(dtype=float)
```

---

**[C2] `EWMA_LAMBDA = 0.70` vs. BLUEPRINT/docstring que especifican 0.94 — `02_features.py:31`**

Con λ = 0.70 (alpha = 0.30), la vida media efectiva del EWMA sobre retornos mensuales es 1.94 meses. El estimador ignora prácticamente todo el historial de 36 meses y se comporta como un promedio de 2-3 meses: alta varianza, inestable, no es "EWMA" en ningún sentido útil para estimación de retornos esperados. El propio docstring de `_mu_ewma` dice "RiskMetrics λ=0.94 (alpha=0.06)" — la constante fue modificada sin actualizar el docstring ni la lógica.

**Corrección:** `EWMA_LAMBDA: float = 0.94`

---

**[C3] Yields históricos calculados con precios auto-ajustados — sesgo sistemático al alza — `05_backtest.py:154-178`**

`build_historical_yields` usa precios de `auto_adjust=True`. El ajuste retroactivo reduce los precios históricos en acumulado de dividendos. Para un stock con ~3% yield anual y 9 años de historia, el precio ajustado de 2016 es aproximadamente 24% menor que el precio real de 2016, inflando el yield histórico calculado en ~30%.

Consecuencia: el backtest sobreestima sistemáticamente el yield del portafolio y reporta que la restricción ≥3% se cumple con más frecuencia de lo que ocurriría en producción real.

**Corrección:** Descargar precios sin ajustar (`auto_adjust=False`) solo para el denominador del yield histórico. Los precios `auto_adjust=True` se siguen usando exclusivamente para retornos totales (μ).

---

**[C4] Frecuencia del CV contradice el BLUEPRINT — `05_backtest.py:44,268`**

Dos problemas combinados:
1. `CV_RETRAIN_FREQ = 12` — el CV corre anualmente, no mensualmente como especifica el BLUEPRINT.
2. `compute_features(mu_method=mu_method)` — se pasa un método explícito, por lo que el auto-CV interno de `compute_features` (`mu_method=None`) nunca se activa.

BLUEPRINT §3.1 y §9 son explícitos: "CV se ejecuta en cada rebalanceo mensual" porque el régimen de mercado puede cambiar mes a mes. Con CV anual, el modelo puede usar un estimador subóptimo hasta 11 meses después de un cambio de régimen (ej: COVID marzo 2020).

**Corrección:** Cambiar `CV_RETRAIN_FREQ = 1` y llamar `compute_features(mu_method=None)` para que el CV interno seleccione el método en cada rebalanceo.

---

### MODERADOS — Inconsistencias metodológicas o riesgos financieros no controlados

---

**[M1] σ para señales: GARCH vs. rolling 252d — divergencia del requerimiento literal**

`04_signals.py` usa GARCH(1,1) condicional. BLUEPRINT §3.3 y §9 especifican explícitamente "σ rolling 252d" para stop-loss/take-profit. GARCH es metodológicamente superior, pero diverge del documento que se presentará al cliente y al panel de Actinver. Riesgo de inconsistencia entre código y presentación.

**Recomendación:** Elegir explícitamente una de las dos opciones y documentarla con coherencia en BLUEPRINT, código y presentación.

---

**[M2] Fallback EWMA de GARCH usa span=30 días — `02_features.py:263`**

GARCH se estima sobre todo el historial (~3000 días). El fallback para tickers sin convergencia usa `ewm(span=30)` — 30 días de memoria efectiva. Asimetría enorme: tickers con GARCH funcional tienen estimadores de vol robustos; tickers con fallback tienen estimadores de vol altamente ruidosos.

**Corrección:** Usar `ewm(span=252)` como mínimo para el fallback.

---

**[M3] BLUEPRINT afirma factibilidad garantizada del yield — incorrecto**

BLUEPRINT §3.2: "Con `Σwᵢ ≤ 1` el QP siempre tiene solución factible." Esto es incorrecto si el yield máximo disponible en el universo es < 3% (posible en bull markets con yields comprimidos). El optimizador maneja esto correctamente con el pre-check greedy y fallback, pero el BLUEPRINT debe corregirse y el riesgo de infeasibilidad debe reportarse con claridad al cliente.

---

**[M4] Sin restricción sectorial — concentración en sectores rate-sensitive**

El QP con restricción de yield ≥3% y cap de 5% por acción puede concentrar sin límite en Utilities, Telecom y Energy. En 2022, estos tres sectores cayeron en correlación alta con la subida de tasas. Sin constraint sectorial (ej: máx 25% en cualquier sector GICS), el "riesgo controlado" del portafolio es incompleto.

**Recomendación:** Añadir a `_build_constraints()` restricciones `Σ_{i∈sector_s} wᵢ ≤ sector_cap` usando el campo `sectorKey` de yfinance.

---

**[M5] Sin filtro de sostenibilidad del dividendo**

El pipeline usa yield TTM sin verificar payout ratio, dividend growth histórico ni cobertura por flujo de caja libre. AT&T recortó su dividendo 47% en mayo 2022. Para el cliente de Actinver que asume dividendos estables, esto es un riesgo de presentación significativo.

**Recomendación:** Añadir a `01_universe.py` filtros `payout_ratio < 0.80` y `dividend_growth_years >= 5` (al menos 5 años de dividendos no decrecientes).

---

**[M6] `MonthlyRecord` no registra yield realizado — campo clave faltante — `05_backtest.py`**

`MonthlyRecord` tiene `proj_yield` (proyectado) pero no `actual_yield` (dividendos realmente cobrados con los pesos del período anterior). La métrica "Dividend forecast accuracy: RMSE entre yield proyectado y pagado" del BLUEPRINT §4 no puede calcularse sin este campo.

**Corrección:** En el bucle mensual, calcular `actual_yield_t = Σwᵢ_prev × (div_en_t / price_{t-1})` y añadirlo a `MonthlyRecord`.

---

**[M7] Benchmark insuficiente para estrategia dividend equity**

Solo SPY como benchmark. Alpha vs. SPY puede reflejar simplemente el factor yield/valor y no skill del modelo. La comparación que importa es contra ETFs de dividendos pasivos (SCHD, VYM, DVY) que implementan estrategias similares con 0.06% de gastos anuales.

**Recomendación:** Añadir descarga de SCHD y VYM en `05_backtest.py:main()` para la tabla de métricas comparativas.

---

### MENORES — Cosmético, documentación o mejora opcional

---

**[m1] Docstring incorrecto en `01_universe.py:8`**
Dice "Dividend yield TTM >= 3%" pero la función implementa `min_yield=0.005` (0.5%), que es lo correcto. Corregir el docstring del módulo.

**[m2] `features.sigma_garch` devuelto pero no consumido en el backtest loop**
El backtester calcula `sigma_t` directamente de `garch_vols` duplicando la lógica. Simplificar usando `features.sigma_garch` o eliminar el campo del FeatureSet.

**[m3] `ME` vs. `BME` inconsistencia cosmética entre módulos**
`02_features.py` usa `resample("BME")`, `05_backtest.py` usa `resample("ME")`. Impacto práctico mínimo (yfinance solo entrega días hábiles), pero los índices de fechas difieren y pueden confundir al comparar DataFrames entre módulos.

**[m4] `lambda_risk = 0.5` no calibrado**
El parámetro de aversión al riesgo λ en `½w'Σw − λμ'w` está fijo en 0.5 sin justificación ni análisis de sensibilidad. Un λ diferente puede cambiar el balance yield/varianza significativamente.

**[m5] Sin costos de transacción**
Los retornos del backtest son brutos. Para la presentación debería estimarse el impacto (aunque sea 5-10bp por rebalanceo sobre el turnover promedio).

**[m6] Universo estático durante el backtest**
Se carga el universo actual del CSV y se usa para todo el período 2016-2026. El universo real varía mes a mes (empresas que superan o caen por debajo de $100B de capitalización). Parcialmente mitigado por el filtro de mega-caps, pero introduce momentum implícito hacia las empresas que crecieron.

---

## RECOMENDACIONES METODOLÓGICAS ADICIONALES

Las siguientes no son bugs sino mejoras que fortalecerían el pipeline y la tesis de inversión:

**1. Métrica de CV orientada a portafolio**
Reemplazar MSE como métrica de selección del estimador de μ por el **Sharpe realizado en el período test** dado el universo y los pesos que produciría ese estimador. MSE penaliza errores en retornos absolutos pero no en el ranking relativo entre acciones, que es lo que determina la calidad de la asignación de portafolio.

**2. Filtros de calidad del dividendo**
Añadir al screening de `01_universe.py`:
- `payout_ratio < 0.80` (dividendo sostenible con earnings)
- `dividend_growth_years >= 5` (al menos 5 años de dividendos no decrecientes — estándar "Dividend Achiever")

Esto reduciría exposición a MO, T y PFE y mejoraría la credibilidad del yield ante el cliente.

**3. Restricción sectorial en el QP**
Añadir a `_build_constraints()` en `03_optimizer.py` restricciones `Σ_{i∈sector_s} wᵢ ≤ 0.25` por sector GICS. Previene concentración en sectores rate-sensitive que correlacionan negativamente en entornos de subida de tasas.

**4. Expanding window vs. rolling en CV**
El CV actual usa ventanas de entrenamiento fijas de 36 meses (rolling). Una **expanding window** (usa todo el historial disponible hasta cada fold) aprovecha más datos en folds tardíos y es más apropiada para series de tiempo donde el tamaño muestral importa más que la estacionariedad.

**5. Añadir SCHD/VYM como benchmark**
Una única línea adicional en `05_backtest.py:main()` descargando SCHD y VYM. Sin esto, la pregunta del cliente "¿por qué no simplemente comprar SCHD?" no tiene respuesta cuantificada.

**6. Tracking de dividendos reales mes a mes**
En el bucle de backtest, sumar `actual_yield_t = Σwᵢ_prev × (div_en_t / price_{t-1})` para calcular el yield realizado por mes. Este campo es la prueba más directa de que la estrategia cumple la promesa al cliente y es necesario para el informe de validación.

**7. Análisis de sensibilidad de `lambda_risk`**
Correr el backtest con `lambda_risk ∈ {0.1, 0.5, 1.0, 2.0}` y reportar la frontera yield-varianza resultante. Permite al cliente elegir el punto de la frontera eficiente que corresponde a su apetito de riesgo.

**8. Stress testing por régimen de tasas**
Separar el análisis de resultados en tres sub-períodos: (a) tasas bajas 2016-2021, (b) subida de tasas 2022-2023, (c) normalización 2024-2026. Esto permite mostrar la robustez (o no) de la estrategia ante el riesgo de duration que afecta diferencialmente a los sectores de alto yield.

---

## TABLA RESUMEN EJECUTIVO

| ID | Módulo | Descripción | Severidad |
|----|--------|-------------|-----------|
| C1 | `05_backtest.py:308` | `prices.iloc[-1]` como fallback usa precios del futuro para ~30% de los meses | **CRÍTICO** |
| C2 | `02_features.py:31` | `EWMA_LAMBDA=0.70` produce estimador de μ con vida media de ~2 meses (debe ser 0.94) | **CRÍTICO** |
| C3 | `05_backtest.py:154-178` | Yields históricos calculados con precios auto-ajustados → sobreestima yields ~30% | **CRÍTICO** |
| C4 | `05_backtest.py:44,268` | CV cada 12 meses y `mu_method` explícito contradicen el diseño mensual del BLUEPRINT | **CRÍTICO** |
| M1 | `04_signals.py` | GARCH vs. "rolling 252d" especificado en BLUEPRINT §3.3 para señales SL/TP | Moderado |
| M2 | `02_features.py:263` | Fallback EWMA de GARCH usa `span=30` vs. historial completo de GARCH | Moderado |
| M3 | `03_optimizer.py` + BLUEPRINT | BLUEPRINT afirma factibilidad garantizada del yield — incorrecto en yields comprimidos | Moderado |
| M4 | `03_optimizer.py` | Sin restricción sectorial → concentración en Utilities/Telecom correlados en tasas | Moderado |
| M5 | `01_universe.py` | Sin filtro de payout ratio ni dividend growth → exposición a dividendos insostenibles | Moderado |
| M6 | `05_backtest.py` | `MonthlyRecord` no registra yield realizado — métrica clave del BLUEPRINT §4 faltante | Moderado |
| M7 | `05_backtest.py:main()` | Solo SPY como benchmark; falta SCHD/VYM para comparar vs. estrategia pasiva | Moderado |
| m1 | `01_universe.py:8` | Docstring dice "yield ≥ 3%" pero implementa 0.5% (código correcto, doc incorrecto) | Menor |
| m2 | `05_backtest.py` | `features.sigma_garch` calculado pero no consumido en el loop; `sigma_t` se recalcula | Menor |
| m3 | `02_features.py` / `05_backtest.py` | `BME` vs. `ME` inconsistente entre módulos (impacto práctico mínimo) | Menor |
| m4 | `03_optimizer.py:37` | `lambda_risk=0.5` sin justificación ni análisis de sensibilidad | Menor |
| m5 | `05_backtest.py` | Sin costos de transacción — retornos son brutos | Menor |
| m6 | `01_universe.py` | Universo estático durante el backtest, no recalibrado mensualmente | Menor |

**Prioridad de corrección antes de cualquier presentación de resultados:** C3 → C1 → C2 → C4.
Los cuatro críticos deben resolverse antes de que los números del backtest sean citables.
Los moderados M4 y M5 deben abordarse antes de la presentación al cliente por ser riesgos financieros observables que el panel de Actinver puede señalar.
