# Test de latencia: cuello de botella real del pipeline

## Metodología

Se midió el tiempo de procesamiento por muestra (35 segmentos de test),
con la señal ya cargada en memoria (simula señal ya adquirida por el
receptor, excluyendo I/O de disco). Se separó cada etapa del pipeline:
extracción de features/espectrograma vs. predicción del modelo.

## Resultados (ms por muestra, media ± desvío estándar)

| Etapa                          | Tiempo medio (ms) | Desvío (ms) |
|---------------------------------|-------------------|-------------|
| Extracción de features (RF/SVM) | 26359             | 27605       |
| Extracción de espectrograma (CNN)| 2508             | 5618        |
| Predicción Random Forest        | 492               | 2303        |
| Predicción SVM                  | 87                | 460         |
| Predicción CNN                  | 543               | 2428        |

**Totales de pipeline completo (extracción + predicción):**

| Modelo         | Total (ms/muestra) |
|----------------|---------------------|
| Random Forest  | 26852               |
| SVM            | 26446               |
| CNN            | 3051                |

## Hallazgo

El cuello de botella de latencia **no es el modelo, es la extracción de
features estadísticas** que alimenta a RF/SVM. Esa etapa (~26s) es ~10x
más lenta que la extracción del espectrograma STFT (~2.5s) que usa la CNN,
probablemente porque las funciones `skew()` y `kurtosis()` de scipy no
están optimizadas para arrays de 20 millones de puntos, mientras que el
STFT se apoya en FFT, algorítmicamente mucho más eficiente.

Esto genera una paradoja: el modelo con mejor accuracy y mejor robustez a
ruido (Random Forest, ver `robustness_snr.png`) es también el más lento
en el pipeline end-to-end, no por el modelo en sí (492 ms) sino por el
preprocesamiento previo (26359 ms). La CNN, con peor accuracy, es ~9x más
rápida en total porque su extracción de features (STFT) es mucho más
eficiente.

## Implicancia práctica

Si la latencia es un requisito crítico del sistema (detección en tiempo
real), hay margen de optimización considerable en la extracción de
features estadísticas, sin necesidad de cambiar de modelo:

- Reemplazar `scipy.stats.skew`/`kurtosis` por cálculos vectorizados
  manuales (fórmulas de momentos estadísticos con numpy puro)
- Reducir la cantidad de features calculadas, evaluando cuáles aportan
  menos poder discriminativo
- Evaluar cálculo de features sobre una submuestra de la señal en vez de
  los 20M puntos completos

## Conclusión

La elección de modelo (RF vs CNN) no debe basarse solo en accuracy o
robustez — la latencia del pipeline completo depende fuertemente de la
etapa de preprocesamiento, que puede optimizarse independientemente del
modelo elegido.
