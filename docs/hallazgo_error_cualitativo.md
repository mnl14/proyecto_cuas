# Análisis de error cualitativo: casos límite AR vs Bebop

## Metodología

Se identificaron los segmentos de test mal clasificados por el Random
Forest, y se compararon sus 12 features contra el promedio de su clase
real y el promedio de la clase con la que el modelo los confundió.

## Resultados

Solo **2 de 35 segmentos de test** (94% accuracy) fueron mal
clasificados, y **ambos pertenecen al mismo BUI** (`10100`, AR Drone,
modo on_connected) — no es un error disperso entre distintas grabaciones,
sino una sesión de captura específica que resultó atípica.

| Feature             | Valor caso | Media AR (real) | Media Bebop (predicho) |
|----------------------|-----------|------------------|--------------------------|
| std / rms           | 348 / 256 | 106              | 495                      |
| kurtosis             | 146 / 194 | 198              | 69                       |
| spectral_centroid    | ~5.5M     | 5.53M            | 5.51M                    |

## Hallazgo

Las **features de energía/amplitud** (std, rms, energy, peak_to_peak) de
esta sesión de AR están muy por encima del rango típico de AR, y más
cerca del rango típico de Bebop — esto es lo que arrastra la predicción
hacia la clase incorrecta.

Sin embargo, la **kurtosis** (forma de la distribución de la señal) sí es
consistente con AR en ambos casos, no con Bebop. Es decir, la "forma"
estadística de la señal identifica correctamente el drone, pero el modelo
le da más peso combinado a las features de energía, que en este caso
particular son engañosas.

El **spectral_centroid** resultó casi idéntico entre AR y Bebop en
general, sugiriendo que aporta poco poder discriminativo entre estas dos
clases específicas (aunque puede ser útil para otras distinciones, como
separar de Phantom o background).

## Interpretación

La variabilidad de energía **entre sesiones de grabación de la misma
clase** (ej. distancia dron-receptor, ganancia del receptor en esa
captura puntual) puede ser mayor que la diferencia de energía **entre
clases distintas**. Esto es una limitación conocida de depender de
features de amplitud/energía absoluta en señales RF: no son invariantes
a la distancia o configuración de captura.

## Implicancia para el sistema C-UAS

Un sistema real debería considerar:
- Normalizar features de energía por alguna referencia de potencia total
  recibida, para reducir sensibilidad a la distancia/ganancia
- Dar más peso relativo a features de forma (kurtosis, skewness) que a
  features de magnitud absoluta, ya que las primeras parecen más robustas
  entre sesiones de captura distintas
- Recolectar más sesiones de grabación por clase (no solo más segmentos
  de la misma sesión) para capturar mejor esta variabilidad inter-sesión
