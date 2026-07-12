# SPRT simplificado: decisión secuencial con umbral de confianza

## Metodología

Sobre el modelo Random Forest (el más preciso y robusto del proyecto), se
evaluó cada señal de test en 10 checkpoints crecientes (10%, 20%, ...,
100% de la señal). En cada checkpoint se calculan features estadísticas
sobre la porción de señal disponible hasta ese punto y se obtiene la
confianza del modelo (`predict_proba`). Se simuló la decisión secuencial
para 6 umbrales de confianza distintos (0.60 a 0.99): el sistema "corta"
la señal y decide apenas la confianza supera el umbral en algún checkpoint;
si nunca lo supera, decide con el 100% de la señal.

## Resultados

| Umbral | Accuracy | F1-macro | Fracción promedio de señal usada |
|--------|----------|----------|-----------------------------------|
| 0.60   | 0.886    | 0.912    | 0.16                               |
| 0.70   | 0.943    | 0.953    | 0.21                               |
| 0.80   | 0.943    | 0.953    | 0.24                               |
| 0.90   | 0.914    | 0.933    | 0.39                               |
| 0.95   | 0.914    | 0.933    | 0.42                               |
| 0.99   | 0.914    | 0.933    | 0.61                               |

(Referencia: con el 100% de la señal, RF alcanza accuracy 0.94 / F1 0.96
en el set de test completo, ver comparación baseline vs CNN.)

## Hallazgo

Con umbral **0.70–0.80**, el sistema iguala prácticamente el desempeño de
usar la señal completa (accuracy 0.943 / F1 0.953 vs 0.94 / 0.96), pero
usando en promedio solo **21–24% de la señal**. Esto implica una
reducción proporcional similar en el tiempo de extracción de features,
que era el cuello de botella identificado en el test de latencia.

**Resultado contraintuitivo:** subir el umbral de confianza por encima de
0.80 no solo aumenta la fracción de señal necesaria (hasta 61% en
umbral 0.99), sino que **empeora la accuracy** (0.914 vs 0.943). La
confianza del modelo no crece de forma monótona a lo largo de la señal:
en algunos segmentos ambiguos, un checkpoint intermedio puede mostrar un
pico de confianza alta pero incorrecta. Con umbral bajo, el sistema ya
había cortado antes con la predicción correcta; con umbral alto, el
sistema atraviesa ese pico erróneo y termina decidiendo mal, o sigue
esperando sin ninguna ganancia real de precisión.

## Conclusión

El punto óptimo de operación es un umbral de confianza moderado
(~0.70–0.80), no el más alto posible. Esto es relevante para el diseño
del sistema real: maximizar el umbral de confianza no maximiza la
precisión ni minimiza la latencia — existe un punto de equilibrio que
debe determinarse empíricamente, como se hizo acá.

## Implicancia para el sistema C-UAS

Un sistema de detección de drones en producción podría operar con
decisión secuencial (SPRT) en vez de esperar siempre la señal completa,
logrando la misma precisión con una fracción del tiempo de procesamiento
— crítico en un escenario de detección en tiempo real donde la latencia
importa tanto como la exactitud.
