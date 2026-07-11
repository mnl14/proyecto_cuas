# Comparación de modelos: Baseline (RF/SVM) vs CNN

## Contexto

Clasificación de señales RF de drones (DroneRF dataset) en 4 clases:
AR Drone, Bebop Drone, Phantom Drone, Background (sin drone).

- Dataset: 227 segmentos (158 train / 34 val / 35 test), split estratificado 70/15/15
- Baseline: Random Forest y SVM sobre 12 features estadísticas (dominio tiempo + frecuencia)
- CNN: 3 capas convolucionales sobre espectrograma STFT, redimensionado a 128x128

## Resultados (test set)

| Modelo         | F1-macro | Accuracy |
|----------------|----------|----------|
| Random Forest  | 0.96     | 0.94     |
| SVM            | 0.95     | 0.94     |
| CNN            | 0.84     | 0.77     |

Detalle por clase — recall más bajo en ambos modelos ocurre en `ar`/`bebop`:

| Clase      | RF recall (test) | CNN recall (test) |
|------------|-------------------|--------------------|
| ar         | 0.83              | 0.83               |
| bebop      | 1.00              | 0.54               |
| background | 1.00              | 1.00               |
| phantom    | 1.00              | 1.00               |

## Hallazgo

El baseline (Random Forest) superó a la CNN por un margen amplio (F1-macro 0.96 vs 0.84).

**Causas probables:**

1. **Tamaño de dataset insuficiente para deep learning.** 158 muestras de
   entrenamiento no alcanzan para que una CNN aprenda representaciones
   convolucionales robustas desde cero, mientras que RF/SVM generalizan
   mejor con pocos datos cuando las features de entrada ya están bien
   diseñadas manualmente.

2. **Pérdida de resolución temporal en el resize.** El espectrograma original
   (513 x ~39000 frames) se redujo a 128x128 para que fuera viable
   computacionalmente. Esa compresión descarta gran parte del detalle
   temporal fino, justo donde probablemente están las diferencias sutiles
   entre AR y Bebop.

3. **Ambos modelos comparten el mismo patrón de error** (confusión AR↔Bebop),
   lo que sugiere una similitud real en la firma RF de esos dos drones,
   agravada en la CNN por la pérdida de información del punto 2.

## Conclusión

Para este dataset y este tamaño de muestra, el enfoque de features
estadísticas + modelo clásico (Random Forest) es superior a la CNN. Esto es
consistente con la literatura: deep learning no siempre supera a modelos
clásicos cuando el dataset es chico, y aquí el diseño manual de features
capturó mejor la información relevante que la arquitectura convolucional
bajo las restricciones de cómputo actuales.

## Próximos pasos posibles (no ejecutados aún)

- Aumentar resolución del resize (ej. 256x256) si el hardware lo permite
- Data augmentation sobre espectrogramas para compensar el dataset chico
- Segmentar cada señal larga en sub-ventanas más cortas para generar más
  muestras de entrenamiento por segmento original
