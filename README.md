# proyecto_cuas — Detección y clasificación de drones por señal RF

Sistema de clasificación de drones (C-UAS: Counter-Unmanned Aerial
Systems) basado en análisis de señales de radiofrecuencia, usando el
dataset público [DroneRF](https://data.mendeley.com/datasets/f4c2b4n755/1).

## Problema

Detectar y clasificar drones (tipo y modo de vuelo) a partir de su firma
RF, sin depender de cámaras ni radar, evaluando no solo precisión sino
robustez ante ruido y latencia de decisión — condiciones relevantes para
un sistema de detección en tiempo real.

## Dataset

**DroneRF** (Mendeley Data): 227 segmentos de captura RF de 3 drones
(AR Drone, Bebop, Phantom) más actividad de fondo sin drone, en distintos
modos de vuelo (conectado, hovering, flying, flying+video). Cada segmento
combina dos bandas de captura (L/H, 40 MHz cada una).

## Pipeline

```
Dataset raw (CSV) → Indexado + labels (BUI) → Split 70/15/15 estratificado
   → Features estadísticas (12) ──────→ Random Forest / SVM
   → Espectrograma STFT (resize 128x128) → CNN
   → Evaluación (F1, matriz de confusión)
   → Test de robustez (SNR -10 a 20 dB)
   → Test de latencia (features vs predicción)
   → SPRT simplificado (decisión secuencial)
   → Análisis de error cualitativo
```

## Resultados principales

| Modelo         | F1-macro (test) | Accuracy (test) |
|----------------|------------------|-------------------|
| Random Forest  | 0.96             | 0.94              |
| SVM            | 0.95             | 0.94              |
| CNN            | 0.84             | 0.77              |

**Random Forest fue el modelo elegido**: mejor accuracy, mucho más
robusto ante ruido (mantiene F1 ~0.59 incluso a SNR=-10dB, vs ~0.08 de
SVM/CNN en la misma condición), aunque es el más lento en el pipeline
completo por el costo de extracción de features estadísticas.

Con **SPRT simplificado** (decisión secuencial con umbral de confianza
~0.70-0.80), se logra el mismo desempeño que con la señal completa
usando solo ~21-24% de la señal, mitigando el problema de latencia.

Ver documentación detallada de cada hallazgo en `docs/`:
- [`hallazgo_baseline_vs_cnn.md`](docs/hallazgo_baseline_vs_cnn.md) — por qué RF superó a la CNN
- [`hallazgo_latencia.md`](docs/hallazgo_latencia.md) — cuello de botella real del pipeline
- [`hallazgo_sprt.md`](docs/hallazgo_sprt.md) — trade-off accuracy vs tiempo de decisión
- [`hallazgo_error_cualitativo.md`](docs/hallazgo_error_cualitativo.md) — análisis de casos límite AR/Bebop

## Estructura del repositorio

```
proyecto_cuas/
├── data/
│   ├── raw/dronerf/          # dataset original (no versionado, ver abajo)
│   ├── processed/            # espectrogramas, features, índice
│   └── splits/                # train/val/test.csv
├── src/
│   ├── preprocessing/         # carga, split, STFT, features
│   ├── models/
│   │   ├── baseline/           # Random Forest / SVM
│   │   └── cnn/                 # CNN sobre espectrograma
│   └── evaluation/             # robustez, latencia, SPRT, error cualitativo
├── results/
│   ├── models/                 # modelos entrenados (.joblib, .pt)
│   ├── metrics/                # reportes y CSVs de resultados
│   └── figures/                # gráficos generados
└── docs/                       # documentos de hallazgos + este README
```

## Cómo reproducir

```powershell
# 1. Entorno
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Dataset: descargar de Mendeley y extraer en data/raw/dronerf/
#    (los .rar requieren 7-Zip)

# 3. Pipeline completo, en orden:
python src/preprocessing/load_data.py
python src/preprocessing/split_data.py
python src/preprocessing/compute_spectrograms.py
python src/preprocessing/extract_features.py
python src/models/baseline/train_baseline.py
python src/models/cnn/train_cnn.py
python src/evaluation/robustness_test.py
python src/evaluation/latency_test.py
python src/evaluation/sprt_test.py
python src/evaluation/error_analysis.py
```

**Nota:** el dataset (~3 GB) no está incluido en el repositorio. Descargar
desde [Mendeley Data](https://data.mendeley.com/datasets/f4c2b4n755/1) y
extraer en `data/raw/dronerf/`.

## Stack técnico

Python 3.14, scikit-learn (Random Forest, SVM), PyTorch (CNN), scipy
(STFT, features estadísticas), pandas, numpy, matplotlib.
