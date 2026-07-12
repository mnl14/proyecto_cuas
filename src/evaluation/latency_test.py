import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from scipy.stats import skew, kurtosis
from scipy.signal import stft
from sklearn.preprocessing import LabelEncoder
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.cnn.train_cnn import DroneCNN, resize_spectrogram, CLASSES

SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("results/models")
METRICS_DIR = Path("results/metrics")

N_FFT = 1024
HOP = 512
SAMPLE_RATE = 40_000_000

FEATURE_COLS = [
    "mean", "std", "var", "skewness", "kurtosis", "rms",
    "peak_to_peak", "mean_abs", "max_abs", "energy",
    "spectral_centroid", "spectral_energy",
]


def load_signal(path_h: str, path_l: str) -> np.ndarray:
    with open(path_h) as f:
        h = np.fromstring(f.readline(), sep=",", dtype=np.float32)
    with open(path_l) as f:
        l = np.fromstring(f.readline(), sep=",", dtype=np.float32)
    return np.concatenate([l, h])


def extract_features(signal: np.ndarray) -> dict:
    abs_signal = np.abs(signal)
    fft_mag = np.abs(np.fft.rfft(signal))
    return {
        "mean": float(np.mean(signal)), "std": float(np.std(signal)),
        "var": float(np.var(signal)), "skewness": float(skew(signal)),
        "kurtosis": float(kurtosis(signal)), "rms": float(np.sqrt(np.mean(signal ** 2))),
        "peak_to_peak": float(np.ptp(signal)), "mean_abs": float(np.mean(abs_signal)),
        "max_abs": float(np.max(abs_signal)), "energy": float(np.sum(signal.astype(np.float64) ** 2)),
        "spectral_centroid": float(np.sum(np.arange(len(fft_mag)) * fft_mag) / (np.sum(fft_mag) + 1e-10)),
        "spectral_energy": float(np.sum(fft_mag ** 2)),
    }


def compute_spectrogram(signal: np.ndarray) -> np.ndarray:
    _, _, Zxx = stft(signal, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT - HOP)
    return np.abs(Zxx).astype(np.float32)


if __name__ == "__main__":
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    rf = joblib.load(MODELS_DIR / "random_forest.joblib")
    svm = joblib.load(MODELS_DIR / "svm.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_encoder = LabelEncoder()
    label_encoder.fit(CLASSES)
    cnn = DroneCNN(n_classes=len(CLASSES)).to(device)
    cnn.load_state_dict(torch.load(MODELS_DIR / "cnn_best.pt", map_location=device))
    cnn.eval()

    # Precargar todas las señales en memoria (simula señal ya adquirida)
    print("Cargando señales de test en memoria...")
    signals = []
    for i, row in test_df.iterrows():
        signals.append(load_signal(row["path_h"], row["path_l"]))
        print(f"  cargada {i+1}/{len(test_df)}")

    times = {
        "feature_extraction": [],
        "random_forest_predict": [],
        "svm_predict": [],
        "spectrogram_extraction": [],
        "cnn_predict": [],
    }

    for idx, signal in enumerate(signals):
        # --- Extracción de features (compartida por RF y SVM) ---
        t0 = time.perf_counter()
        feats = extract_features(signal)
        X = np.array([[feats[c] for c in FEATURE_COLS]])
        times["feature_extraction"].append(time.perf_counter() - t0)

        # --- Predicción Random Forest (features ya calculadas) ---
        t0 = time.perf_counter()
        rf.predict(X)
        times["random_forest_predict"].append(time.perf_counter() - t0)

        # --- Predicción SVM (features ya calculadas) ---
        t0 = time.perf_counter()
        X_s = scaler.transform(X)
        svm.predict(X_s)
        times["svm_predict"].append(time.perf_counter() - t0)

        # --- Extracción de espectrograma (STFT + resize) ---
        t0 = time.perf_counter()
        spec = compute_spectrogram(signal)
        spec_resized = resize_spectrogram(spec)
        times["spectrogram_extraction"].append(time.perf_counter() - t0)

        # --- Predicción CNN (espectrograma ya calculado) ---
        t0 = time.perf_counter()
        x_tensor = torch.from_numpy(spec_resized).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            cnn(x_tensor).argmax(dim=1)
        times["cnn_predict"].append(time.perf_counter() - t0)

        print(f"  procesada {idx+1}/{len(signals)}")

    rows = []
    for stage_name, ts in times.items():
        ts = np.array(ts)
        rows.append({
            "stage": stage_name,
            "mean_ms": ts.mean() * 1000,
            "std_ms": ts.std() * 1000,
            "min_ms": ts.min() * 1000,
            "max_ms": ts.max() * 1000,
        })
        print(f"{stage_name}: {ts.mean()*1000:.2f} ms/muestra (±{ts.std()*1000:.2f})")

    # Totales de pipeline completo (extracción + predicción) por modelo
    rf_total = np.array(times["feature_extraction"]) + np.array(times["random_forest_predict"])
    svm_total = np.array(times["feature_extraction"]) + np.array(times["svm_predict"])
    cnn_total = np.array(times["spectrogram_extraction"]) + np.array(times["cnn_predict"])

    for model_name, ts in [("TOTAL_random_forest", rf_total), ("TOTAL_svm", svm_total), ("TOTAL_cnn", cnn_total)]:
        rows.append({
            "stage": model_name,
            "mean_ms": ts.mean() * 1000,
            "std_ms": ts.std() * 1000,
            "min_ms": ts.min() * 1000,
            "max_ms": ts.max() * 1000,
        })
        print(f"{model_name}: {ts.mean()*1000:.2f} ms/muestra (±{ts.std()*1000:.2f})")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(METRICS_DIR / "latency_report.csv", index=False)
    print("\nGuardado en results/metrics/latency_report.csv")