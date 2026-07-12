"""
Test de robustez: inyecta ruido gaussiano sintético a distintos niveles de SNR
sobre las señales de test, y mide cómo degrada el desempeño de cada modelo.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from scipy.stats import skew, kurtosis
from scipy.signal import stft
from scipy.ndimage import zoom
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.cnn.train_cnn import DroneCNN, resize_spectrogram, TARGET_SIZE, CLASSES

SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("results/models")
METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")

SNR_LEVELS_DB = [-10, -5, 0, 5, 10, 15, 20, None]  # None = señal limpia
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


def add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    if snr_db is None:
        return signal
    signal_power = np.mean(signal.astype(np.float64) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape).astype(np.float32)
    return signal + noise


def extract_features(signal: np.ndarray) -> dict:
    abs_signal = np.abs(signal)
    fft_mag = np.abs(np.fft.rfft(signal))
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "var": float(np.var(signal)),
        "skewness": float(skew(signal)),
        "kurtosis": float(kurtosis(signal)),
        "rms": float(np.sqrt(np.mean(signal ** 2))),
        "peak_to_peak": float(np.ptp(signal)),
        "mean_abs": float(np.mean(abs_signal)),
        "max_abs": float(np.max(abs_signal)),
        "energy": float(np.sum(signal.astype(np.float64) ** 2)),
        "spectral_centroid": float(
            np.sum(np.arange(len(fft_mag)) * fft_mag) / (np.sum(fft_mag) + 1e-10)
        ),
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

    results = []

    for snr in SNR_LEVELS_DB:
        y_true = []
        rf_preds, svm_preds, cnn_preds = [], [], []

        for i, row in test_df.iterrows():
            signal = load_signal(row["path_h"], row["path_l"])
            noisy = add_noise(signal, snr)
            y_true.append(row["drone"])

            feats = extract_features(noisy)
            X = np.array([[feats[c] for c in FEATURE_COLS]])
            rf_preds.append(rf.predict(X)[0])
            svm_preds.append(svm.predict(scaler.transform(X))[0])

            spec = compute_spectrogram(noisy)
            spec_resized = resize_spectrogram(spec)
            x_tensor = torch.from_numpy(spec_resized).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_idx = cnn(x_tensor).argmax(dim=1).item()
            cnn_preds.append(label_encoder.inverse_transform([pred_idx])[0])

            print(f"SNR={snr} | {i+1}/{len(test_df)} procesado")

        snr_label = "clean" if snr is None else snr
        for model_name, preds in [("random_forest", rf_preds), ("svm", svm_preds), ("cnn", cnn_preds)]:
            acc = accuracy_score(y_true, preds)
            f1 = f1_score(y_true, preds, average="macro", zero_division=0)
            results.append({"snr_db": snr_label, "model": model_name, "accuracy": acc, "f1_macro": f1})
            print(f"  {model_name}: acc={acc:.3f} f1={f1:.3f}")

    results_df = pd.DataFrame(results)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(METRICS_DIR / "robustness_snr.csv", index=False)

    # Plot
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for model_name in results_df["model"].unique():
        sub = results_df[results_df["model"] == model_name]
        # ordenar por snr, dejando "clean" al final
        sub_numeric = sub[sub["snr_db"] != "clean"].sort_values("snr_db")
        sub_clean = sub[sub["snr_db"] == "clean"]
        x = list(sub_numeric["snr_db"]) + (["clean"] if len(sub_clean) else [])
        y = list(sub_numeric["f1_macro"]) + (list(sub_clean["f1_macro"]) if len(sub_clean) else [])
        plt.plot(range(len(x)), y, marker="o", label=model_name)
    plt.xticks(range(len(x)), x)
    plt.xlabel("SNR (dB)")
    plt.ylabel("F1-macro")
    plt.title("Robustez ante ruido: F1-macro vs SNR")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "robustness_snr.png", dpi=150)

    print("\nResultados guardados en results/metrics/robustness_snr.csv")
    print("Gráfico guardado en results/figures/robustness_snr.png")