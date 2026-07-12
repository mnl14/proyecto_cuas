"""
Genera espectrogramas (STFT) a partir de los segmentos RF indexados.
Guarda los espectrogramas como arrays .npy y una muestra visual .png de control.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import stft
import matplotlib.pyplot as plt

SPLITS_DIR = Path("data/splits")
OUT_DIR = Path("data/processed/spectrograms")

N_FFT = 1024
HOP = 512
SAMPLE_RATE = 40_000_000  # 40 MHz, según especificación del receptor DroneRF


def load_signal(path_h: str, path_l: str) -> np.ndarray:
    with open(path_h) as f:
        h = np.fromstring(f.readline(), sep=",", dtype=np.float32)
    with open(path_l) as f:
        l = np.fromstring(f.readline(), sep=",", dtype=np.float32)
    return np.concatenate([l, h])


def compute_spectrogram(signal: np.ndarray) -> np.ndarray:
    _, _, Zxx = stft(signal, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT - HOP)
    return np.abs(Zxx).astype(np.float32)


def process_split(split_name: str):
    df = pd.read_csv(SPLITS_DIR / f"{split_name}.csv")
    out_dir = OUT_DIR / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, row in df.iterrows():
        signal = load_signal(row["path_h"], row["path_l"])
        spec = compute_spectrogram(signal)
        out_path = out_dir / f"{row['drone']}_{row['mode']}_{row['bui']}_{row['segment_num']}.npy"
        np.save(out_path, spec)
        print(f"[{split_name}] {i+1}/{len(df)} -> {out_path.name} shape={spec.shape}")

    return df


def save_sample_plot(split_name: str = "train"):
    """Guarda un espectrograma de ejemplo como PNG para inspección visual."""
    df = pd.read_csv(SPLITS_DIR / f"{split_name}.csv")
    row = df.iloc[0]
    signal = load_signal(row["path_h"], row["path_l"])
    spec = compute_spectrogram(signal)

    plt.figure(figsize=(10, 5))
    plt.imshow(20 * np.log10(spec + 1e-10), aspect="auto", origin="lower", cmap="viridis")
    plt.title(f"Espectrograma ejemplo: {row['drone']} - {row['mode']}")
    plt.xlabel("Tiempo (frames)")
    plt.ylabel("Frecuencia (bins)")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.savefig("results/figures/spectrogram_sample.png", dpi=150)
    print("\nMuestra visual guardada en results/figures/spectrogram_sample.png")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        process_split(split)

    save_sample_plot("train")