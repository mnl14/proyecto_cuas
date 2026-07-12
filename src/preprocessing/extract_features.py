"""
Extrae features estadísticas por segmento a partir de la señal cruda (L+H),
para alimentar los modelos baseline (Random Forest / SVM).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

SPLITS_DIR = Path("data/splits")
OUT_DIR = Path("data/processed/features")


def load_signal(path_h: str, path_l: str) -> np.ndarray:
    with open(path_h) as f:
        h = np.fromstring(f.readline(), sep=",", dtype=np.float32)
    with open(path_l) as f:
        l = np.fromstring(f.readline(), sep=",", dtype=np.float32)
    return np.concatenate([l, h])


def extract_features(signal: np.ndarray) -> dict:
    """Features estadísticas básicas en el dominio del tiempo."""
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


def process_split(split_name: str) -> pd.DataFrame:
    df = pd.read_csv(SPLITS_DIR / f"{split_name}.csv")
    rows = []

    for i, row in df.iterrows():
        signal = load_signal(row["path_h"], row["path_l"])
        feats = extract_features(signal)
        feats["drone"] = row["drone"]
        feats["mode"] = row["mode"]
        feats["bui"] = row["bui"]
        feats["segment_num"] = row["segment_num"]
        rows.append(feats)
        print(f"[{split_name}] {i+1}/{len(df)} procesado")

    result = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{split_name}_features.csv"
    result.to_csv(out_path, index=False)
    print(f"Guardado: {out_path}")
    return result


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        process_split(split)