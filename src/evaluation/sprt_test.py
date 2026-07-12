from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("results/models")
METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")

CHECKPOINTS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
THRESHOLDS = [0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

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


if __name__ == "__main__":
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    rf = joblib.load(MODELS_DIR / "random_forest.joblib")

    print("Cargando señales y precalculando features por checkpoint...")
    # Precalcular, por señal, las features y prob en cada checkpoint (se reusa entre thresholds)
    per_signal_checkpoints = []  # lista de listas [(frac, proba_dict, pred_class)]
    y_true_all = []

    for i, row in test_df.iterrows():
        signal = load_signal(row["path_h"], row["path_l"])
        y_true_all.append(row["drone"])
        n = len(signal)

        cp_results = []
        for frac in CHECKPOINTS:
            partial = signal[: int(n * frac)]
            feats = extract_features(partial)
            X = np.array([[feats[c] for c in FEATURE_COLS]])
            proba = rf.predict_proba(X)[0]
            pred_class = rf.classes_[np.argmax(proba)]
            confidence = np.max(proba)
            cp_results.append((frac, confidence, pred_class))
        per_signal_checkpoints.append(cp_results)
        print(f"  procesada {i+1}/{len(test_df)}")

    # Para cada umbral, simular la decisión secuencial
    results = []
    for threshold in THRESHOLDS:
        preds = []
        fracs_used = []
        for cp_results in per_signal_checkpoints:
            decided = False
            for frac, confidence, pred_class in cp_results:
                if confidence >= threshold:
                    preds.append(pred_class)
                    fracs_used.append(frac)
                    decided = True
                    break
            if not decided:
                # nunca alcanzó el umbral: usar la decisión al 100%
                preds.append(cp_results[-1][2])
                fracs_used.append(1.00)

        acc = accuracy_score(y_true_all, preds)
        f1 = f1_score(y_true_all, preds, average="macro", zero_division=0)
        avg_frac = np.mean(fracs_used)
        results.append({
            "threshold": threshold,
            "accuracy": acc,
            "f1_macro": f1,
            "avg_fraction_signal_used": avg_frac,
        })
        print(f"threshold={threshold}: acc={acc:.3f} f1={f1:.3f} frac_promedio={avg_frac:.2f}")

    results_df = pd.DataFrame(results)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(METRICS_DIR / "sprt_results.csv", index=False)

    # Plot: accuracy vs fracción de señal usada (tiempo de decisión relativo)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["avg_fraction_signal_used"], results_df["accuracy"], marker="o", label="Accuracy")
    plt.plot(results_df["avg_fraction_signal_used"], results_df["f1_macro"], marker="s", label="F1-macro")
    for _, r in results_df.iterrows():
        plt.annotate(f"thr={r['threshold']}", (r["avg_fraction_signal_used"], r["accuracy"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel("Fracción promedio de señal usada (proxy de tiempo de decisión)")
    plt.ylabel("Score")
    plt.title("SPRT simplificado: Accuracy/F1 vs tiempo de decisión")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sprt_accuracy_vs_time.png", dpi=150)

    print("\nResultados guardados en results/metrics/sprt_results.csv")
    print("Gráfico guardado en results/figures/sprt_accuracy_vs_time.png")