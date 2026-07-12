"""
Entrena y evalúa modelos baseline (Random Forest y SVM) sobre las
features estadísticas extraídas de las señales RF.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

FEATURES_DIR = Path("data/processed/features")
MODELS_DIR = Path("results/models")
METRICS_DIR = Path("results/metrics")

FEATURE_COLS = [
    "mean", "std", "var", "skewness", "kurtosis", "rms",
    "peak_to_peak", "mean_abs", "max_abs", "energy",
    "spectral_centroid", "spectral_energy",
]

LABEL_COL = "drone"  # clasificación por tipo de drone (background incluido)


def load_split(name: str):
    df = pd.read_csv(FEATURES_DIR / f"{name}_features.csv")
    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values
    return X, y


def evaluate(model, X, y, split_name: str, model_name: str):
    preds = model.predict(X)
    report = classification_report(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds, labels=sorted(set(y)))
    f1 = f1_score(y, preds, average="macro", zero_division=0)

    print(f"\n=== {model_name} | {split_name} ===")
    print(f"F1-macro: {f1:.4f}")
    print(report)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / f"{model_name}_{split_name}_report.txt"
    with open(out_path, "w") as f:
        f.write(f"F1-macro: {f1:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion matrix (orden alfabético de clases):\n")
        f.write(str(cm))

    return f1


if __name__ == "__main__":
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)  # RF no necesita escalado
    evaluate(rf, X_val, y_val, "val", "random_forest")
    evaluate(rf, X_test, y_test, "test", "random_forest")
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")

    # --- SVM ---
    svm = SVC(kernel="rbf", class_weight="balanced", random_state=42)
    svm.fit(X_train_s, y_train)
    evaluate(svm, X_val_s, y_val, "val", "svm")
    evaluate(svm, X_test_s, y_test, "test", "svm")
    joblib.dump(svm, MODELS_DIR / "svm.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    print("\nModelos y reportes guardados en results/models y results/metrics")