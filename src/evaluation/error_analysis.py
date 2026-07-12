from pathlib import Path
import numpy as np
import pandas as pd
import joblib

FEATURES_DIR = Path("data/processed/features")
MODELS_DIR = Path("results/models")
METRICS_DIR = Path("results/metrics")

FEATURE_COLS = [
    "mean", "std", "var", "skewness", "kurtosis", "rms",
    "peak_to_peak", "mean_abs", "max_abs", "energy",
    "spectral_centroid", "spectral_energy",
]

if __name__ == "__main__":
    train_df = pd.read_csv(FEATURES_DIR / "train_features.csv")
    test_df = pd.read_csv(FEATURES_DIR / "test_features.csv")
    rf = joblib.load(MODELS_DIR / "random_forest.joblib")

    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["drone"].values

    preds = rf.predict(X_test)
    probas = rf.predict_proba(X_test)

    test_df = test_df.copy()
    test_df["pred"] = preds
    test_df["confidence"] = probas.max(axis=1)

    misclassified = test_df[test_df["drone"] != test_df["pred"]]

    print(f"Total mal clasificados: {len(misclassified)} de {len(test_df)}\n")

    class_means = train_df.groupby("drone")[FEATURE_COLS].mean()

    report_lines = []
    for _, row in misclassified.iterrows():
        line = (
            f"BUI={row['bui']} segmento={row['segment_num']} | "
            f"real={row['drone']} ({row['mode']}) -> predicho={row['pred']} "
            f"(confianza={row['confidence']:.2f})"
        )
        print(line)
        report_lines.append(line)

        real_mean = class_means.loc[row["drone"]]
        pred_mean = class_means.loc[row["pred"]]

        print("  Feature       | valor caso | media_real | media_pred")
        report_lines.append("  Feature       | valor caso | media_real | media_pred")
        for feat in FEATURE_COLS:
            val = row[feat]
            print(f"  {feat:<14}| {val:.4g}   | {real_mean[feat]:.4g}   | {pred_mean[feat]:.4g}")
            report_lines.append(f"  {feat:<14}| {val:.4g} | {real_mean[feat]:.4g} | {pred_mean[feat]:.4g}")
        print()
        report_lines.append("")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "error_analysis.txt", "w") as f:
        f.write(f"Total mal clasificados: {len(misclassified)} de {len(test_df)}\n\n")
        f.write("\n".join(report_lines))

    print("Reporte guardado en results/metrics/error_analysis.txt")