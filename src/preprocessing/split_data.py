"""
Split train/val/test del índice DroneRF, estratificado por (drone, mode)
para mantener proporciones de clase y evitar leakage entre segmentos.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

INDEX_PATH = "data/processed/index_dronerf.csv"
SPLITS_DIR = "data/splits"

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42


def stratified_split(df: pd.DataFrame):
    df = df.copy()
    df["strata"] = df["drone"] + "_" + df["mode"]

    train, temp = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        stratify=df["strata"],
        random_state=RANDOM_STATE,
    )

    val_relative = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    val, test = train_test_split(
        temp,
        train_size=val_relative,
        stratify=temp["strata"],
        random_state=RANDOM_STATE,
    )

    return train.drop(columns="strata"), val.drop(columns="strata"), test.drop(columns="strata")


if __name__ == "__main__":
    df = pd.read_csv(INDEX_PATH)
    train, val, test = stratified_split(df)

    train.to_csv(f"{SPLITS_DIR}/train.csv", index=False)
    val.to_csv(f"{SPLITS_DIR}/val.csv", index=False)
    test.to_csv(f"{SPLITS_DIR}/test.csv", index=False)

    print(f"Train: {len(train)} segmentos")
    print(train.groupby(["drone", "mode"]).size())
    print(f"\nVal: {len(val)} segmentos")
    print(val.groupby(["drone", "mode"]).size())
    print(f"\nTest: {len(test)} segmentos")
    print(test.groupby(["drone", "mode"]).size())