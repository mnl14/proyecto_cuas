"""
Carga y etiquetado del dataset DroneRF.
Empareja segmentos L/H por BUI y genera un índice con metadata + labels.
"""

import re
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/dronerf/DroneRF")

DRONE_MAP = {
    "000": "background",
    "100": "bebop",
    "101": "ar",
    "110": "phantom",
    "111": "phantom",
}

MODE_MAP = {
    "00": "on_connected",
    "01": "hovering",
    "10": "flying",
    "11": "flying_video",
}

FILENAME_RE = re.compile(r"^(\d{5})([HL])_(\d+)\.csv$")


@dataclass
class Segment:
    bui: str
    band: str          # 'H' o 'L'
    segment_num: int
    drone: str
    mode: str
    path: Path


def parse_filename(path: Path) -> Segment | None:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    bui, band, seg_num = m.groups()
    drone_bits = bui[:3]
    mode_bits = bui[3:]
    drone = DRONE_MAP.get(drone_bits, "unknown")
    mode = MODE_MAP.get(mode_bits, "unknown") if drone != "background" else "background"
    return Segment(bui, band, int(seg_num), drone, mode, path)


def build_index(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Recorre raw_dir y arma un índice de segmentos con su path y label."""
    rows = []
    for csv_path in raw_dir.rglob("*.csv"):
        seg = parse_filename(csv_path)
        if seg is None:
            print(f"[WARN] No matchea el patrón de nombre: {csv_path.name}")
            continue
        rows.append(seg.__dict__)
    df = pd.DataFrame(rows)
    return df


def pair_lh(df: pd.DataFrame) -> pd.DataFrame:
    """Empareja bandas L y H del mismo bui+segment_num en una sola fila."""
    pivot = df.pivot_table(
        index=["bui", "segment_num", "drone", "mode"],
        columns="band",
        values="path",
        aggfunc="first",
    ).reset_index()
    pivot = pivot.rename(columns={"H": "path_h", "L": "path_l"})
    missing = pivot[pivot["path_h"].isna() | pivot["path_l"].isna()]
    if len(missing):
        print(f"[WARN] {len(missing)} segmentos sin par L/H completo")
    return pivot.dropna(subset=["path_h", "path_l"])


def load_signal(path_h: Path, path_l: Path) -> np.ndarray:
    """Carga y concatena las bandas H y L de un segmento en un solo array."""
    h = pd.read_csv(path_h, header=None).values.flatten()
    l = pd.read_csv(path_l, header=None).values.flatten()
    return np.concatenate([l, h])


if __name__ == "__main__":
    df = build_index()
    print(f"Total archivos indexados: {len(df)}")
    print(df["drone"].value_counts())

    pairs = pair_lh(df)
    print(f"\nSegmentos completos (L+H): {len(pairs)}")
    print(pairs.groupby(["drone", "mode"]).size())

    pairs.to_csv("data/processed/index_dronerf.csv", index=False)
    print("\nÍndice guardado en data/processed/index_dronerf.csv")