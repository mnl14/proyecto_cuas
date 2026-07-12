"""
Entrena y evalúa una CNN sobre los espectrogramas generados en compute_spectrograms.py.

Nota de diseño: los espectrogramas originales son muy grandes (513 x ~39000
frames), inmanejables en memoria/cómputo para una CNN estándar. Se
redimensionan a un tamaño fijo (128x128) antes de entrenar.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

SPEC_DIR = Path("data/processed/spectrograms")
MODELS_DIR = Path("results/models")
METRICS_DIR = Path("results/metrics")
TARGET_SIZE = (128, 128)

CLASSES = ["ar", "background", "bebop", "phantom"]


def resize_spectrogram(spec: np.ndarray, target_size=TARGET_SIZE) -> np.ndarray:
    spec_db = 20 * np.log10(spec + 1e-10)
    zoom_factors = (target_size[0] / spec_db.shape[0], target_size[1] / spec_db.shape[1])
    resized = zoom(spec_db, zoom_factors, order=1)
    # Normalización por muestra (z-score)
    resized = (resized - resized.mean()) / (resized.std() + 1e-8)
    return resized.astype(np.float32)


class SpectrogramDataset(Dataset):
    def __init__(self, split_name: str, label_encoder: LabelEncoder):
        self.files = sorted((SPEC_DIR / split_name).glob("*.npy"))
        self.label_encoder = label_encoder
        self.labels = [f.name.split("_")[0] for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spec = np.load(self.files[idx])
        spec = resize_spectrogram(spec)
        label = self.label_encoder.transform([self.labels[idx]])[0]
        return torch.from_numpy(spec).unsqueeze(0), label


class DroneCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 128->64
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64->32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32->16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def evaluate(model, loader, device, label_encoder, split_name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    y_true = label_encoder.inverse_transform(all_labels)
    y_pred = label_encoder.inverse_transform(all_preds)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)

    print(f"\n=== CNN | {split_name} ===")
    print(f"F1-macro: {f1:.4f}")
    print(report)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / f"cnn_{split_name}_report.txt", "w") as f:
        f.write(f"F1-macro: {f1:.4f}\n\n{report}")

    return f1


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    label_encoder = LabelEncoder()
    label_encoder.fit(CLASSES)

    train_ds = SpectrogramDataset("train", label_encoder)
    val_ds = SpectrogramDataset("val", label_encoder)
    test_ds = SpectrogramDataset("test", label_encoder)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)
    test_loader = DataLoader(test_ds, batch_size=8)

    model = DroneCNN(n_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    N_EPOCHS = 30
    best_val_f1 = 0.0

    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_ds)
        val_f1 = evaluate(model, val_loader, device, label_encoder, f"val_epoch{epoch+1}")
        print(f"Epoch {epoch+1}/{N_EPOCHS} - loss: {avg_loss:.4f} - val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODELS_DIR / "cnn_best.pt")

    # Evaluación final con el mejor modelo guardado
    model.load_state_dict(torch.load(MODELS_DIR / "cnn_best.pt"))
    evaluate(model, val_loader, device, label_encoder, "val")
    evaluate(model, test_loader, device, label_encoder, "test")

    print("\nModelo y reportes guardados en results/models y results/metrics")