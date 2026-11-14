import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from training.dataset_loader import build_dataset_manifest, train_val_test_split


class RetinalDataset(Dataset):
    def __init__(self, samples: Sequence[Dict[str, object]], image_size: int = 224):
        self.samples = list(samples)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = str(sample["image_path"])
        label = int(sample["label"])

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.tensor(np.transpose(image_array, (2, 0, 1)), dtype=torch.float32)
        return image_tensor, torch.tensor(label, dtype=torch.long)


def build_efficientnet_b0(num_classes: int = 5):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def train_model(train_loader: DataLoader, val_loader: DataLoader, epochs: int = 5, device: str = "cpu"):
    model = build_efficientnet_b0(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                val_loss += float(criterion(logits, labels).item())

        print(f"Epoch {epoch + 1}/{epochs} - train_loss={running_loss:.4f} - val_loss={val_loss:.4f}")

    output_path = Path("model/dr_model.pth")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DR classification model")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to a retinal-image dataset directory")
    parser.add_argument("--csv", type=str, default="", help="Optional CSV file with image IDs and labels")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")

    manifest = build_dataset_manifest(str(data_dir), args.csv)
    if not manifest:
        raise ValueError("No labeled retinal images were found in the dataset directory.")

    train_samples, val_samples, _ = train_val_test_split(manifest, val_ratio=0.2, test_ratio=0.1)
    train_loader = DataLoader(RetinalDataset(train_samples), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(RetinalDataset(val_samples), batch_size=args.batch_size, shuffle=False)

    print(f"Training samples: {len(train_samples)} | validation samples: {len(val_samples)}")
    train_model(train_loader, val_loader, epochs=args.epochs, device=args.device)
    print("Training completed. Model saved to model/dr_model.pth")
