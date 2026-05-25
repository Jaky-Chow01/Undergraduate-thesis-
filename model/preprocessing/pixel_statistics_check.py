import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# ── config ────────────────────────────────────────────────────
LABEL_COLS = ["N", "D", "G", "C", "A", "H", "M", "O"]

def get_image_path(image_name: str) -> str:
    train_path = f"../real_dataset/ODIR-5K/ODIR-5K/Training Images/{image_name}"
    test_path  = f"../real_dataset/ODIR-5K/ODIR-5K/Testing Images/{image_name}"
    return train_path if os.path.exists(train_path) else test_path

baseline_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3013, 0.1916, 0.1041],
                         std=[0.3105, 0.2125, 0.1388])
])

class FundusDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        image = Image.open(get_image_path(self.df.loc[idx, "image"])).convert("RGB")
        label = torch.tensor(self.df.loc[idx, LABEL_COLS].values.astype(np.float32))
        if self.transform:
            image = self.transform(image)
        return image, label

# ── function ──────────────────────────────────────────────────
def check_pixel_stats(loader: DataLoader, n_batches: int = 5, split_name: str = "train"):
    means, stds = [], []
    for i, (images, _) in enumerate(loader):
        if i >= n_batches:
            break
        means.append(images.mean(dim=[0, 2, 3]))
        stds.append(images.std(dim=[0, 2, 3]))

    mean = torch.stack(means).mean(0).numpy().round(3)
    std  = torch.stack(stds).mean(0).numpy().round(3)

    print(f"\nPixel statistics — {split_name} (first {n_batches} batches)")
    print(f"  Post-normalization mean (R,G,B): {mean}")
    print(f"  Post-normalization std  (R,G,B): {std}")

    mean_ok = np.all(np.abs(mean) < 0.1)
    std_ok  = np.all(np.abs(std - 1.0) < 0.2)

    if mean_ok:
        print("  ✓  Mean is close to 0 — normalization looks correct.")
    else:
        print("  ⚠  Mean is far from 0 — check your normalization values.")

    if std_ok:
        print("  ✓  Std is close to 1 — normalization looks correct.")
    else:
        print("  ⚠  Std is far from 1 — check your normalization values.")


# ── run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    df     = pd.read_csv("../real_dataset/train_seed42.csv")
    loader = DataLoader(FundusDataset(df, baseline_transform),
                        batch_size=32, shuffle=False, num_workers=0)
    check_pixel_stats(loader, n_batches=5, split_name="train_seed42")
