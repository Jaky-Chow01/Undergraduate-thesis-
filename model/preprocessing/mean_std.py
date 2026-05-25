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

# NO normalization here — raw pixels only
raw_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()   # converts to [0,1]
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


def compute_mean_std(df: pd.DataFrame, batch_size: int = 32) -> tuple:
    """
    Compute per-channel mean and std over the entire dataset.
    Must be run on TRAIN SET ONLY — never val or test.
    """
    loader = DataLoader(
        FundusDataset(df, raw_transform),
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    mean = torch.zeros(3)
    std  = torch.zeros(3)
    n_pixels = 0

    print("Computing mean...")
    for images, _ in loader:
        # images: [B, C, H, W]
        b, c, h, w = images.shape
        n_pixels  += b * h * w
        mean      += images.sum(dim=[0, 2, 3])

    mean /= n_pixels

    print("Computing std...")
    for images, _ in loader:
        b, c, h, w = images.shape
        std += ((images - mean.view(1, 3, 1, 1)) ** 2).sum(dim=[0, 2, 3])

    std = torch.sqrt(std / n_pixels)

    return mean.numpy(), std.numpy()


if __name__ == "__main__":
    # compute ONLY on training set
    train_df = pd.read_csv("../real_dataset/train_seed42.csv")

    mean, std = compute_mean_std(train_df)

    print(f"\n── Your dataset's actual statistics ──")
    print(f"  Mean (R,G,B): {mean.round(4).tolist()}")
    print(f"  Std  (R,G,B): {std.round(4).tolist()}")

    print(f"\n── Replace in preprocessing.py ──")
    print(f"  transforms.Normalize(")
    print(f"      mean={mean.round(4).tolist()},")
    print(f"      std ={std.round(4).tolist()}")
    print(f"  )")

    print(f"\n── For reference — ImageNet values (what you were using) ──")
    print(f"  mean=[0.485, 0.456, 0.406]")
    print(f"  std =[0.229, 0.224, 0.225]")
