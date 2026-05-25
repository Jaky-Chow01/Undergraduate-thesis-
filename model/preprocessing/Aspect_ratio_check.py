import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ── config ────────────────────────────────────────────────────
def get_image_path(image_name: str) -> str:
    train_path = f"../real_dataset/ODIR-5K/ODIR-5K/Training Images/{image_name}"
    test_path  = f"../real_dataset/ODIR-5K/ODIR-5K/Testing Images/{image_name}"
    return train_path if os.path.exists(train_path) else test_path

# ── function ──────────────────────────────────────────────────
def check_aspect_ratios(df: pd.DataFrame, n: int = 100, split_name: str = "train"):
    ratios = []
    for i in range(min(n, len(df))):
        img = Image.open(get_image_path(df.loc[i, "image"]))
        w, h = img.size
        ratios.append(w / h)

    ratios = np.array(ratios)
    non_square = (np.abs(ratios - 1) > 0.05).sum()

    print(f"\nAspect ratio check — {split_name} (first {len(ratios)} images)")
    print(f"  Mean  : {ratios.mean():.3f}")
    print(f"  Min   : {ratios.min():.3f}")
    print(f"  Max   : {ratios.max():.3f}")
    print(f"  Non-square (|ratio-1| > 0.05): {non_square} / {len(ratios)}")

    if non_square > 0:
        print("  ⚠  Non-square images detected — Resize(224,224) will distort them.")
        print("     Consider Resize(224) on the shorter edge + padding instead.")
    else:
        print("  ✓  All images are approximately square — Resize(224,224) is safe.")

    # histogram
    plt.figure(figsize=(7, 4))
    plt.hist(ratios, bins=20, color="steelblue", edgecolor="white")
    plt.axvline(1.0, color="red", linestyle="--", label="Square (ratio=1)")
    plt.xlabel("Width / Height ratio")
    plt.ylabel("Count")
    plt.title(f"Aspect Ratio Distribution — {split_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"aspect_ratio_{split_name}.png", dpi=150)
    print(f"  Saved: aspect_ratio_{split_name}.png")
    plt.show()


# ── run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    for seed in [42, 99, 202]:
        df = pd.read_csv(f"../real_dataset/train_seed{seed}.csv")
        check_aspect_ratios(df, n=100, split_name=f"train_seed{seed}")
