import os
import random
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EXPERIMENT_NAME = "baseline_cnn_comparison"
LABEL_COLS      = ["N", "D", "G", "C", "A", "H", "M", "O"]
NUM_CLASSES     = len(LABEL_COLS)
BATCH_SIZE      = 32
NUM_EPOCHS      = 50
EARLY_STOP_PAT  = 10
LR              = 1e-4
THRESHOLD       = 0.5

# IMAGE PATHS — adjust these to your actual directory structure
TRAIN_IMG_DIR = "real_dataset/ODIR-5K/ODIR-5K/Training Images"
TEST_IMG_DIR  = "real_dataset/ODIR-5K/ODIR-5K/Testing Images"


# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────
# TRANSFORM — VANILLA BASELINE ONLY
# Minimal preprocessing: resize + normalize. No augmentation, no cropping.
# This is the most vanilla, reproducible baseline.
# ─────────────────────────────────────────────
baseline_transform = transforms.Compose([
    transforms.Resize((224, 224)),             # vanilla resize to fixed size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.3013, 0.1916, 0.1041],         # your dataset's actual mean
        std=[0.3105, 0.2125, 0.1388]           # your dataset's actual std
    )
])


# ─────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────
class FundusDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image"]
        
        # Try training directory first, then testing directory
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            # fallback to test directory if not in training
            img_path = os.path.join(TEST_IMG_DIR, img_name)
        
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(
            self.df.loc[idx, LABEL_COLS].values.astype(np.float32)
        )

        if self.transform:
            image = self.transform(image)

        return image, label


# ─────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────
def create_model(model_name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    if model_name == "resnet50":
        model    = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "densenet121":
        model            = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model               = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


# ─────────────────────────────────────────────
# IMBALANCE HANDLING
# ─────────────────────────────────────────────
def compute_pos_weight(df: pd.DataFrame) -> torch.Tensor:
    labels     = df[LABEL_COLS].values
    pos        = labels.sum(axis=0)
    neg        = len(labels) - pos
    return torch.tensor(neg / (pos + 1e-6), dtype=torch.float32)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_bin = (y_pred >= THRESHOLD).astype(int)
    per_class = {}
    aucs, f1s, sens, specs = [], [], [], []

    for i, col in enumerate(LABEL_COLS):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auc = float("nan")

        f1  = f1_score(y_true[:, i], y_bin[:, i], zero_division=0)
        tn, fp, fn, tp = confusion_matrix(
            y_true[:, i], y_bin[:, i], labels=[0, 1]
        ).ravel()
        sensitivity = tp / (tp + fn + 1e-6)
        specificity = tn / (tn + fp + 1e-6)

        per_class[col] = dict(auc=auc, f1=f1,
                              sensitivity=sensitivity,
                              specificity=specificity)
        aucs.append(auc); f1s.append(f1)
        sens.append(sensitivity); specs.append(specificity)

    macro = dict(auc=np.nanmean(aucs), f1=np.mean(f1s),
                 sensitivity=np.mean(sens), specificity=np.mean(specs))

    return {"per_class": per_class, "macro": macro}


# ─────────────────────────────────────────────
# INFERENCE HELPER
# ─────────────────────────────────────────────
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = torch.sigmoid(model(images.to(device)))
            all_preds.append(preds.cpu())
            all_labels.append(labels)
    return torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()


# ─────────────────────────────────────────────
# TRAINING — one model × one seed
# ─────────────────────────────────────────────
def train_one_seed(seed: int, model_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  {model_name.upper()}  |  Seed {seed}")
    print(f"{'='*60}")

    set_seed(seed)

    # data — same baseline_transform for all splits, NO AUGMENTATION
    train_df = pd.read_csv(f"real_dataset/train_seed{seed}.csv")
    val_df   = pd.read_csv(f"real_dataset/val{seed}.csv")
    test_df  = pd.read_csv(f"real_dataset/test_seed{seed}.csv")

    train_loader = DataLoader(
        FundusDataset(train_df, TRAIN_IMG_DIR, baseline_transform),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader   = DataLoader(
        FundusDataset(val_df, TRAIN_IMG_DIR, baseline_transform),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader  = DataLoader(
        FundusDataset(test_df, TEST_IMG_DIR, baseline_transform),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = create_model(model_name).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=compute_pos_weight(train_df).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_auc  = 0.0
    epochs_no_imp = 0
    ckpt_path     = f"{EXPERIMENT_NAME}_{model_name}_seed{seed}.pth"

    # learning curve history
    history = {"epoch": [], "train_loss": [], "train_auc": [], "val_auc": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # training AUC
        y_true_train, y_pred_train = run_inference(model, train_loader, device)
        train_auc = compute_metrics(y_true_train, y_pred_train)["macro"]["auc"]

        # validation AUC
        y_true_val, y_pred_val = run_inference(model, val_loader, device)
        val_auc = compute_metrics(y_true_val, y_pred_val)["macro"]["auc"]

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"Loss {avg_loss:.4f} | "
              f"Train AUC {train_auc:.4f} | "
              f"Val AUC {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc  = val_auc
            epochs_no_imp = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= EARLY_STOP_PAT:
                print(f"  Early stopping at epoch {epoch}")
                break

    # save learning curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{model_name} | Seed {seed}", fontsize=12)

    ax1.plot(history["epoch"], history["train_loss"], label="Train Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss"); ax1.legend(); ax1.grid(True)

    ax2.plot(history["epoch"], history["train_auc"], label="Train AUC")
    ax2.plot(history["epoch"], history["val_auc"],   label="Val AUC")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro AUC")
    ax2.set_title("AUC Curves"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    curve_path = f"{EXPERIMENT_NAME}_{model_name}_seed{seed}_curves.png"
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"  Learning curves saved: {curve_path}")

    # test with best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    y_true_test, y_pred_test = run_inference(model, test_loader, device)
    test_metrics = compute_metrics(y_true_test, y_pred_test)

    print(f"\n  ── Test results (seed {seed}) ──")
    for k, v in test_metrics["macro"].items():
        print(f"  Macro {k:<14}: {v:.4f}")
    print("  Per-class AUC:")
    for col, m in test_metrics["per_class"].items():
        print(f"    {col}: {m['auc']:.4f}")

    return test_metrics, history


# ─────────────────────────────────────────────
# EXPERIMENT LOOP
# ─────────────────────────────────────────────
model_names = ["resnet50", "densenet121", "efficientnet_b0"]
seeds       = [42, 99, 202]
all_results = {m: [] for m in model_names}

for model_name in model_names:
    for seed in seeds:
        test_metrics, history = train_one_seed(seed, model_name)
        all_results[model_name].append(test_metrics)


# ─────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────
MACRO_KEYS = ["auc", "f1", "sensitivity", "specificity"]

rows = []
for model_name, seed_metrics in all_results.items():
    row = {"Model": model_name}
    for key in MACRO_KEYS:
        vals = [m["macro"][key] for m in seed_metrics]
        row[f"{key}_mean"] = np.mean(vals)
        row[f"{key}_std"]  = np.std(vals)
    rows.append(row)

summary_df = pd.DataFrame(rows).set_index("Model")

print("\n\n" + "="*70)
print("  FINAL RESULTS  (mean ± std over 3 seeds)")
print("="*70)
for model_name, row in summary_df.iterrows():
    print(f"\n{model_name}")
    for key in MACRO_KEYS:
        print(f"  {key:<14}: {row[f'{key}_mean']:.4f} ± {row[f'{key}_std']:.4f}")

summary_df.to_csv(f"{EXPERIMENT_NAME}_summary.csv")

per_class_rows = []
for model_name, seed_metrics in all_results.items():
    row = {"Model": model_name}
    for col in LABEL_COLS:
        aucs = [m["per_class"][col]["auc"] for m in seed_metrics]
        row[col] = f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}"
    per_class_rows.append(row)

per_class_df = pd.DataFrame(per_class_rows).set_index("Model")
per_class_df.to_csv(f"{EXPERIMENT_NAME}_per_class_auc.csv")

print(f"\nSaved: {EXPERIMENT_NAME}_summary.csv")
print(f"Saved: {EXPERIMENT_NAME}_per_class_auc.csv")
print("\nPer-class AUC (mean ± std):")
print(per_class_df.to_string())
