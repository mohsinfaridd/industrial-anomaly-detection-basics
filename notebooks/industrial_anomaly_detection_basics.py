# %% [markdown]
# # Industrial Image Anomaly Detection — Kaggle Basics
#
# This notebook teaches the essential workflow using one **MVTec AD** category:
#
# 1. Explore normal images, defects, and pixel masks.
# 2. Create a normal-only train/validation split without using test labels.
# 3. Train a convolutional autoencoder (reconstruction baseline).
# 4. Train a simplified PaDiM-style pretrained-feature baseline.
# 5. Evaluate image-level detection and pixel-level localization.
# 6. Inspect confusion matrices, ROC/PR curves, heatmaps, and per-defect results.
#
# Add MVTec AD as a Kaggle input. One category should look like:
#
# ```text
# bottle/
# ├── train/good/
# ├── test/good/
# ├── test/<defect_type>/
# └── ground_truth/<defect_type>/
# ```
#
# Official references:
# - MVTec AD: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
# - Torchvision ResNet-18: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
# - Scikit-learn metrics: https://scikit-learn.org/stable/api/sklearn.metrics.html

# %%
from __future__ import annotations

import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

warnings.filterwarnings("ignore", category=UserWarning)
SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


seed_everything()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch:", torch.__version__)
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## 1. Configuration
#
# Change `category` to another MVTec category. Leave `manual_category_path` empty so the notebook searches `/kaggle/input` automatically.

# %%
@dataclass
class Config:
    category: str = "bottle"
    manual_category_path: str = ""
    image_size: int = 128
    batch_size: int = 32
    num_workers: int = 2
    validation_fraction: float = 0.20
    normal_threshold_quantile: float = 0.99
    autoencoder_epochs: int = 12
    autoencoder_learning_rate: float = 1e-3
    run_autoencoder: bool = True
    run_feature_baseline: bool = True
    top_fraction: float = 0.01
    max_pixels_for_metric: int = 1_000_000
    output_dir: str = "/kaggle/working/industrial_anomaly_outputs"


CFG = Config()
Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)
CFG

# %% [markdown]
# ## 2. Locate and index the dataset
#
# Labels are binary: `0 = normal`, `1 = anomaly`. Defect folder names are retained for analysis; this is not multi-class classification.

# %%
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_category_folder(path: Path) -> bool:
    return path.is_dir() and (path / "train").is_dir() and (path / "test").is_dir()


def find_category_folder(category: str, manual_path: str = "") -> Path:
    if manual_path.strip():
        candidate = Path(manual_path).expanduser().resolve()
        if not is_category_folder(candidate):
            raise FileNotFoundError(f"Invalid category folder: {candidate}")
        return candidate

    candidates = []
    for root in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if root.exists():
            for path in root.rglob(category):
                if path.name.lower() == category.lower() and is_category_folder(path):
                    candidates.append(path.resolve())

    if not candidates:
        raise FileNotFoundError(
            f"Could not find '{category}'. Add MVTec AD as Kaggle input or set "
            "CFG.manual_category_path to the exact category folder."
        )
    candidates.sort(key=lambda p: (not (p / "ground_truth").exists(), len(str(p))))
    return candidates[0]


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def locate_mask(root: Path, image_path: Path, defect_type: str) -> Optional[Path]:
    if defect_type == "good":
        return None
    mask_dir = root / "ground_truth" / defect_type
    for candidate in [
        mask_dir / f"{image_path.stem}_mask.png",
        mask_dir / f"{image_path.stem}.png",
    ]:
        if candidate.exists():
            return candidate
    matches = list(mask_dir.glob(f"{image_path.stem}*"))
    return matches[0] if matches else None


def build_index(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = [
        {"path": str(p), "label": 0, "defect_type": "good", "mask_path": None}
        for p in list_images(root / "train" / "good")
    ]
    test = []
    for p in list_images(root / "test"):
        defect = p.parent.name
        mask = locate_mask(root, p, defect)
        test.append({
            "path": str(p),
            "label": 0 if defect == "good" else 1,
            "defect_type": defect,
            "mask_path": str(mask) if mask else None,
        })
    train_df = pd.DataFrame(train)
    test_df = pd.DataFrame(test)
    if train_df.empty or test_df.empty:
        raise ValueError("Training or test images are missing.")
    return train_df, test_df


CATEGORY_ROOT = find_category_folder(CFG.category, CFG.manual_category_path)
train_all_df, test_df = build_index(CATEGORY_ROOT)
train_df, val_normal_df = train_test_split(
    train_all_df,
    test_size=CFG.validation_fraction,
    random_state=SEED,
    shuffle=True,
)
train_df = train_df.reset_index(drop=True)
val_normal_df = val_normal_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Category folder:", CATEGORY_ROOT)
print("Training normal:", len(train_df))
print("Validation normal:", len(val_normal_df))
print("Test images:", len(test_df))
display(test_df["defect_type"].value_counts().rename("images").to_frame())

# %% [markdown]
# ### Why split normal training images?
#
# Training images teach the model normality. Validation-normal images select a threshold. Test labels are used only for final evaluation, preventing test leakage.

# %% [markdown]
# ## 3. Explore images and ground-truth masks

# %%
def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_mask(mask_path: Optional[str], size: tuple[int, int]) -> np.ndarray:
    if not mask_path:
        return np.zeros(size, dtype=np.uint8)
    with Image.open(mask_path) as image:
        image = image.convert("L").resize((size[1], size[0]), Image.NEAREST)
        return (np.asarray(image) > 0).astype(np.uint8)


def show_examples(frame: pd.DataFrame, max_defects: int = 4) -> None:
    normal = frame[frame.label == 0].head(2)
    defects = frame[frame.label == 1].groupby("defect_type", group_keys=False).head(1).head(max_defects)
    rows = pd.concat([normal, defects], ignore_index=True)
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 4 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, 0)
    for i, row in rows.iterrows():
        image = load_rgb(row.path)
        mask = load_mask(row.mask_path, image.shape[:2])
        axes[i, 0].imshow(image); axes[i, 0].set_title(row.defect_type); axes[i, 0].axis("off")
        axes[i, 1].imshow(mask, cmap="gray"); axes[i, 1].set_title("Ground-truth mask"); axes[i, 1].axis("off")
        axes[i, 2].imshow(image); axes[i, 2].imshow(mask, cmap="Reds", alpha=0.45)
        axes[i, 2].set_title("Overlay"); axes[i, 2].axis("off")
    plt.tight_layout(); plt.show()


show_examples(test_df)

# %% [markdown]
# ## 4. PyTorch datasets

# %%
class IndustrialDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor,
            "label": int(row.label),
            "path": str(row.path),
            "defect_type": str(row.defect_type),
        }


def make_loader(frame: pd.DataFrame, transform, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        IndustrialDataset(frame, transform),
        batch_size=CFG.batch_size,
        shuffle=shuffle,
        num_workers=CFG.num_workers,
        pin_memory=DEVICE.type == "cuda",
    )


ae_transform = transforms.Compose([
    transforms.Resize((CFG.image_size, CFG.image_size)),
    transforms.ToTensor(),
])
train_loader_ae = make_loader(train_df, ae_transform, shuffle=True)
val_loader_ae = make_loader(val_normal_df, ae_transform)
test_loader_ae = make_loader(test_df, ae_transform)

# %% [markdown]
# # Part A — Reconstruction baseline
#
# The autoencoder sees only normal images. The channel-averaged squared reconstruction error becomes an anomaly map.

# %%
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def top_fraction_mean(maps: torch.Tensor, fraction: float) -> torch.Tensor:
    flat = maps.flatten(start_dim=1)
    k = max(1, int(math.ceil(flat.shape[1] * fraction)))
    return torch.topk(flat, k=k, dim=1).values.mean(dim=1)


def train_autoencoder(model: nn.Module, loader: DataLoader) -> pd.DataFrame:
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.autoencoder_learning_rate)
    history = []
    for epoch in range(1, CFG.autoencoder_epochs + 1):
        model.train(); total = 0.0; count = 0
        for batch in loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(images)
            loss = F.mse_loss(reconstruction, images)
            loss.backward(); optimizer.step()
            total += loss.item() * len(images); count += len(images)
        epoch_loss = total / count
        history.append({"epoch": epoch, "train_mse": epoch_loss})
        print(f"Epoch {epoch:02d}/{CFG.autoencoder_epochs} | MSE={epoch_loss:.6f}")
    return pd.DataFrame(history)


@torch.inference_mode()
def infer_autoencoder(model: nn.Module, loader: DataLoader) -> dict:
    model.eval().to(DEVICE)
    scores, maps, labels, paths, defects = [], [], [], [], []
    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        reconstruction = model(images)
        error_maps = ((images - reconstruction) ** 2).mean(dim=1)
        scores.append(top_fraction_mean(error_maps, CFG.top_fraction).cpu().numpy())
        maps.append(error_maps.cpu().numpy())
        labels.append(np.asarray(batch["label"]))
        paths.extend(batch["path"]); defects.extend(batch["defect_type"])
    return {
        "scores": np.concatenate(scores), "maps": np.concatenate(maps),
        "labels": np.concatenate(labels).astype(int), "paths": paths, "defects": defects,
    }


ae_result = None
ae_threshold = None
if CFG.run_autoencoder:
    autoencoder = ConvAutoencoder()
    ae_history = train_autoencoder(autoencoder, train_loader_ae)
    ae_history.plot(x="epoch", y="train_mse", marker="o", figsize=(7, 4), title="Autoencoder training loss")
    plt.grid(alpha=0.25); plt.show()

    val_ae = infer_autoencoder(autoencoder, val_loader_ae)
    ae_result = infer_autoencoder(autoencoder, test_loader_ae)
    ae_threshold = float(np.quantile(val_ae["scores"], CFG.normal_threshold_quantile))
    torch.save(autoencoder.state_dict(), Path(CFG.output_dir) / "autoencoder_state_dict.pt")
    print("Autoencoder threshold:", ae_threshold)

# %% [markdown]
# # Part B — Simplified PaDiM-style feature baseline
#
# Pretrained ResNet-18 produces a spatial feature map. The code estimates the mean and diagonal variance of normal features at every spatial position. Standardized distance becomes the anomaly map.
#
# The first run may need Kaggle Internet enabled to download pretrained weights.

# %%
class ResNetLayer2(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception as exc:
            raise RuntimeError(
                "Could not load pretrained ResNet-18 weights. Enable Kaggle Internet for the first run."
            ) from exc
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


feature_transform = ResNet18_Weights.DEFAULT.transforms()
train_loader_feature = make_loader(train_df, feature_transform)
val_loader_feature = make_loader(val_normal_df, feature_transform)
test_loader_feature = make_loader(test_df, feature_transform)


@torch.inference_mode()
def fit_feature_distribution(extractor: nn.Module, loader: DataLoader):
    extractor.eval().to(DEVICE)
    total_sum = total_square = None
    count = 0
    for batch in loader:
        features = extractor(batch["image"].to(DEVICE, non_blocking=True)).float()
        batch_sum = features.sum(0)
        batch_square = (features ** 2).sum(0)
        total_sum = batch_sum if total_sum is None else total_sum + batch_sum
        total_square = batch_square if total_square is None else total_square + batch_square
        count += len(features)
    mean = total_sum / count
    variance = (total_square / count - mean ** 2).clamp_min(1e-6)
    return mean.detach(), variance.detach()


@torch.inference_mode()
def infer_features(extractor, mean, variance, loader, output_size=224) -> dict:
    extractor.eval().to(DEVICE); mean = mean.to(DEVICE); variance = variance.to(DEVICE)
    scores, maps, labels, paths, defects = [], [], [], [], []
    for batch in loader:
        features = extractor(batch["image"].to(DEVICE, non_blocking=True)).float()
        distance = ((features - mean) ** 2 / variance).mean(dim=1, keepdim=True)
        anomaly_maps = F.interpolate(distance, (output_size, output_size), mode="bilinear", align_corners=False).squeeze(1)
        scores.append(top_fraction_mean(anomaly_maps, CFG.top_fraction).cpu().numpy())
        maps.append(anomaly_maps.cpu().numpy())
        labels.append(np.asarray(batch["label"]))
        paths.extend(batch["path"]); defects.extend(batch["defect_type"])
    return {
        "scores": np.concatenate(scores), "maps": np.concatenate(maps),
        "labels": np.concatenate(labels).astype(int), "paths": paths, "defects": defects,
    }


feature_result = None
feature_threshold = None
if CFG.run_feature_baseline:
    extractor = ResNetLayer2()
    feature_mean, feature_variance = fit_feature_distribution(extractor, train_loader_feature)
    val_feature = infer_features(extractor, feature_mean, feature_variance, val_loader_feature)
    feature_result = infer_features(extractor, feature_mean, feature_variance, test_loader_feature)
    feature_threshold = float(np.quantile(val_feature["scores"], CFG.normal_threshold_quantile))
    torch.save(
        {"mean": feature_mean.cpu(), "variance": feature_variance.cpu(), "layer": "layer2"},
        Path(CFG.output_dir) / "feature_distribution.pt",
    )
    print("Feature threshold:", feature_threshold)

# %% [markdown]
# # Part C — Evaluation
#
# Important image metrics include anomaly precision/recall/F1, normal recall, balanced accuracy, ROC-AUC, and average precision. Pixel metrics compare heatmaps with ground-truth masks.

# %%
def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float):
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "anomaly_precision": precision_score(labels, predictions, pos_label=1, zero_division=0),
        "anomaly_recall": recall_score(labels, predictions, pos_label=1, zero_division=0),
        "normal_recall": recall_score(labels, predictions, pos_label=0, zero_division=0),
        "anomaly_f1": f1_score(labels, predictions, pos_label=1, zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(labels, scores),
        "average_precision": average_precision_score(labels, scores),
        "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1]),
    }
    return metrics, predictions


def pixel_metrics(frame: pd.DataFrame, maps: np.ndarray) -> dict:
    labels_parts, score_parts = [], []
    height, width = maps.shape[-2:]
    for i, row in frame.iterrows():
        labels_parts.append(load_mask(row.mask_path, (height, width)).reshape(-1))
        score_parts.append(maps[i].reshape(-1))
    labels = np.concatenate(labels_parts).astype(np.uint8)
    scores = np.concatenate(score_parts).astype(np.float32)
    if len(labels) > CFG.max_pixels_for_metric:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(labels), CFG.max_pixels_for_metric, replace=False)
        labels, scores = labels[idx], scores[idx]
    return {
        "pixel_roc_auc": roc_auc_score(labels, scores),
        "pixel_average_precision": average_precision_score(labels, scores),
        "pixel_count_used": len(labels),
    }


def evaluate(name: str, result: dict, threshold: float):
    metrics, predictions = binary_metrics(result["labels"], result["scores"], threshold)
    metrics.update(pixel_metrics(test_df, result["maps"]))
    metrics["method"] = name
    frame = test_df.copy()
    frame["score"] = result["scores"]
    frame["prediction"] = predictions
    return metrics, predictions, frame


metrics_rows = []
outputs = {}
if ae_result is not None:
    m, p, f = evaluate("Convolutional Autoencoder", ae_result, ae_threshold)
    metrics_rows.append(m); outputs[m["method"]] = {"result": ae_result, "pred": p, "frame": f, "threshold": ae_threshold}
if feature_result is not None:
    m, p, f = evaluate("Diagonal Gaussian Features", feature_result, feature_threshold)
    metrics_rows.append(m); outputs[m["method"]] = {"result": feature_result, "pred": p, "frame": f, "threshold": feature_threshold}

metrics_df = pd.DataFrame(metrics_rows).set_index("method")
display(metrics_df.round(4))

# %% [markdown]
# ## 5. Confusion matrices and classification reports

# %%
fig, axes = plt.subplots(1, len(outputs), figsize=(6 * len(outputs), 5))
if len(outputs) == 1:
    axes = [axes]
for ax, (name, output) in zip(axes, outputs.items()):
    matrix = confusion_matrix(output["result"]["labels"], output["pred"], labels=[0, 1])
    image = ax.imshow(matrix, cmap="Blues"); ax.figure.colorbar(image, ax=ax)
    ax.set_xticks([0, 1], ["Normal", "Anomaly"]); ax.set_yticks([0, 1], ["Normal", "Anomaly"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name)
    for r in range(2):
        for c in range(2):
            ax.text(c, r, matrix[r, c], ha="center", va="center", fontsize=13)
plt.tight_layout(); plt.show()

for name, output in outputs.items():
    print("=" * 80, "\n", name)
    print(classification_report(
        output["result"]["labels"], output["pred"],
        target_names=["normal", "anomaly"], zero_division=0,
    ))

# %% [markdown]
# ## 6. ROC, precision-recall, and score distributions

# %%
plt.figure(figsize=(7, 5))
for name, output in outputs.items():
    labels, scores = output["result"]["labels"], output["result"]["scores"]
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(labels, scores):.3f})")
plt.plot([0, 1], [0, 1], "--", label="Random")
plt.xlabel("False-positive rate"); plt.ylabel("True-positive rate"); plt.title("Image-level ROC")
plt.legend(); plt.grid(alpha=0.25); plt.show()

plt.figure(figsize=(7, 5))
for name, output in outputs.items():
    labels, scores = output["result"]["labels"], output["result"]["scores"]
    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.plot(recall, precision, label=f"{name} (AP={average_precision_score(labels, scores):.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Image-level precision-recall")
plt.legend(); plt.grid(alpha=0.25); plt.show()

for name, output in outputs.items():
    frame = output["frame"]
    plt.figure(figsize=(8, 4))
    plt.hist(frame.loc[frame.label == 0, "score"], bins=20, alpha=0.65, label="Normal")
    plt.hist(frame.loc[frame.label == 1, "score"], bins=20, alpha=0.65, label="Anomaly")
    plt.axvline(output["threshold"], linestyle="--", label="Validation-normal threshold")
    plt.xlabel("Anomaly score"); plt.ylabel("Images"); plt.title(name); plt.legend(); plt.show()

# %% [markdown]
# ## 7. Visualize anomaly heatmaps
#
# Heatmaps are normalized per image only for display. Raw values are used for metrics.

# %%
def normalize_for_display(anomaly_map: np.ndarray) -> np.ndarray:
    low, high = np.percentile(anomaly_map, [1, 99])
    return np.zeros_like(anomaly_map) if high <= low else np.clip((anomaly_map - low) / (high - low), 0, 1)


def show_predictions(method: str, normal_count=2, anomaly_count=4):
    output = outputs[method]
    result, predictions = output["result"], output["pred"]
    indices = np.concatenate([
        np.where(result["labels"] == 0)[0][:normal_count],
        np.where(result["labels"] == 1)[0][:anomaly_count],
    ])
    fig, axes = plt.subplots(len(indices), 4, figsize=(15, 4 * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, 0)
    for row_index, index in enumerate(indices):
        row = test_df.iloc[index]
        image = load_rgb(row.path)
        mask = load_mask(row.mask_path, image.shape[:2])
        amap = normalize_for_display(result["maps"][index])
        amap = np.asarray(Image.fromarray((amap * 255).astype(np.uint8)).resize(
            (image.shape[1], image.shape[0]), Image.BILINEAR
        )) / 255.0
        truth = "anomaly" if row.label else "normal"
        pred = "anomaly" if predictions[index] else "normal"
        axes[row_index, 0].imshow(image); axes[row_index, 0].set_title(f"{row.defect_type}\ntruth={truth}, pred={pred}")
        axes[row_index, 1].imshow(mask, cmap="gray"); axes[row_index, 1].set_title("Ground truth")
        axes[row_index, 2].imshow(amap, cmap="inferno"); axes[row_index, 2].set_title(f"Heatmap\nscore={result['scores'][index]:.4f}")
        axes[row_index, 3].imshow(image); axes[row_index, 3].imshow(amap, cmap="inferno", alpha=0.45); axes[row_index, 3].set_title("Overlay")
        for col in range(4): axes[row_index, col].axis("off")
    fig.suptitle(method, fontsize=16, y=1.01); plt.tight_layout(); plt.show()


for method in outputs:
    show_predictions(method)

# %% [markdown]
# ## 8. Per-defect performance
#
# For anomalous defect groups, `detection_rate` is recall for that defect type. For `good`, it is the false-positive rate.

# %%
for name, output in outputs.items():
    table = output["frame"].groupby("defect_type").agg(
        images=("path", "count"),
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        detection_rate=("prediction", "mean"),
    ).sort_values("mean_score", ascending=False)
    print("\n", name)
    display(table.round(4))

# %% [markdown]
# ## 9. Save reproducible results

# %%
output_dir = Path(CFG.output_dir)
metrics_df.reset_index().to_csv(output_dir / "method_comparison_metrics.csv", index=False)
for name, output in outputs.items():
    safe = name.lower().replace(" ", "_").replace("-", "_")
    output["frame"].to_csv(output_dir / f"{safe}_test_scores.csv", index=False)

summary = {
    "category": CFG.category,
    "category_root": str(CATEGORY_ROOT),
    "device": str(DEVICE),
    "seed": SEED,
    "configuration": CFG.__dict__,
    "metrics": metrics_df.reset_index().to_dict(orient="records"),
}
with (output_dir / "run_summary.json").open("w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

print("Saved to:", output_dir)
for path in sorted(output_dir.iterdir()):
    print("-", path.name)

# %% [markdown]
# # What this notebook covers
#
# You should now understand:
#
# - normal-only training;
# - image-level detection versus pixel localization;
# - reconstruction error;
# - pretrained feature distance;
# - validation-based thresholding;
# - test leakage;
# - precision, anomaly recall, normal recall, F1, balanced accuracy, ROC-AUC, and average precision;
# - confusion matrices and defect-specific error analysis;
# - saving scores, metrics, model state, and configuration.
#
# ## Exercises
#
# 1. Change `bottle` to another category.
# 2. Compare threshold quantiles `0.95`, `0.975`, and `0.99`.
# 3. Compare maximum, top-1%, and whole-map-mean image scores.
# 4. Repeat with three random seeds and report mean ± standard deviation.
# 5. Replace the simplified feature model with full PaDiM or PatchCore.
# 6. Wrap dataset inspection, training, evaluation, critique, and reporting as tools for an agentic-AI system.
#
# ## Limitations
#
# - The autoencoder is intentionally small.
# - The feature method uses diagonal variance rather than full covariance.
# - Pixel metrics may use a reproducible sample for speed.
# - One category is not enough for a publication claim.
# - Research-grade work should evaluate all categories, stronger baselines, repeated runs, runtime, memory, and ablations.
