from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PathRemapConfig:
    data_root: Path | None = None
    path_prefix_from: str | None = None
    path_prefix_to: str | None = None


def _resolve_path(raw_path: str, label: str, remap: PathRemapConfig) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    rewritten = raw_path
    if remap.path_prefix_from and remap.path_prefix_to and raw_path.startswith(remap.path_prefix_from):
        rewritten = remap.path_prefix_to + raw_path[len(remap.path_prefix_from) :]
        candidate = Path(rewritten)
        if candidate.exists():
            return candidate

    if remap.data_root:
        parts = Path(rewritten).parts
        if "merged_dataset" in parts:
            rel_idx = parts.index("merged_dataset") + 1
            relative_path = Path(*parts[rel_idx:])
            candidate = remap.data_root / relative_path
        else:
            candidate = remap.data_root / label / Path(rewritten).name
    return candidate


class MushroomCsvDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        class_to_idx: dict[str, int] | None = None,
        transform: Callable | None = None,
        remap: PathRemapConfig | None = None,
        strict_paths: bool = True,
    ) -> None:
        self.csv_path = csv_path
        self.transform = transform
        self.samples: list[tuple[Path, int, str]] = []
        self.remap = remap or PathRemapConfig()
        rows: list[tuple[str, str]] = []
        with csv_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                rows.append((row["image_path"], row["label"]))

        if class_to_idx is None:
            classes = sorted({label for _, label in rows})
            self.class_to_idx = {label: idx for idx, label in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        missing_count = 0
        for raw_path, label in rows:
            resolved = _resolve_path(raw_path, label, self.remap)
            if strict_paths and not resolved.exists():
                missing_count += 1
                continue
            label_idx = self.class_to_idx[label]
            self.samples.append((resolved, label_idx, raw_path))

        if strict_paths and missing_count > 0:
            raise FileNotFoundError(
                f"{csv_path} has {missing_count} missing files after remapping. "
                "Check your path remapping settings."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, target, raw_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target, str(image_path), raw_path


def build_transforms(model_name: str, aug_profile: str = "baseline") -> tuple[Callable, Callable]:
    input_size = 224
    if aug_profile == "strong":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            ]
        )
    elif aug_profile == "baseline":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    else:
        raise ValueError(f"Unsupported aug_profile: {aug_profile}")
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    # Keep API explicit for possible model-specific transforms later.
    if model_name in {"resnet18", "vit_b_16", "custom_cnn", "custom_vit"}:
        return train_transform, eval_transform
    raise ValueError(f"Unsupported model_name: {model_name}")


def create_csv_dataloaders(
    *,
    model_name: str,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    batch_size: int,
    num_workers: int,
    remap: PathRemapConfig,
    strict_paths: bool = True,
    pin_memory: bool = False,
    aug_profile: str = "baseline",
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    train_transform, eval_transform = build_transforms(model_name, aug_profile=aug_profile)
    train_dataset = MushroomCsvDataset(train_csv, transform=train_transform, remap=remap, strict_paths=strict_paths)
    val_dataset = MushroomCsvDataset(
        val_csv,
        class_to_idx=train_dataset.class_to_idx,
        transform=eval_transform,
        remap=remap,
        strict_paths=strict_paths,
    )
    test_dataset = MushroomCsvDataset(
        test_csv,
        class_to_idx=train_dataset.class_to_idx,
        transform=eval_transform,
        remap=remap,
        strict_paths=strict_paths,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx


def resolve_waste_root(dataset_root: Path) -> Path:
    candidates = [
        dataset_root,
        dataset_root / "DATASET",
        dataset_root / "DATASET" / "DATASET",
    ]
    for candidate in candidates:
        train_dir = candidate / "TRAIN"
        test_dir = candidate / "TEST"
        if train_dir.exists() and test_dir.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find TRAIN/TEST folders under {dataset_root}. "
        "Expected either <root>/TRAIN and <root>/TEST or nested DATASET/."
    )


def create_waste_dataloaders(
    *,
    model_name: str,
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
    val_split: float = 0.1,
    seed: int = 42,
    pin_memory: bool = False,
    aug_profile: str = "baseline",
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int], dict[str, int | str]]:
    if not (0.0 < val_split < 1.0):
        raise ValueError("--val-split must be between 0 and 1.")

    resolved_root = resolve_waste_root(dataset_root)
    train_transform, eval_transform = build_transforms(model_name, aug_profile=aug_profile)

    train_base = ImageFolder(resolved_root / "TRAIN", transform=train_transform)
    eval_base = ImageFolder(resolved_root / "TRAIN", transform=eval_transform)
    test_base = ImageFolder(resolved_root / "TEST", transform=eval_transform)

    if train_base.class_to_idx != test_base.class_to_idx:
        raise ValueError("Class mapping mismatch between TRAIN and TEST folders.")

    total_train = len(train_base)
    val_size = max(1, int(total_train * val_split))
    train_size = total_train - val_size
    if train_size <= 0:
        raise ValueError("Validation split is too large for current train set.")

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(total_train, generator=generator).tolist()
    val_indices = shuffled_indices[:val_size]
    train_indices = shuffled_indices[val_size:]

    train_dataset = Subset(train_base, train_indices)
    val_dataset = Subset(eval_base, val_indices)
    test_dataset = test_base

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    info = {
        "resolved_dataset_root": str(resolved_root),
        "train_samples": train_size,
        "val_samples": val_size,
        "test_samples": len(test_dataset),
    }
    return train_loader, val_loader, test_loader, train_base.class_to_idx, info


def class_distribution(dataset: Dataset) -> dict[int, int]:
    dist: dict[int, int] = {}
    if isinstance(dataset, Subset):
        base = dataset.dataset
        if hasattr(base, "targets"):
            for idx in dataset.indices:
                label_idx = int(base.targets[idx])
                dist[label_idx] = dist.get(label_idx, 0) + 1
        return dist

    samples = getattr(dataset, "samples", None)
    if samples is not None:
        for sample in samples:
            # CSV dataset sample format: (path, label_idx, raw_path)
            # ImageFolder sample format: (path, label_idx)
            label_idx = int(sample[1])
            dist[label_idx] = dist.get(label_idx, 0) + 1
    return dist

