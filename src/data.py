from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from torch.utils.data import DataLoader, Dataset
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
                "Check --data-root and/or --path-prefix-from/--path-prefix-to."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, target, raw_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target, str(image_path), raw_path


def build_transforms(model_name: str) -> tuple[Callable, Callable]:
    input_size = 224
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    # Keep API explicit for possible model-specific transforms later.
    if model_name in {"resnet18", "vit_b_16"}:
        return train_transform, eval_transform
    raise ValueError(f"Unsupported model_name: {model_name}")


def create_dataloaders(
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
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    train_transform, eval_transform = build_transforms(model_name)
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


def class_distribution(dataset: Dataset) -> dict[int, int]:
    dist: dict[int, int] = {}
    samples = getattr(dataset, "samples", [])
    for _, label_idx, _ in samples:
        dist[label_idx] = dist.get(label_idx, 0) + 1
    return dist

