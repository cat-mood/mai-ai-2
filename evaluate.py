from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn

from src.data import create_waste_dataloaders
from src.engine import measure_inference_time, run_eval_epoch
from src.models import build_model
from src.utils import get_device, save_json

MODEL_CHOICES = ["resnet18", "vit_b_16", "custom_cnn", "custom_vit"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained waste baseline model")
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("waste/DATASET"))
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_to_idx: dict[str, int] = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    _, _, test_loader, _, dataset_info = create_waste_dataloaders(
        model_name=args.model,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model, num_classes=len(class_to_idx), pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    criterion = nn.CrossEntropyLoss()

    metrics, targets, preds, probs = run_eval_epoch(
        model,
        test_loader,
        criterion,
        device,
        max_batches=args.max_eval_batches,
    )
    metrics["inference_time_per_image_sec"] = measure_inference_time(model, test_loader, device)

    out_dir = args.output_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(targets, preds, labels=list(range(len(class_to_idx))))
    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[idx_to_class[i] for i in range(len(class_to_idx))])
    disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    predictions_path = out_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["sample_idx", "target_idx", "target_label", "pred_idx", "pred_label", "confidence"])
        for idx, (target, pred) in enumerate(zip(targets, preds)):
            confidence = float(np.max(probs[idx])) if probs.size else 0.0
            writer.writerow([idx, target, idx_to_class[target], pred, idx_to_class[pred], f"{confidence:.6f}"])

    save_json(
        {
            "model": args.model,
            "device": str(device),
            "checkpoint": str(args.checkpoint),
            "test_metrics": metrics,
            **dataset_info,
        },
        out_dir / "metrics.json",
    )
    print(f"Saved evaluation artifacts to: {out_dir}")


if __name__ == "__main__":
    main()

