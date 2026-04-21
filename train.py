from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.data import class_distribution, create_waste_dataloaders
from src.engine import run_eval_epoch, run_train_epoch
from src.models import build_model
from src.utils import get_device, save_json, set_seed

MODEL_CHOICES = ["resnet18", "vit_b_16", "custom_cnn", "custom_vit"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train waste binary classifier baselines")
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("waste/DATASET"))
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--aug-profile", choices=["baseline", "strong"], default="baseline")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scheduler", choices=["cosine", "cosine_warmup"], default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Example: cuda, mps, cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    use_amp = device.type == "cuda"

    train_loader, val_loader, test_loader, class_to_idx, dataset_info = create_waste_dataloaders(
        model_name=args.model,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        pin_memory=device.type == "cuda",
        aug_profile=args.aug_profile,
    )
    dist = class_distribution(train_loader.dataset)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model(args.model, num_classes=len(class_to_idx), pretrained=args.pretrained).to(device)
    class_weights_tensor = None
    if args.use_class_weights:
        counts = [float(dist.get(class_idx, 1)) for class_idx in range(len(class_to_idx))]
        total = sum(counts)
        weights = [total / (len(counts) * c) for c in counts]
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        if args.warmup_epochs < 1:
            raise ValueError("--warmup-epochs must be >= 1 when --scheduler cosine_warmup is used.")

        def lr_lambda(epoch_idx: int) -> float:
            current_epoch = epoch_idx + 1
            if current_epoch <= args.warmup_epochs:
                return current_epoch / args.warmup_epochs
            return 1.0

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - args.warmup_epochs),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs],
        )
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    run_dir = args.output_dir / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(class_to_idx, run_dir / "class_to_idx.json")
    save_json(
        {
            "train_distribution": dist,
            "device": str(device),
            "task_type": "binary" if len(class_to_idx) == 2 else "multiclass",
            "aug_profile": args.aug_profile,
            "label_smoothing": args.label_smoothing,
            "use_class_weights": args.use_class_weights,
            "scheduler": args.scheduler,
            "warmup_epochs": args.warmup_epochs,
            **dataset_info,
        },
        run_dir / "run_info.json",
    )

    history: list[dict[str, float | int]] = []
    best_val_macro_f1 = -1.0
    patience_counter = 0
    best_ckpt_path = run_dir / "best.ckpt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            max_batches=args.max_train_batches,
        )
        val_metrics, _, _, _ = run_eval_epoch(
            model,
            val_loader,
            criterion,
            device,
            max_batches=args.max_eval_batches,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_top5_accuracy": val_metrics["top5_accuracy"],
        }
        history.append(row)
        print(
            f"[{args.model}] epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            torch.save(
                {
                    "model_name": args.model,
                    "state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics, _, _, _ = run_eval_epoch(
        model,
        test_loader,
        criterion,
        device,
        max_batches=args.max_eval_batches,
    )

    save_json({"history": history}, run_dir / "history.json")
    save_json(
        {
            "model": args.model,
            "device": str(device),
            "num_classes": len(class_to_idx),
            "best_val_macro_f1": best_val_macro_f1,
            "test_metrics": test_metrics,
            "best_epoch": checkpoint["epoch"],
            "idx_to_class": idx_to_class,
        },
        run_dir / "metrics.json",
    )
    print(f"Saved artifacts to: {run_dir}")


if __name__ == "__main__":
    main()

