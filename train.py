from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.data import PathRemapConfig, class_distribution, create_dataloaders
from src.engine import run_eval_epoch, run_train_epoch
from src.models import build_model
from src.utils import get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mushroom classifier baseline models")
    parser.add_argument("--model", choices=["resnet18", "vit_b_16"], required=True)
    parser.add_argument("--train-csv", type=Path, default=Path("mushroom1/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("mushroom1/val.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("mushroom1/test.csv"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--path-prefix-from", type=str, default=None)
    parser.add_argument("--path-prefix-to", type=str, default=None)
    parser.add_argument("--strict-paths", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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

    remap = PathRemapConfig(
        data_root=args.data_root,
        path_prefix_from=args.path_prefix_from,
        path_prefix_to=args.path_prefix_to,
    )
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        model_name=args.model,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        remap=remap,
        strict_paths=args.strict_paths,
        pin_memory=device.type == "cuda",
    )
    dist = class_distribution(train_loader.dataset)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model(args.model, num_classes=len(class_to_idx), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    run_dir = args.output_dir / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(class_to_idx, run_dir / "class_to_idx.json")
    save_json({"train_distribution": dist, "device": str(device)}, run_dir / "run_info.json")

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

