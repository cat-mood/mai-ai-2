from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build concise lab report from model metrics")
    parser.add_argument("--resnet-metrics", type=Path, required=True)
    parser.add_argument("--vit-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/report.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resnet = load_metrics(args.resnet_metrics)
    vit = load_metrics(args.vit_metrics)

    rows = []
    for model_name, payload in [("resnet18", resnet), ("vit_b_16", vit)]:
        metrics = payload["test_metrics"]
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics.get("accuracy", 0.0),
                "macro_f1": metrics.get("macro_f1", 0.0),
                "weighted_f1": metrics.get("weighted_f1", 0.0),
                "top5_accuracy": metrics.get("top5_accuracy", 0.0),
                "inference_time_per_image_sec": metrics.get("inference_time_per_image_sec", 0.0),
            }
        )

    best = max(rows, key=lambda x: x["macro_f1"])
    output = [
        "# Лабораторная работа 1: baseline CNN vs Transformer",
        "",
        "## Сравнение моделей",
        "",
        "| Модель | Accuracy | Macro F1 | Weighted F1 | Top-5 Accuracy | Inference time (sec/image) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        output.append(
            "| {model} | {accuracy:.4f} | {macro_f1:.4f} | {weighted_f1:.4f} | {top5_accuracy:.4f} | "
            "{inference_time_per_image_sec:.6f} |".format(**row)
        )

    output.extend(
        [
            "",
            "## Вывод",
            "",
            (
                f"По метрике Macro F1 лучшей моделью стала `{best['model']}` "
                f"({best['macro_f1']:.4f})."
            ),
            "Для улучшения baseline можно попробовать tuning scheduler/lr, class weights и аугментации.",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output), encoding="utf-8")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()

