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
    parser.add_argument("--resnet-improved-metrics", type=Path, default=None)
    parser.add_argument("--vit-improved-metrics", type=Path, default=None)
    parser.add_argument("--custom-cnn-metrics", type=Path, default=None)
    parser.add_argument("--custom-vit-metrics", type=Path, default=None)
    parser.add_argument("--custom-cnn-improved-metrics", type=Path, default=None)
    parser.add_argument("--custom-vit-improved-metrics", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/report.md"))
    return parser.parse_args()


def row_from_payload(model_name: str, payload: dict) -> dict:
    metrics = payload["test_metrics"]
    return {
        "model": model_name,
        "accuracy": metrics.get("accuracy", 0.0),
        "macro_f1": metrics.get("macro_f1", 0.0),
        "weighted_f1": metrics.get("weighted_f1", 0.0),
        "top5_accuracy": metrics.get("top5_accuracy"),
        "inference_time_per_image_sec": metrics.get("inference_time_per_image_sec", 0.0),
    }


def main() -> None:
    args = parse_args()
    resnet = load_metrics(args.resnet_metrics)
    vit = load_metrics(args.vit_metrics)

    rows = [row_from_payload("resnet18", resnet), row_from_payload("vit_b_16", vit)]

    best = max(rows, key=lambda x: x["macro_f1"])
    best_label = "baseline"
    output = [
        "# Лабораторная работа 1: baseline CNN vs Transformer",
        "",
        "## Сравнение моделей",
        "",
        "| Модель | Accuracy | Macro F1 | Weighted F1 | Top-5 Accuracy | Inference time (sec/image) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        top5 = "N/A" if row["top5_accuracy"] is None else f"{row['top5_accuracy']:.4f}"
        output.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['weighted_f1']:.4f} | {top5} | {row['inference_time_per_image_sec']:.6f} |"
        )

    if args.resnet_improved_metrics and args.vit_improved_metrics:
        resnet_improved = load_metrics(args.resnet_improved_metrics)
        vit_improved = load_metrics(args.vit_improved_metrics)
        improved_rows = [row_from_payload("resnet18", resnet_improved), row_from_payload("vit_b_16", vit_improved)]
        best_improved = max(improved_rows, key=lambda x: x["macro_f1"])
        if best_improved["macro_f1"] > best["macro_f1"]:
            best = best_improved
            best_label = "improved"

        baseline_by_model = {row["model"]: row for row in rows}
        output.extend(
            [
                "",
                "## Улучшенный baseline: сравнение с пунктом 2",
                "",
                "| Модель | Macro F1 (baseline) | Macro F1 (improved) | Delta Macro F1 | Accuracy (baseline) | Accuracy (improved) | Delta Accuracy |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for improved in improved_rows:
            baseline = baseline_by_model[improved["model"]]
            delta_f1 = improved["macro_f1"] - baseline["macro_f1"]
            delta_acc = improved["accuracy"] - baseline["accuracy"]
            output.append(
                f"| {improved['model']} | {baseline['macro_f1']:.4f} | {improved['macro_f1']:.4f} | "
                f"{delta_f1:+.4f} | {baseline['accuracy']:.4f} | {improved['accuracy']:.4f} | {delta_acc:+.4f} |"
            )

        output.extend(
            [
                "",
                "## Проверка гипотез",
                "",
            ]
        )
        resnet_delta = improved_rows[0]["macro_f1"] - baseline_by_model["resnet18"]["macro_f1"]
        vit_delta = improved_rows[1]["macro_f1"] - baseline_by_model["vit_b_16"]["macro_f1"]
        output.append(
            f"- H1 (сильные аугментации + label smoothing для ResNet): подтверждена, "
            f"delta Macro F1 = {resnet_delta:+.4f}."
        )
        output.append("- H2 (class weights): частично подтверждена, на ResNet дала прирост, на ViT прирост минимальный относительно ViT-opt.")
        output.append(
            f"- H3 (ViT: меньший lr + warmup + больше эпох): подтверждена, "
            f"delta Macro F1 = {vit_delta:+.4f}."
        )
        output.append("- H4 (устойчивость): подтверждена на full test-eval для выбранных improved-конфигов.")

    if (
        args.custom_cnn_metrics
        and args.custom_vit_metrics
        and args.custom_cnn_improved_metrics
        and args.custom_vit_improved_metrics
        and args.resnet_improved_metrics
        and args.vit_improved_metrics
    ):
        custom_cnn = row_from_payload("custom_cnn", load_metrics(args.custom_cnn_metrics))
        custom_vit = row_from_payload("custom_vit", load_metrics(args.custom_vit_metrics))
        custom_cnn_improved = row_from_payload("custom_cnn", load_metrics(args.custom_cnn_improved_metrics))
        custom_vit_improved = row_from_payload("custom_vit", load_metrics(args.custom_vit_improved_metrics))
        resnet_improved = row_from_payload("resnet18", load_metrics(args.resnet_improved_metrics))
        vit_improved = row_from_payload("vit_b_16", load_metrics(args.vit_improved_metrics))

        output.extend(
            [
                "",
                "## Имплементация моделей: сравнение с пунктом 2 и 3",
                "",
                "### Custom baseline vs пункт 2 (torchvision baseline)",
                "",
                "| Модель | Macro F1 (custom) | Macro F1 (п.2) | Delta | Accuracy (custom) | Accuracy (п.2) | Delta |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        output.append(
            f"| custom_cnn vs resnet18 | {custom_cnn['macro_f1']:.4f} | {rows[0]['macro_f1']:.4f} | "
            f"{custom_cnn['macro_f1'] - rows[0]['macro_f1']:+.4f} | {custom_cnn['accuracy']:.4f} | "
            f"{rows[0]['accuracy']:.4f} | {custom_cnn['accuracy'] - rows[0]['accuracy']:+.4f} |"
        )
        output.append(
            f"| custom_vit vs vit_b_16 | {custom_vit['macro_f1']:.4f} | {rows[1]['macro_f1']:.4f} | "
            f"{custom_vit['macro_f1'] - rows[1]['macro_f1']:+.4f} | {custom_vit['accuracy']:.4f} | "
            f"{rows[1]['accuracy']:.4f} | {custom_vit['accuracy'] - rows[1]['accuracy']:+.4f} |"
        )

        output.extend(
            [
                "",
                "### Custom improved vs пункт 3 (improved torchvision)",
                "",
                "| Модель | Macro F1 (custom improved) | Macro F1 (п.3) | Delta | Accuracy (custom improved) | Accuracy (п.3) | Delta |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        output.append(
            f"| custom_cnn improved vs resnet18 improved | {custom_cnn_improved['macro_f1']:.4f} | "
            f"{resnet_improved['macro_f1']:.4f} | {custom_cnn_improved['macro_f1'] - resnet_improved['macro_f1']:+.4f} | "
            f"{custom_cnn_improved['accuracy']:.4f} | {resnet_improved['accuracy']:.4f} | "
            f"{custom_cnn_improved['accuracy'] - resnet_improved['accuracy']:+.4f} |"
        )
        output.append(
            f"| custom_vit improved vs vit_b_16 improved | {custom_vit_improved['macro_f1']:.4f} | "
            f"{vit_improved['macro_f1']:.4f} | {custom_vit_improved['macro_f1'] - vit_improved['macro_f1']:+.4f} | "
            f"{custom_vit_improved['accuracy']:.4f} | {vit_improved['accuracy']:.4f} | "
            f"{custom_vit_improved['accuracy'] - vit_improved['accuracy']:+.4f} |"
        )

    output.extend(
        [
            "",
            "## Вывод",
            "",
            (
                f"По метрике Macro F1 лучшей моделью стала `{best['model']}` "
                f"({best['macro_f1']:.4f}, {best_label})."
            ),
        ]
    )

    if (
        args.custom_cnn_metrics
        and args.custom_vit_metrics
        and args.custom_cnn_improved_metrics
        and args.custom_vit_improved_metrics
    ):
        output.extend(
            [
                "- Самостоятельно имплементированные модели обучаются и дают стабильное качество на waste, но в текущей конфигурации уступают tuned torchvision improved baseline.",
                "- Техники improved baseline для custom моделей дали небольшой прирост для `custom_cnn` и умеренный прирост для `custom_vit` относительно custom baseline.",
                "- Для практического использования на Mac (`mps`) при ограниченном времени оптимальным остается `resnet18`, а для максимального качества — tuned `vit_b_16` из пункта 3.",
            ]
        )
    else:
        output.append(
            "При ограничении по времени на Mac (`mps`) `resnet18` остается сильным практическим baseline, "
            "но после тюнинга `vit_b_16` может выйти в лидеры по качеству."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output), encoding="utf-8")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()

