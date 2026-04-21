# Лабораторная работа 1: baseline CNN vs Transformer

## Сравнение моделей

| Модель | Accuracy | Macro F1 | Weighted F1 | Top-5 Accuracy | Inference time (sec/image) |
|---|---:|---:|---:|---:|---:|
| resnet18 | 0.9292 | 0.9284 | 0.9292 | N/A | 0.001156 |
| vit_b_16 | 0.8778 | 0.8743 | 0.8767 | N/A | 0.011309 |

## Улучшенный baseline: сравнение с пунктом 2

| Модель | Macro F1 (baseline) | Macro F1 (improved) | Delta Macro F1 | Accuracy (baseline) | Accuracy (improved) | Delta Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| resnet18 | 0.9284 | 0.9327 | +0.0043 | 0.9292 | 0.9343 | +0.0052 |
| vit_b_16 | 0.8743 | 0.9676 | +0.0933 | 0.8778 | 0.9682 | +0.0903 |

## Проверка гипотез

- H1 (сильные аугментации + label smoothing для ResNet): подтверждена, delta Macro F1 = +0.0043.
- H2 (class weights): частично подтверждена, на ResNet дала прирост, на ViT прирост минимальный относительно ViT-opt.
- H3 (ViT: меньший lr + warmup + больше эпох): подтверждена, delta Macro F1 = +0.0933.
- H4 (устойчивость): подтверждена на full test-eval для выбранных improved-конфигов.

## Имплементация моделей: сравнение с пунктом 2 и 3

### Custom baseline vs пункт 2 (torchvision baseline)

| Модель | Macro F1 (custom) | Macro F1 (п.2) | Delta | Accuracy (custom) | Accuracy (п.2) | Delta |
|---|---:|---:|---:|---:|---:|---:|
| custom_cnn vs resnet18 | 0.8794 | 0.9284 | -0.0489 | 0.8818 | 0.9292 | -0.0474 |
| custom_vit vs vit_b_16 | 0.8761 | 0.8743 | +0.0019 | 0.8802 | 0.8778 | +0.0024 |

### Custom improved vs пункт 3 (improved torchvision)

| Модель | Macro F1 (custom improved) | Macro F1 (п.3) | Delta | Accuracy (custom improved) | Accuracy (п.3) | Delta |
|---|---:|---:|---:|---:|---:|---:|
| custom_cnn improved vs resnet18 improved | 0.8809 | 0.9327 | -0.0518 | 0.8854 | 0.9343 | -0.0489 |
| custom_vit improved vs vit_b_16 improved | 0.8808 | 0.9676 | -0.0868 | 0.8842 | 0.9682 | -0.0840 |

## Вывод

По метрике Macro F1 лучшей моделью стала `vit_b_16` (0.9676, improved).
- Самостоятельно имплементированные модели обучаются и дают стабильное качество на waste, но в текущей конфигурации уступают tuned torchvision improved baseline.
- Техники improved baseline для custom моделей дали небольшой прирост для `custom_cnn` и умеренный прирост для `custom_vit` относительно custom baseline.
- Для практического использования на Mac (`mps`) при ограниченном времени оптимальным остается `resnet18`, а для максимального качества — tuned `vit_b_16` из пункта 3.

Приложение (воспроизводимые конфигурации экспериментов): [experiments/improved_baseline_configs.md](experiments/improved_baseline_configs.md).