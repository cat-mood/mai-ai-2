# Классификация изображений

Датасет: https://www.kaggle.com/datasets/techsash/waste-classification-data

Основной отчет: [report.md](report.md)

Работу выполнил Голубев Тимофей Дмитриевич

Группа М8О-406Б-22

## Бинарная классификация отходов (ResNet18 vs ViT-B/16)

### 1) Установка зависимостей

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Загрузка и подготовка датасета

Вариант A (рекомендуется): через `curl`.

Выполните команды:

```bash
mkdir -p waste
curl -L \
  "https://www.kaggle.com/api/v1/datasets/download/techsash/waste-classification-data" \
  -o waste/waste-classification-data.zip
unzip -o waste/waste-classification-data.zip -d waste
```

Вариант B: скачать архив вручную по ссылке и распаковать в папку `waste/`.

После распаковки ожидается структура:

Если папка вложена на один уровень глубже, она определяется автоматически:

```text
waste/DATASET/
  TRAIN/
    O/
    R/
  TEST/
    O/
    R/
```

Классы в задаче:

- `R` -> перерабатываемые отходы
- `O` -> неперерабатываемые/органические отходы

### 3) Обучение baseline-моделей

ResNet18:

```bash
python train.py \
  --model resnet18 \
  --dataset-root "waste/DATASET" \
  --epochs 10 \
  --batch-size 64 \
  --output-dir artifacts
```

ViT-B/16:

```bash
python train.py \
  --model vit_b_16 \
  --dataset-root "waste/DATASET" \
  --epochs 10 \
  --batch-size 32 \
  --output-dir artifacts
```

Устройство выбирается автоматически. При необходимости можно принудительно задать backend через `--device`.

Если нужна инициализация ImageNet-весами и есть доступ к интернету, добавьте `--pretrained`.

Для быстрого smoke-прогона (несколько батчей) добавьте:

```bash
--max-train-batches 20 --max-eval-batches 5 --num-workers 0
```

Валидационная выборка берется из `TRAIN` (по умолчанию `--val-split 0.1`).

### 4) Оценка и сохранение артефактов

```bash
python evaluate.py \
  --model resnet18 \
  --checkpoint artifacts/resnet18/best.ckpt \
  --dataset-root "waste/DATASET" \
  --output-dir artifacts
```

Артефакты по каждой модели сохраняются в `artifacts/<model>/`:

- `best.ckpt`
- `metrics.json`
- `predictions.csv`
- `confusion_matrix.png`
- `run_info.json` (определенный путь к датасету и размеры сплитов)

### 5) Сборка краткого отчета

```bash
python make_report.py \
  --resnet-metrics artifacts/resnet18/metrics.json \
  --vit-metrics artifacts/vit_b_16/metrics.json \
  --output artifacts/report.md
```

Для сравнения baseline и improved:

```bash
python make_report.py \
  --resnet-metrics artifacts/resnet18/metrics.json \
  --vit-metrics artifacts/vit_b_16/metrics.json \
  --resnet-improved-metrics artifacts_improved/final_resnet/resnet18/metrics.json \
  --vit-improved-metrics artifacts_improved/final_vit/vit_b_16/metrics.json \
  --output artifacts/report.md
```

## Параметры improved baseline

`train.py` поддерживает следующие опции для проверки гипотез:

- `--aug-profile baseline|strong`
- `--label-smoothing <float>`
- `--use-class-weights`
- `--scheduler cosine|cosine_warmup`
- `--warmup-epochs <int>`

Воспроизводимые команды экспериментов вынесены в:
`experiments/improved_baseline_configs.md`.

## Custom-модели (самостоятельная реализация)

Поддерживаемые имена custom-моделей:

- `custom_cnn`
- `custom_vit`

Примеры обучения custom baseline:

```bash
python train.py \
  --model custom_cnn \
  --dataset-root "waste/DATASET" \
  --epochs 5 \
  --batch-size 64 \
  --output-dir artifacts_custom/baseline
```

```bash
python train.py \
  --model custom_vit \
  --dataset-root "waste/DATASET" \
  --epochs 4 \
  --batch-size 32 \
  --output-dir artifacts_custom/baseline
```

Примеры обучения custom improved:

```bash
python train.py \
  --model custom_cnn \
  --dataset-root "waste/DATASET" \
  --aug-profile strong \
  --label-smoothing 0.05 \
  --use-class-weights \
  --scheduler cosine_warmup \
  --warmup-epochs 1 \
  --output-dir artifacts_custom/improved
```

```bash
python train.py \
  --model custom_vit \
  --dataset-root "waste/DATASET" \
  --aug-profile strong \
  --label-smoothing 0.05 \
  --use-class-weights \
  --scheduler cosine_warmup \
  --warmup-epochs 1 \
  --output-dir artifacts_custom/improved
```

Сборка полного сравнительного отчета (п.2, п.3, custom):

```bash
python make_report.py \
  --resnet-metrics artifacts/resnet18/metrics.json \
  --vit-metrics artifacts/vit_b_16/metrics.json \
  --resnet-improved-metrics artifacts_improved/final_resnet/resnet18/metrics.json \
  --vit-improved-metrics artifacts_improved/final_vit/vit_b_16/metrics.json \
  --custom-cnn-metrics artifacts_custom/baseline/custom_cnn/metrics.json \
  --custom-vit-metrics artifacts_custom/baseline/custom_vit/metrics.json \
  --custom-cnn-improved-metrics artifacts_custom/improved/custom_cnn/metrics.json \
  --custom-vit-improved-metrics artifacts_custom/improved/custom_vit/metrics.json \
  --output artifacts/report.md
```

## Воспроизводимый запуск (end-to-end)

Используйте команды ниже без изменений, чтобы воспроизвести тот же пайплайн и структуру артефактов.
Во всех запусках зафиксированы `--seed 42` и `--num-workers 0`.

```bash
source .venv/bin/activate
export TORCH_HOME="$(pwd)/.cache/torch"
export MPLCONFIGDIR="$(pwd)/.cache/matplotlib"
export XDG_CACHE_HOME="$(pwd)/.cache"

# 1) Baseline torchvision (пункт 2)
python train.py --model resnet18 --dataset-root "waste/DATASET" --epochs 10 --batch-size 64 --lr 3e-4 --seed 42 --num-workers 0 --output-dir artifacts
python train.py --model vit_b_16 --dataset-root "waste/DATASET" --epochs 10 --batch-size 32 --lr 3e-4 --seed 42 --num-workers 0 --output-dir artifacts
python evaluate.py --model resnet18 --checkpoint artifacts/resnet18/best.ckpt --dataset-root "waste/DATASET" --val-split 0.1 --seed 42 --num-workers 0 --output-dir artifacts
python evaluate.py --model vit_b_16 --checkpoint artifacts/vit_b_16/best.ckpt --dataset-root "waste/DATASET" --val-split 0.1 --seed 42 --num-workers 0 --output-dir artifacts

# 2) Improved torchvision baseline (пункт 3)
python train.py --model resnet18 --dataset-root "waste/DATASET" --epochs 10 --batch-size 64 --lr 3e-4 --aug-profile strong --label-smoothing 0.05 --use-class-weights --scheduler cosine_warmup --warmup-epochs 1 --seed 42 --num-workers 0 --output-dir artifacts_improved/final_resnet
python train.py --model vit_b_16 --dataset-root "waste/DATASET" --epochs 10 --batch-size 32 --lr 1e-4 --aug-profile strong --label-smoothing 0.05 --use-class-weights --scheduler cosine_warmup --warmup-epochs 1 --seed 42 --num-workers 0 --output-dir artifacts_improved/final_vit
python evaluate.py --model resnet18 --checkpoint artifacts_improved/final_resnet/resnet18/best.ckpt --dataset-root "waste/DATASET" --val-split 0.1 --seed 42 --num-workers 0 --output-dir artifacts_improved/final_resnet
python evaluate.py --model vit_b_16 --checkpoint artifacts_improved/final_vit/vit_b_16/best.ckpt --dataset-root "waste/DATASET" --val-split 0.1 --seed 42 --num-workers 0 --output-dir artifacts_improved/final_vit

# 3) Custom baseline + custom improved (пункт 4)
python train.py --model custom_cnn --dataset-root "waste/DATASET" --epochs 5 --batch-size 64 --lr 3e-4 --seed 42 --num-workers 0 --output-dir artifacts_custom/baseline
python train.py --model custom_vit --dataset-root "waste/DATASET" --epochs 4 --batch-size 32 --lr 1e-4 --seed 42 --num-workers 0 --output-dir artifacts_custom/baseline
python evaluate.py --model custom_cnn --checkpoint artifacts_custom/baseline/custom_cnn/best.ckpt --dataset-root "waste/DATASET" --val-split 0.1 --seed 42 --num-workers 0 --output-dir artifacts_custom/baseline
python evaluate.py --model custom_vit --checkpoint artifacts_custom/baseline/custom_vit/best.ckpt --dataset-root "waste/DATASET" --val-split 0.1 --seed 42 --num-workers 0 --output-dir artifacts_custom/baseline

python train.py --model custom_cnn --dataset-root "waste/DATASET" --epochs 5 --batch-size 64 --lr 3e-4 --aug-profile strong --label-smoothing 0.05 --use-class-weights --scheduler cosine_warmup --warmup-epochs 1 --seed 42 --num-workers 0 --output-dir artifacts_custom/improved
python train.py --model custom_vit --dataset-root "waste/DATASET" --epochs 4 --batch-size 32 --lr 1e-4 --aug-profile strong --label-smoothing 0.05 --use-class-weights --scheduler cosine_warmup --warmup-epochs 1 --seed 42 --num-workers 0 --output-dir artifacts_custom/improved
python evaluate.py --model custom_cnn --checkpoint artifacts_custom/improved/custom_cnn/best.ckpt --dataset-root "waste/DATASET" --val-split 0.1 --seed 42 --num-workers 0 --output-dir artifacts_custom/improved
python evaluate.py --model custom_vit --checkpoint artifacts_custom/improved/custom_vit/best.ckpt --dataset-root "waste/DATASET" --val-split 0.1 --seed 42 --num-workers 0 --output-dir artifacts_custom/improved

# 4) Финальный объединенный отчет
python make_report.py \
  --resnet-metrics artifacts/resnet18/metrics.json \
  --vit-metrics artifacts/vit_b_16/metrics.json \
  --resnet-improved-metrics artifacts_improved/final_resnet/resnet18/metrics.json \
  --vit-improved-metrics artifacts_improved/final_vit/vit_b_16/metrics.json \
  --custom-cnn-metrics artifacts_custom/baseline/custom_cnn/metrics.json \
  --custom-vit-metrics artifacts_custom/baseline/custom_vit/metrics.json \
  --custom-cnn-improved-metrics artifacts_custom/improved/custom_cnn/metrics.json \
  --custom-vit-improved-metrics artifacts_custom/improved/custom_vit/metrics.json \
  --output artifacts/report.md
```

## Общий вывод

- На датасете waste лучшую итоговую метрику показывает `vit_b_16` с improved-настройками (пункт 3).
- `resnet18` дает сильный и стабильный baseline, а его improved-версия улучшает качество относительно пункта 2.
- Самостоятельно реализованные `custom_cnn` и `custom_vit` обучаются корректно и дают конкурентный baseline-уровень.
- Применение техник improved baseline к custom-моделям дает прирост, но в текущей конфигурации они уступают improved torchvision-моделям.
- Практическая рекомендация: для максимального качества использовать tuned `vit_b_16`; для более простого и быстрого baseline — `resnet18`.
