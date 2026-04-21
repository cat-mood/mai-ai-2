# Матрица экспериментов improved baseline (waste, бинарная классификация)

Фиксированные настройки для честного сравнения:

- `dataset-root`: `waste/DATASET`
- `seed`: `42`
- `device`: `mps`
- `num-workers`: `0`
- целевая метрика оптимизации: `val_macro_f1`

## E1 — Регуляризация ResNet (H1)

```bash
python train.py \
  --model resnet18 \
  --dataset-root "waste/DATASET" \
  --pretrained \
  --epochs 5 \
  --batch-size 64 \
  --lr 3e-4 \
  --aug-profile strong \
  --label-smoothing 0.05 \
  --output-dir artifacts_improved/e1_resnet_aug
```

## E2 — Регуляризация ResNet + веса классов (H2)

```bash
python train.py \
  --model resnet18 \
  --dataset-root "waste/DATASET" \
  --pretrained \
  --epochs 5 \
  --batch-size 64 \
  --lr 3e-4 \
  --aug-profile strong \
  --label-smoothing 0.05 \
  --use-class-weights \
  --output-dir artifacts_improved/e2_resnet_aug_weights
```

## E3 — Оптимизированный scheduler для ViT (H3)

```bash
python train.py \
  --model vit_b_16 \
  --dataset-root "waste/DATASET" \
  --pretrained \
  --epochs 4 \
  --batch-size 16 \
  --lr 3e-5 \
  --scheduler cosine_warmup \
  --warmup-epochs 1 \
  --output-dir artifacts_improved/e3_vit_opt
```

## E4 — ViT optimized + веса классов (H2/H3)

```bash
python train.py \
  --model vit_b_16 \
  --dataset-root "waste/DATASET" \
  --pretrained \
  --epochs 4 \
  --batch-size 16 \
  --lr 3e-5 \
  --scheduler cosine_warmup \
  --warmup-epochs 1 \
  --use-class-weights \
  --output-dir artifacts_improved/e4_vit_opt_weights
```

## E5 — Финальные improved-конфиги (валидация H4)

Выберите лучший запуск для каждой модели по `val_macro_f1`, затем выполните:

```bash
python evaluate.py \
  --model resnet18 \
  --checkpoint artifacts_improved/<best_resnet_run>/resnet18/best.ckpt \
  --dataset-root "waste/DATASET" \
  --num-workers 0 \
  --device mps \
  --output-dir artifacts_improved/final_resnet
```

```bash
python evaluate.py \
  --model vit_b_16 \
  --checkpoint artifacts_improved/<best_vit_run>/vit_b_16/best.ckpt \
  --dataset-root "waste/DATASET" \
  --num-workers 0 \
  --device mps \
  --output-dir artifacts_improved/final_vit
```
