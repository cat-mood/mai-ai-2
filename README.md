# AI Classification

Dataset: https://www.kaggle.com/datasets/zlatan599/mushroom1

## Baseline experiments (ResNet18 vs ViT-B/16)

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare image paths

CSV files contain Kaggle-style absolute paths (`/kaggle/working/merged_dataset/...`).

Use one of these options:

- `--data-root` if your local images are in `.../merged_dataset/<class>/<image>`
- `--path-prefix-from` and `--path-prefix-to` for explicit prefix rewrite

### 3) Train baseline models

ResNet18:

```bash
python train.py \
  --model resnet18 \
  --data-root "/absolute/path/to/merged_dataset" \
  --epochs 10 \
  --batch-size 64 \
  --output-dir artifacts
```

ViT-B/16:

```bash
python train.py \
  --model vit_b_16 \
  --data-root "/absolute/path/to/merged_dataset" \
  --epochs 10 \
  --batch-size 32 \
  --output-dir artifacts
```

Device is selected automatically by priority: `cuda -> mps -> cpu`.

If you need ImageNet initialization and internet is available, add `--pretrained`.

For a fast smoke run (few batches only), add:

```bash
--max-train-batches 20 --max-eval-batches 5 --num-workers 0
```

### 4) Evaluate and save artifacts

```bash
python evaluate.py \
  --model resnet18 \
  --checkpoint artifacts/resnet18/best.ckpt \
  --data-root "/absolute/path/to/merged_dataset" \
  --output-dir artifacts
```

Artifacts per model are saved under `artifacts/<model>/`:

- `best.ckpt`
- `metrics.json`
- `predictions.csv`
- `confusion_matrix.png`

### 5) Build concise lab report

```bash
python make_report.py \
  --resnet-metrics artifacts/resnet18/metrics.json \
  --vit-metrics artifacts/vit_b_16/metrics.json \
  --output artifacts/report.md
```
