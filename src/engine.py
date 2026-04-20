from __future__ import annotations

import time
from collections.abc import Iterable
from contextlib import nullcontext

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from tqdm import tqdm


def _topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    k = min(k, logits.size(1))
    topk = torch.topk(logits, k=k, dim=1).indices
    correct = topk.eq(targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def run_train_epoch(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    all_targets: list[int] = []
    all_preds: list[int] = []
    progress = tqdm(dataloader, desc="train", leave=False)
    for batch_idx, (images, targets, *_rest) in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        autocast_context = torch.autocast(device_type="cuda", enabled=True) if use_amp else nullcontext()
        with autocast_context:
            logits = model(images)
            loss = criterion(logits, targets)

        if scaler and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits, dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())
        running_loss += loss.item() * images.size(0)

    n = len(all_targets)
    return {
        "loss": running_loss / max(n, 1),
        "accuracy": accuracy_score(all_targets, all_preds) if n else 0.0,
        "macro_f1": f1_score(all_targets, all_preds, average="macro") if n else 0.0,
        "weighted_f1": f1_score(all_targets, all_preds, average="weighted") if n else 0.0,
    }


@torch.no_grad()
def run_eval_epoch(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[dict[str, float], list[int], list[int], np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_targets: list[int] = []
    all_preds: list[int] = []
    top5_sum = 0.0
    probs_list: list[np.ndarray] = []

    for batch_idx, (images, targets, *_rest) in enumerate(tqdm(dataloader, desc="eval", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_targets.extend(targets.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        probs_list.append(probs.cpu().numpy())
        top5_sum += _topk_accuracy(logits, targets, k=5) * images.size(0)
        running_loss += loss.item() * images.size(0)

    n = len(all_targets)
    metrics = {
        "loss": running_loss / max(n, 1),
        "accuracy": accuracy_score(all_targets, all_preds) if n else 0.0,
        "macro_f1": f1_score(all_targets, all_preds, average="macro") if n else 0.0,
        "weighted_f1": f1_score(all_targets, all_preds, average="weighted") if n else 0.0,
        "top5_accuracy": top5_sum / max(n, 1),
    }
    probabilities = np.concatenate(probs_list, axis=0) if probs_list else np.zeros((0, 0))
    return metrics, all_targets, all_preds, probabilities


@torch.no_grad()
def measure_inference_time(model: nn.Module, dataloader: Iterable, device: torch.device, batches: int = 20) -> float:
    model.eval()
    total_time = 0.0
    total_images = 0
    for batch_idx, (images, *_rest) in enumerate(dataloader):
        if batch_idx >= batches:
            break
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - start
        total_images += images.size(0)
    if total_images == 0:
        return 0.0
    return total_time / total_images

