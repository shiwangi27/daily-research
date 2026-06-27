"""Pluggable loss functions and evaluation metrics."""
from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE = -100


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, cfg) -> torch.Tensor:
    """Loss selected by config. Accepts (B, C) class logits or (B, T, V) sequence logits
    (flattened to (B*T, V)). Targets == IGNORE are dropped (prompt-loss masking / padding)."""
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)

    keep = targets != IGNORE
    if not keep.all():
        logits, targets = logits[keep], targets[keep]

    kind = cfg.loss
    if kind == "cross_entropy":
        return F.cross_entropy(logits, targets)
    if kind == "label_smoothing":
        return F.cross_entropy(logits, targets, label_smoothing=cfg.label_smoothing)
    if kind == "focal":
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** cfg.focal_gamma * ce).mean()
    raise ValueError(f"Unknown loss '{kind}'")


def classification_metrics(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    preds, labels = preds.tolist(), labels.tolist()
    n = len(labels)
    acc = sum(int(p == l) for p, l in zip(preds, labels)) / n if n else 0.0
    f1s = []
    for c in range(num_classes):
        tp = sum(int(p == c and l == c) for p, l in zip(preds, labels))
        fp = sum(int(p == c and l != c) for p, l in zip(preds, labels))
        fn = sum(int(p != c and l == c) for p, l in zip(preds, labels))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return {"accuracy": round(acc, 4), "macro_f1": round(sum(f1s) / num_classes, 4)}


def sequence_metrics(logits: torch.Tensor, targets: torch.Tensor, answer_start: int) -> dict:
    preds = logits.argmax(-1)
    ans_pred, ans_true = preds[:, answer_start:], targets[:, answer_start:]
    token_acc = (ans_pred == ans_true).float().mean().item()
    exact = (ans_pred == ans_true).all(dim=1).float().mean().item()
    return {"token_accuracy": round(token_acc, 4), "exact_match": round(exact, 4)}
