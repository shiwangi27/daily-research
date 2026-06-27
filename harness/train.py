"""Config-driven training/eval loop (modality-agnostic).

Usage:
    python3 -m harness.train experiments/<dir>/config.yaml

Writes result.json next to the config. Plain PyTorch loop so optimizer, scheduler,
gradient clipping, and loss are all explicit and easy to swap one knob at a time.
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

from .config import ExperimentConfig, load_config
from .data import Task, load_task
from .metrics import classification_metrics, compute_loss, sequence_metrics
from .model import build_model, count_params
from .utils import Timer, capture_env, config_hash, set_seed


def _build_optimizer(model, tc):
    params = [p for p in model.parameters() if p.requires_grad]
    name = tc.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=tc.lr, momentum=tc.momentum, weight_decay=tc.weight_decay)
    if name == "adafactor":
        try:
            from transformers.optimization import Adafactor
        except ImportError as exc:  # optional dependency
            raise SystemExit(
                "optimizer 'adafactor' needs transformers: python3 -m pip install transformers"
            ) from exc
        return Adafactor(params, lr=tc.lr, scale_parameter=False, relative_step=False, warmup_init=False)
    if name == "lion":
        return _Lion(params, lr=tc.lr, weight_decay=tc.weight_decay)
    raise ValueError(f"Unknown optimizer '{tc.optimizer}'")


class _Lion(torch.optim.Optimizer):
    """Minimal Lion optimizer (Chen et al., 2023): update is the sign of an interpolated
    momentum. Included so we can A/B it against AdamW with no extra dependency."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                update = exp_avg.mul(b1).add_(p.grad, alpha=1 - b1).sign_()
                p.add_(update, alpha=-group["lr"])
                exp_avg.mul_(b2).add_(p.grad, alpha=1 - b2)
        return loss


def _build_scheduler(optimizer, tc, total_steps):
    warmup = int(tc.warmup_ratio * total_steps)
    if tc.scheduler == "none":
        return None

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        if tc.scheduler == "linear":
            return max(0.0, 1.0 - progress)
        if tc.scheduler == "cosine":
            return 0.5 * (1 + math.cos(math.pi * progress))
        raise ValueError(f"Unknown scheduler '{tc.scheduler}'")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, task: Task, cfg: ExperimentConfig, gen) -> dict:
    model.eval()
    losses, all_logits, all_targets, all_preds = [], [], [], []
    with torch.no_grad():
        for xb, yb in task.batches("eval", cfg.train.batch_size, gen):
            logits = model(xb)
            losses.append(compute_loss(logits, yb, cfg.train).item())
            if task.kind == "classification":
                all_preds.append(logits.argmax(-1))
                all_targets.append(yb)
            else:
                all_logits.append(logits)
                all_targets.append(yb)
    metrics = {"eval_loss": round(sum(losses) / len(losses), 4)}
    if task.kind == "classification":
        metrics.update(
            classification_metrics(torch.cat(all_preds), torch.cat(all_targets), task.num_classes)
        )
    else:
        metrics.update(
            sequence_metrics(torch.cat(all_logits), torch.cat(all_targets), task.answer_start)
        )
    return metrics


def train(config_path: str) -> dict:
    cfg = load_config(config_path)
    if cfg.needs_gpu and not torch.cuda.is_available():
        raise SystemExit(f"[{cfg.name}] flagged needs_gpu but no GPU here. Run externally.")

    set_seed(cfg.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    gen = torch.Generator().manual_seed(cfg.seed)

    print(f"== {cfg.name} ==  {cfg.description}")
    task = load_task(cfg.data)
    model = build_model(cfg.model, task)
    params = count_params(model)
    print(f"task={task.name}({task.kind})  params: {params['trainable']:,} trainable / {params['total']:,} total")

    tc = cfg.train
    n_train = task.train_x.shape[0]
    steps_per_epoch = math.ceil(n_train / tc.batch_size)
    optimizer = _build_optimizer(model, tc)
    scheduler = _build_scheduler(optimizer, tc, steps_per_epoch * tc.epochs)

    history = []
    with Timer() as timer:
        for epoch in range(tc.epochs):
            model.train()
            running = 0.0
            for xb, yb in task.batches("train", tc.batch_size, gen):
                logits = model(xb)
                loss = compute_loss(logits, yb, tc)
                optimizer.zero_grad()
                loss.backward()
                if tc.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], tc.grad_clip
                    )
                optimizer.step()
                if scheduler:
                    scheduler.step()
                running += loss.item()
            entry = {"epoch": epoch, "train_loss": round(running / steps_per_epoch, 4)}
            if tc.eval_every_epoch or epoch == tc.epochs - 1:
                entry.update(evaluate(model, task, cfg, gen))
            history.append(entry)
            print(f"epoch {epoch}: {entry}")

    final = dict(history[-1])

    if cfg.model.quantize_dynamic:
        qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        final["quantized_int8"] = evaluate(qmodel, task, cfg, gen)
        print(f"quantized int8 eval: {final['quantized_int8']}")

    result = {
        "name": cfg.name,
        "hypothesis": cfg.hypothesis,
        "compare_to": cfg.compare_to,
        "config": cfg.to_dict(),
        "config_hash": config_hash(cfg.to_dict()),
        "params": params,
        "wall_seconds": timer.seconds,
        "history": history,
        "final": final,
        "env": capture_env(),
    }

    out_path = os.path.join(os.path.dirname(config_path), "result.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {out_path}  ({timer.seconds}s)")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 -m harness.train <config.yaml>")
    train(sys.argv[1])
