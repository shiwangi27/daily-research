"""Config-driven training/eval loop, organized by post-training `stage`.

Usage:
    python3 -m harness.train experiments/<dir>/config.yaml

Stages:
    pretrain    next-token LM on the raw corpus (the base model)
    sft         instruction tuning; prompt-loss masking is the key knob (train.mask_prompt)
    supervised  classification / legacy copy-sort sequence tasks
    rm/dpo/grpo  scheduled — informative stub until implemented on their day

Chaining: set `init_from: experiments/<dir>` to warm-start from that experiment's saved
`model.pt`. Each run saves its own `model.pt` (gitignored) so a later stage can build on it.
Writes result.json next to the config. Plain PyTorch so every knob stays explicit.
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


# --------------------------------------------------------------------------- optim
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
        except ImportError as exc:
            raise SystemExit("optimizer 'adafactor' needs: python3 -m pip install transformers") from exc
        return Adafactor(params, lr=tc.lr, scale_parameter=False, relative_step=False, warmup_init=False)
    if name == "lion":
        return _Lion(params, lr=tc.lr, weight_decay=tc.weight_decay)
    raise ValueError(f"Unknown optimizer '{tc.optimizer}'")


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


# --------------------------------------------------------------------------- eval
def _eval_loss(model, task, cfg, gen) -> float:
    losses = []
    with torch.no_grad():
        for xb, yb in task.batches("eval", cfg.train.batch_size, gen):
            losses.append(compute_loss(model(xb), yb, cfg.train).item())
    return round(sum(losses) / len(losses), 4)


def _arith_exact_match(model, task: Task, batch_size: int) -> dict:
    """Greedy-decode answers from the eval prompts and check them programmatically."""
    tok = task.tokenizer
    correct, total = 0, len(task.eval_answers)
    P = task.eval_prompts.shape[1]
    with torch.no_grad():
        for start in range(0, total, batch_size):
            prompts = task.eval_prompts[start : start + batch_size]
            out = model.generate(prompts, max_new_tokens=task.answer_len, eos_id=tok.eos_id)
            gen_tokens = out[:, P:].tolist()
            for row, true_val in zip(gen_tokens, task.eval_answers[start : start + batch_size]):
                cut = row.index(tok.eos_id) if tok.eos_id in row else len(row)
                text = tok.decode(row[:cut])
                if task.reverse_answer:
                    text = text[::-1]
                correct += int(text.isdigit() and int(text) == true_val)
    return {"exact_match": round(correct / total, 4)}


def evaluate(model, task: Task, cfg, gen) -> dict:
    model.eval()
    metrics = {"eval_loss": _eval_loss(model, task, cfg, gen)}
    if task.kind in ("classification", "textcls"):
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in task.batches("eval", cfg.train.batch_size, gen):
                preds.append(model(xb).argmax(-1)); labels.append(yb)
        metrics.update(classification_metrics(torch.cat(preds), torch.cat(labels), task.num_classes))
    elif task.kind == "sequence":
        with torch.no_grad():
            logits = torch.cat([model(xb) for xb, _ in task.batches("eval", cfg.train.batch_size, gen)])
        metrics.update(sequence_metrics(logits, task.eval_y, task.answer_start))
    elif task.kind == "lm":
        metrics.update(_arith_exact_match(model, task, cfg.train.batch_size))
    return metrics


# --------------------------------------------------------------------------- build
def _make_model(cfg: ExperimentConfig, task: Task):
    if task.kind == "lm":
        model = build_model(cfg.model, vocab=task.vocab_size)
    elif task.kind == "textcls":
        model = build_model(cfg.model, vocab=task.vocab_size, num_classes=task.num_classes,
                            pad_id=task.tokenizer.pad_id, classifier=True)
    elif task.kind == "sequence":
        model = build_model(cfg.model, vocab=cfg.data.vocab + 2)
    else:
        model = build_model(cfg.model, input_dim=task.input_dim, num_classes=task.num_classes)

    if cfg.init_from:
        ckpt = cfg.init_from
        if os.path.isdir(ckpt):
            ckpt = os.path.join(ckpt, "model.pt")
        state = torch.load(ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"init_from {ckpt}  (missing={len(missing)} unexpected={len(unexpected)})")
    return model


# --------------------------------------------------------------------------- main
def train(config_path: str) -> dict:
    cfg = load_config(config_path)
    if cfg.needs_gpu and not torch.cuda.is_available():
        raise SystemExit(f"[{cfg.name}] flagged needs_gpu but no GPU here. Run externally.")
    if cfg.stage in {"rm", "dpo", "grpo"}:
        raise SystemExit(
            f"stage '{cfg.stage}' is scheduled but not yet implemented — it's an upcoming "
            f"one-knob day. See schedule.md. (pretrain/sft/supervised work today.)"
        )

    set_seed(cfg.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    gen = torch.Generator().manual_seed(cfg.seed)

    print(f"== {cfg.name} == [{cfg.stage}]  {cfg.description}")
    stage_for_data = cfg.stage if cfg.stage in {"pretrain", "sft"} else "supervised"
    task = load_task(cfg.data, stage=stage_for_data, mask_prompt=cfg.train.mask_prompt)
    model = _make_model(cfg, task)
    params = count_params(model)
    print(f"task={task.name}({task.kind})  params: {params['trainable']:,} / {params['total']:,}")

    tc = cfg.train
    steps_per_epoch = math.ceil(task.train_x.shape[0] / tc.batch_size)
    optimizer = _build_optimizer(model, tc)
    scheduler = _build_scheduler(optimizer, tc, steps_per_epoch * tc.epochs)

    # Optional inverse-frequency class weights for imbalanced classification (training loss only).
    weight = None
    if task.kind == "textcls" and tc.class_weight == "balanced":
        counts = torch.bincount(task.train_y, minlength=task.num_classes).float()
        weight = counts.sum() / (len(counts) * counts.clamp(min=1))
        print(f"class counts {counts.int().tolist()} -> weights {[round(w, 2) for w in weight.tolist()]}")

    history = []
    with Timer() as timer:
        for epoch in range(tc.epochs):
            model.train()
            running = 0.0
            for xb, yb in task.batches("train", tc.batch_size, gen):
                loss = compute_loss(model(xb), yb, tc, weight=weight)
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

    out_dir = os.path.dirname(config_path)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))  # gitignored; for chaining
    result = {
        "name": cfg.name, "stage": cfg.stage, "hypothesis": cfg.hypothesis,
        "compare_to": cfg.compare_to, "config": cfg.to_dict(),
        "config_hash": config_hash(cfg.to_dict()), "params": params,
        "wall_seconds": timer.seconds, "history": history, "final": final,
        "env": capture_env(),
    }
    with open(os.path.join(out_dir, "result.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {out_dir}/result.json  ({timer.seconds}s)")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 -m harness.train <config.yaml>")
    train(sys.argv[1])
