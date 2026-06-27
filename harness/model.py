"""Models built from scratch (no pretrained downloads).

  * MLP        — vector classification.
  * TinyGPT    — decoder-only transformer for the sequence tasks.

Plus a from-scratch LoRA wrapper for nn.Linear, so we can study parameter-efficient
fine-tuning without a black-box library. Freezing and dynamic int8 quantization are
applied by the harness around these.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .data import Task


def count_params(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# --------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """Wrap a (frozen) Linear with a low-rank trainable update: W x + (alpha/r) B A x."""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * (1.0 / math.sqrt(base.in_features)))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + self.scaling * F.linear(self.drop(x) @ self.A.t(), self.B)


def apply_lora(model: nn.Module, cfg) -> None:
    """Replace matching nn.Linear modules with LoRALinear in place."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and any(t in name for t in cfg.target_modules):
            setattr(model, name, LoRALinear(child, cfg.r, cfg.alpha, cfg.dropout))
        else:
            apply_lora(child, cfg)


# --------------------------------------------------------------------------- MLP
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, depth: int, num_classes: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, num_classes)

    def forward(self, x):
        return self.head(self.body(x))


# --------------------------------------------------------------------------- TinyGPT
class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab: int, d_model: int, n_heads: int, n_layers: int, block_size: int, dropout: float):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        mask = torch.triu(torch.full((t, t), float("-inf"), device=idx.device), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln_f(x))


def build_model(cfg: ModelConfig, task: Task) -> nn.Module:
    if cfg.arch == "mlp":
        model: nn.Module = MLP(task.input_dim, cfg.hidden, cfg.depth, task.num_classes, cfg.dropout)
    elif cfg.arch == "tiny_gpt":
        model = TinyGPT(task.vocab_size, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.block_size, cfg.dropout)
    else:
        raise ValueError(f"Unknown arch '{cfg.arch}'")

    if cfg.lora.enabled:
        apply_lora(model, cfg.lora)

    if cfg.freeze:
        for name, p in model.named_parameters():
            if any(f in name for f in cfg.freeze):
                p.requires_grad = False

    return model
