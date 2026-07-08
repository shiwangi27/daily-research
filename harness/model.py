"""Models built from scratch (no pretrained downloads).

  * NanoLM  — a *modern* decoder-only transformer matching the LLM core recipe:
              RoPE positions, RMSNorm (no learnable params), ReLU^2 MLP, QK-norm,
              no biases, untied embeddings, optional grouped-query attention.
              This is the architecture nanochat/Llama-style models use, shrunk to
              run on CPU. It has a `generate()` for sampling (needed by GRPO/eval).
  * MLP     — vector classification, for the fast optimization side-track.

Plus a from-scratch LoRA wrapper for nn.Linear (parameter-efficient post-training)
and freezing. Dynamic int8 quantization is applied by the harness around these.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


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


# --------------------------------------------------------------------------- modern bits
def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm with no learnable parameters (nanochat-style)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _rope_tables(seq_len: int, head_dim: int, device, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)              # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)       # (T, head_dim)
    return emb.cos()[None, None], emb.sin()[None, None]  # (1,1,T,head_dim)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = d_model // n_heads
        self.attn_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.attn_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.attn_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.attn_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x):
        b, t, _ = x.shape
        q = self.attn_q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.attn_k(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.attn_v(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = rms_norm(q), rms_norm(k)                       # QK-norm
        cos, sin = _rope_tables(t, self.head_dim, x.device)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

        if self.n_kv_heads != self.n_heads:                   # grouped-query attention
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.attn_o(out)


class MLPBlock(nn.Module):
    """ReLU^2 feed-forward (squared ReLU), no biases."""

    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model, bias=False)
        self.proj = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x):
        return self.proj(F.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, dropout):
        super().__init__()
        self.attn = Attention(d_model, n_heads, n_kv_heads, dropout)
        self.mlp = MLPBlock(d_model)

    def forward(self, x):
        x = x + self.attn(rms_norm(x))       # pre-norm
        x = x + self.mlp(rms_norm(x))
        return x


class NanoLM(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_kv_heads, n_layers, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, n_kv_heads, dropout) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab, bias=False)   # untied from wte

    def forward(self, idx):
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(rms_norm(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, eos_id=None, temperature=0.0, top_k=None):
        """Autoregressive decode. temperature==0 -> greedy. Stops a row at eos_id."""
        self.eval()
        done = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.block_size:])[:, -1, :]
            if temperature == 0.0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None:
                done |= nxt.squeeze(1) == eos_id
                if done.all():
                    break
        return idx


class NanoLMClassifier(nn.Module):
    """NanoLM trunk + a classification head over the last real (non-pad) token's hidden state.

    Causal attention means the last token has attended to the whole sequence, so its state is a
    natural sentence summary (the GPT-style approach to sequence classification)."""

    def __init__(self, vocab, num_classes, d_model, n_heads, n_kv_heads, n_layers, block_size, dropout, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.wte = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, n_kv_heads, dropout) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, idx):
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        x = rms_norm(x)
        last = (idx != self.pad_id).sum(1).clamp(min=1) - 1     # index of last real token
        pooled = x[torch.arange(x.shape[0]), last]
        return self.head(pooled)


# --------------------------------------------------------------------------- MLP (side-track)
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, depth, num_classes, dropout):
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


class RewardModel(nn.Module):
    """An LM body with a scalar value head — for reward modeling (Bradley-Terry).

    Reuses a NanoLM trunk; reads the hidden state at the last real token and maps it
    to a single scalar. Built lazily by the harness when stage == 'rm'.
    """

    def __init__(self, backbone: NanoLM, d_model: int):
        super().__init__()
        self.wte = backbone.wte
        self.blocks = backbone.blocks
        self.value = nn.Linear(d_model, 1, bias=False)

    def forward(self, idx, last_idx):
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        x = rms_norm(x)
        gathered = x[torch.arange(x.shape[0]), last_idx]   # (B, d_model)
        return self.value(gathered).squeeze(-1)            # (B,)


def build_model(cfg: ModelConfig, *, vocab=None, input_dim=None, num_classes=None,
                pad_id=None, classifier=False) -> nn.Module:
    if classifier:                               # NanoLM trunk + classification head
        n_kv = cfg.n_kv_heads or cfg.n_heads
        model: nn.Module = NanoLMClassifier(
            vocab, num_classes, cfg.d_model, cfg.n_heads, n_kv, cfg.n_layers,
            cfg.block_size, cfg.dropout, pad_id,
        )
    elif cfg.arch in ("nanolm", "tiny_gpt"):     # tiny_gpt kept as an alias
        n_kv = cfg.n_kv_heads or cfg.n_heads
        model = NanoLM(
            vocab, cfg.d_model, cfg.n_heads, n_kv, cfg.n_layers, cfg.block_size, cfg.dropout
        )
    elif cfg.arch == "mlp":
        model = MLP(input_dim, cfg.hidden, cfg.depth, num_classes, cfg.dropout)
    else:
        raise ValueError(f"Unknown arch '{cfg.arch}'")

    if cfg.lora.enabled:
        apply_lora(model, cfg.lora)
    if cfg.freeze:
        for name, p in model.named_parameters():
            if any(f in name for f in cfg.freeze):
                p.requires_grad = False
    return model
