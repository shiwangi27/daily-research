"""Task/data loading — fully offline (no network egress in this environment).

Two modalities:
  * classification: bundled sklearn datasets or synthetic, vector inputs -> class label.
  * sequence: locally generated algorithmic tasks (copy/sort) for a tiny GPT.

Everything is deterministic given the data seed, so runs are reproducible from git alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch

from .config import DataConfig

CLASSIFICATION = {"digits", "wine", "breast_cancer", "synth_cls"}
SEQUENCE = {"copy", "sort"}

# Sequence tasks reserve two special tokens beyond the value vocab.
PAD, SEP = 0, 1
SPECIAL = 2


@dataclass
class Task:
    kind: str                       # "classification" | "sequence"
    name: str
    train_x: torch.Tensor
    train_y: torch.Tensor
    eval_x: torch.Tensor
    eval_y: torch.Tensor
    input_dim: Optional[int] = None       # classification: feature count
    num_classes: Optional[int] = None     # classification: class count
    vocab_size: Optional[int] = None      # sequence: token vocabulary
    answer_start: Optional[int] = None    # sequence: index where the answer region begins

    def batches(self, split: str, batch_size: int, generator: torch.Generator):
        x = self.train_x if split == "train" else self.eval_x
        y = self.train_y if split == "train" else self.eval_y
        n = x.shape[0]
        order = torch.randperm(n, generator=generator) if split == "train" else torch.arange(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            yield x[idx], y[idx]


def _classification(cfg: DataConfig) -> Task:
    from sklearn import datasets as skd
    from sklearn.model_selection import train_test_split

    if cfg.task == "digits":
        bunch = skd.load_digits()
        X, y = bunch.data.astype("float32") / 16.0, bunch.target
    elif cfg.task == "wine":
        bunch = skd.load_wine()
        X, y = bunch.data.astype("float32"), bunch.target
    elif cfg.task == "breast_cancer":
        bunch = skd.load_breast_cancer()
        X, y = bunch.data.astype("float32"), bunch.target
    elif cfg.task == "synth_cls":
        X, y = skd.make_classification(
            n_samples=cfg.max_train + cfg.max_eval,
            n_features=cfg.n_features,
            n_informative=cfg.n_informative,
            n_classes=cfg.n_classes,
            random_state=cfg.seed,
        )
        X = X.astype("float32")
    else:  # pragma: no cover
        raise ValueError(cfg.task)

    # Standardize features (helps optimization land in a sane regime).
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Xtr, Xev, ytr, yev = train_test_split(
        X, y, test_size=0.3, random_state=cfg.seed, stratify=y
    )
    Xtr, ytr = Xtr[: cfg.max_train], ytr[: cfg.max_train]
    Xev, yev = Xev[: cfg.max_eval], yev[: cfg.max_eval]
    return Task(
        kind="classification",
        name=cfg.task,
        train_x=torch.tensor(Xtr),
        train_y=torch.tensor(ytr, dtype=torch.long),
        eval_x=torch.tensor(Xev),
        eval_y=torch.tensor(yev, dtype=torch.long),
        input_dim=X.shape[1],
        num_classes=int(np.unique(y).size),
    )


def _sequence(cfg: DataConfig) -> Task:
    """Build a next-token LM dataset for an algorithmic task.

    Layout per example (length 2*seq_len + 1):
        [ src_0 .. src_{L-1}  SEP  ans_0 .. ans_{L-1} ]
    The model is trained to predict every next token; we only *score* the answer region.
    Value tokens occupy ids [SPECIAL, SPECIAL+vocab).
    """
    rng = np.random.default_rng(cfg.seed)
    L, V = cfg.seq_len, cfg.vocab
    n = cfg.max_train + cfg.max_eval

    src = rng.integers(0, V, size=(n, L)) + SPECIAL
    if cfg.task == "copy":
        ans = src.copy()
    elif cfg.task == "sort":
        ans = np.sort(src, axis=1)
    else:  # pragma: no cover
        raise ValueError(cfg.task)

    sep = np.full((n, 1), SEP)
    seq = np.concatenate([src, sep, ans], axis=1).astype("int64")  # (n, 2L+1)

    # inputs are the sequence; targets are the sequence shifted left by one.
    x = seq[:, :-1]
    y = seq[:, 1:]
    answer_start = L  # position in the target where the answer region begins

    xt, yt = torch.tensor(x), torch.tensor(y)
    return Task(
        kind="sequence",
        name=cfg.task,
        train_x=xt[: cfg.max_train],
        train_y=yt[: cfg.max_train],
        eval_x=xt[cfg.max_train : cfg.max_train + cfg.max_eval],
        eval_y=yt[cfg.max_train : cfg.max_train + cfg.max_eval],
        vocab_size=V + SPECIAL,
        answer_start=answer_start,
    )


def load_task(cfg: DataConfig) -> Task:
    if cfg.task in CLASSIFICATION:
        return _classification(cfg)
    if cfg.task in SEQUENCE:
        return _sequence(cfg)
    raise ValueError(f"Unknown task '{cfg.task}'. Known: {sorted(CLASSIFICATION | SEQUENCE)}")
