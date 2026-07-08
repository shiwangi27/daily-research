"""Task/data loading — fully offline (no network egress in this environment).

Primary world: `arith`, a *verifiable* toy-reasoning task (a OP b = c) rendered as
fixed-width strings so batches are uniform and the answer can be checked programmatically
(enabling RLVR/GRPO with no reward model). Char-level tokenizer, generated locally.

Also kept: vector classification (sklearn bundled datasets) as a fast optimization
side-track, and the copy/sort sequence sanity tasks.

Everything is deterministic given the data seed -> reproducible from git alone.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .config import DataConfig

CLASSIFICATION = {"digits", "wine", "breast_cancer", "synth_cls"}
SEQUENCE = {"copy", "sort"}
LM = {"arith"}
TEXTCLS = {"textcls"}

ARITH_CHARS = "0123456789+-="
SPECIALS = ["<pad>", "<bos>", "<eos>"]


class WordTokenizer:
    """Tiny word-level tokenizer built from scratch on the training texts (no downloads).

    Lowercases, splits words/punctuation, keeps the `max_vocab` most frequent tokens, maps
    the rest to <unk>, and pads/truncates to `max_len`. Fit on TRAIN only (no eval leakage).
    """

    def __init__(self, texts, max_vocab: int, max_len: int):
        self.max_len = max_len
        counter: Counter = Counter()
        for t in texts:
            counter.update(self._split(t))
        self.itos = ["<pad>", "<unk>"] + [w for w, _ in counter.most_common(max_vocab - 2)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id, self.unk_id = 0, 1

    @staticmethod
    def _split(t: str) -> list[str]:
        return re.findall(r"[a-z0-9]+|[^\sa-z0-9]", str(t).lower())

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, t: str) -> list[int]:
        ids = [self.stoi.get(w, self.unk_id) for w in self._split(t)][: self.max_len]
        return ids + [self.pad_id] * (self.max_len - len(ids))


class CharTokenizer:
    def __init__(self, chars: str):
        self.itos = SPECIALS + list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if i >= len(SPECIALS))


@dataclass
class Task:
    kind: str                         # "classification" | "sequence" | "lm"
    name: str
    train_x: torch.Tensor
    train_y: torch.Tensor
    eval_x: torch.Tensor
    eval_y: torch.Tensor
    # classification
    input_dim: Optional[int] = None
    num_classes: Optional[int] = None
    # sequence (copy/sort)
    answer_start: Optional[int] = None
    # lm (arith) / textcls
    tokenizer: Optional[object] = None            # CharTokenizer or WordTokenizer
    eval_prompts: Optional[torch.Tensor] = None   # (Ne, P) prompt token ids for generation
    eval_answers: Optional[list[int]] = None      # ground-truth integer answers
    answer_len: Optional[int] = None              # tokens to generate (answer digits + eos)
    reverse_answer: bool = False                  # answer digits emitted least-significant first

    @property
    def vocab_size(self) -> Optional[int]:
        return self.tokenizer.vocab_size if self.tokenizer else None

    def batches(self, split: str, batch_size: int, generator: torch.Generator):
        x = self.train_x if split == "train" else self.eval_x
        y = self.train_y if split == "train" else self.eval_y
        n = x.shape[0]
        order = torch.randperm(n, generator=generator) if split == "train" else torch.arange(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            yield x[idx], y[idx]


# --------------------------------------------------------------------------- classification
def _classification(cfg: DataConfig) -> Task:
    from sklearn import datasets as skd
    from sklearn.model_selection import train_test_split

    if cfg.task == "digits":
        b = skd.load_digits(); X, y = b.data.astype("float32") / 16.0, b.target
    elif cfg.task == "wine":
        b = skd.load_wine(); X, y = b.data.astype("float32"), b.target
    elif cfg.task == "breast_cancer":
        b = skd.load_breast_cancer(); X, y = b.data.astype("float32"), b.target
    else:  # synth_cls
        X, y = skd.make_classification(
            n_samples=cfg.max_train + cfg.max_eval, n_features=cfg.n_features,
            n_informative=cfg.n_informative, n_classes=cfg.n_classes, random_state=cfg.seed,
        )
        X = X.astype("float32")

    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Xtr, Xev, ytr, yev = train_test_split(X, y, test_size=0.3, random_state=cfg.seed, stratify=y)
    return Task(
        kind="classification", name=cfg.task,
        train_x=torch.tensor(Xtr[: cfg.max_train]), train_y=torch.tensor(ytr[: cfg.max_train], dtype=torch.long),
        eval_x=torch.tensor(Xev[: cfg.max_eval]), eval_y=torch.tensor(yev[: cfg.max_eval], dtype=torch.long),
        input_dim=X.shape[1], num_classes=int(np.unique(y).size),
    )


# --------------------------------------------------------------------------- legacy copy/sort
def _sequence(cfg: DataConfig) -> Task:
    rng = np.random.default_rng(cfg.seed)
    L, V, SPC = cfg.seq_len, cfg.vocab, 2
    n = cfg.max_train + cfg.max_eval
    src = rng.integers(0, V, size=(n, L)) + SPC
    ans = src.copy() if cfg.task == "copy" else np.sort(src, axis=1)
    seq = np.concatenate([src, np.full((n, 1), 1), ans], axis=1).astype("int64")
    x, y = torch.tensor(seq[:, :-1]), torch.tensor(seq[:, 1:])
    return Task(
        kind="sequence", name=cfg.task,
        train_x=x[: cfg.max_train], train_y=y[: cfg.max_train],
        eval_x=x[cfg.max_train : cfg.max_train + cfg.max_eval],
        eval_y=y[cfg.max_train : cfg.max_train + cfg.max_eval],
        answer_start=L, tokenizer=None,
    )
    # NB: copy/sort don't use the char tokenizer; vocab_size handled in train via V+SPC.


# --------------------------------------------------------------------------- arith (LM)
def _arith(cfg: DataConfig, stage: str, mask_prompt: bool) -> Task:
    """Build a fixed-width arithmetic LM dataset.

    Layout (D = max_digits):  <bos> A..A op B..B = C..C <eos>
    operands zero-padded to width D, answer zero-padded to width D+1.
    pretrain  -> loss on every token.   sft -> optionally mask the prompt region.
    """
    rng = np.random.default_rng(cfg.seed)
    tok = CharTokenizer(ARITH_CHARS)
    D = cfg.max_digits
    n = cfg.max_train + cfg.max_eval

    a = rng.integers(0, 10 ** D, n)
    b = rng.integers(0, 10 ** D, n)
    if cfg.op == "add":
        opch, c = "+", a + b
    elif cfg.op == "sub":
        opch = "-"
        a, b = np.maximum(a, b), np.minimum(a, b)   # keep answers non-negative
        c = a - b
    else:
        raise ValueError(f"unknown op '{cfg.op}'")
    aw, cw = D, D + 1

    inputs, targets, prompt_ids_eval, answers_eval = [], [], [], []
    for i in range(n):
        prompt = f"{a[i]:0{aw}d}{opch}{b[i]:0{aw}d}="
        answer = f"{c[i]:0{cw}d}"
        if cfg.reverse_answer:
            answer = answer[::-1]
        pid = [tok.bos_id] + tok.encode(prompt)        # length P
        rid = tok.encode(answer) + [tok.eos_id]        # length R
        ids = pid + rid
        x = ids[:-1]
        y = ids[1:]
        ans_start = len(pid) - 1                        # first target index of the answer region
        if stage == "sft" and mask_prompt:
            y = [-100] * ans_start + y[ans_start:]
        inputs.append(x); targets.append(y)
        if i >= cfg.max_train:
            prompt_ids_eval.append(pid)
            answers_eval.append(int(c[i]))

    X = torch.tensor(inputs, dtype=torch.long)
    Y = torch.tensor(targets, dtype=torch.long)
    P = len(pid)
    return Task(
        kind="lm", name=f"{cfg.op}{D}",
        train_x=X[: cfg.max_train], train_y=Y[: cfg.max_train],
        eval_x=X[cfg.max_train:], eval_y=Y[cfg.max_train:],
        tokenizer=tok,
        eval_prompts=torch.tensor(prompt_ids_eval, dtype=torch.long),
        eval_answers=answers_eval,
        answer_len=cw + 1,                              # answer digits + eos
        answer_start=P - 1,
        reverse_answer=cfg.reverse_answer,
    )


# --------------------------------------------------------------------------- text classification
def _synth_textcls(cfg: DataConfig):
    """Offline stand-in for a real text-classification dataset: short 'sentences' built from
    per-class signature words mixed with shared filler, with a configurable class imbalance.
    Lets us validate the whole classify pipeline (and imbalance remedies) without egress."""
    rng = np.random.default_rng(cfg.seed)
    K = cfg.n_classes
    filler = [f"w{i}" for i in range(40)]
    sig = {k: [f"c{k}_{j}" for j in range(6)] for k in range(K)}
    n = cfg.max_train + cfg.max_eval

    probs = np.array([cfg.imbalance ** (-k) for k in range(K)], dtype=float)
    probs /= probs.sum()
    labels = rng.choice(K, size=n, p=probs)

    texts = []
    for k in labels:
        L = int(rng.integers(max(8, cfg.max_len // 2), cfg.max_len))   # long -> signal diluted
        words = list(rng.choice(filler, size=L))
        src = int(k) if rng.random() > 0.15 else int(rng.integers(0, K))  # 15% label noise
        words[rng.integers(0, L)] = rng.choice(sig[src])                  # a single weak signal token
        texts.append(" ".join(words))

    s = cfg.max_train
    return texts[:s], labels[:s].tolist(), texts[s:], labels[s:].tolist(), K


def _hf_textcls(cfg: DataConfig):
    """Load a real text-classification dataset from HuggingFace. Requires `huggingface.co`
    egress (blocked by default here) and a parquet-native repo (datasets>=5 dropped scripts).
    UNVALIDATED in the offline sandbox — exercise it in an HF-enabled session."""
    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # force classic CDN (cdn-lfs.*)
    from datasets import load_dataset

    ds = load_dataset(cfg.hf_repo, cfg.hf_config) if cfg.hf_config else load_dataset(cfg.hf_repo)
    splits = list(ds.keys())
    train = ds["train"] if "train" in ds else ds[splits[0]]
    if "validation" in ds:
        ev = ds["validation"]
    elif "test" in ds:
        ev = ds["test"]
    else:  # single split -> deterministic holdout
        parts = train.train_test_split(test_size=0.2, seed=cfg.seed)
        train, ev = parts["train"], parts["test"]

    tf, lf = cfg.text_field, cfg.label_field
    tr_t, tr_y = list(train[tf])[: cfg.max_train], list(train[lf])[: cfg.max_train]
    ev_t, ev_y = list(ev[tf])[: cfg.max_eval], list(ev[lf])[: cfg.max_eval]
    K = len(set(tr_y) | set(ev_y))
    return tr_t, [int(y) for y in tr_y], ev_t, [int(y) for y in ev_y], K


def _textcls(cfg: DataConfig) -> Task:
    if cfg.source == "hf":
        tr_t, tr_y, ev_t, ev_y, K = _hf_textcls(cfg)
    else:
        tr_t, tr_y, ev_t, ev_y, K = _synth_textcls(cfg)

    tok = WordTokenizer(tr_t, cfg.max_vocab, cfg.max_len)   # fit on train only
    Xtr = torch.tensor([tok.encode(t) for t in tr_t], dtype=torch.long)
    Xev = torch.tensor([tok.encode(t) for t in ev_t], dtype=torch.long)
    return Task(
        kind="textcls", name=f"{cfg.source}:{cfg.hf_repo or 'synth'}",
        train_x=Xtr, train_y=torch.tensor(tr_y, dtype=torch.long),
        eval_x=Xev, eval_y=torch.tensor(ev_y, dtype=torch.long),
        tokenizer=tok, num_classes=K,
    )


def load_task(cfg: DataConfig, stage: str = "supervised", mask_prompt: bool = True) -> Task:
    if cfg.task in TEXTCLS:
        return _textcls(cfg)
    if cfg.task in LM:
        return _arith(cfg, stage, mask_prompt)
    if cfg.task in CLASSIFICATION:
        return _classification(cfg)
    if cfg.task in SEQUENCE:
        return _sequence(cfg)
    known = sorted(TEXTCLS | LM | CLASSIFICATION | SEQUENCE)
    raise ValueError(f"Unknown task '{cfg.task}'. Known: {known}")
