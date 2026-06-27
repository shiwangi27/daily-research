"""Config schema for experiments.

A config is a YAML file with three sections — data / model / train — plus top-level
metadata. Everything has a default so a minimal config is short, and a full config is
fully declarative (no code in experiment folders).

This environment has NO external network egress for models/datasets (HuggingFace and
pytorch.org are blocked by policy). So data is bundled-in-sklearn or generated locally,
and models are built from scratch. Techniques (optimizers, LR, loss, LoRA, quantization)
are exercised at tiny CPU scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import yaml


@dataclass
class DataConfig:
    # task selects both the data source and the modality.
    #   classification (vector in): digits | wine | breast_cancer | synth_cls
    #   sequence (token LM):        copy | sort
    task: str = "sort"
    max_train: int = 4000
    max_eval: int = 1000
    seed: int = 0                   # data-generation / split seed (separate from train seed)
    # synthetic classification params (task == synth_cls)
    n_features: int = 40
    n_informative: int = 10
    n_classes: int = 4
    # sequence-task params (task in {copy, sort})
    seq_len: int = 10               # length of the source sequence
    vocab: int = 12                 # number of distinct value tokens


@dataclass
class LoraConfig:
    enabled: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    # substring match against Linear module names to wrap (e.g. ["attn", "fc"])
    target_modules: list[str] = field(default_factory=lambda: ["attn"])


@dataclass
class ModelConfig:
    # arch: mlp (vector classification) | tiny_gpt (sequence LM)
    arch: str = "tiny_gpt"
    # --- mlp ---
    hidden: int = 128
    depth: int = 2
    dropout: float = 0.0
    # --- tiny_gpt ---
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    block_size: int = 32            # max context length
    # --- shared techniques ---
    lora: LoraConfig = field(default_factory=LoraConfig)
    freeze: list[str] = field(default_factory=list)  # substrings of params to freeze
    quantize_dynamic: bool = False  # post-training dynamic int8 (eval-only on CPU)


@dataclass
class TrainConfig:
    epochs: int = 6
    batch_size: int = 64
    loss: str = "cross_entropy"     # cross_entropy | label_smoothing | focal
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    optimizer: str = "adamw"        # adamw | sgd | adafactor | lion
    lr: float = 3.0e-3
    momentum: float = 0.9           # sgd only
    weight_decay: float = 0.01
    scheduler: str = "cosine"       # none | linear | cosine
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    eval_every_epoch: bool = True


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    description: str = ""
    hypothesis: str = ""            # what we expect to learn / the one knob varied
    seed: int = 1234
    needs_gpu: bool = False
    compare_to: Optional[str] = None  # name/dir of the baseline this is measured against
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(dc_cls, raw: Optional[dict[str, Any]]):
    """Build a dataclass from a raw dict, rejecting unknown keys (typos fail loud)."""
    if raw is None:
        return dc_cls()
    fields = {f.name for f in dc_cls.__dataclass_fields__.values()}
    for key in raw:
        if key not in fields:
            raise ValueError(f"Unknown config key '{key}' for {dc_cls.__name__}")
    return dc_cls(**raw)


def load_config(path: str) -> ExperimentConfig:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    data = _merge(DataConfig, raw.pop("data", None))

    model_raw = dict(raw.pop("model", None) or {})
    lora = _merge(LoraConfig, model_raw.pop("lora", None))
    model = _merge(ModelConfig, model_raw)
    model.lora = lora

    train = _merge(TrainConfig, raw.pop("train", None))

    cfg = ExperimentConfig(**raw)
    cfg.data = data
    cfg.model = model
    cfg.train = train
    return cfg
