"""Config schema for experiments.

A config is a YAML file with three sections — data / model / train — plus top-level
metadata including the post-training `stage`. Everything has a default so a minimal
config is short, and a full config is fully declarative (no code in experiment folders).

No external network egress is available (HuggingFace and pytorch.org are blocked by
policy). Data is generated locally; models are built and trained from scratch. The point
is to exercise the real LLM post-training stack — SFT, reward modeling, DPO, GRPO — at a
tiny CPU scale on a *verifiable* task where rewards are programmatic.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import yaml


@dataclass
class DataConfig:
    # task: the model's "world".
    #   textcls                     -> real-world text classification (HF dataset or synthetic)
    #   arith                       -> verifiable toy reasoning (a OP b = c), programmatic reward
    #   copy | sort                 -> simple sequence tasks (legacy sanity checks)
    #   digits|wine|breast_cancer|synth_cls -> vector classification (fast optimization side-track)
    task: str = "arith"
    max_train: int = 8000
    max_eval: int = 1000
    seed: int = 0                   # data-generation seed (separate from the train seed)
    # arith params
    op: str = "add"                 # add | sub
    max_digits: int = 2             # operands drawn in [0, 10**max_digits)
    reverse_answer: bool = True     # emit answer least-significant-digit first (helps carries)
    # text-classification params (task == textcls)
    source: str = "synth"           # "synth" (offline) | "hf" (HuggingFace dataset)
    hf_repo: Optional[str] = None   # e.g. "takala/financial_phrasebank"
    hf_config: Optional[str] = None # e.g. "sentences_allagree"
    text_field: str = "text"
    label_field: str = "label"
    max_len: int = 48               # token truncation length
    max_vocab: int = 5000           # word-tokenizer vocabulary cap
    imbalance: float = 6.0          # synth only: class-frequency skew (majority:minority)
    # synthetic vector-classification params (task == synth_cls)
    n_features: int = 40
    n_informative: int = 10
    n_classes: int = 4
    # legacy sequence-task params (task in {copy, sort})
    seq_len: int = 10
    vocab: int = 12


@dataclass
class LoraConfig:
    enabled: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["attn_q", "attn_v"])


@dataclass
class ModelConfig:
    # arch: nanolm (modern decoder-only LM) | mlp (vector classification)
    arch: str = "nanolm"
    # --- nanolm (RoPE, RMSNorm, ReLU^2, QK-norm, no-bias, untied embeddings) ---
    d_model: int = 128
    n_heads: int = 4
    n_kv_heads: Optional[int] = None   # None -> = n_heads (no GQA); set lower for GQA
    n_layers: int = 3
    block_size: int = 48               # max context length
    dropout: float = 0.0
    # --- mlp ---
    hidden: int = 128
    depth: int = 2
    # --- shared techniques ---
    lora: LoraConfig = field(default_factory=LoraConfig)
    freeze: list[str] = field(default_factory=list)
    quantize_dynamic: bool = False


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 64
    loss: str = "cross_entropy"     # cross_entropy | label_smoothing | focal (supervised stages)
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    optimizer: str = "adamw"        # adamw | sgd | adafactor | lion
    lr: float = 3.0e-3
    momentum: float = 0.9
    weight_decay: float = 0.01
    scheduler: str = "cosine"       # none | linear | cosine
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    eval_every_epoch: bool = True
    # --- classification ---
    class_weight: str = "none"      # none | balanced (inverse-frequency, for imbalanced sets)
    # --- SFT ---
    mask_prompt: bool = True        # mask loss on prompt tokens (the key SFT knob)
    # --- DPO (implemented on a scheduled day) ---
    dpo_beta: float = 0.1
    # --- GRPO / RLVR (implemented on a scheduled day) ---
    grpo_group: int = 8             # samples per prompt
    grpo_kl: float = 0.0            # KL-to-reference coefficient
    sample_temperature: float = 1.0


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    description: str = ""
    hypothesis: str = ""
    # stage: which part of the pipeline this experiment runs.
    #   classify -> real-world text classification (chase a benchmark)
    #   pretrain -> next-token LM on raw corpus
    #   sft      -> instruction tuning with prompt-loss masking
    #   rm       -> reward model (Bradley-Terry)         [scheduled]
    #   dpo      -> direct preference optimization        [scheduled]
    #   grpo     -> RL with verifiable reward             [scheduled]
    #   supervised -> vector classification / legacy seq tasks
    stage: str = "pretrain"
    init_from: Optional[str] = None  # path to a base checkpoint (e.g. a pretrain/sft result dir)
    seed: int = 1234
    needs_gpu: bool = False
    compare_to: Optional[str] = None
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(dc_cls, raw: Optional[dict[str, Any]]):
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
