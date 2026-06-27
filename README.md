# daily-research

A personal lab for **everyday LLM-research and self-improvement**: build the real
**post-training stack** (SFT → reward modeling → DPO → GRPO) on a *modern* tiny transformer,
from scratch, one small experiment at a time, driven by a team of collaborating research
agents. Think "nano-nanochat" — the LLM pipeline shrunk to run on a CPU in seconds.

## The idea

- **One small thing, daily.** Each day we change *exactly one knob* (prompt-loss masking,
  DPO's β, the KL coefficient, LoRA rank, an architecture component, …), run a fast
  experiment, and write down what happened. Controlled ablations, not heroics.
- **An agent team does the work.** A lead PhD researcher writes goals and the daily plan;
  data scientists find/prepare data; senior researchers build and train models; an error
  analyst digs into failures. They are real Claude Code subagents (`.claude/agents/`).
- **Everything is reproducible from git.** The container is ephemeral — code, configs, and
  results all live in the repo. A day's experiment = a config + a logged result.

## Compute & environment reality

This runs **CPU-only** (no GPU) in an ephemeral container, and **egress is restricted** —
`huggingface.co` and `download.pytorch.org` are blocked by org policy; only PyPI is reachable.
So we **don't download pretrained models or datasets**. Instead we **build the LLM stack from
scratch on offline data**:

- **`NanoLM` — the LLM core recipe, shrunk:** RoPE positions, RMSNorm (no learnable params),
  ReLU² MLP, QK-norm, no biases, untied embeddings, optional GQA, greedy `generate()`. The
  same architecture family as nanochat/Llama, at ~0.6M params.
- **A *verifiable* world:** arithmetic (`a + b = c`). The reward is a programmatic check, so we
  get **RLVR/GRPO with no reward model and no human labels**, and ground-truth SFT/preference
  data for free. (A fast vector-classification side-track on sklearn `digits` also exists.)
- **The real post-training stack** as pipeline `stage`s: `pretrain` → `sft` (with prompt-loss
  masking) → `rm` (Bradley-Terry reward model) → `dpo` → `grpo`. Plus from-scratch **LoRA**,
  **dynamic int8 quantization**, and a full optimizer/scheduler/loss toolkit.

Every run finishes in *seconds-to-minutes*. A config can be flagged `needs_gpu: true` to be run
elsewhere (e.g. fine-tuning a real HF model) and have its results committed back; using real
HuggingFace models/datasets here would need `huggingface.co` allow-listed in the egress policy.

## Layout

```
.claude/agents/      # the research team (lead, data scientist, researcher, error analyst)
.claude/skills/      # repeatable workflows (run-experiment, analyze-results, new-experiment)
harness/             # config-driven training/eval engine (data, model, train, metrics)
experiments/         # one folder per experiment: config.yaml + (committed) results
results/             # aggregated metrics, plots
lab_notebook.md      # the running narrative — what we tried, what we learned
schedule.md          # the rolling daily plan, written by the lead researcher
```

## A day in the loop

1. **Start a session.** The lead researcher reviews `lab_notebook.md` + `schedule.md` and
   picks today's one experiment.
2. **Prepare.** Data scientist confirms the dataset; researcher writes/updates a
   `config.yaml` under `experiments/<date>-<slug>/`.
3. **Run.** `python3 -m harness.train experiments/<date>-<slug>/config.yaml` (or the
   `run-experiment` skill). Results land next to the config as `result.json`.
4. **Analyze.** The error analyst / researcher compares against the baseline and the prior
   day, updates `lab_notebook.md` with a verdict and a hypothesis for tomorrow.
5. **Commit.** Code + config + result + notebook entry are committed together.

## Quickstart

```bash
python3 -m pip install -r requirements.txt
python3 -m harness.train experiments/_template/config.yaml             # ~1s smoke test
python3 -m harness.train experiments/2026-06-27-pretrain-base/config.yaml   # ~24s, the base LM
python3 -m harness.train experiments/2026-06-27-sft-masked/config.yaml      # SFT (prompt-masked)
```

See `CLAUDE.md` for conventions the agent team follows.
