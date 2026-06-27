# daily-research

A personal lab for **everyday model-training research and self-improvement**: fine-tune
small open-source models on open datasets, one small experiment at a time, driven by a
team of collaborating research agents.

## The idea

- **One small thing, daily.** Each day we change *exactly one knob* (learning rate,
  optimizer, LoRA rank, quantization, a layer trick, a loss function, …), run a fast
  experiment, and write down what happened. Controlled ablations, not heroics.
- **An agent team does the work.** A lead PhD researcher writes goals and the daily plan;
  data scientists find/prepare data; senior researchers build and train models; an error
  analyst digs into failures. They are real Claude Code subagents (`.claude/agents/`).
- **Everything is reproducible from git.** The container is ephemeral — code, configs, and
  results all live in the repo. A day's experiment = a config + a logged result.

## Compute & environment reality

This runs **CPU-only** (no GPU) in an ephemeral container, and **egress is restricted** —
`huggingface.co` and `download.pytorch.org` are blocked by org policy; only PyPI is reachable.
So we **don't download pretrained models or datasets**. Instead we **train small models from
scratch on offline data**:

- **Models from scratch:** a tiny MLP and a **TinyGPT** (a real decoder-only transformer,
  ~100K params) — which exercises *architecture & layer optimization* directly.
- **Offline data:** scikit-learn's *bundled* datasets (`digits`, `wine`, `breast_cancer`) and
  locally-generated **algorithmic tasks** (copy / sort) for the sequence model.
- **All the techniques still apply:** optimizers (AdamW/SGD/Lion/Adafactor), LR schedules,
  loss functions (CE / label-smoothing / focal), weight decay, grad clipping, **LoRA**
  (implemented from scratch on `nn.Linear`), and **dynamic int8 quantization**.

Every run finishes in *seconds-to-minutes*. A config can be flagged `needs_gpu: true` to be
run elsewhere (e.g. on a real HF model) and have its results committed back. If you want to
use real HuggingFace models/datasets here, the egress policy would need `huggingface.co`
allow-listed.

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
python3 -m harness.train experiments/_template/config.yaml      # ~0.4s smoke test
python3 -m harness.train experiments/2026-06-27-baseline/config.yaml   # ~8s, the baseline
```

See `CLAUDE.md` for conventions the agent team follows.
