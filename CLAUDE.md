# CLAUDE.md — conventions for the research team

This repo is a daily model-training lab. Read this before doing work here.

## Prime directives

1. **One knob per day.** Every experiment changes exactly *one* variable versus a named
   baseline. If you're tempted to change two things, that's two experiments. The whole point
   is clean causal attribution.
2. **It must run on CPU in seconds-to-minutes.** No GPU. And **no network egress for
   models/data** — `huggingface.co` and `pytorch.org` are blocked by policy; only PyPI works.
   So everything trains **from scratch on offline data** (bundled sklearn datasets +
   locally-generated tasks). Do not add code that downloads models/datasets — it will 403.
   If something genuinely needs a GPU or a real pretrained model, set `needs_gpu: true` and
   stop — queue it for an external run, don't brute-force it here.
3. **Reproducible from git.** The container is ephemeral. Commit code, config, AND results
   together. Set seeds. Pin the data subset size. Never rely on uncommitted local state.
4. **Write down what you learned.** An experiment without a `lab_notebook.md` entry didn't
   happen. The notebook is the product; the models are disposable.

## How an experiment is defined

One experiment = one folder `experiments/<YYYY-MM-DD>-<slug>/` containing:

- `config.yaml` — fully specifies data, model, training, and what knob this varies.
- `result.json` — written by the harness: metrics, timing, environment, git SHA.
- (optional) `notes.md` — error analyst / researcher commentary.

Run it with:

```bash
python3 -m harness.train experiments/<dir>/config.yaml
```

Always `python3 -m` — this box has a second pip/python pair, and the deps live in the
`python3` interpreter. The config schema lives in `harness/config.py`. Keep configs
declarative — if you need new behavior, extend the harness, don't hand-write training loops
in experiment folders.

The project is a "nano-nanochat": the LLM **post-training stack** on a modern tiny transformer
(`NanoLM`: RoPE, RMSNorm, ReLU², QK-norm, no-bias, untied embeddings), in a **verifiable**
world (`arith`). Experiments declare a pipeline `stage`: `pretrain` → `sft` → `rm` → `dpo` →
`grpo` (rm/dpo/grpo are stubs to be built on their scheduled days). Chain stages with
`init_from: experiments/<dir>` (loads that run's gitignored `model.pt`). Separately, `stage:
classify` fine-tunes a `NanoLMClassifier` on real-world text (`task: textcls`, `source: hf` or
`synth`) to chase public benchmarks — the current focus (needs `huggingface.co` egress). A
vector-classification side-track (`mlp` on sklearn data, `stage: supervised`) exists too.
Adding a new stage/arch/task/optimizer/loss means extending `harness/` (and `harness/config.py`)
— itself a fine "one thing" for a day.

## Baselines

The first baseline is `experiments/2026-06-27-baseline/`. Every later experiment names the
baseline it compares against in its config (`compare_to:`). When you beat a baseline
meaningfully and repeatably, promote the new config to be the baseline and note it.

## Daily workflow (who does what)

- **lead-researcher** — owns `schedule.md` and the goal. Picks today's one experiment,
  frames the hypothesis, delegates. Updates the schedule as findings come in.
- **data-scientist** — owns data: sourcing, subset selection, leakage/balance checks,
  tokenization choices. Confirms the dataset is sane before training.
- **senior-researcher** — owns the model/training: writes the config, extends the harness if
  needed, runs the experiment, reports raw results.
- **error-analyst** — owns failure analysis: reads `result.json`, inspects misclassified /
  high-loss examples, forms the hypothesis for tomorrow.

The lead delegates to the others via the Agent tool. Keep each subagent's scope tight.

## Conventions

- Python 3.11, PyTorch CPU, scikit-learn, numpy, PyYAML. (transformers is optional — only the
  `adafactor` optimizer needs it.) No HF model/dataset downloads.
- Seeds: always set `seed: 1234` (or per-config) and seed torch + numpy + python `random`.
- Determinism over speed in the harness; speed comes from small models, not sloppy science.
- Results JSON always includes: metrics, wall-clock seconds, param counts, git SHA, config hash.
- Don't commit large artifacts. Tiny model checkpoints (<5MB) are OK if useful; otherwise
  results-only. Add big stuff to `.gitignore`.
- Prefer extending `harness/` over copy-pasting. The harness is the shared instrument.

## Schedule

`schedule.md` is the rolling plan. It's a living document — the lead rewrites the next few
days based on what we learn, rather than executing a frozen list.
