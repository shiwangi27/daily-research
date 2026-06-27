---
name: new-experiment
description: Scaffold a new daily experiment that changes exactly one knob versus a baseline. Use when planning the day's experiment, e.g. "set up today's experiment trying the cosine scheduler" or "create an experiment for LoRA rank 16".
---

# new-experiment

Create a one-knob experiment folder, correctly derived from a baseline.

## Steps

1. **Identify the baseline.** Usually `experiments/2026-06-27-baseline` or the current promoted
   baseline. Read its `config.yaml`.

2. **Create the folder:** `experiments/<YYYY-MM-DD>-<slug>/` where `<slug>` names the knob, e.g.
   `2026-06-28-cosine-sched`, `2026-06-29-lora-r16`, `2026-06-30-lr-1e-3`.

3. **Write `config.yaml` by copying the baseline and changing ONE field.** Then update metadata:
   - `name`: short unique id matching the slug.
   - `hypothesis`: a falsifiable sentence — what you expect and why.
   - `compare_to`: the baseline folder name.
   Leave every other field identical to the baseline. The diff between the two configs should be
   the single knob plus metadata.

4. **Self-check the diff:**
   ```bash
   diff <(python3 -c "import yaml,sys;print(yaml.safe_dump(yaml.safe_load(open('experiments/<baseline>/config.yaml'))))") \
        <(python3 -c "import yaml,sys;print(yaml.safe_dump(yaml.safe_load(open('experiments/<new>/config.yaml'))))")
   ```
   Confirm only the intended knob (and metadata) differ.

5. Hand off to `run-experiment`.

## Knob menu (one per day)

| Theme            | Knobs to vary                                                          |
|------------------|------------------------------------------------------------------------|
| Optimization     | `optimizer` (adamw/sgd/adafactor/lion), `lr`, `momentum`, `weight_decay`|
| LR schedule      | `scheduler` (none/linear/cosine), `warmup_ratio`                        |
| Regularization   | `loss` (label_smoothing/focal), `label_smoothing`, `lora.dropout`      |
| Efficiency       | `lora.enabled`/`r`/`alpha`, `freeze_encoder`, `quantize_dynamic`        |
| Capacity/data    | `model.name`, `data.max_train`, `data.max_length`, `batch_size`, `epochs`|

Adding a *new* knob (a new optimizer/loss/model family) means extending `harness/` first — that
is itself a worthy "one thing" for a day.
