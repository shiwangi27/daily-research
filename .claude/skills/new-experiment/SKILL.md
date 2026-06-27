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

The `stage` is usually fixed by where we are in the pipeline; vary one knob *within* it.

| Theme            | Knobs to vary                                                            |
|------------------|--------------------------------------------------------------------------|
| SFT mechanics    | `train.mask_prompt`, chat-template/special tokens, `epochs` (overfit)     |
| Preference (DPO) | `train.dpo_beta` (the KL strength), reference = base vs SFT               |
| RL (GRPO/RLVR)   | `train.grpo_kl`, `train.grpo_group`, `train.sample_temperature`          |
| PEFT             | `model.lora.enabled`/`r`/`alpha`/`target_modules`, `model.freeze`         |
| Architecture     | RoPE↔learned, RMSNorm↔LayerNorm, ReLU²↔GELU, QK-norm on/off, GQA `n_kv_heads` |
| Optimization     | `optimizer` (adamw/sgd/adafactor/lion), `lr`, `scheduler`, `warmup_ratio`, `weight_decay` |
| Efficiency       | `model.quantize_dynamic`                                                  |
| Task/data        | `data.max_digits`, `data.op`, `data.reverse_answer`, `data.max_train`, `batch_size` |

Adding a *new* stage or knob (the `rm`/`dpo`/`grpo` stages, a new architecture component, a new
optimizer/loss) means extending `harness/` first — itself a worthy "one thing" for a day.
