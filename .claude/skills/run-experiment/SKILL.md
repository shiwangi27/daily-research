---
name: run-experiment
description: Run a daily training experiment end-to-end — validate the config changes one knob vs its baseline, execute the harness on CPU, and report results. Use when the user says "run today's experiment", "train the model", or points at an experiment config.
---

# run-experiment

Execute one experiment cleanly and report results.

## Steps

1. **Locate the config.** It's `experiments/<dir>/config.yaml`. If the user named a knob but no
   config exists yet, create the folder by copying the `compare_to` baseline config and changing
   the single field (see the `new-experiment` skill).

2. **Pre-flight checks** (cheap, do them every time):
   - Confirm `compare_to` points at an existing experiment with a `result.json`.
   - Diff this config against the baseline config and confirm **exactly one knob changed**. If
     more than one differs, stop and flag it — that breaks causal attribution.
   - Confirm `needs_gpu` is false (no GPU here). If true, stop and tell the user to run externally.

3. **Run it:**
   ```bash
   python3 -m harness.train experiments/<dir>/config.yaml
   ```
   Always `python3 -m` — the deps live in that interpreter.

4. **Verify output.** Confirm `result.json` was written next to the config and parses. Read the
   `final` metrics and `wall_seconds`.

5. **Report.** Summarize: the one knob, final accuracy/macro-F1/eval-loss, delta vs baseline,
   wall-time, trainable params. Hand off to the error-analyst for interpretation if this is part
   of the daily loop.

## Guardrails

- If a run exceeds a few minutes, it's mis-scoped for CPU — shrink `max_train` or the model and
  note it, don't just wait.
- Never edit unrelated config fields to "make it work". The config is the experiment.
- Commit config + result.json together; results without their config are unreproducible.
