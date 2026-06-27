---
name: analyze-results
description: Interpret an experiment's result.json against its baseline, decide signal vs noise, inspect failing examples, and write the lab_notebook.md entry. Use after a run, e.g. "analyze today's results" or "did the cosine schedule help?".
---

# analyze-results

Turn a `result.json` into an honest, recorded conclusion.

## Steps

1. **Load both results.** Today's `result.json` and its `compare_to` baseline's. Extract
   `final` metrics, `wall_seconds`, `params.trainable`.

2. **Compute deltas.** accuracy, macro_f1, eval_loss, wall-time, trainable params. Be precise
   (e.g. "+0.8 acc, −0.02 eval_loss, 1.4× faster, 6% of params trainable").

3. **Signal vs noise.** Std error of accuracy ≈ `sqrt(p*(1-p)/N)` for N eval examples. If the
   accuracy delta is within ~1–2× that, label it a wash and recommend a seed-averaged re-run
   before believing it. Tiny eval sets lie.

4. **Inspect errors.** Write a short throwaway script: reload the eval set, run the model, dump
   the highest-loss / misclassified examples. Read them. Look for class-concentrated failures,
   length effects, degenerate all-one-class predictions.

5. **Write the notebook entry.** Append to `lab_notebook.md` using this shape:

   ```markdown
   ## <YYYY-MM-DD> — <one-knob title>
   - **Hypothesis:** ...
   - **Setup:** <model> on <dataset> (<train>/<eval> subset), changed only <knob> vs <baseline>.
   - **Result:** acc X (Δ +/−), macro-F1 Y, eval-loss Z, Ns wall. <signal/noise call>
   - **From the errors:** <what the failing examples showed>
   - **Verdict:** helped / hurt / wash — and the mechanism.
   - **Next:** <the single most informative next knob, with reason>
   ```

6. **Maybe promote.** If today beat the baseline repeatably (re-run with a 2nd seed), recommend
   the lead promote this config to baseline and note it.

## Guardrails

- Never record a conclusion you couldn't defend from the data. "Uninterpretable, re-run" is a
  valid entry.
- One knob only — if the diff touched more than one field, the comparison is void; say so.
