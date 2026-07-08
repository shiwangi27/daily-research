# Research plan — 2026-07-08 (Week 1, Day 1: chat template / special tokens)

**Status: planning only.** Nothing has been run today. This document frames the one
experiment for the next execution session; `schedule.md` Day 1 stays unchecked until
`experiments/2026-07-08-sft-role-token/result.json` exists and has been reviewed.

## Where we are

- Baseline for SFT comparisons: `experiments/2026-06-27-sft-masked` (`mask_prompt: true`,
  10 epochs, exact-match **1.000**, groks by epoch 5, eval-loss 0.004).
- Current format (`harness/data.py::_arith`): every example is
  `<bos> A..A <op> B..B = C..C <eos>`, char-level tokenizer, vocab =
  `<pad><bos><eos>` + `0123456789+-=`. The prompt/answer boundary is marked *only* by the
  ordinary character `"="` — there is no dedicated, reserved "start-of-answer" /
  role token distinct from the arithmetic alphabet. `mask_prompt` finds the boundary via
  `ans_start = len(pid) - 1`, i.e. by position, not by a special symbol the model attends to
  as "role" information.
- So today's schedule item, "add explicit role markers — does formatting consistency
  change generalization to held-out prompts?", maps concretely onto: **does giving the
  prompt→answer boundary its own reserved special-token embedding (instead of overloading
  the ordinary vocab character `"="`) change anything?** That is a faithful, minimal reading
  of "chat template / special tokens" in a world with no natural-language turns.

## Today's one experiment

**Name:** `2026-07-08-sft-role-token` (to be created next session by `senior-researcher`,
not today).

**Compare to:** `experiments/2026-06-27-sft-masked` (same op, digits, split sizes, model
size, optimizer, schedule, seed — the current best SFT config).

**The one knob:** a new `data.explicit_turn_token: bool` field. When `true`, insert one new
special token, `<assistant>`, between the prompt and the answer:

- baseline (`sft-masked`, `explicit_turn_token: false`): `<bos> A op B = | C.. <eos>`
  (boundary = plain `"="`)
- variant (today, `explicit_turn_token: true`): `<bos> A op B = <assistant> | C.. <eos>`
  (boundary = a dedicated embedding, reserved only for this role, never reused elsewhere)

Everything else held fixed: `op: add`, `max_digits: 2`, `reverse_answer: true`,
`max_train: 6000`, `max_eval: 1000`, data `seed: 0`; model `d_model/n_heads/n_layers/
block_size` unchanged (adding one token keeps the sequence at 11 ids, well under
`block_size: 16` — no resize needed); `train.*` identical to `sft-masked` (10 epochs,
adamw, lr 3e-3, cosine, warmup 0.1, `mask_prompt: true`, same seed 1234). `mask_prompt`'s
boundary shifts automatically (it's computed from `len(pid)`, and `pid` now includes the
new token), so masking semantics stay equivalent — the answer region is still the only
thing that gets gradient.

## Hypothesis (falsifiable)

`"="` already unambiguously and consistently marks the prompt/answer boundary — it appears
exactly once, always in the same position, in every example, regardless of `op` (`+` or
`-`). A dedicated `<assistant>` special token is therefore *redundant* information.

**Prediction:** exact-match and epoch-to-grok on `2026-07-08-sft-role-token` will be within
noise of `sft-masked` (EM difference ≤ ~1-2 pp given eval_n=1000; grok epoch ±1). This would
be a clean negative: explicit role tokens don't help when the implicit boundary is already
unambiguous — the interesting content of "chat templates matter" in real LLMs is more about
train/inference format *consistency* than about token specialness per se.

**Disconfirmed if:** the new run groks meaningfully sooner (e.g. epoch 3-4 instead of 5) or
closes the residual gap to 100% EM faster / more stably across a second seed. That would be
the actually interesting result — it would say the model benefits from a boundary marker
with its own reserved embedding (not entangled with any other role in the vocab) more than
from an equally-positioned ordinary character, and would motivate a fuller `<user>/
<assistant>` wrap next.

## Concrete config sketch (do not create yet — for `senior-researcher` next session)

`experiments/2026-07-08-sft-role-token/config.yaml`:

```yaml
name: sft-role-token
stage: sft
description: SFT with a dedicated <assistant> special token marking the prompt->answer
  boundary, instead of relying only on the plain "=" character. Tests whether an explicit,
  reserved role-boundary embedding adds anything beyond an already-unambiguous positional cue.
hypothesis: "= already unambiguously marks the prompt/answer boundary; a redundant
  <assistant> special token will not move exact-match or grok-epoch beyond noise
  vs sft-masked."
seed: 1234
needs_gpu: false
compare_to: 2026-06-27-sft-masked

data: {task: arith, op: add, max_digits: 2, reverse_answer: true, max_train: 6000,
       max_eval: 1000, seed: 0, explicit_turn_token: true}
model: {arch: nanolm, d_model: 128, n_heads: 4, n_layers: 3, block_size: 16, dropout: 0.0,
        lora: {enabled: false}}
train:
  epochs: 10
  batch_size: 64
  loss: cross_entropy
  optimizer: adamw
  lr: 3.0e-3
  weight_decay: 0.01
  scheduler: cosine
  warmup_ratio: 0.1
  grad_clip: 1.0
  mask_prompt: true
```

Run a second seed (e.g. `seed: 5678`, same `data.seed: 0`) before drawing any conclusion —
`sft-masked` vs `sft-unmasked` already showed the final-EM gap can be small (1.0 vs 0.976),
so a single-seed comparison here would not be trustworthy either way.

## Harness extension needed (flag now, implement next session)

`harness/data.py` and `harness/config.py` do not currently support a role/turn token at
all — this is new, not a config toggle on existing behavior. Concretely:

1. `harness/config.py::DataConfig` — add `explicit_turn_token: bool = False`.
2. `harness/data.py::SPECIALS` — add `"<assistant>"` to the specials list (vocab_size +1;
   nothing else hard-codes vocab size, everything reads `tok.vocab_size`, so this should be
   a clean +1 that flows through automatically).
3. `harness/data.py::_arith` — when `cfg.explicit_turn_token`, append
   `tok.stoi["<assistant>"]` to `pid` right after the `"="` token, before appending the
   answer/eos region. `ans_start = len(pid) - 1` is already computed *after* `pid` is built,
   so `mask_prompt` and generation-start logic should need no further change — but
   `senior-researcher` must verify this by hand on one example, and must double check that
   `eval_prompts` (used for greedy `generate()` at eval time) includes the new token too, so
   train-time and inference-time formats match exactly. (This is the actual crux of "chat
   templates matter" in real LLMs: a train/inference format mismatch silently breaks
   generation — worth calling out explicitly in the harness docstring when this lands.)
4. No changes needed to `harness/model.py` or `harness/train.py` if vocab size is read
   dynamically from the tokenizer (spot-check this assumption first).

This is a small, contained harness change in service of one data-format knob — consistent
with "extend the harness, don't hand-write training loops in experiment folders."

## What "done" looks like for today

Today = this plan only. Next session:
1. `data-scientist` — sanity check that `explicit_turn_token: true` produces well-formed
   sequences (spot-check a few decoded examples), no leakage/off-by-one in `ans_start`, and
   confirm the extra token doesn't silently overflow `block_size`.
2. `senior-researcher` — implement the 3-line harness extension above, write the config,
   run at seed 1234 *and* a second seed, report EM / grok-epoch / eval-loss for both vs
   `sft-masked`.
3. `error-analyst` — read `result.json` for both seeds, decide if the EM/grok-epoch delta
   exceeds noise (use the `sft-masked` vs `sft-unmasked` gap, ~2.4pp EM, as a rough noise
   reference), and propose whether Day 1 continues into a fuller `<user>/<assistant>`
   wrap or whether we move on to Day 2 (SFT epochs & overfitting) with a documented null
   result.
4. Lead writes the `lab_notebook.md` entry and updates `schedule.md` (check off Day 1) once
   results are in.
