# Lab notebook

The running narrative of the lab. Newest entries on top. Each entry is one day / one knob.
Written by the lead researcher, with the error analyst's interpretation folded in.

---

## 2026-07-08 — Day 1: real-world direction + text-classification capability

**Decision.** Pivot the near-term goal from synthetic arithmetic to **real-world domain data
with a public benchmark to chase** (user's call). Target domain: **financial** — the
**Financial PhraseBank** (3-class news-sentence sentiment; FinBERT scores **0.86 acc / 0.84 F1**
full, **0.97 / 0.95** on the all-agree subset). We train *from scratch* (no pretraining), so we
expect to land well below FinBERT — the point is to measure the gap and close it one knob/day.

**Egress note.** HuggingFace is now allow-listed, but the change only applies to **new
sessions** — this sandbox is still blocked (403). So today's work is the part that needs no
egress: building + validating the machinery. The real FPB baseline runs next session.

**Harness capability added (the day's "one thing").** Text classification on `NanoLM`:
- `NanoLMClassifier` — the LM trunk + a head over the **last non-pad token** (causal pooling).
- `WordTokenizer` — a from-scratch word-level tokenizer fit on the training split only.
- `stage: classify`, plus imbalance remedies: **inverse-frequency class weights**
  (`train.class_weight: balanced`) and **focal loss**, and macro-F1 reporting.
- A `source: hf` adapter (`datasets.load_dataset`) for real data — written, but UNVALIDATED
  until an HF-enabled session (datasets 5.x dropped script loaders → we target parquet repos).

**Offline validation (`2026-07-08-textcls-validation`).** A synthetic imbalanced (6:1) task as
a proxy. Result is exactly the dynamic we care about:

| loss / weighting        | accuracy | macro-F1 |
|-------------------------|----------|----------|
| cross-entropy (plain)   | 0.888    | **0.463** |
| cross-entropy, balanced | 0.913    | **0.562** |
| focal                   | 0.895    | 0.474    |

- **Lesson previewed:** under imbalance, **accuracy lies** — 0.89 acc hides a 0.46 macro-F1
  (minorities missed). Inverse-frequency weighting buys **+0.10 macro-F1**. This is the exact
  playbook we'll run on Financial PhraseBank, which is similarly skewed (~60/28/12).

**Next session (HF-enabled):** run `2026-07-08-fpb-baseline` (from-scratch NanoLM on FPB
all-agree), record accuracy + macro-F1 vs FinBERT's 0.97/0.95, then attack the gap: tokenizer
(BPE vs word), class weighting, focal, depth/width, LoRA, label smoothing — one per day.

---

## 2026-06-27 — Day 0: post-training pipeline bring-up (modern arch + base + SFT masking)

**Reframing (why this looks different from a first sketch).** An initial harness trained
toy models (an MLP; a TinyGPT on a "sort" task). That teaches optimizer mechanics, not *LLM
fine-tuning*. We pivoted to the thing actually worth learning: the **post-training stack**
(SFT → reward modeling → DPO → GRPO) on a **modern** transformer, in a **verifiable** world.
Grounded in: Karpathy's **nanochat** (the minimal full ChatGPT pipeline — RoPE, RMSNorm,
ReLU², QK-norm, no-bias, untied embeddings); **TinyStories** (1–35M-param models learn real
structure from scratch); and CPU-scale **DPO/GRPO** being feasible.

**Environment constraints (unchanged):** CPU-only, and egress-blocked (`huggingface.co` /
`pytorch.org` → 403). So everything trains from scratch on locally-generated data. We chose a
**verifiable** task (arithmetic) precisely because the reward is a programmatic check — no
reward model or human labels needed for the RL signal, and ground-truth SFT/preference data
is free.

**Architecture.** Rebuilt the model as `NanoLM`, the LLM core recipe shrunk to CPU: **RoPE**
rotary positions, **RMSNorm** (no learnable params), **ReLU² MLP**, **QK-norm**, **no biases**,
**untied** embeddings, optional GQA, and a greedy `generate()`. ~0.6M params at default size.

**Base model (`2026-06-27-pretrain-base`).** Next-token pretraining on 2-digit addition.
Key trick: emit the answer **least-significant-digit-first** — carries propagate left-to-right
that way, which a tiny causal model can actually learn. Result: a textbook **grokking** curve,
exact-match 0.06 → 0.33 → **0.89** → **0.97** over epochs 3–7, **~24 s** on CPU. Without the
reversal, the same model was stuck near 3% — a clean lesson in input/output formatting.

**Day-0 experiment — prompt-loss masking (`sft-masked` vs `sft-unmasked`).** One knob:
`train.mask_prompt`. Identical otherwise (same data/seed/budget, 10 epochs).

| run          | exact-match (final) | groks by | eval-loss |
|--------------|---------------------|----------|-----------|
| sft-masked   | **1.000**           | epoch 5  | **0.004** |
| sft-unmasked | 0.976               | epoch 7  | 0.946     |

- EM trajectories — masked: `… 0.44, 0.87, 0.995, 1.0 …`; unmasked: `… 0.04, 0.26, 0.81, 0.93, 0.98`.
- **Verdict / mechanism:** masking the prompt focuses all gradient on the answer, so the model
  groks ~2 epochs sooner and reaches 100%. The deeper lesson is the **loss**: unmasked SFT
  loss floors at ~0.95 because it keeps trying to predict the *random* prompt digits (irreducible
  entropy ≈ log 10 per digit) — so you **cannot read task success off an unmasked SFT loss**.
  This is exactly why real SFT masks the prompt.
- **Signal vs noise:** the EM gap at the budget end is small (1.0 vs 0.976), but the
  *convergence-speed* and *loss-interpretability* gaps are large and unambiguous.

**Harness state:** `pretrain`, `sft`, and `supervised` (classification/legacy seq) stages work.
`rm` / `dpo` / `grpo` are stubbed with informative errors — they're the next scheduled days.

**Next (Day 1):** chat templating / explicit role tokens — does formatting consistency improve
generalization to held-out prompts? Then harder arithmetic (3-digit) to open headroom for RL.

### How to reproduce
```bash
python3 -m pip install -r requirements.txt
python3 -m harness.train experiments/2026-06-27-pretrain-base/config.yaml
python3 -m harness.train experiments/2026-06-27-sft-masked/config.yaml
python3 -m harness.train experiments/2026-06-27-sft-unmasked/config.yaml
```
