# Research schedule — rolling daily plan

**Owner:** lead-researcher. A *living* curriculum, not a frozen list. After each day's
finding, the lead rewrites the next few days. One informative knob per day.

**Primary track:** from-scratch **TinyGPT** learning to **sort** a token sequence (the
LLM-flavored core — real transformer, real next-token training).
**Fast side track:** **MLP on sklearn `digits`** for pure-optimization days that need to run
in under a second.
**Current baseline:** `experiments/2026-06-27-baseline` — TinyGPT/sort, AdamW, cosine, lr 3e-3,
8 epochs → exact-match **0.983** in ~8 s.

The point of a daily ablation is to build *intuition for the mechanism*, on a model small
enough to iterate in seconds. Numbers are secondary to what you learn about the lever.

---

## Week 1 — Optimization & learning rate
The biggest practical lever. Map this setup's sensitivity before touching anything else.

- [x] **Day 0 — Harness + baseline.** TinyGPT/sort reference established. (done 2026-06-27)
- [ ] **Day 1 — LR up.** lr 3e-3 → 1e-2. Does sort tolerate it or destabilize early?
- [ ] **Day 2 — LR down.** lr 1e-3. Map the speed/stability trade; find the rough optimum.
- [ ] **Day 3 — Optimizer: SGD+momentum** at a tuned LR. Feel what adaptivity was buying.
- [ ] **Day 4 — Optimizer: Lion** (use ~3–10× smaller LR). Note the LR coupling.
- [ ] **Day 5 — Optimizer: Adafactor.** Does the memory-lean optimizer match AdamW here?
- [ ] **Day 6 — Warmup ratio:** 0.0 vs 0.1 vs 0.3. How much does warmup matter at this scale?
- [ ] **Day 7 — Synthesis.** Write up the optimizer/LR landscape; promote a new baseline if won.

## Week 2 — Schedules, regularization & loss
- [ ] Schedule: cosine vs linear vs none (same warmup).
- [ ] Weight decay: 0.0 vs 0.01 vs 0.1 — does the sort model overfit without it?
- [ ] Dropout in the GPT blocks: 0.0 vs 0.1.
- [ ] Loss: label smoothing 0.1 — effect on a low-entropy algorithmic target.
- [ ] Loss: focal (γ=2) — meaningful here, or a no-op when classes are balanced?
- [ ] Grad clip: 1.0 vs none vs 0.1 — does clipping change the early curve?
- [ ] Synthesis.

## Week 3 — Architecture & layer optimization
- [ ] Depth: n_layers 1 vs 2 vs 4 (compute vs exact-match).
- [ ] Width: d_model 32 vs 64 vs 128.
- [ ] Heads: n_heads 1 vs 4 vs 8 at fixed d_model.
- [ ] Harder task: seq_len 10 → 16, vocab 12 → 20 — where does the baseline break?
- [ ] Pre-LN vs post-LN block ordering (extend the harness; itself the day's "one thing").
- [ ] Tied vs untied input/output embeddings.
- [ ] Synthesis: the capacity/quality frontier.

## Week 4 — Parameter-efficient FT & quantization
- [ ] Freeze all but the head (`freeze: [...]`) — cheapest baseline; how much is lost?
- [ ] LoRA r=8 on the attention/MLP linears vs full training — quality at a fraction of params.
- [ ] LoRA rank sweep: r=2 / 4 / 8 / 16.
- [ ] LoRA alpha/scaling effect; LoRA target-module choice (attn vs mlp vs head).
- [ ] Dynamic int8 quantization — accuracy vs the full-precision baseline.
- [ ] Compose: quantize a LoRA-trained model.
- [ ] Synthesis + plan the next track.

---

## Side track (fast, <1 s/run) — MLP on sklearn `digits`
A pure-optimization sandbox for days when you want to sweep LR/optimizer/weight-decay quickly
on a vector-classification problem. Mirror the Week-1 knobs here for contrast with the GPT.

## Backlog / "needs GPU" or "needs egress" (run externally, commit results back)
- Real fine-tune: a small HF model (e.g. DistilBERT) on SST-2 — needs `huggingface.co` egress.
- Char-level nanoGPT on TinyStories; ablate depth vs width at larger scale.
- Knowledge distillation between two from-scratch GPTs of different sizes.
