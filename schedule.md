# Research schedule — rolling daily plan

**Owner:** lead-researcher. A *living* curriculum, not a frozen list. After each day's
finding, the lead rewrites the next few days. One informative knob per day.

**The project:** a "nano-nanochat" — build the real LLM **post-training stack** (SFT →
reward modeling → DPO → GRPO) on a *modern* tiny transformer, entirely offline on CPU, in a
**verifiable** world (arithmetic) where rewards are programmatic (no human labels, no
reward-model dependency for the RL signal).

**Architecture (the LLM core recipe, shrunk):** `NanoLM` — RoPE positions, RMSNorm (no
learnable params), ReLU² MLP, QK-norm, no biases, untied embeddings, optional GQA, KV-less
greedy `generate()`. ~0.6M params at the default size.

**Base model:** `experiments/2026-06-27-pretrain-base` — 2-digit addition, answers emitted
least-significant-digit-first, **97% exact-match** in ~24 s. RM/DPO/GRPO stages `init_from`
this.

The point of a daily ablation is intuition for the *mechanism*. Numbers are secondary to
what you learn about the lever.

**Active tracks:** (A) **real-world benchmark chase**, (B) **post-training stack** on
arithmetic, and (C) **BabySim developmental RL** (below). Each day advances one track by one knob.

---

## Track C — BabySim: developmental / curriculum RL (visual)

A 2-D baby learns motor milestones week-by-week with policy-gradient RL; output is a comic-book
artifact you can watch. `babysim/sim.py` (env + REINFORCE) → `babysim/render.py` (comic).

- [x] **v1 env + 5 milestones + comic artifact.** Kinematic body, REINFORCE per skill, all
      milestones 4–5★. Surfaced two real RL lessons (log-reward for dense gradient; drop the
      `1/σ` NES step to avoid blow-up). (done 2026-07-10)
- [ ] **Dynamics, not kinematics** — torques + gravity integration + real falling (the honest
      version of "balance"). The day the baby can actually topple.
- [ ] **Curriculum gating** — lock week *k+1* until week *k* clears its success threshold;
      measure whether the ordering helps vs training each in isolation.
- [ ] **Algorithm ablation** — REINFORCE vs PPO vs GRPO on the same milestone (sample
      efficiency, stability). Reuse the harness optimizer/KL machinery.
- [ ] **Reward shaping** — sparse (success only) vs dense (shaped) — how much shaping is needed?
- [ ] **Exploration** — entropy bonus / σ schedule and its effect on wobble→mastery speed.
- [ ] **Transfer & forgetting** — does a policy that can stand learn to walk faster? Does
      learning to walk break sitting? (continual-learning angle)
- [ ] Richer visualization — animate full training rollouts, a "growth chart" of reward curves.

---

## Track A — Real-world benchmark chase (CURRENT FOCUS)

Fine-tune a from-scratch `NanoLM` classifier on a real domain dataset and see how far behind
the leaderboard we are, then close the gap one knob at a time. **Target: Financial PhraseBank**
(3-class financial-news sentiment; FinBERT ≈ **0.97 acc / 0.95 F1** on the all-agree subset).
Requires an HF-enabled session (`huggingface.co` allow-listed).

- [x] **Add text-classification to the harness** — `NanoLMClassifier`, `WordTokenizer`,
      `stage: classify`, class-weighting + focal, macro-F1. Validated offline on a synthetic
      imbalanced task (weighting: +0.10 macro-F1). (done 2026-07-08)
- [ ] **FPB baseline** — run `2026-07-08-fpb-baseline`; record acc + macro-F1 vs FinBERT. The gap.
- [ ] **Tokenizer** — word-level vs a trained BPE (add BPE to the harness); effect on a tiny model.
- [ ] **Class imbalance** — `class_weight: balanced` vs focal vs label smoothing on macro-F1.
- [ ] **Capacity** — depth/width sweep; where does more capacity stop helping without pretraining?
- [ ] **Regularization** — dropout / weight decay against the small-data overfitting.
- [ ] **LoRA** — adapters vs full fine-tuning at equal budget.
- [ ] **Reality check** — a real *pretrained* small encoder as an upper-bound reference (needs egress);
      quantify how much of the gap is "from scratch vs pretrained."
- [ ] Synthesis: our best from-scratch number vs the leaderboard, and what moved it most.

Backlog datasets for this track: Twitter Financial Sentiment (bigger, parquet-native),
LexGLUE LEDGAR / Overruling (legal), for a domain-transfer comparison later.

---

## Track B — Post-training stack on arithmetic

## Week 1 — SFT mechanics (the stuff people get wrong)
- [x] **Day 0 — Pipeline bring-up + base model + prompt-loss masking.** Masked vs unmasked
      SFT: masking groks ~2 epochs sooner and makes the loss actually readable. (done 06-27)
- [ ] **Day 1 — Chat template / special tokens.** Add explicit role markers; does formatting
      consistency change generalization to held-out prompts?
- [ ] **Day 2 — SFT epochs & overfitting.** 1 vs 3 vs 10 epochs — where does a tiny SFT set
      start to memorize? (LLMs overfit SFT fast.)
- [ ] **Day 3 — Harder task = headroom.** 3-digit addition (or add+sub mix). Re-establish the
      base so later stages (RL) have room to improve over SFT.
- [ ] **Day 4 — LoRA-SFT.** Train only LoRA adapters on the attention projections vs full FT —
      quality at a fraction of trainable params. (Harness already has from-scratch LoRA.)
- [ ] **Day 5 — Synthesis.** Write up SFT mechanics; lock the base + SFT recipe.

## Week 2 — Reward modeling (extend the harness: `stage: rm`)
- [ ] Implement the `rm` stage: scalar value head + **Bradley-Terry** pairwise loss on
      (chosen, rejected) pairs. Chosen = correct answer; rejected = an off-by-k corruption.
- [ ] Reward-model accuracy: does it rank correct > incorrect on held-out pairs?
- [ ] Reward calibration / margin: how separated are the scalars?
- [ ] **Reward hacking demo:** show a policy can score high reward while being wrong if the RM
      is weak — the motivation for verifiable rewards.
- [ ] LoRA vs full reward model. · Synthesis.

## Week 3 — Direct Preference Optimization (extend the harness: `stage: dpo`)
- [ ] Implement `dpo`: reference model = frozen SFT, DPO loss over preference pairs.
- [ ] **β (KL strength) sweep:** 0.05 / 0.1 / 0.5 — the single most important DPO knob. Watch
      drift/reward-hacking at low β and under-fitting at high β.
- [ ] DPO from SFT vs DPO from base — does the reference quality matter?
- [ ] DPO vs the reward-model+nothing baseline; measure KL from the reference. · Synthesis.

## Week 4 — GRPO / RLVR (extend the harness: `stage: grpo`)
- [ ] Implement `grpo`: sample G completions/prompt, **verifiable reward** (answer correct?),
      group-normalized advantage, KL-to-reference penalty.
- [ ] **KL coefficient** sweep — the leash that stops policy collapse.
- [ ] Group size G (4/8/16) — variance vs compute of the advantage estimate.
- [ ] Sampling temperature's effect on exploration & final accuracy.
- [ ] **The payoff:** does RLVR lift exact-match *above* SFT on the hard (carry-heavy) cases?
- [ ] Synthesis: SFT vs DPO vs GRPO on the same base — the headline comparison.

## Ongoing side-quests (slot in when a stage is between milestones)
- **Architecture-fidelity ablations:** turn each modern component off vs the base — learned
  positions instead of RoPE, LayerNorm instead of RMSNorm, GELU instead of ReLU², no QK-norm,
  GQA vs MHA. Which actually earn their place at this scale?
- **Quantization:** dynamic int8 on the SFT/DPO model — accuracy vs the fp32 baseline.
- **Eval methodology:** win-rate vs the reference, mean reward, KL-from-reference, held-out
  preference accuracy. Getting eval right is half the battle.
- **Tokenization:** train a small BPE vs the char tokenizer; vocab-size effects.

## Backlog / "needs GPU or egress" (run externally, commit results back)
- A real HF base (e.g. SmolLM/Qwen-0.5B) fine-tuned with TRL — needs `huggingface.co` egress.
- TinyStories-scale natural-language base for a model that actually "talks."
