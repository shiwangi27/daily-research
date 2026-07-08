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

---

## Week 1 — SFT mechanics (the stuff people get wrong)
- [x] **Day 0 — Pipeline bring-up + base model + prompt-loss masking.** Masked vs unmasked
      SFT: masking groks ~2 epochs sooner and makes the loss actually readable. (done 06-27)
- [ ] **Day 1 — Chat template / special tokens.** Add explicit role markers; does formatting
      consistency change generalization to held-out prompts? Planned in
      `research_plan_2026-07-08.md`: one new `data.explicit_turn_token` knob (a dedicated
      `<assistant>` special token at the prompt/answer boundary) vs `sft-masked`, where the
      boundary today is only the plain `"="` char. Not yet run.
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
