# Lab notebook

The running narrative of the lab. Newest entries on top. Each entry is one day / one knob.
Written by the lead researcher, with the error analyst's interpretation folded in.

---

## 2026-07-12 — BabySim physics: real dynamics + learning to stand (RL)

BabySim moves from *kinematic posing* to *real dynamics*. First tried an articulated
Position-Based-Dynamics soft body (point masses + bones + angle muscles) — it was unstable
(interacting angle constraints blew up). Pivoted to the robust, textbook substrate: a
**torque-controlled inverted pendulum** in reduced coordinates (`babysim/physics.py`), the
standard balance-control task.

- **Real gravity:** uncontrolled, she topples (fall at step ~59). A hand-tuned PD "ankle
  muscle" balances and recovers from a shove — so the dynamics are genuine, not scripted.
- **Learning to stand (REINFORCE):** a linear torque policy `τ = W·[sinθ, cosθ, ω] + b`
  trained by policy gradient goes from toppling (~71 steps) to balancing the full horizon
  (**200/200**), discovering `W ≈ [−43 on lean, −24 on angular velocity]` — an LQR-style
  balance controller, learned from scratch against gravity.
- **RL lessons (again):** REINFORCE needed a much larger LR/exploration than intuition
  suggested, because the useful control gains are large (~30–40) while the informative state
  feature (lean) is small — slow to grow without enough step size. Same "optimizer stability
  vs reward design" theme as the kinematic milestones.

This is the seed of the physics RL track. Next: render the *learned* balance in the comic
(the baby genuinely wobbling and catching herself), then extend to a multi-link body and the
curriculum-gated REINFORCE→PPO→GRPO comparisons.

---

## 2026-07-10 — BabySim: a developmental-RL environment + comic-book visualization

**New direction (user's pick of RL problem):** simulate how a baby learns motor skills week
by week, and *visualize* it comic-book style. This is genuinely a **curriculum / developmental
RL** setup — skills emerge in a fixed order, each a shaped reward, and "getting better over
weeks" is literally the learning curve. Fully offline, CPU, seconds to train.

**Environment (`babysim/sim.py`).** A 2-D side-view baby as a pelvis-rooted kinematic chain
(torso, head, arm, leg). Five milestones, each a task reward: lift head (head height), reach
(hand→rattle distance), sit (trunk upright + COM balance), pull-to-stand (upright + feet on
floor + balance), first steps (forward stride + balance). Each is trained by **REINFORCE**
(Gaussian policy over the controlled joint angles).

**Two RL lessons already surfaced building it:**
1. **Vanishing gradients from squashed rewards.** `exp(-k·err²)` is ~flat far from the goal →
   no learning signal. Fix: optimize the **log-reward** (dense quadratic). Reach/sit went from
   stuck to solved.
2. **Step-size blow-up.** The textbook NES `1/σ` scaling explodes as σ anneals → the policy
   slams joints into their limits (grotesque poses). Dropping `1/σ` + flooring σ + modest lr
   made all five milestones converge (final rewards 0.85–1.00; reach lands exactly on the toy).
   A clean reminder that RL results are as much optimizer-stability as reward design.

**Visualization (`babysim/render.py` → comic artifact).** Chose a **self-contained comic-book
HTML page**: a week-by-week panel strip (each panel = the mastered pose, drawn as a cute comic
baby with badge + star rating) plus an **animated stage** that replays a skill, interpolating
the training snapshots so you watch exploration wobble anneal into mastery. Format rationale:
game-like, shareable, theme-aware, no assets. Published as an Artifact.

**Status:** poses reach 4–5★ across all milestones. This is a *kinematic* first cut (policy sets
joint angles; balance is a COM-over-base reward, not true rigid-body dynamics).

**Next (the real RL curriculum):** make it dynamics-based (torques + gravity integration, real
falling), add **curriculum gating** (must master week *k* to unlock *k+1*), then run the
technique ablations as daily experiments: REINFORCE vs PPO vs GRPO, reward shaping (sparse vs
dense), exploration/entropy, and **skill transfer / catastrophic forgetting** across milestones.

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
