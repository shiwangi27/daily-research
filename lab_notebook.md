# Lab notebook

The running narrative of the lab. Newest entries on top. Each entry is one day / one knob.
Written by the lead researcher, with the error analyst's interpretation folded in.

---

## 2026-06-27 — Day 0: harness bring-up + first baseline

- **Goal:** Stand up the daily-research harness and establish a reproducible baseline to
  measure everything against.
- **Environment reality (important):** This box is **CPU-only** (4 cores, 15 GB) and
  **egress-restricted** — `huggingface.co` and `download.pytorch.org` are blocked by org
  policy (403); only PyPI is reachable. So we cannot pull pretrained models or HF datasets
  here. We pivoted to **training small models from scratch on offline data** (bundled sklearn
  datasets + locally-generated algorithmic tasks). This still exercises every technique we
  care about — optimizers, LR schedules, losses, LoRA, quantization, architecture, layers —
  and is fully reproducible from git, which suits the ephemeral container.
- **Baseline (`experiments/2026-06-27-baseline`):** a from-scratch **TinyGPT** (104K params,
  2 layers, d=64, 4 heads) trained to **sort** a length-10 sequence over a 12-token vocab.
  AdamW, lr 3e-3, cosine schedule, 10% warmup, grad-clip 1.0, 8 epochs, 4k train / 1k eval.
- **Result:** converges cleanly to **token-acc 0.998 / exact-match 0.983** in **8.3 s** on CPU.
  Learning curve is textbook: exact-match 0.0 → 0.28 → 0.60 → 0.90 → 0.95 → 0.98 over epochs.
- **Verdict:** Solid reference point. The task is *learnable but not trivial* (epoch-0
  exact-match is 0, so there's clear headroom for ablations to move the curve), and 8 s/run
  means we can do real one-knob-per-day science fast.
- **Sanity checks done:** smoke config runs in 0.36 s; LoRA-from-scratch and dynamic int8
  quantization paths both execute and report metrics.
- **Next (Day 1 / Week 1):** Begin the optimization track. First knob: **learning rate** —
  push lr from 3e-3 → 1e-2 and see whether the sort task tolerates a more aggressive rate or
  destabilizes early. Hypothesis: at d=64/2 layers it'll still converge but with a noisier
  early curve; expected exact-match roughly on par or slightly worse.

### How to reproduce
```bash
python3 -m pip install -r requirements.txt
python3 -m harness.train experiments/2026-06-27-baseline/config.yaml
```
