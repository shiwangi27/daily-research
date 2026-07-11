"""BabySim — a tiny developmental-RL environment.

A 2D side-view baby (pelvis-rooted kinematic chain) learns a *curriculum* of motor
milestones, each defined by a task reward + a success threshold. A milestone is trained
with REINFORCE (a Gaussian policy over the controlled joint angles), on the *log* reward so
the gradient stays dense; the policy's noise anneals over "weeks", so early attempts wobble
and late ones are steady — the developmental arc we visualize.

Angle convention (radians, measured from +y "up", positive tilts toward +x "forward"):
limbs point straight DOWN at angle π. So hip=π is a leg hanging down; hip=π/2 is a thigh
held forward horizontally. Feet/hands are kept near the ground by the reward, not by hacks.

Fully offline, CPU-only, deterministic given the seed.
"""
from __future__ import annotations

import json
import os

import numpy as np

PI = np.pi
L = dict(torso=1.02, neck=0.20, head=0.52, uarm=0.50, farm=0.44, thigh=0.60, shin=0.56)


def _d(ang):
    return np.array([np.sin(ang), np.cos(ang)])          # unit vector, ang from +y


def fk(a: dict) -> dict:
    """Forward kinematics -> named 2D points (y up)."""
    pelvis = np.array([a["px"], a["py"]], float)
    torso = a["torso"]
    chest = pelvis + L["torso"] * _d(torso)
    neck = chest + L["neck"] * _d(torso)
    head = neck + L["head"] * _d(torso + a["head"])       # head tilt relative to torso
    elbow = chest + L["uarm"] * _d(a["shoulder"])         # shoulder absolute (π = down)
    hand = elbow + L["farm"] * _d(a["shoulder"] + a["elbow"])
    knee = pelvis + L["thigh"] * _d(a["hip"])             # hip absolute (π = down)
    foot = knee + L["shin"] * _d(a["hip"] + a["knee"])
    return dict(pelvis=pelvis, chest=chest, neck=neck, head=head,
                elbow=elbow, hand=hand, knee=knee, foot=foot)


def com(p: dict) -> np.ndarray:
    w = dict(head=2.0, chest=1.5, pelvis=1.5, knee=0.5, foot=0.3, hand=0.2)
    return sum(p[k] * wt for k, wt in w.items()) / sum(w.values())


def _base(**over):
    b = dict(px=0.0, py=1.0, torso=0.0, head=0.0, shoulder=PI, elbow=0.0, hip=PI, knee=0.0)
    b.update(over)
    return b


JOINT_LIMITS = dict(torso=(-0.4, 1.6), head=(-1.8, 1.0), shoulder=(-0.2, 3.4),
                    elbow=(-0.2, 2.6), hip=(0.4, 3.6), knee=(-0.2, 2.4))


def clamp(a: dict) -> dict:
    for j, (lo, hi) in JOINT_LIMITS.items():
        if j in a:
            a[j] = float(np.clip(a[j], lo, hi))
    return a


# Targets read off hand-authored "mastered" poses, so each reward optimum is reachable and
# looks right (uprightness/balance genuinely optimal, rather than fighting the fixed limbs).
RATTLE = fk(_base(py=0.34, torso=1.5, head=-0.95, hip=0.55, knee=1.5, shoulder=0.15, elbow=0.15))["hand"]
SIT_COMX = float(com(fk(_base(py=0.32, hip=PI / 2, knee=0.15, shoulder=2.5, elbow=0.7, torso=0.0)))[0])


def cost_head(a):                                  # prone: lift the head as high as it goes
    return (fk(a)["head"][1] - 0.98) ** 2
def cost_reach(a):                                 # get the hand to the toy
    return float(np.sum((fk(a)["hand"] - RATTLE) ** 2))
def cost_sit(a):                                   # torso upright, COM over the sitting base
    p = fk(a); c = com(p)
    return a["torso"] ** 2 + 3.0 * (c[0] - SIT_COMX) ** 2
def cost_stand(a):                                 # upright, feet on the floor, balanced
    p = fk(a); c = com(p)
    return a["torso"] ** 2 + 2.5 * p["foot"][1] ** 2 + 3.0 * (c[0] - p["foot"][0]) ** 2
def cost_walk(a):                                  # a leg swung forward, foot down, trunk over base
    p = fk(a)
    return (a["hip"] - 2.3) ** 2 + 0.5 * (a["knee"] - 0.6) ** 2 + (a["torso"] - 0.12) ** 2 \
        + 2.0 * p["foot"][1] ** 2
def cost_crawl(a):                                 # on all fours, head up to look ahead
    return (fk(a)["head"][1] - 1.45) ** 2


def _reward(costfn, scale):
    return lambda a: float(np.exp(-scale * costfn(a)))


MILESTONES = [
    dict(key="head", week=6, title="Lifts head", badge="HEAD CONTROL",
         blurb="On the mat, tummy down — the very first push against gravity.",
         base=_base(py=0.30, torso=1.4, shoulder=2.3, elbow=0.5, hip=1.7, knee=0.6),
         control=["head"], reward=_reward(cost_head, 1.5), success=0.8),
    dict(key="reach", week=18, title="Reaches for the rattle", badge="EYE–HAND",
         blurb="On her back, knees up, arms shoot up toward a bright, jingly toy overhead.",
         base=_base(py=0.34, torso=1.5, head=-0.95, hip=0.55, knee=1.5, shoulder=1.3, elbow=0.4),
         control=["shoulder", "elbow"], reward=_reward(cost_reach, 1.0), success=0.75),
    dict(key="sit", week=28, title="Sits unsupported", badge="BALANCE",
         blurb="Spine stacks over the hips and — wobble, wobble — holds.",
         base=_base(py=0.32, hip=PI / 2, knee=0.15, shoulder=2.5, elbow=0.7, torso=0.7),
         control=["torso"], reward=_reward(cost_sit, 3.0), success=0.75),
    dict(key="crawl", week=34, title="Crawls", badge="ON THE MOVE",
         blurb="Up on all fours, rocking — then a hand forward, a knee forward, and off they go.",
         base=_base(py=0.64, torso=1.30, head=0.3, shoulder=3.0, elbow=0.5, hip=3.0, knee=1.5),
         control=["head"], reward=_reward(cost_crawl, 1.5), success=0.7),
    dict(key="stand", week=44, title="Pulls to stand", badge="UPRIGHT",
         blurb="Legs straighten, hips rise — the world looks different from up here.",
         base=_base(py=1.16, shoulder=2.9, elbow=0.3, hip=3.0, knee=0.5),
         control=["torso", "hip", "knee"], reward=_reward(cost_stand, 1.5), success=0.7),
    dict(key="walk", week=54, title="First steps", badge="LOCOMOTION",
         blurb="One foot forward, catch the fall, again — walking is controlled falling.",
         base=_base(py=1.14, shoulder=2.7, elbow=0.3, hip=3.0, knee=0.3),
         control=["hip", "knee", "torso"], reward=_reward(cost_walk, 1.2), success=0.6),
]


def train_milestone(m, iters=240, batch=64, lr=0.12, sigma0=0.6, sigma_floor=0.12, seed=0):
    """Gaussian-policy REINFORCE on the log reward (dense gradient), with a floored,
    gently-annealed exploration noise. Returns the score curve and pose snapshots from
    wobbly (early) to steady (mastered)."""
    rng = np.random.default_rng(seed)
    ctrl = m["control"]
    theta = np.array([m["base"][j] for j in ctrl], float)
    reward = m["reward"]
    lo = np.array([JOINT_LIMITS[j][0] for j in ctrl])
    hi = np.array([JOINT_LIMITS[j][1] for j in ctrl])

    def pose_from(vec):
        a = dict(m["base"]); a.update({j: float(v) for j, v in zip(ctrl, vec)})
        return clamp(a)

    curve, snaps = [], []
    checkpoints = np.linspace(0, iters - 1, 6).astype(int)
    for it in range(iters):
        sigma = max(sigma_floor, sigma0 * (0.1 / sigma0) ** (it / (iters - 1)))
        eps = rng.standard_normal((batch, len(ctrl)))
        rewards = np.array([reward(pose_from(theta + sigma * e)) for e in eps])
        obj = np.log(rewards + 1e-9)
        adv = (obj - obj.mean()) / (obj.std() + 1e-8)
        theta = np.clip(theta + lr * (adv[:, None] * eps).mean(0), lo, hi)  # no 1/sigma -> stable
        curve.append(reward(pose_from(theta)))
        if it in checkpoints:
            snaps.append(_pts(pose_from(theta + sigma * rng.standard_normal(len(ctrl)))))
    final_r = reward(pose_from(theta))
    angles = {k: round(float(v), 4) for k, v in pose_from(theta).items()}
    return dict(curve=[round(c, 3) for c in curve[::4]], angles=angles, mastered=_pts(pose_from(theta)),
                final=round(final_r, 3), stars=int(np.clip(round(1 + 4 * final_r), 1, 5)),
                success=bool(final_r >= m["success"]))


def _pts(a):
    return {k: [round(float(v[0]), 3), round(float(v[1]), 3)] for k, v in fk(a).items()}


def generate(path, seed=0):
    out = []
    for m in MILESTONES:
        res = train_milestone(m, seed=seed)
        out.append(dict(key=m["key"], week=m["week"], title=m["title"], badge=m["badge"],
                        blurb=m["blurb"], **res))
        print(f"{m['week']:>3}w  {m['title']:<24} final={res['final']:.2f} "
              f"{'MASTERED' if res['success'] else 'learning '}  {'★' * res['stars']}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"milestones": out, "rattle": [round(float(RATTLE[0]), 3), round(float(RATTLE[1]), 3)],
                   "L": L, "seed": seed}, fh)
    print(f"wrote {path}")
    return out


if __name__ == "__main__":
    import sys
    generate(sys.argv[1] if len(sys.argv) > 1 else "babysim/poses.json")
