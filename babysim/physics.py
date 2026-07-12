"""BabySim physics — reduced-coordinate dynamics for balance/standing.

Robust, textbook physics: the standing baby is modelled as a torque-controlled inverted
pendulum (body pivoting at the ankle). Gravity destabilises it; a controller (later, an RL
policy) applies ankle torque to stay upright. Uncontrolled -> she topples. This is the classic
control substrate (CartPole/Acrobot family) and integrates stably — the right foundation for
meaningful RL (REINFORCE/PPO/GRPO), unlike a fiddly articulated soft-body sim.

theta = lean from vertical (0 = upright, +/- = falling forward/back). y is up.
Offline, CPU, deterministic.
"""
from __future__ import annotations

import numpy as np


class InvertedPendulum:
    """Body as a single rigid segment of length L, mass m, pivoting at the ankle.

    theta'' = (g/L) sin(theta) + tau/(m L^2) - damp*theta'
    A torque tau (the "ankle muscle") is the control. Falls past |theta| = fall_angle.
    """

    def __init__(self, L=1.7, m=1.0, g=9.8, damp=0.05, dt=0.02, tau_max=6.0, fall_angle=1.2):
        self.L, self.m, self.g, self.damp, self.dt = L, m, g, damp, dt
        self.tau_max, self.fall_angle = tau_max, fall_angle
        self.reset()

    def reset(self, theta=0.0, omega=0.0):
        self.theta, self.omega = float(theta), float(omega)
        self.t = 0
        return self.state()

    def state(self):
        return np.array([np.sin(self.theta), np.cos(self.theta), self.omega], float)

    @property
    def fallen(self):
        return abs(self.theta) > self.fall_angle

    def step(self, tau):
        tau = float(np.clip(tau, -self.tau_max, self.tau_max))
        acc = (self.g / self.L) * np.sin(self.theta) + tau / (self.m * self.L ** 2) - self.damp * self.omega
        self.omega += acc * self.dt          # semi-implicit Euler (stable)
        self.theta += self.omega * self.dt
        self.t += 1
        # reward: upright & still & low effort; big penalty when fallen
        r = np.cos(self.theta) - 0.02 * self.omega ** 2 - 0.001 * tau ** 2
        return self.state(), (r - 5.0 if self.fallen else r), self.fallen


def train_balance(iters=300, batch=64, lr=1.2, sigma=3.0, max_steps=200, seed=0):
    """REINFORCE a linear torque policy tau = W·state + b to keep the pendulum upright.
    Returns (W, b, curve) where curve is mean balanced-steps per iteration — it should climb
    from a quick topple to the full horizon as the baby learns to stand."""
    rng = np.random.default_rng(seed)
    W, b = np.zeros(3), 0.0
    curve = []
    for _ in range(iters):
        traj, rets, lens = [], [], []
        for _e in range(batch):
            p = InvertedPendulum(); p.reset(theta=rng.uniform(-0.2, 0.2))
            s = p.state(); S, A, Mu, R = [], [], [], 0.0
            for _t in range(max_steps):
                mu = float(W @ s + b); a = mu + sigma * rng.standard_normal()
                s2, r, done = p.step(a)
                S.append(s); A.append(a); Mu.append(mu); R += r; s = s2
                if done:
                    break
            traj.append((S, A, Mu)); rets.append(R); lens.append(len(S))
        adv = (np.array(rets) - np.mean(rets)) / (np.std(rets) + 1e-8)
        gW, gb = np.zeros(3), 0.0
        for (S, A, Mu), ad in zip(traj, adv):
            for s, a, mu in zip(S, A, Mu):
                coef = (a - mu) / sigma ** 2 * ad
                gW += coef * s; gb += coef
        W += lr * gW / batch; b += lr * gb / batch
        curve.append(float(np.mean(lens)))
    return W, b, curve


if __name__ == "__main__":  # smoke: no control topples; a PD "muscle" balances a shove
    def run(controller, label):
        p = InvertedPendulum(); p.reset(theta=0.15)      # a lean/shove
        for _ in range(300):
            tau = controller(p)
            p.step(tau)
            if p.fallen:
                break
        print(f"{label:18s} -> {'FELL at step '+str(p.t) if p.fallen else 'balanced 300 steps'}"
              f"  (final lean {np.degrees(p.theta):+.1f} deg)")
    run(lambda p: 0.0, "no control")
    run(lambda p: -32.0 * p.theta - 7.0 * p.omega, "PD muscle")

    print("\nlearning to stand (REINFORCE) ...")
    W, b, curve = train_balance()
    print(f"  balanced steps: {curve[0]:.0f} -> {np.mean(curve[-10:]):.0f} / 200"
          f"   learned gains W={[round(float(x), 1) for x in W]}")
