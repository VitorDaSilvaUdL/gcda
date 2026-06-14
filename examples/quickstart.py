"""Quickstart: a GO case and a NO-GO case on synthetic data.

    python examples/quickstart.py
"""
import numpy as np

import daxis


def make_domain(axis, n=400, p=20, seed=0):
    """Two Gaussian classes separated along the unit vector ``axis``."""
    rng = np.random.default_rng(seed)
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    y = rng.integers(0, 2, n)
    X = rng.standard_normal((n, p)) + 2.0 * (y[:, None] - 0.5) * axis
    return X, y


def ax(p, *idx_weights):
    """Build an axis in R^p from (index, weight) pairs."""
    v = np.zeros(p)
    for i, w in idx_weights:
        v[i] = w
    return v


def stack(*doms):
    X = np.vstack([d[0] for d in doms])
    y = np.concatenate([d[1] for d in doms])
    dom = np.concatenate([[f"domain_{i}"] * len(d[1]) for i, d in enumerate(doms)])
    return X, y, dom


P = 20
print("\n### GO: three domains that share (nearly) the same discriminant axis ###")
X, y, dom = stack(make_domain(ax(P, (0, 1.0)), seed=1),
                  make_domain(ax(P, (0, 1.0), (1, 0.15)), seed=2),   # tiny tilt
                  make_domain(ax(P, (0, 1.0), (1, -0.15)), seed=3))
res = daxis.daxis_score(X, y, dom)
print(res.report())
res.plot("quickstart_go.png")
print("[saved] quickstart_go.png")

print("\n### NO-GO: three mutually orthogonal discriminant axes ###")
X, y, dom = stack(make_domain(ax(P, (0, 1.0)), seed=1),
                  make_domain(ax(P, (1, 1.0)), seed=2),
                  make_domain(ax(P, (2, 1.0)), seed=3))
res = daxis.daxis_score(X, y, dom)
print(res.report())
res.plot("quickstart_nogo.png")
print("[saved] quickstart_nogo.png")
