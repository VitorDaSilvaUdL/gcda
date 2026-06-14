"""Behavioural tests for the DAXIS core: aligned domains -> GO, orthogonal -> NO-GO."""
import numpy as np
import pytest

import daxis


def _binary_domain(n=300, p=20, angle=0.0, seed=0):
    """A domain whose discriminant axis is the base axis rotated by ``angle``."""
    rng = np.random.default_rng(seed)
    d = np.zeros(p)
    d[0], d[1] = np.cos(angle), np.sin(angle)
    y = rng.integers(0, 2, n)
    X = rng.standard_normal((n, p)) + 2.0 * (y[:, None] - 0.5) * d
    return X, y


def _stack(domains):
    X = np.vstack([d[0] for d in domains])
    y = np.concatenate([d[1] for d in domains])
    dom = np.concatenate([[f"d{i}"] * len(d[1]) for i, d in enumerate(domains)])
    return X, y, dom


def test_go_when_aligned():
    X, y, dom = _stack([_binary_domain(angle=0.0, seed=1),
                        _binary_domain(angle=0.0, seed=2)])
    r = daxis.daxis_score(X, y, dom, n_boot=100, n_perm=100)
    assert r.score > 0.6
    assert r.regime == "GO"
    assert r.p_value < 0.05


def test_nogo_when_orthogonal():
    X, y, dom = _stack([_binary_domain(angle=0.0, seed=1),
                        _binary_domain(angle=np.pi / 2, seed=2)])
    r = daxis.daxis_score(X, y, dom, n_boot=100, n_perm=100)
    assert abs(r.score) < 0.3
    assert r.regime in ("NO-GO", "BORDERLINE")


def test_matrix_is_symmetric_unit_diagonal():
    X, y, dom = _stack([_binary_domain(seed=s) for s in range(4)])
    r = daxis.daxis_score(X, y, dom, n_boot=50, n_perm=50)
    assert r.matrix.shape == (4, 4)
    assert np.allclose(np.diag(r.matrix), 1.0)
    assert np.allclose(r.matrix, r.matrix.T)


def test_multiclass_classwise_aligned():
    p = 15
    centers = np.random.default_rng(0).standard_normal((3, p))

    def dom(seed):
        rl = np.random.default_rng(seed)
        y = rl.integers(0, 3, 300)
        X = rl.standard_normal((300, p)) + 2.0 * centers[y]
        return X, y

    X, y, dm = _stack([dom(1), dom(2)])
    r = daxis.daxis_score(X, y, dm, mode="classwise", n_boot=50, n_perm=50)
    assert r.mode == "classwise"
    assert r.score > 0.5


def test_subspace_mode_runs():
    p = 12
    centers = np.random.default_rng(0).standard_normal((3, p))

    def dom(seed):
        rl = np.random.default_rng(seed)
        y = rl.integers(0, 3, 400)
        X = rl.standard_normal((400, p)) + 2.0 * centers[y]
        return X, y

    X, y, dm = _stack([dom(1), dom(2)])
    r = daxis.daxis_score(X, y, dm, mode="subspace", n_boot=30, n_perm=30)
    assert r.mode == "subspace"
    assert 0.0 <= r.score <= 1.0


def test_report_and_serialisation():
    X, y, dom = _stack([_binary_domain(seed=1), _binary_domain(seed=2)])
    r = daxis.daxis_score(X, y, dom, n_boot=50, n_perm=50)
    text = r.report()
    assert "DAXIS" in text and "regime" in text
    d = r.to_dict()
    assert set(["score", "regime", "matrix"]).issubset(d)


def test_missing_class_raises():
    X, y, dom = _stack([_binary_domain(seed=1), _binary_domain(seed=2)])
    y[dom == "d0"] = 0  # wipe one class from a domain
    with pytest.raises(ValueError):
        daxis.daxis_score(X, y, dom, n_boot=0, n_perm=0)
