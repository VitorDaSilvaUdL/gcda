"""Edge cases: determinism, the no-resampling path, validation, CI width."""
import numpy as np
import pytest

import daxis


def _two_domains(seed=0, n=200, p=10, angle=0.1):
    rng = np.random.default_rng(seed)
    out_X, out_y, out_d = [], [], []
    for i, a in enumerate([0.0, angle]):
        axis = np.zeros(p)
        axis[0], axis[1] = np.cos(a), np.sin(a)
        y = rng.integers(0, 2, n)
        X = rng.standard_normal((n, p)) + 2.0 * (y[:, None] - 0.5) * axis
        out_X.append(X)
        out_y.append(y)
        out_d.append(np.array([f"d{i}"] * n))
    return np.vstack(out_X), np.concatenate(out_y), np.concatenate(out_d)


def test_deterministic_with_seed():
    X, y, d = _two_domains()
    a = daxis.daxis_score(X, y, d, n_boot=80, n_perm=80, random_state=42)
    b = daxis.daxis_score(X, y, d, n_boot=80, n_perm=80, random_state=42)
    assert a.score == b.score
    assert a.ci == b.ci
    assert a.p_value == b.p_value


def test_no_resampling_path():
    X, y, d = _two_domains()
    r = daxis.daxis_score(X, y, d, n_boot=0, n_perm=0)
    assert np.isnan(r.ci[0]) and np.isnan(r.ci[1])
    assert np.isnan(r.p_value)
    assert r.regime == "BORDERLINE"          # cannot decide without the CI/null
    assert -1.0 <= r.score <= 1.0


def test_wider_confidence_level_is_wider():
    X, y, d = _two_domains()
    narrow = daxis.daxis_score(X, y, d, n_boot=200, n_perm=0, ci=80, random_state=3)
    wide = daxis.daxis_score(X, y, d, n_boot=200, n_perm=0, ci=99, random_state=3)
    assert (wide.ci[1] - wide.ci[0]) > (narrow.ci[1] - narrow.ci[0])


def test_invalid_mode_raises():
    X, y, d = _two_domains()
    with pytest.raises(ValueError):
        daxis.daxis_score(X, y, d, mode="nope", n_boot=0, n_perm=0)


def test_binary_mode_with_three_classes_raises():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 8))
    y = rng.integers(0, 3, 300)
    d = np.array(["a"] * 150 + ["b"] * 150)
    with pytest.raises(ValueError):
        daxis.daxis_score(X, y, d, mode="binary", n_boot=0, n_perm=0)


def test_single_domain_raises():
    X, y, _ = _two_domains()
    d = np.array(["only"] * len(y))
    with pytest.raises(ValueError):
        daxis.daxis_score(X, y, d, n_boot=0, n_perm=0)


def test_length_mismatch_raises():
    X, y, d = _two_domains()
    with pytest.raises(ValueError):
        daxis.daxis_score(X, y[:-1], d, n_boot=0, n_perm=0)
