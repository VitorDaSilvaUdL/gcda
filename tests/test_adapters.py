"""Tests for the data-container adapters (arrays, DataFrame, CSV)."""
import numpy as np
import pytest

import gcda


def _toy(seed=0, n=200, p=10):
    rng = np.random.default_rng(seed)
    blocks_X, blocks_y, blocks_d = [], [], []
    for i, tilt in enumerate([0.0, 0.1, -0.1]):
        axis = np.zeros(p)
        axis[0], axis[1] = 1.0, tilt
        y = rng.integers(0, 2, n)
        X = rng.standard_normal((n, p)) + 2.0 * (y[:, None] - 0.5) * axis
        blocks_X.append(X)
        blocks_y.append(y)
        blocks_d.append(np.array([f"s{i}"] * n))
    return (np.vstack(blocks_X), np.concatenate(blocks_y),
            np.concatenate(blocks_d))


def test_from_arrays_is_alias():
    X, y, d = _toy()
    a = gcda.from_arrays(X, y, d, n_boot=50, n_perm=50, random_state=1)
    b = gcda.gcda_score(X, y, d, n_boot=50, n_perm=50, random_state=1)
    assert a.score == b.score


def test_from_dataframe_matches_arrays():
    pd = pytest.importorskip("pandas")
    X, y, d = _toy()
    df = pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])
    df["y"] = y
    df["dom"] = d
    r_df = gcda.from_dataframe(df, label_col="y", domain_col="dom",
                              n_boot=50, n_perm=50, random_state=7)
    r_ar = gcda.gcda_score(X, y, d, n_boot=50, n_perm=50, random_state=7)
    assert abs(r_df.score - r_ar.score) < 1e-9
    assert r_df.domains == r_ar.domains


def test_from_dataframe_autodetects_features():
    pd = pytest.importorskip("pandas")
    X, y, d = _toy()
    df = pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])
    df["label"] = y
    df["domain"] = d
    # default label_col/domain_col, feature_cols auto = everything else
    r = gcda.from_dataframe(df, n_boot=30, n_perm=30)
    assert r.n_features == X.shape[1]


def test_csv_round_trip(tmp_path):
    pd = pytest.importorskip("pandas")
    X, y, d = _toy()
    df = pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])
    df["y"] = y
    df["site"] = d
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    r = gcda.from_dataframe(pd.read_csv(path), label_col="y", domain_col="site",
                           n_boot=30, n_perm=30)
    assert r.regime in ("GO", "BORDERLINE", "NO-GO")
    assert r.matrix.shape == (3, 3)
