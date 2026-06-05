"""Tests for GCDAResult: report text, serialisation, plotting, verdict."""
import json

import numpy as np
import pytest

import gcda


def _result(seed=0):
    rng = np.random.default_rng(seed)
    p = 12
    blocks_X, blocks_y, blocks_d = [], [], []
    for i in range(3):
        axis = np.zeros(p)
        axis[0], axis[1] = 1.0, 0.1 * (i - 1)
        y = rng.integers(0, 2, 200)
        X = rng.standard_normal((200, p)) + 2.0 * (y[:, None] - 0.5) * axis
        blocks_X.append(X)
        blocks_y.append(y)
        blocks_d.append(np.array([f"d{i}"] * 200))
    return gcda.gcda_score(np.vstack(blocks_X), np.concatenate(blocks_y),
                           np.concatenate(blocks_d), n_boot=60, n_perm=60)


def test_report_contains_key_fields():
    text = _result().report()
    for token in ("GCDA report", "GCDA score", "CI", "regime", "aligned"):
        assert token in text


def test_to_dict_is_json_serialisable():
    r = _result()
    d = r.to_dict()
    for key in ("score", "ci_low", "ci_high", "p_value", "regime", "matrix",
                "domains", "mode"):
        assert key in d
    s = json.dumps(d)                      # must not raise
    assert json.loads(s)["regime"] == r.regime


def test_plot_writes_file(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "matrix.png"
    path = _result().plot(str(out))
    assert out.exists() and out.stat().st_size > 0
    assert str(out) == path


def test_verdict_matches_regime():
    r = _result()
    assert r.go == (r.regime == "GO")
    assert r.regime.split("-")[0].lower() in r.verdict().lower()


def test_str_is_compact():
    s = str(_result())
    assert s.startswith("GCDAResult(") and "score=" in s
