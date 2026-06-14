"""Core DAXIS computation.

DAXIS (Discriminant-Axis Alignment) reads, *before any adaptation
training*, whether several domains share a class-discriminant direction.  For
each domain ``b`` it forms the unit class-discriminant direction

    d_b = (mu_b^+ - mu_b^-) / || mu_b^+ - mu_b^- ||                  (binary)

and reports the mean off-diagonal pairwise cosine ``mean_{b != b'} <d_b, d_b'>``
as the *DAXIS score*.  A score near ``+1`` means a shared boundary (adaptation
can transfer it -- a GO); near ``0`` means the boundaries are unrelated (a
NO-GO).  Multiclass problems use principal angles between per-domain LDA
subspaces; high-dimensional deep features use a class-wise one-vs-rest variant.

The only heavy dependency is scikit-learn (for standardisation and LDA).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from .report import DAXISResult

EPS = 1e-12
MODES = ("auto", "binary", "classwise", "subspace")


# -- geometry ------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > EPS else v


def _binary_dir(X, y, classes):
    """Unit mean-difference direction, positive class minus negative class."""
    return _unit(X[y == classes[1]].mean(0) - X[y == classes[0]].mean(0))


def _classwise_dirs(X, y, classes):
    """One-vs-rest unit mean-difference direction for every class."""
    return {k: _unit(X[y == k].mean(0) - X[y != k].mean(0)) for k in classes}


def _subspace(X, y, dim):
    """Orthonormal basis of the per-domain LDA discriminant subspace."""
    lda = LinearDiscriminantAnalysis(n_components=dim)
    lda.fit(X, y)
    Q, _ = np.linalg.qr(lda.scalings_[:, :dim])
    return Q


def _princ_cos(Q1, Q2):
    """Mean cosine of principal angles between two subspaces."""
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(np.clip(s, 0.0, 1.0).mean())


def _pair(Ri, Rj, mode, classes):
    if mode == "binary":
        return float(np.dot(Ri, Rj))
    if mode == "classwise":
        return float(np.mean([np.dot(Ri[k], Rj[k]) for k in classes]))
    return _princ_cos(Ri, Rj)  # subspace


def _rep(Xd, yd, mode, classes, K):
    if mode == "binary":
        return _binary_dir(Xd, yd, classes)
    if mode == "classwise":
        return _classwise_dirs(Xd, yd, classes)
    return _subspace(Xd, yd, K - 1)


def _has_all_classes(yd, classes):
    return set(np.asarray(classes).tolist()).issubset(set(np.unique(yd).tolist()))


def _offdiag_mean(R, doms, mode, classes):
    n = len(doms)
    return float(np.mean([_pair(R[doms[a]], R[doms[b]], mode, classes)
                          for a in range(n) for b in range(a + 1, n)]))


# -- public --------------------------------------------------------------
def daxis_score(X, y, domain, mode="auto", n_boot=500, n_perm=500,
               standardize=True, random_state=0, ci=95.0):
    """Compute the DAXIS GO / NO-GO diagnostic.

    Parameters
    ----------
    X : array (n_samples, n_features)
        Feature matrix (raw features, or a frozen backbone's embeddings).
    y : array (n_samples,)
        Class labels.
    domain : array (n_samples,)
        Domain / environment id per sample (strings or ints).
    mode : {"auto", "binary", "classwise", "subspace"}
        How the per-domain discriminant geometry is formed.  ``"auto"`` picks
        ``"binary"`` for 2 classes and ``"classwise"`` for more.  Use
        ``"subspace"`` for a few well-sampled multiclass domains, ``"classwise"``
        for high-dimensional deep features.
    n_boot, n_perm : int
        Bootstrap resamples (confidence interval) and label permutations
        (null).  Set to 0 to skip.
    standardize : bool
        Z-score the features first (recommended).
    random_state : int
        Seed for the bootstrap and permutation.
    ci : float
        Confidence level, percent.

    Returns
    -------
    DAXISResult
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    X = np.asarray(X, float)
    y = np.asarray(y)
    domain = np.asarray(domain)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_samples, n_features)")
    if not (len(X) == len(y) == len(domain)):
        raise ValueError("X, y, domain must have the same length")
    if standardize:
        X = StandardScaler().fit_transform(X)

    classes = np.unique(y)
    K = len(classes)
    if K < 2:
        raise ValueError("need at least 2 classes")
    if mode == "auto":
        mode = "binary" if K == 2 else "classwise"
    if mode == "binary" and K != 2:
        raise ValueError("binary mode needs exactly 2 classes; "
                         "use mode='classwise' or 'subspace'")

    doms = list(dict.fromkeys(domain.tolist()))
    if len(doms) < 2:
        raise ValueError("need at least 2 domains")
    idx = {d: np.where(domain == d)[0] for d in doms}
    for d in doms:
        if not _has_all_classes(y[idx[d]], classes):
            raise ValueError(f"domain {d!r} is missing one of the classes "
                             f"{classes.tolist()}; DAXIS needs every class in "
                             "every domain")

    # point estimate + full pairwise matrix
    R = {d: _rep(X[idx[d]], y[idx[d]], mode, classes, K) for d in doms}
    n = len(doms)
    M = np.eye(n)
    for a in range(n):
        for b in range(a + 1, n):
            M[a, b] = M[b, a] = _pair(R[doms[a]], R[doms[b]], mode, classes)
    score = _offdiag_mean(R, doms, mode, classes)

    rng = np.random.default_rng(random_state)

    # bootstrap CI -- resample examples within each domain
    boot = []
    for _ in range(int(n_boot)):
        Rb, ok = {}, True
        for d in doms:
            ii = rng.choice(idx[d], size=idx[d].size, replace=True)
            if not _has_all_classes(y[ii], classes):
                ok = False
                break
            Rb[d] = _rep(X[ii], y[ii], mode, classes, K)
        if ok:
            boot.append(_offdiag_mean(Rb, doms, mode, classes))
    # Centre the interval on the point estimate and use the bootstrap standard
    # error (a normal-approximation CI).  The cosine is bounded above by 1, so a
    # raw percentile interval is biased low and can sit below the point estimate;
    # the SE interval keeps the estimate inside its own CI.  Clipped to [-1, 1].
    if boot:
        se = float(np.std(boot, ddof=1))
        z = float(norm.ppf(0.5 + ci / 200.0))
        ci_lo = float(np.clip(score - z * se, -1.0, 1.0))
        ci_hi = float(np.clip(score + z * se, -1.0, 1.0))
    else:
        ci_lo = ci_hi = float("nan")

    # permutation null -- shuffle labels within each domain
    null = []
    for _ in range(int(n_perm)):
        Rp = {d: _rep(X[idx[d]], rng.permutation(y[idx[d]]), mode, classes, K)
              for d in doms}
        null.append(_offdiag_mean(Rp, doms, mode, classes))
    null = np.asarray(null, float)
    if null.size:
        p_value = float((np.sum(null >= score) + 1) / (null.size + 1))
        null_mean = float(null.mean())
        null_hi = float(np.percentile(null, 95))
    else:
        p_value = null_mean = null_hi = float("nan")

    # regime: GO must clear both 0 and the null; NO-GO sits at/under the null
    floor = 0.05 if np.isnan(null_hi) else max(0.05, null_hi)
    if not np.isnan(ci_hi) and ci_hi <= floor:
        regime = "NO-GO"
    elif not np.isnan(ci_lo) and ci_lo >= 0.35 and ci_lo > floor:
        regime = "GO"
    else:
        regime = "BORDERLINE"

    return DAXISResult(
        score=score, ci=(ci_lo, ci_hi), ci_level=float(ci), p_value=p_value,
        matrix=M, domains=[str(d) for d in doms], mode=mode,
        n_classes=int(K), n_samples=int(X.shape[0]), n_features=int(X.shape[1]),
        null_mean=null_mean, null_hi=null_hi, regime=regime,
    )
