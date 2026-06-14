"""DAXISResult: the data object returned by :func:`daxis.daxis_score`.

It carries the DAXIS score, its bootstrap confidence interval, the permutation
null, the full pairwise cosine matrix and the GO / NO-GO verdict, and knows how
to print a detailed report and draw the cosine-matrix heatmap.  It depends only
on ``numpy`` (``matplotlib`` is imported lazily inside :meth:`plot`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DAXISResult:
    """Outcome of a DAXIS analysis.

    Attributes
    ----------
    score : float
        Mean off-diagonal class-discriminant cosine across domains
        (the DAXIS score).  ``+1`` shared boundary, ``0`` unrelated.
    ci : (float, float)
        Bootstrap confidence interval on ``score``.
    ci_level : float
        Confidence level of ``ci`` (e.g. ``95``).
    p_value : float
        Permutation-null p-value (label shuffle within each domain).
    matrix : np.ndarray
        ``(n_domains, n_domains)`` pairwise cosine matrix (diagonal 1).
    domains : list of str
        Domain names, ordered as in ``matrix``.
    mode : str
        ``"binary"``, ``"classwise"`` or ``"subspace"``.
    n_classes, n_samples, n_features : int
        Shapes of the analysed data.
    null_mean, null_hi : float
        Mean and 95th percentile of the permutation null.
    regime : str
        ``"GO"``, ``"NO-GO"`` or ``"BORDERLINE"``.
    """

    score: float
    ci: Tuple[float, float]
    ci_level: float
    p_value: float
    matrix: np.ndarray
    domains: List[str]
    mode: str
    n_classes: int
    n_samples: int
    n_features: int
    null_mean: float
    null_hi: float
    regime: str

    # -- convenience ------------------------------------------------------
    @property
    def go(self) -> bool:
        """True only for a clean GO verdict."""
        return self.regime == "GO"

    def verdict(self) -> str:
        """One-paragraph plain-language recommendation."""
        if self.regime == "GO":
            return ("GO -- the domains share a class-discriminant axis. An "
                    "alignment-style domain adaptation (CORAL, MMD, subspace "
                    "alignment, adversarial) can move the source decision "
                    "boundary onto the target.")
        if self.regime == "NO-GO":
            return ("NO-GO -- the per-domain boundaries are not aligned "
                    "(cosine indistinguishable from chance). No amount of "
                    "marginal alignment recovers a shared linear rule; collect "
                    "target labels, re-define the label, or change the feature "
                    "space instead of running adaptation.")
        return ("BORDERLINE -- weak shared structure. Adaptation may give a "
                "small gain; validate on a small labelled target sample before "
                "committing compute.")

    # -- reporting --------------------------------------------------------
    def report(self, top: int = 5) -> str:
        """Return a detailed, human-readable multi-line report."""
        lo, hi = self.ci
        n = len(self.domains)
        out = [
            "=" * 66,
            "  DAXIS report  --  Discriminant-Axis Alignment",
            "=" * 66,
            f"  data      : {self.n_samples} samples | {self.n_features} features"
            f" | {self.n_classes} classes | {n} domains",
            f"  mode      : {self.mode}",
            f"  DAXIS score: {self.score:+.3f}"
            f"   ({self.ci_level:.0f}% CI [{lo:+.3f}, {hi:+.3f}])",
            f"  null      : mean {self.null_mean:+.3f} | 95th pct "
            f"{self.null_hi:+.3f} | p = {self.p_value:.3f}",
            f"  regime    : {self.regime}",
            "-" * 66,
        ]
        for line in _wrap(self.verdict(), 62):
            out.append("  " + line)
        out.append("-" * 66)

        pairs = [(self.matrix[a, b], self.domains[a], self.domains[b])
                 for a in range(n) for b in range(a + 1, n)]
        pairs.sort(reverse=True)
        out.append("  most aligned domain pairs:")
        for c, a, b in pairs[:top]:
            out.append(f"    {c:+.3f}   {a}  <->  {b}")
        if len(pairs) > top:
            out.append("  least aligned domain pairs:")
            for c, a, b in pairs[-top:][::-1]:
                out.append(f"    {c:+.3f}   {a}  <->  {b}")
        out.append("=" * 66)
        return "\n".join(out)

    def __str__(self) -> str:
        lo, hi = self.ci
        return (f"DAXISResult(score={self.score:+.3f}, "
                f"CI=[{lo:+.3f},{hi:+.3f}], p={self.p_value:.3f}, "
                f"regime={self.regime})")

    def to_dict(self) -> dict:
        """Serialisable dict (for JSON / logging)."""
        lo, hi = self.ci
        return dict(score=self.score, ci_low=lo, ci_high=hi,
                    ci_level=self.ci_level, p_value=self.p_value,
                    regime=self.regime, mode=self.mode,
                    n_classes=self.n_classes, n_samples=self.n_samples,
                    n_features=self.n_features, domains=list(self.domains),
                    matrix=np.asarray(self.matrix).tolist(),
                    null_mean=self.null_mean, null_hi=self.null_hi)

    # -- figure -----------------------------------------------------------
    def plot(self, path: Optional[str] = None, ax=None):
        """Draw the pairwise cosine matrix as a heatmap.

        Saves to ``path`` if given (returns the path), otherwise draws on
        ``ax`` (or a new axis) and returns the axis.
        """
        import matplotlib
        if path is not None and ax is None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        own = ax is None
        n = len(self.domains)
        if own:
            fig, ax = plt.subplots(figsize=(1.6 + 0.55 * n, 1.4 + 0.55 * n),
                                   dpi=150)
        im = ax.imshow(self.matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(self.domains, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(self.domains, fontsize=8)
        for a in range(n):
            for b in range(n):
                ax.text(b, a, f"{self.matrix[a, b]:.2f}", ha="center",
                        va="center", fontsize=7,
                        color="white" if abs(self.matrix[a, b]) > 0.6 else "black")
        ax.set_title(f"DAXIS {self.score:+.2f}  [{self.regime}]", fontsize=10)
        if own:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label="discriminant cosine")
            fig.tight_layout()
            if path:
                fig.savefig(path, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                return path
        return ax


def _wrap(text: str, width: int) -> List[str]:
    """Tiny word-wrap so the report stays inside ``width`` columns."""
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out
