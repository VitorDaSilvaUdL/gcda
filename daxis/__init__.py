"""DAXIS -- Discriminant-Axis Alignment.

A one-call GO / NO-GO diagnostic for domain adaptation.  Given labelled
features from two or more domains it measures whether the per-domain
class-discriminant directions point the same way; if they do, alignment-style
adaptation can transfer the decision boundary (GO), if they do not, no marginal
alignment will help (NO-GO) -- and you learn this *before* training anything.

Quickstart
----------
>>> import daxis
>>> res = daxis.daxis_score(X, y, domain)      # X:(n,p) y:(n,) domain:(n,)
>>> print(res.report())
>>> res.plot("daxis_matrix.png")

From a PyTorch DataLoader or tf.data.Dataset:

>>> res = daxis.from_torch(loader, backbone=net, device="cuda")
>>> res = daxis.from_tensorflow(ds, backbone=net)
"""
from .adapters import (from_arrays, from_dataframe, from_tensorflow,
                       from_torch)
from .core import daxis_score
from .report import DAXISResult

score = daxis_score  # short alias: daxis.score(X, y, domain)

__all__ = ["daxis_score", "score", "DAXISResult", "from_arrays",
           "from_dataframe", "from_torch", "from_tensorflow"]
__version__ = "0.1.0"
