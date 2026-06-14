"""Adapters -- build ``(X, y, domain)`` from common data containers, then run
DAXIS.  The deep-learning frameworks are imported lazily, so ``import daxis``
never requires torch or tensorflow to be installed.

Conventions
-----------
* A PyTorch batch is ``(x, y)`` or ``(x, y, metadata)``.  Pass a
  ``domain_fn(batch) -> array`` when the domain is not the third element
  (e.g. WILDS returns metadata whose first column is the domain).
* ``backbone`` is any callable mapping a batch of inputs to a feature tensor.
  Pass a frozen, ``eval()``-mode network; if ``None`` the raw inputs are
  flattened and used directly (fine for already-extracted features).
"""
from __future__ import annotations

import numpy as np

from .core import daxis_score


def from_arrays(X, y, domain, **kw):
    """Thin alias of :func:`daxis.daxis_score` for explicit array inputs."""
    return daxis_score(X, y, domain, **kw)


def from_dataframe(df, feature_cols=None, label_col="label",
                   domain_col="domain", **kw):
    """Run DAXIS on a :class:`pandas.DataFrame`.

    ``feature_cols`` defaults to every column except the label and domain.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in (label_col, domain_col)]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[label_col].to_numpy()
    domain = df[domain_col].to_numpy()
    return daxis_score(X, y, domain, **kw)


# -- PyTorch -------------------------------------------------------------
def _collect_torch(loader, backbone, device, domain_fn, label_index,
                   max_batches):
    import torch

    feats, labs, doms = [], [], []
    if backbone is not None:
        backbone = backbone.to(device).eval()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            if not isinstance(batch, (tuple, list)):
                raise TypeError("expected the loader to yield (x, y[, meta]) "
                                "tuples")
            x = batch[0].to(device)
            f = backbone(x) if backbone is not None else x
            if f.ndim > 2:
                f = f.flatten(1)
            feats.append(np.asarray(f.detach().cpu().numpy()))
            labs.append(np.asarray(batch[label_index]).ravel())
            if domain_fn is not None:
                doms.append(np.asarray(domain_fn(batch)).ravel())
            elif len(batch) > 2:
                meta = np.asarray(batch[2])
                doms.append(meta.reshape(len(meta), -1)[:, 0])
            else:
                raise ValueError("no domain in the batch; pass "
                                 "domain_fn=lambda batch: <domain array>")
    return (np.concatenate(feats, 0),
            np.concatenate(labs, 0),
            np.concatenate(doms, 0))


def from_torch(loader, backbone=None, device="cpu", domain_fn=None,
               label_index=1, max_batches=None, **kw):
    """Run DAXIS on the features a :class:`torch.utils.data.DataLoader` yields.

    Example
    -------
    >>> import torchvision, torch, daxis
    >>> net = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    >>> net.fc = torch.nn.Identity()              # 2048-d features
    >>> res = daxis.from_torch(loader, backbone=net, device="cuda",
    ...                       domain_fn=lambda b: b[2][:, 0], mode="classwise")
    >>> print(res.report())
    """
    X, y, domain = _collect_torch(loader, backbone, device, domain_fn,
                                  label_index, max_batches)
    return daxis_score(X, y, domain, **kw)


# -- TensorFlow ----------------------------------------------------------
def from_tensorflow(ds, backbone=None, domain_fn=None, max_batches=None, **kw):
    """Run DAXIS on the features a ``tf.data.Dataset`` yields.

    The dataset is expected to yield ``(x, y)`` or ``(x, y, meta)``.  Pass
    ``domain_fn(batch) -> array`` if the domain lives elsewhere.
    """
    feats, labs, doms = [], [], []
    for bi, batch in enumerate(ds):
        if max_batches and bi >= max_batches:
            break
        x, y = batch[0], batch[1]
        f = backbone(x) if backbone is not None else x
        f = np.asarray(f)
        feats.append(f.reshape(f.shape[0], -1))
        labs.append(np.asarray(y).ravel())
        if domain_fn is not None:
            doms.append(np.asarray(domain_fn(batch)).ravel())
        elif len(batch) > 2:
            doms.append(np.asarray(batch[2]).ravel())
        else:
            raise ValueError("no domain in the batch; pass domain_fn")
    return daxis_score(np.concatenate(feats, 0),
                      np.concatenate(labs, 0),
                      np.concatenate(doms, 0), **kw)
