# GCDA — Geometric Cross-Domain Adaptability

**A one-call GO / NO-GO diagnostic for domain adaptation — computed *before* you train anything.**

You have labelled data from several domains (sites, sensors, styles, hospitals,
sessions…) and you are about to spend compute on domain adaptation to make a
model transfer. GCDA answers, in one pass over your features, the question
adaptation methods *assume* but never check:

> Do the domains even share a class-boundary that alignment could line up?

If they do → **GO** (alignment-style DA can transfer the boundary). If they
don't → **NO-GO** (no marginal alignment will help; you need labels, a new
label definition, or new features). You learn this in seconds, not after a
week of failed adaptation runs.

```python
import gcda
res = gcda.gcda_score(X, y, domain)   # X:(n,p)  y:(n,)  domain:(n,)
print(res.report())
res.plot("gcda_matrix.png")
```

```
==================================================================
  GCDA report  --  Geometric Cross-Domain Adaptability
==================================================================
  data      : 1200 samples | 2048 features | 2 classes | 3 domains
  mode      : binary
  GCDA score: +0.78   (95% CI [+0.71, +0.84])
  null      : mean +0.01 | 95th pct +0.12 | p = 0.002
  regime    : GO
------------------------------------------------------------------
  GO -- the domains share a class-discriminant axis. An alignment-
  style domain adaptation (CORAL, MMD, subspace alignment,
  adversarial) can move the source decision boundary onto target.
------------------------------------------------------------------
  most aligned domain pairs:
    +0.83   photo  <->  art_painting
    ...
==================================================================
```

---

## Install

```bash
pip install gcda                 # core: numpy, scipy, scikit-learn
pip install gcda[plot]           # + matplotlib for .plot()
pip install gcda[torch]          # + the PyTorch adapter
pip install gcda[tf]             # + the TensorFlow adapter
```

Local dev install from this folder:

```bash
pip install -e .[dev]
pytest
```

## What it does (30 seconds)

For each domain `b`, GCDA forms the **unit class-discriminant direction**

```
d_b = (μ_b⁺ − μ_b⁻) / ‖μ_b⁺ − μ_b⁻‖          (binary case)
```

and reports the **mean pairwise cosine** between domains' directions — the
*GCDA score*. Cosine ≈ **+1**: every domain separates the classes along the
same axis, so a shift is a mere displacement adaptation can undo (**GO**).
Cosine ≈ **0**: the boundaries point in unrelated directions, the shift is in
the label rule itself, and alignment is provably limited (**NO-GO**). It is a
cheap, directly estimable proxy for the Ben-David joint-risk term λ.

Every score ships with a **bootstrap confidence interval** and a
**within-domain label-permutation null**, because in high-dimensional feature
spaces random labels already yield a positive cosine — the defensible signal is
a CI that clears that null.

## Pick the mode for your dataset structure

GCDA auto-selects, but you can force it:

| Your data | Mode | What it compares |
|---|---|---|
| 2 classes, tabular / low-dim features | `binary` (auto) | mean-difference direction `d_b` |
| Many classes, a few well-sampled domains | `subspace` | principal angles between per-domain **LDA subspaces** |
| Many classes **and/or** high-dim deep features (CNN/transformer embeddings) | `classwise` (auto for K>2) | per-class one-vs-rest directions, averaged |

```python
gcda.gcda_score(X, y, domain, mode="auto")        # binary if 2 classes, else classwise
gcda.gcda_score(X, y, domain, mode="subspace")    # multiclass, few domains
gcda.gcda_score(X, y, domain, mode="classwise")   # deep features
```

## Plug it into your training pipeline

### PyTorch `DataLoader`

Freeze a backbone, let GCDA pull the features out of the loader you already have:

```python
import torch, torchvision as tv, gcda

net = tv.models.resnet50(weights="IMAGENET1K_V2")
net.fc = torch.nn.Identity()                      # 2048-d features

res = gcda.from_torch(
    loader,                                        # yields (x, y) or (x, y, meta)
    backbone=net, device="cuda",
    domain_fn=lambda batch: batch[2],              # where the domain lives
    mode="classwise",
)
print(res.report())
```

WILDS-style loaders (batch = `x, y, metadata`) work out of the box — the
domain defaults to the first metadata column, or pass your own `domain_fn`.

### TensorFlow `tf.data.Dataset`

```python
import gcda
res = gcda.from_tensorflow(ds, backbone=feature_model,
                           domain_fn=lambda b: b[2])
```

### pandas / CSV

```python
import pandas as pd, gcda
res = gcda.from_dataframe(df, label_col="y", domain_col="site")
```

### Command line

```bash
gcda features.csv --label y --domain site --plot matrix.png --json out.json
gcda features.npz --mode classwise          # npz with arrays X, y, domain
```

## Reading the verdict

| Regime | Meaning | What to do |
|---|---|---|
| **GO** | CI clears 0 and the null; shared axis | run alignment DA (CORAL/MMD/DANN/subspace) |
| **NO-GO** | CI sits at/under the null | get target labels, re-define the label, or change features — **don't** burn compute on alignment |
| **BORDERLINE** | weak shared structure | validate on a small labelled target set first |

The verdict is geometry, not a promise of accuracy: GCDA isolates *whether a
shared boundary exists to be aligned*. It is complementary to distance-based
transferability scores (which measure how far the domains are) — GCDA measures
whether moving them closer can possibly help.

## API

```
gcda.gcda_score(X, y, domain, mode="auto", n_boot=500, n_perm=500,
                standardize=True, random_state=0, ci=95.0) -> GCDAResult
gcda.from_torch(loader, backbone=None, device="cpu", domain_fn=None, ...)
gcda.from_tensorflow(ds, backbone=None, domain_fn=None, ...)
gcda.from_dataframe(df, feature_cols=None, label_col="label", domain_col="domain", ...)

GCDAResult.score / .ci / .p_value / .regime / .matrix / .domains
GCDAResult.report()     -> str
GCDAResult.plot(path)   -> saves the cosine heatmap
GCDAResult.to_dict()    -> JSON-serialisable
```

## Citation

```bibtex
@article{dasilva2026gcda,
  title   = {GCDA: A Geometric Cross-Domain Adaptability Diagnostic for Deciding
             Whether Domain Adaptation Can Help, Before Training},
  author  = {da Silva, Vitor and Tom\`as, Rosana and Roig, Concepci\'o and
             Cores and Guirado and L\'erida and Gin\'e},
  journal = {Submitted to IEEE Access},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
