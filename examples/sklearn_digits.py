"""A real-data example with zero downloads: scikit-learn handwritten digits.

We build three *domains* from the 8x8 digit images and ask DAXIS whether the
class-discriminant geometry survives the shift:

  * ``original``    -- the raw images,
  * ``noisy``       -- images + mild Gaussian noise and an intensity rescale
                       (a covariate shift that keeps the same boundary),
  * ``scrambled``   -- images with a fixed pixel permutation (destroys the
                       shared discriminant axis).

DAXIS should report a high cosine between ``original`` and ``noisy`` and a low
one to ``scrambled`` -- a GO for the first pair, a NO-GO for the second.

    python examples/sklearn_digits.py
"""
import numpy as np
from sklearn.datasets import load_digits

import daxis

rng = np.random.default_rng(0)
X, y = load_digits(return_X_y=True)          # (1797, 64), 10 classes
p = X.shape[1]

# domain 1: original
X1, y1 = X, y
# domain 2: covariate shift -- rescale intensity + Gaussian noise, same labels
X2 = 0.8 * X + 6.0 * rng.standard_normal(X.shape)
y2 = y.copy()
# domain 3: a fixed pixel permutation -- same labels, scrambled discriminant
perm = rng.permutation(p)
X3 = X[:, perm]
y3 = y.copy()

Xa = np.vstack([X1, X2, X3])
ya = np.concatenate([y1, y2, y3])
dom = np.array(["original"] * len(y1) + ["noisy"] * len(y2) + ["scrambled"] * len(y3))

res = daxis.daxis_score(Xa, ya, dom, mode="classwise")
print(res.report())
print("\noriginal<->noisy     :", round(res.matrix[0, 1], 3), "(expect high -- GO)")
print("original<->scrambled :", round(res.matrix[0, 2], 3), "(expect low  -- NO-GO)")
res.plot("digits_daxis_matrix.png")
print("[saved] digits_daxis_matrix.png")
