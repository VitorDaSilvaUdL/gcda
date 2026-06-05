"""Run GCDA from a pandas DataFrame / CSV -- the tabular workflow.

Shows the two equivalent entry points (a DataFrame in memory, and the same data
written to CSV and read back), plus the command-line form.

    python examples/pandas_csv.py
"""
import numpy as np
import pandas as pd

import gcda


def make_site(axis, n=300, p=8, seed=0):
    rng = np.random.default_rng(seed)
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    y = rng.integers(0, 2, n)
    X = rng.standard_normal((n, p)) + 2.0 * (y[:, None] - 0.5) * axis
    return X, y


# three measurement "sites" that share a discriminant axis (a GO case)
P = 8
blocks = []
for i, tilt in enumerate([0.0, 0.1, -0.1]):
    axis = np.zeros(P)
    axis[0], axis[1] = 1.0, tilt
    X, y = make_site(axis, seed=i + 1)
    df = pd.DataFrame(X, columns=[f"f{j}" for j in range(P)])
    df["y"] = y
    df["site"] = f"site_{i}"
    blocks.append(df)
data = pd.concat(blocks, ignore_index=True)

# 1) straight from the DataFrame (feature columns auto-detected)
res = gcda.from_dataframe(data, label_col="y", domain_col="site")
print(res.report())

# 2) the same via a CSV on disk
data.to_csv("sites.csv", index=False)
res_csv = gcda.from_dataframe(pd.read_csv("sites.csv"), label_col="y", domain_col="site")
assert abs(res.score - res_csv.score) < 1e-9
print("\nCSV round-trip score matches:", round(res_csv.score, 3))
print("Command line equivalent:\n  gcda sites.csv --label y --domain site --plot m.png")
