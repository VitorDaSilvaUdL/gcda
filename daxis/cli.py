"""``daxis`` command-line entry point.

    daxis data.csv --label y --domain env --mode auto --plot m.png

Accepts a CSV (one row per sample; a label column, a domain column and the
rest features) or an ``.npz`` with arrays ``X``, ``y``, ``domain``.
"""
from __future__ import annotations

import argparse
import json


def _build_parser():
    ap = argparse.ArgumentParser(
        prog="daxis",
        description="Discriminant-Axis Alignment -- a GO/NO-GO "
                    "diagnostic for domain adaptation, before training.")
    ap.add_argument("data", help="path to a .csv or .npz file")
    ap.add_argument("--label", default="label", help="label column (CSV)")
    ap.add_argument("--domain", default="domain", help="domain column (CSV)")
    ap.add_argument("--features", nargs="*", default=None,
                    help="feature columns (CSV); default = all the rest")
    ap.add_argument("--mode", default="auto",
                    choices=["auto", "binary", "classwise", "subspace"])
    ap.add_argument("--boot", type=int, default=500, help="bootstrap resamples")
    ap.add_argument("--perm", type=int, default=500, help="null permutations")
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--plot", default=None, help="save the cosine heatmap here")
    ap.add_argument("--json", default=None, help="write the result as JSON here")
    return ap


def main(argv=None):
    args = _build_parser().parse_args(argv)
    kw = dict(mode=args.mode, n_boot=args.boot, n_perm=args.perm,
              standardize=not args.no_standardize)

    if args.data.endswith(".npz"):
        import numpy as np
        from .core import daxis_score
        z = np.load(args.data, allow_pickle=True)
        res = daxis_score(z["X"], z["y"], z["domain"], **kw)
    else:
        import pandas as pd
        from .adapters import from_dataframe
        df = pd.read_csv(args.data)
        res = from_dataframe(df, args.features, args.label, args.domain, **kw)

    print(res.report())
    if args.plot:
        res.plot(args.plot)
        print(f"[plot] {args.plot}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res.to_dict(), fh, indent=2)
        print(f"[json] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
