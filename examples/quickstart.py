"""End-to-end example: score two folders with otadtk.

Run::

    python examples/quickstart.py --ref ref/ --eval eval/

The script demonstrates three usage modes that mirror the paper:

  1. ``OTAD.score``           — single aggregate score.
  2. ``OTAD.diagnose``        — per-sample c_j (top offenders + CSV).
  3. ``OTAD.score_individual``— per-file scores for inspection.
"""
from __future__ import annotations

import argparse

from otadtk import OTAD


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="panns")
    parser.add_argument("--variant", default="agnostic", choices=["agnostic", "native", "raw"])
    parser.add_argument("--ref", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--cj-csv", default="cj.csv",
                        help="Where to write per-file c_j (default: cj.csv).")
    args = parser.parse_args()

    otad = OTAD(model=args.model, variant=args.variant, epsilon=args.epsilon)

    score = otad.score(args.ref, args.eval)
    print(f"Aggregate OTAD = {score:.4f}")

    diag = otad.diagnose(args.ref, args.eval)
    print("\nTop-5 worst evaluation files (largest c_j):")
    for f, cj in diag.top_k(5):
        print(f"  cj = {cj:.4f}   {f}")

    out = diag.to_csv(args.cj_csv)
    print(f"\nPer-file c_j written to {out}")


if __name__ == "__main__":
    main()
