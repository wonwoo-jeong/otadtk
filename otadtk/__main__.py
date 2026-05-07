"""otadtk command-line entry point.

Usage
-----

    otadtk <model> <ref_dir> <eval_dir> [options]

Options
-------

    --variant {agnostic,native,raw}        OTAD adapter variant (default: agnostic).
    --epsilon FLOAT                         Sinkhorn regularisation (default: 0.1).
    --device {cuda,cpu}                     Compute device (default: auto).
    --csv PATH                              Append a single-row CSV record.
    --diagnose                              Compute per-sample c_j; print top-k offenders.
    --diag-out PATH                         When --diagnose, write c_j CSV here.
    --top-k INT                             Print the top-k worst offenders.
    --indiv PATH                            Write per-file scores to PATH (CSV).
    --fad / --kad                           Compute FAD or KAD instead of OTAD (for comparison).
    --bandwidth FLOAT                       KAD bandwidth (defaults to test-set median).
    --workers INT                           Parallelism for the embedding cache.
    --force-emb-encode                      Recompute the embedding cache.
    --adapter-checkpoint PATH               Override the adapter checkpoint location.

Example
-------

    otadtk panns ref/ eval/                                   # OTAD-agnostic
    otadtk panns ref/ eval/ --variant raw --epsilon 0.05      # raw OTAD
    otadtk vggish ref/ eval/ --diagnose --diag-out cj.csv     # per-sample diagnostics
    otadtk panns ref/ eval/ --kad                             # KAD baseline
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .otad import OTAD


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="otadtk",
        description="Optimal Transport Audio Distance Toolkit",
    )
    p.add_argument("model", help="Encoder name (use otadtk-list-models to see options)")
    p.add_argument("ref", help="Reference (ground-truth) directory or single audio file")
    p.add_argument("eval", help="Evaluation (generated) directory or single audio file")

    p.add_argument("--variant", choices=["agnostic", "native", "raw"], default="agnostic")
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--device", choices=["cuda", "cpu"], default=None)
    p.add_argument("--workers", type=int, default=1)

    p.add_argument("--csv", type=str, default=None,
                   help="Append a single-row record (model,ref,eval,score,time) to this CSV.")
    p.add_argument("--indiv", type=str, default=None,
                   help="Write per-file OTAD scores (cj) to this CSV.")
    p.add_argument("--diagnose", action="store_true",
                   help="Compute per-sample c_j (also dumps the top offenders).")
    p.add_argument("--diag-out", type=str, default=None,
                   help="Where to write the c_j CSV (implies --diagnose).")
    p.add_argument("--top-k", type=int, default=10,
                   help="Top-k offenders to print under --diagnose.")

    p.add_argument("--fad", action="store_true", help="Compute FAD instead of OTAD.")
    p.add_argument("--kad", action="store_true", help="Compute KAD instead of OTAD.")
    p.add_argument("--bandwidth", type=float, default=None,
                   help="KAD bandwidth; default is the median over the eval set.")

    p.add_argument("--force-emb-encode", action="store_true",
                   help="Recompute the embedding cache even if files exist.")
    p.add_argument("--adapter-checkpoint", type=str, default=None,
                   help="Override the adapter checkpoint path.")
    p.add_argument("--version", action="version", version=f"otadtk {__version__}")
    return p


def _setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _append_csv(path: Path, row: list) -> None:
    path = Path(path)
    is_new = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as fp:
        w = csv.writer(fp)
        if is_new:
            w.writerow(["metric", "model", "variant", "ref", "eval", "score", "time"])
        w.writerow(row)


def main() -> int:  # pragma: no cover - exercised by integration tests
    args = _build_parser().parse_args()
    _setup_logger()
    log = logging.getLogger("otadtk")
    if args.fad and args.kad:
        log.error("Choose at most one of --fad / --kad")
        return 2

    do_diagnose = args.diagnose or args.diag_out
    metric_name = "FAD" if args.fad else ("KAD" if args.kad else "OTAD")

    if not args.fad and not args.kad:
        # ---------------------- OTAD ----------------------
        otad = OTAD(
            model=args.model,
            variant=args.variant,
            epsilon=args.epsilon,
            device=args.device,
            adapter_checkpoint=args.adapter_checkpoint,
            force_recompute=args.force_emb_encode,
        )
        if do_diagnose:
            diag = otad.diagnose(args.ref, args.eval, workers=args.workers)
            log.info(
                f"OTAD = {diag.score:.6f}  ({otad.model_name}, variant={args.variant}, eps={args.epsilon})"
            )
            if args.top_k > 0:
                log.info(f"Top-{args.top_k} offenders by c_j:")
                for f, c in diag.top_k(args.top_k):
                    log.info(f"  {c:.6f}  {f}")
            if args.diag_out:
                diag.to_csv(args.diag_out)
                log.info(f"Per-sample c_j written to {args.diag_out}")
            score = diag.score
            if args.indiv:
                otad.score_individual(args.ref, args.eval, csv=args.indiv,
                                      workers=args.workers)
        else:
            score = otad.score(args.ref, args.eval, workers=args.workers)
            log.info(
                f"OTAD = {score:.6f}  ({otad.model_name}, variant={args.variant}, eps={args.epsilon})"
            )
            if args.indiv:
                otad.score_individual(args.ref, args.eval, csv=args.indiv,
                                      workers=args.workers)
    else:
        # ---------------------- FAD / KAD ----------------------
        from .emb_loader import EmbeddingLoader
        from .fad import compute_fad
        from .kad import compute_kad
        from .model_loader import get_encoder

        encoder = get_encoder(args.model, device=args.device or "cuda")
        loader = EmbeddingLoader(encoder, force_recompute=args.force_emb_encode)
        embs_ref, _ = loader.load_embeddings(Path(args.ref), workers=args.workers)
        embs_eval, _ = loader.load_embeddings(Path(args.eval), workers=args.workers)
        if args.fad:
            score = float(compute_fad(embs_ref, embs_eval))
            log.info(f"FAD = {score:.6f}  ({encoder.name})")
        else:
            score = float(
                compute_kad(embs_ref, embs_eval, bandwidth=args.bandwidth)
            )
            log.info(f"KAD = {score:.6f}  ({encoder.name})")

    if args.csv:
        _append_csv(
            Path(args.csv),
            [metric_name, args.model, args.variant, args.ref, args.eval, score, time.time()],
        )

    print(f"{metric_name}: {score:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
