"""otadtk-embeds — populate the on-disk embedding cache without computing OTAD.

Mirrors ``kadtk-embeds``: a one-shot way to amortise encoding before scoring
multiple metric variants over the same dataset.

Usage:
    otadtk-embeds -m vggish -d ref/ eval/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .emb_loader import EmbeddingLoader
from .model_loader import get_encoder, list_models


def main() -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(prog="otadtk-embeds")
    parser.add_argument(
        "-m", "--model", required=True, choices=list_models(),
        help="Encoder name to extract embeddings with.",
    )
    parser.add_argument(
        "-d", "--dirs", nargs="+", required=True,
        help="One or more audio directories (or single audio files).",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    encoder = get_encoder(args.model, device=args.device or "cuda")
    loader = EmbeddingLoader(encoder, force_recompute=args.force)
    for d in args.dirs:
        loader.cache_directory(Path(d), workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
