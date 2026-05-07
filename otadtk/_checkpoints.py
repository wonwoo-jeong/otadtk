"""Adapter checkpoint resolution.

Checkpoints are resolved in the following priority order:

  1. Explicit ``override`` (e.g. CLI ``--adapter-checkpoint``).
  2. Package-bundled location ``otadtk/checkpoints/<encoder>/adapter_<loss>_best.pt``.
     The package ships these by default — see ``pyproject.toml`` ``package-data`` —
     so ``pip install otadtk`` and a fresh clone of the repository both work
     out of the box, with no network access required.
  3. ``$OTADTK_CHECKPOINTS`` environment variable (a directory laid out the same
     way).  Useful if the user retrains adapters and wants to keep them outside
     the install tree.

If none of the above resolves, we raise a descriptive error rather than fall
back to a network download — every official distribution bundles the eighteen
adapter checkpoints, so a missing file means the package was stripped on
purpose and the caller should either restore it, point ``OTADTK_CHECKPOINTS``
at a local copy, or pass ``--adapter-checkpoint``.

The mapping between OTAD variants and on-disk loss tags follows the training
script naming scheme:

    agnostic  -> triplet_contrastive
    native    -> sinkhorn_native
"""
from __future__ import annotations

import hashlib
import importlib.resources as resources
import os
from pathlib import Path

VARIANT_TO_LOSS = {
    "agnostic": "triplet_contrastive",
    "native": "sinkhorn_native",
}


def _bundled_path(encoder: str, variant: str) -> Path | None:
    """Return the package-bundled checkpoint path, if any."""
    loss = VARIANT_TO_LOSS[variant]
    fname = f"adapter_{loss}_best.pt"
    try:
        # importlib.resources gives us a path that works equally for installed
        # wheels, source checkouts, and zipped distributions.
        ref = resources.files("otadtk").joinpath("checkpoints", encoder, fname)
        if ref.is_file():
            return Path(str(ref))
    except (ModuleNotFoundError, FileNotFoundError, AttributeError):
        return None
    return None


def _env_path(encoder: str, variant: str) -> Path | None:
    base = os.environ.get("OTADTK_CHECKPOINTS")
    if not base:
        return None
    loss = VARIANT_TO_LOSS[variant]
    p = Path(base) / encoder / f"adapter_{loss}_best.pt"
    return p if p.exists() else None


def resolve_adapter_checkpoint(
    encoder: str,
    variant: str,
    override: str | os.PathLike | None = None,
) -> Path:
    if variant not in VARIANT_TO_LOSS:
        raise ValueError(f"Unknown OTAD variant: {variant!r}")

    if override is not None:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    bundled = _bundled_path(encoder, variant)
    if bundled is not None:
        return bundled

    env = _env_path(encoder, variant)
    if env is not None:
        return env

    loss = VARIANT_TO_LOSS[variant]
    raise FileNotFoundError(
        f"Adapter checkpoint not found for ({encoder}, {variant}). "
        f"All official distributions bundle this file under "
        f"otadtk/checkpoints/{encoder}/adapter_{loss}_best.pt; if you "
        f"installed a stripped wheel, restore the file, set "
        f"OTADTK_CHECKPOINTS=/path/to/dir, or pass --adapter-checkpoint."
    )


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
