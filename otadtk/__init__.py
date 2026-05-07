"""otadtk — Optimal Transport Audio Distance Toolkit.

Public re-exports:
    - ``OTAD``                            : main scoring class.
    - ``compute_otad`` / ``compute_fad``  : functional API on numpy embeddings.
    - ``compute_kad``                     : KAD baseline (kadtk-aligned bandwidth).
    - ``RiemannianAdapter``               : the cost-side primitive.
    - ``OTADDiagnostics``                 : per-sample c_j, AUROC, plotting.
    - ``get_encoder``                     : load an encoder by name.
    - ``list_models``                     : enumerate registered encoders.
"""
from .otad import OTAD, compute_otad
from .fad import compute_fad
from .kad import compute_kad
from .adapter import RiemannianAdapter, load_adapter
from .diagnostics import OTADDiagnostics
from .model_loader import (
    BaseEncoder,
    get_encoder,
    list_models,
    register_encoder,
)

__version__ = "0.1.0"

__all__ = [
    "OTAD",
    "compute_otad",
    "compute_fad",
    "compute_kad",
    "RiemannianAdapter",
    "load_adapter",
    "OTADDiagnostics",
    "BaseEncoder",
    "get_encoder",
    "list_models",
    "register_encoder",
    "__version__",
]
