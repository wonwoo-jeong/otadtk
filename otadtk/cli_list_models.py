"""otadtk-list-models — print the registered encoders."""
from __future__ import annotations

import sys

from .model_loader import _LAZY_MODULE_MAP, list_models


_DESCRIPTION = {
    "vggish":     ("128",  "16 kHz", "Hershey et al., ICASSP 2017 — original FAD encoder."),
    "panns":      ("2048", "32 kHz", "Kong et al., 2020 — PANNs CNN14."),
    "panns_wglm": ("2048", "32 kHz", "Kong et al., 2020 — PANNs Wavegram-LogMel-CNN14 (KAD-paper backbone)."),
    "clap":       ("512",  "48 kHz", "LAION-CLAP music branch."),
    "openl3":     ("512",  "48 kHz", "Cramer et al., 2019 — OpenL3 mel256-music."),
    "audiomae":   ("768",  "16 kHz", "Huang et al., 2022 — AudioMAE-base."),
    "ast":        ("768",  "16 kHz", "Gong et al., 2021 — AST-base."),
    "beats":      ("768",  "16 kHz", "Chen et al., 2023 — BEATs-iter3+."),
    "encodec":    ("128",  "24 kHz", "Defossez et al., 2023 — EnCodec 24 kHz."),
}


def main() -> int:  # pragma: no cover
    print("Registered encoders for otadtk:")
    print(f"  {'name':<14} {'dim':>4} {'sr':<8} description")
    print(f"  {'-'*14} {'-'*4} {'-'*8} {'-'*60}")
    for name in list_models():
        dim, sr, desc = _DESCRIPTION.get(name, ("?", "?", ""))
        print(f"  {name:<14} {dim:>4} {sr:<8} {desc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
