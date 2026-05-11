# otadtk — Optimal Transport Audio Distance Toolkit

A drop-in replacement for FAD / KAD that adds a **learned ground metric**
and **per-sample diagnostics**, with a CLI compatible with
[`fadtk`](https://github.com/microsoft/fadtk) /
[`kadtk`](https://github.com/YoonjinXD/kadtk).

## Features

- **Three metrics behind a uniform interface** — OTAD (default), FAD, KAD;
  swap with a single flag.
- **Three OTAD variants** — `agnostic` (recommended), `native`,
  `raw` (no adapter, ablation).
- **Nine bundled encoders** — VGGish, EnCodec, CLAP, OpenL3, AudioMAE,
  AST, BEATs, PANNs, PANNs-WGLM (lazy-loaded; register your own with
  `@register_encoder`).
- **Self-contained** — all 9 × 2 = 18 pretrained Riemannian adapters
  ship inside the wheel (`otadtk/checkpoints/`, ≈ 43 MB) with SHA-256
  hashes in `MANIFEST.json`. No network access at scoring time.
- **Per-sample diagnostics** — every evaluation file gets a marginal
  transport cost `cj`; the API exposes top-k offenders, contamination
  AUROC, separation ratio, and CSV export so you can plot or analyse
  with whatever tooling you prefer.
- **kadtk-compatible cache** — embeddings live in
  `<dir>/embedding/<model>/<basename>.npy` so `otadtk` and `kadtk` can
  share a working directory.
- **Three CLI entry points** — `otadtk`, `otadtk-embeds`,
  `otadtk-list-models`.
- **CPU & CUDA** — pure-PyTorch primitives; CUDA is auto-used when
  available.

 ## Install
 
> **NOTE — review window.** PyPI distribution is paused during NeurIPS 2026
> double-blind review. Until the camera-ready release, install from source:
>
> ```bash
> git clone https://github.com/wonwoo-jeong/otadtk.git
> cd otadtk && pip install -e ".[all-encoders]"
> ```
> A `pip install otadtk` workflow will be re-enabled at camera-ready
> (see paper Appendix B.6).


Python 3.10–3.12, PyTorch ≥ 2.1.

## CLI

```text
otadtk <model> <ref_dir> <eval_dir> [options]
```

| Option | Effect |
|---|---|
| `--variant {agnostic,native,raw}` | OTAD adapter variant (default `agnostic`) |
| `--epsilon FLOAT` | Sinkhorn regularisation (default `0.1`) |
| `--diagnose` | Compute per-sample `cj` and print the top offenders |
| `--diag-out PATH` | Write per-file `cj` to a CSV (implies `--diagnose`) |
| `--top-k INT` | How many offenders to print (default `10`) |
| `--indiv PATH` | Per-file `cj` CSV (sorted ascending) |
| `--fad` / `--kad` | Compute FAD / KAD instead of OTAD on the same cache |
| `--bandwidth FLOAT` | KAD bandwidth (defaults to test-set median) |
| `--csv PATH` | Append a single-row record `(metric, model, variant, ref, eval, score, time)` |
| `--device {cuda,cpu}` | Override device (default: auto) |
| `--workers INT` | Parallelism for the embedding cache |
| `--force-emb-encode` | Ignore an existing embedding cache |
| `--adapter-checkpoint PATH` | Use a custom adapter checkpoint |

Auxiliary commands:

```bash
otadtk-embeds  -m vggish -d ref/ eval/   # populate the cache only
otadtk-list-models                       # registered encoders + dim/sr
```

## Python

```python
from otadtk import OTAD

otad  = OTAD(model="panns", variant="agnostic", epsilon=0.10)
score = otad.score("ref/", "eval/")                     # scalar OTAD
diag  = otad.diagnose("ref/", "eval/")                  # OTADDiagnostics
print(diag.top_k(10))                                   # worst offenders
diag.to_csv("cj.csv")                                   # file,cj rows
# diag.cj is a numpy array; diag.files is a list[Path] — plot however you like.
otad.score_individual("ref/", "eval/", csv="per_file.csv")

# AUROC against a known contamination flag
flags = [f.name.startswith("CONTAMINATED_") for f in diag.files]
print(diag.auroc(flags), diag.separation_ratio(flags))

# Functional API on numpy banks
from otadtk import compute_otad, compute_fad, compute_kad
score, cj = compute_otad(ref_emb, eval_emb, return_sample_costs=True)
```

## Encoders

| `model` | dim | sr | source |
|---|---:|---:|---|
| `vggish`     |  128 | 16 kHz | torch.hub `harritaylor/torchvggish` |
| `panns`      | 2048 | 32 kHz | PANNs CNN14 |
| `panns_wglm` | 2048 | 32 kHz | PANNs Wavegram-LogMel-CNN14 (KAD-paper backbone) |
| `clap`       |  512 | 48 kHz | LAION-CLAP music branch |
| `openl3`     |  512 | 48 kHz | OpenL3 mel256-music |
| `audiomae`   |  768 | 16 kHz | AudioMAE-base |
| `ast`        |  768 | 16 kHz | AST-base |
| `beats`      |  768 | 16 kHz | BEATs-iter3+ |
| `encodec`    |  128 | 24 kHz | EnCodec 24 kHz |

## Variants

| Variant | Adapter | Use |
|---|---|---|
| `agnostic` | class-contrastive triplets | Default; cross-axis decompositions and MOS evaluation |
| `native`   | trained to minimise Sinkhorn divergence | High-sensitivity outlier scoring |
| `raw`      | identity (no adapter) | Ablation; reproduces the paper's "raw" baseline |

## Citation

```bibtex
@misc{jeong2026otad,
  author        = {Wonwoo Jeong},
  title         = {{OTAD}: Optimal Transport Audio Distance with Learned Riemannian Ground Metrics},
  year          = {2026},
  eprint        = {2605.05554},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS}
}
```

## License

CC-BY-4.0 — see [`LICENSE`](LICENSE).
