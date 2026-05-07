"""Smoke tests that don't require pretrained adapters or real audio.

These exercise the metric primitives on synthetic embedding banks so the
package can be installed and tested in CI without GPU or large downloads.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from otadtk import (
    RiemannianAdapter,
    compute_fad,
    compute_kad,
    compute_otad,
    list_models,
)


@pytest.fixture(scope="module")
def synthetic_banks():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(64, 32)).astype(np.float32)
    same = ref + 0.05 * rng.normal(size=ref.shape).astype(np.float32)
    diff = rng.normal(size=ref.shape).astype(np.float32) + 1.5
    return ref, same, diff


def test_fad_zero_for_identical(synthetic_banks):
    ref, same, _ = synthetic_banks
    assert compute_fad(ref, same) < 1.0
    assert compute_fad(ref, ref) < 1e-6


def test_fad_grows_with_shift(synthetic_banks):
    ref, same, diff = synthetic_banks
    assert compute_fad(ref, same) < compute_fad(ref, diff)


def test_kad_kadtk_aligned(synthetic_banks):
    ref, same, diff = synthetic_banks
    # Unbiased MMD U-statistic can be slightly negative on finite samples — that
    # is precisely the property that lets KAD pass through "no significant
    # difference" with a near-zero value.  We only require it to be small.
    assert abs(compute_kad(ref, ref, scale=False)) < 0.1
    s_same = compute_kad(ref, same)
    s_diff = compute_kad(ref, diff)
    assert s_same < s_diff


def test_otad_runs_without_adapter(synthetic_banks):
    ref, same, diff = synthetic_banks
    s_same = compute_otad(ref, same, epsilon=0.1, device="cpu")
    s_diff = compute_otad(ref, diff, epsilon=0.1, device="cpu")
    assert s_same < s_diff
    assert s_same > -1.0


def test_otad_returns_per_sample_costs(synthetic_banks):
    ref, _, diff = synthetic_banks
    score, cj = compute_otad(
        ref, diff, epsilon=0.1, return_sample_costs=True, device="cpu"
    )
    assert isinstance(score, float)
    assert cj.shape == (diff.shape[0],)


def test_riemannian_adapter_identity_at_init():
    a = RiemannianAdapter(input_dim=32)
    z = torch.randn(8, 32)
    out = a(z)
    assert torch.allclose(out, z, atol=1e-6)


def test_list_models_contains_expected_encoders():
    models = list_models()
    for required in ("vggish", "panns", "panns_wglm", "clap", "openl3"):
        assert required in models
