"""Unit tests for the iSMOTE balancing algorithm (numpy + sklearn only)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ismote import ismote  # noqa: E402


def _imbalanced_blobs(seed=0):
    """3 well-separated classes with counts 80 / 15 / 5."""
    rng = np.random.default_rng(seed)
    X = np.concatenate([
        rng.normal(0.0, 0.1, (80, 3, 16)),
        rng.normal(5.0, 0.1, (15, 3, 16)),
        rng.normal(-5.0, 0.1, (5, 3, 16)),
    ])
    y = np.concatenate([np.zeros(80), np.ones(15), np.full(5, 2)]).astype(int)
    return X, y


def test_ismote_balances_all_classes_to_majority():
    X, y = _imbalanced_blobs()
    Xb, yb = ismote(X, y, k_neighbors=3)
    _, counts = np.unique(yb, return_counts=True)
    assert counts.tolist() == [80, 80, 80]


def test_ismote_preserves_originals():
    X, y = _imbalanced_blobs()
    Xb, yb = ismote(X, y, k_neighbors=3)
    # The first N samples of the output are the untouched originals
    assert np.allclose(Xb[: len(X)], X)
    assert np.array_equal(yb[: len(y)], y)


def test_ismote_synthetics_stay_inside_their_class_cluster():
    X, y = _imbalanced_blobs()
    Xb, yb = ismote(X, y, k_neighbors=3)
    synth_mask = np.arange(len(yb)) >= len(y)
    for cls, center in [(1, 5.0), (2, -5.0)]:
        synth_cls = Xb[synth_mask & (yb == cls)]
        if len(synth_cls):
            # Interpolated samples of a tight cluster stay near its centre
            assert np.abs(synth_cls.mean() - center) < 1.0


def test_ismote_output_shape_matches_input_dims():
    X, y = _imbalanced_blobs()
    Xb, yb = ismote(X, y, k_neighbors=3)
    assert Xb.shape[1:] == X.shape[1:]
    assert len(Xb) == len(yb)
