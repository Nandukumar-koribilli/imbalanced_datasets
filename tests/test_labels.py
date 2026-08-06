"""Tests for UCI HAR label consistency across the pipeline."""
import os
import sys

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


UCI_HAR_LABELS = {
    0: "Walking",
    1: "Walking Upstairs",
    2: "Walking Downstairs",
    3: "Sitting",
    4: "Standing",
    5: "Laying",
}


def test_uci_har_labels_zero_indexed():
    """UCI HAR original labels are 1-6 and must be converted to 0-5."""
    original_labels = np.array([1, 2, 3, 4, 5, 6])
    converted = original_labels - 1

    assert converted.min() == 0
    assert converted.max() == 5
    assert len(np.unique(converted)) == 6


def test_uci_har_label_count():
    """There should be exactly 6 activity classes."""
    assert len(UCI_HAR_LABELS) == 6
    for i in range(6):
        assert i in UCI_HAR_LABELS


def test_ismote_preserves_labels():
    """I-SMOTE must not create labels outside the original label set."""
    from src.ismote import ismote

    X = np.random.rand(100, 9, 128)
    y = np.concatenate([np.zeros(50), np.ones(30), np.full(20, 2)])

    X_res, y_res = ismote(X, y, k_neighbors=5)

    assert set(y_res).issubset(set(y))


def test_label_dtype():
    """Labels should be integer type."""
    labels = np.array([0, 1, 2, 3, 4, 5], dtype=int)
    assert labels.dtype in [np.int32, np.int64, np.int_]
