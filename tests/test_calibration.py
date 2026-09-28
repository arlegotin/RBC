import math

import numpy as np
import pytest

from rbc.calibration import fit_threshold, retained_mask, state_scores
from rbc.types import ValidationError


def test_state_scores_keep_all_worlds_and_include_quantile_ties():
    logits = np.array([0.0, 1.0, 1.0, -2.0], dtype=np.float32)
    assert state_scores(logits).tolist() == [1.0, 0.0, 0.0, 3.0]
    assert retained_mask(logits, 0.0).tolist() == [False, True, True, False]
    assert retained_mask(logits, float("inf")).tolist() == [True, True, True, True]


def test_unattainable_rank_is_infinite_not_clamped():
    assert math.isinf(fit_threshold(np.zeros(10), alpha=0.005))
    scores = np.linspace(0.0, 255.0, 256)
    assert fit_threshold(scores, alpha=0.005) == 255.0
    assert fit_threshold(np.array([1.0, 1.0, 2.0]), alpha=0.5) == 1.0


def test_nonfinite_dense_input_fails_before_calibration():
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValidationError):
            state_scores(np.array([0.0, value]))
        with pytest.raises(ValidationError):
            fit_threshold(np.array([0.0, value]), alpha=0.02)
    with pytest.raises(ValidationError):
        fit_threshold(np.array([0.0, 1.0]), alpha=1.0)
