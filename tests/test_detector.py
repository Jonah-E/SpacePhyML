"""
Unit tests for spacephyml/algorithms/detector.py – OutlierDetector
"""
import numpy as np
import pandas as pd
import pytest
import sys
import types
from pathlib import Path

# The algorithms __init__.py has a broken bare import; register a package
# stub before importing the detector submodule directly.
import spacephyml as _sphy  # noqa: E402 (ensure parent is loaded)
if "spacephyml.algorithms" not in sys.modules or not hasattr(
    sys.modules["spacephyml.algorithms"], "__path__"
):
    _alg_pkg = types.ModuleType("spacephyml.algorithms")
    _alg_pkg.__path__ = [
        str(Path(__file__).parent.parent / "spacephyml" / "algorithms")
    ]
    _alg_pkg.__package__ = "spacephyml.algorithms"
    sys.modules["spacephyml.algorithms"] = _alg_pkg

from spacephyml.algorithms.detector import OutlierDetector, cal_detections  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_samples(n, features=4, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, features)).astype(np.float32)


def _calibrate_detector(detector, n_calib=20, features=4):
    """Push enough clean samples to get the detector out of Init mode."""
    samples = _make_samples(n_calib, features)
    detector(samples)
    return detector


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestOutlierDetectorInit:
    def test_default_construction(self):
        det = OutlierDetector()
        assert det is not None

    def test_history_is_empty_dataframe(self):
        det = OutlierDetector()
        h = det.get_history()
        assert isinstance(h, pd.DataFrame)
        assert len(h) == 0

    def test_custom_params_stored(self):
        det = OutlierDetector(error_threshold=5.0, n_components=2, calib_batch_size=8)
        assert det.error_threshold == 5.0
        assert det.n_components == 2
        assert det.calib_batch_size == 8


# ---------------------------------------------------------------------------
# manual_calibration
# ---------------------------------------------------------------------------

class TestManualCalibration:
    def test_manual_calib_sets_check_mode(self):
        det = OutlierDetector(calib_batch_size=5, n_components=1)
        samples = _make_samples(10)
        det.manual_calibration(samples)
        assert det._calib_mode == 'Check'

    def test_manual_calib_returns_self(self):
        det = OutlierDetector(calib_batch_size=5, n_components=1)
        samples = _make_samples(10)
        assert det.manual_calibration(samples) is det


# ---------------------------------------------------------------------------
# __call__ / basic detection
# ---------------------------------------------------------------------------

class TestOutlierDetectorCall:
    def test_returns_array_of_correct_shape(self):
        det = OutlierDetector(calib_batch_size=5, n_components=1)
        samples = _make_samples(5)
        result = det(samples)
        assert result.shape == (5,)

    def test_history_grows_with_each_call(self):
        det = OutlierDetector(calib_batch_size=5, n_components=1)
        s1 = _make_samples(5)
        s2 = _make_samples(5)
        det(s1)
        det(s2)
        assert len(det.get_history()) == 10

    def test_clean_data_produces_few_outliers(self):
        """After calibration on similar data, very few samples should be flagged."""
        rng = np.random.default_rng(0)
        calib = rng.standard_normal((30, 4)).astype(np.float32)
        test = rng.standard_normal((30, 4)).astype(np.float32)

        det = OutlierDetector(calib_batch_size=20, n_components=2,
                              mean_window=20, error_threshold=5.0)
        det(calib)
        result = det(test)
        # With a generous threshold and same-distribution data, at most 10% should be flagged
        assert result.sum() < len(result) * 0.1

    def test_extreme_outlier_is_detected(self):
        """A sample that is orders of magnitude larger should be flagged."""
        rng = np.random.default_rng(1)
        calib = rng.standard_normal((30, 4)).astype(np.float32)

        det = OutlierDetector(calib_batch_size=20, n_components=2,
                              mean_window=20, error_threshold=2.0)
        det(calib)

        # Fill mean buffer
        normal = rng.standard_normal((30, 4)).astype(np.float32)
        det(normal)

        spike = np.ones((1, 4), dtype=np.float32) * 1000.0
        result = det(spike)
        assert result[0] == 1.0


# ---------------------------------------------------------------------------
# reset_mean / reset_history
# ---------------------------------------------------------------------------

class TestResets:
    def test_reset_mean_clears_buffer(self):
        # Tests the non-EWMA path; _mean_ptr is only used when use_ewma=False (default)
        det = OutlierDetector(calib_batch_size=5, n_components=1)
        _calibrate_detector(det)
        det.reset_mean()
        assert not det._mean_buffer_filled
        assert det._mean_ptr == 0

    def test_reset_history_empties_dataframe(self):
        det = OutlierDetector(calib_batch_size=5, n_components=1)
        det(_make_samples(10))
        det.reset_history()
        assert len(det.get_history()) == 0


# ---------------------------------------------------------------------------
# get_history columns
# ---------------------------------------------------------------------------

class TestGetHistory:
    def test_history_has_required_columns(self):
        det = OutlierDetector(calib_batch_size=5, n_components=1)
        det(_make_samples(10))
        h = det.get_history()
        for col in ('Error', 'Threshold', 'Outlier', 'Calibration', 'Flag'):
            assert col in h.columns

    def test_outlier_column_is_bool(self):
        det = OutlierDetector(calib_batch_size=5, n_components=1)
        det(_make_samples(10))
        h = det.get_history()
        assert h['Outlier'].dtype == bool


# ---------------------------------------------------------------------------
# cal_detections
# ---------------------------------------------------------------------------

class TestCalDetections:
    def _make_history(self):
        times = np.arange(10, dtype=float)
        outliers = np.array([0, 0, 1, 0, 0, 1, 1, 0, 0, 0], dtype=bool)
        flags = outliers.astype(int)
        return pd.DataFrame({
            'Time': times,
            'Outlier': outliers,
            'Flag': flags,
        })

    def test_returns_four_tuples(self):
        data = self._make_history()
        rois = [(2.0, 3.0)]
        result = cal_detections(data, rois)
        assert len(result) == 4

    def test_roi_detection_count(self):
        data = self._make_history()
        rois = [(2.0, 3.0), (8.0, 9.0)]  # first has outlier, second doesn't
        (rois_det, total_rois), *_ = cal_detections(data, rois)
        assert rois_det == 1
        assert total_rois == 2

    def test_total_outlier_count(self):
        data = self._make_history()
        rois = [(0.0, 9.0)]
        _, (outlier_in_roi, total_outliers), *_ = cal_detections(data, rois)
        assert total_outliers == 3  # 3 True values in the full dataset (at positions 2, 5, 6)

    def test_empty_rois(self):
        data = self._make_history()
        (rois_det, total_rois), *_ = cal_detections(data, [])
        assert rois_det == 0
        assert total_rois == 0
