"""
Unit tests for spacephyml/transforms.py
"""
import numpy as np
import pytest
from spacephyml.transforms import (
    Compose,
    FeatureCoupledMinMaxScaler,
    Flatten,
    Log10,
    LogNorm,
    Mean,
    MoveAxis,
    Roll,
    Sum,
    Threshold,
    ZScoreNorm,
)


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_clips_below_low(self):
        t = Threshold((0.1, 10.0))
        x = np.array([-1.0, 0.05, 5.0])
        result = t(x)
        assert np.all(result >= 0.1)
        assert result[2] == 5.0  # in-range value must be unchanged

    def test_clips_above_high(self):
        t = Threshold((0.0, 5.0))
        x = np.array([0.0, 3.0, 99.0])
        result = t(x)
        assert np.all(result <= 5.0)

    def test_values_in_range_unchanged(self):
        t = Threshold((0.0, 10.0))
        x = np.array([1.0, 5.0, 9.0])
        result = t(x.copy())
        np.testing.assert_array_equal(result, x)

    def test_2d_array(self):
        t = Threshold((0.0, 1.0))
        x = np.array([[2.0, -1.0], [0.5, 0.5]])
        result = t(x)
        assert result.max() <= 1.0
        assert result.min() >= 0.0


# ---------------------------------------------------------------------------
# LogNorm
# ---------------------------------------------------------------------------

class TestLogNorm:
    def test_output_range_with_normalization(self):
        ln = LogNorm(normalization=(-2, 0))
        x = np.array([[0.01, 0.1, 1.0]])  # log10 → -2, -1, 0
        result = ln(x)
        np.testing.assert_allclose(result, [[0.0, 0.5, 1.0]], atol=1e-6)

    def test_output_range_without_normalization(self):
        ln = LogNorm()
        x = np.array([1.0, 10.0, 100.0])
        result = ln(x)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

class TestFlatten:
    def test_2d_to_1d(self):
        f = Flatten()
        x = np.ones((4, 8))
        assert f(x).shape == (32,)

    def test_3d_to_1d(self):
        f = Flatten()
        x = np.ones((2, 3, 4))
        assert f(x).shape == (24,)


# ---------------------------------------------------------------------------
# Log10
# ---------------------------------------------------------------------------

class TestLog10:
    def test_nonzero_values(self):
        t = Log10()
        x = np.array([1.0, 10.0, 100.0])
        result = t(x.copy())
        np.testing.assert_allclose(result, [0.0, 1.0, 2.0])

    def test_zero_values_unchanged(self):
        t = Log10()
        x = np.array([0.0, 10.0])
        result = t(x.copy())
        assert result[0] == 0.0


# ---------------------------------------------------------------------------
# Roll
# ---------------------------------------------------------------------------

class TestRoll:
    def test_default_shift(self):
        r = Roll(shift=2, axis=0)
        x = np.array([1, 2, 3, 4])
        result = r(x)
        np.testing.assert_array_equal(result, [3, 4, 1, 2])

    def test_zero_shift_is_identity(self):
        r = Roll(shift=0, axis=0)
        x = np.arange(8, dtype=float)
        np.testing.assert_array_equal(r(x.copy()), x)


# ---------------------------------------------------------------------------
# MoveAxis
# ---------------------------------------------------------------------------

class TestMoveAxis:
    def test_moves_axis(self):
        m = MoveAxis(src=-1, dst=0)
        x = np.ones((3, 4, 5))
        assert m(x).shape == (5, 3, 4)


# ---------------------------------------------------------------------------
# Sum
# ---------------------------------------------------------------------------

class TestSum:
    def test_sum_last_axis(self):
        s = Sum(axis=-1)
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = s(x)
        np.testing.assert_array_equal(result, [6, 15])

    def test_sum_first_axis(self):
        s = Sum(axis=0)
        x = np.ones((4, 3))
        result = s(x)
        np.testing.assert_array_equal(result, np.full(3, 4.0))


# ---------------------------------------------------------------------------
# Mean
# ---------------------------------------------------------------------------

class TestMean:
    def test_mean_last_axis(self):
        m = Mean(axis=-1)
        x = np.array([[0.0, 2.0], [4.0, 6.0]])
        result = m(x)
        np.testing.assert_allclose(result, [1.0, 5.0])


# ---------------------------------------------------------------------------
# ZScoreNorm
# ---------------------------------------------------------------------------

class TestZScoreNorm:
    def test_normalizes_correctly(self):
        z = ZScoreNorm(mean=2.0, std=2.0)
        x = np.array([0.0, 2.0, 4.0])
        result = z(x)
        np.testing.assert_allclose(result, [-1.0, 0.0, 1.0])

    def test_zero_mean_unit_std(self):
        z = ZScoreNorm(mean=0.0, std=1.0)
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(z(x), x)


# ---------------------------------------------------------------------------
# FeatureCoupledMinMaxScaler
# ---------------------------------------------------------------------------

class TestFeatureCoupledMinMaxScaler:
    def _make_data(self):
        # Two feature groups: cols 0-2 and cols 2-4
        return np.array([
            [0.0, 1.0, 10.0, 20.0],
            [1.0, 2.0, 30.0, 40.0],
            [0.5, 1.5, 20.0, 30.0],
        ], dtype=float)

    def test_fit_then_transform_in_range(self):
        features = [(0, 2), (2, 4)]
        scaler = FeatureCoupledMinMaxScaler(features)
        x = self._make_data()
        scaler.fit(x)
        result = scaler.transform(x.copy())
        assert result.min() >= 0.0 - 1e-9
        assert result.max() <= 1.0 + 1e-9

    def test_fit_returns_self(self):
        scaler = FeatureCoupledMinMaxScaler([(0, 2)])
        x = self._make_data()
        assert scaler.fit(x) is scaler

    def test_custom_feature_range(self):
        scaler = FeatureCoupledMinMaxScaler([(0, 4)], feature_range=(-1, 1))
        x = self._make_data()
        scaler.fit(x)
        result = scaler.transform(x.copy())
        assert result.min() >= -1.0 - 1e-9
        assert result.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

class TestCompose:
    def test_applies_transforms_in_order(self):
        # Threshold then Flatten
        compose = Compose(Threshold((0.0, 5.0)), Flatten())
        x = np.array([[-1.0, 3.0, 10.0], [2.0, 0.5, 7.0]])
        result = compose(x)
        assert result.shape == (6,)
        assert result.max() <= 5.0
        assert result.min() >= 0.0

    def test_single_transform(self):
        compose = Compose(Flatten())
        x = np.ones((3, 4))
        assert compose(x).shape == (12,)

    def test_identity_with_no_op(self):
        identity = lambda s: s  # noqa: E731
        compose = Compose(identity)
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(compose(x), x)
