"""
Unit tests for spacephyml/utils/memory.py – FIFOBuffer
"""
import numpy as np
import pytest
from spacephyml.utils.memory import FIFOBuffer


class TestFIFOBufferInit:
    def test_initial_length_is_zero(self):
        buf = FIFOBuffer(5)
        assert len(buf) == 0

    def test_initial_not_full(self):
        buf = FIFOBuffer(4)
        assert not buf.full

    def test_tuple_shape(self):
        buf = FIFOBuffer((4, 3))
        assert buf.length == 4
        assert buf.data.shape == (4, 3)


class TestFIFOBufferAppend:
    def test_append_increments_length(self):
        buf = FIFOBuffer((5, 1))
        buf.append(np.array([1.0]))
        assert len(buf) == 1

    def test_becomes_full_at_capacity(self):
        buf = FIFOBuffer((3, 1))
        for i in range(3):
            buf.append(np.array([float(i)]))
        assert buf.full
        assert len(buf) == 3

    def test_len_stays_at_capacity_when_overfull(self):
        buf = FIFOBuffer((3, 1))
        for i in range(6):
            buf.append(np.array([float(i)]))
        assert len(buf) == 3


class TestFIFOBufferGet:
    def test_get_returns_inserted_values_before_full(self):
        buf = FIFOBuffer((5, 1))
        for i in range(3):
            buf.append(np.array([float(i)]))
        result = buf.get()
        np.testing.assert_array_equal(result[:, 0], [0.0, 1.0, 2.0])

    def test_get_returns_fifo_order_when_full(self):
        """After wrapping, oldest value should be first."""
        buf = FIFOBuffer((3, 1))
        for i in range(4):           # write 0,1,2,3 → wraps; oldest = 1
            buf.append(np.array([float(i)]))
        result = buf.get()
        np.testing.assert_array_equal(result[:, 0], [1.0, 2.0, 3.0])

    def test_2d_buffer_preserves_features(self):
        buf = FIFOBuffer((3, 2))
        buf.append(np.array([1.0, 2.0]))
        buf.append(np.array([3.0, 4.0]))
        result = buf.get()
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result[0], [1.0, 2.0])
        np.testing.assert_array_equal(result[1], [3.0, 4.0])


class TestFIFOBufferReset:
    def test_reset_clears_full_flag(self):
        buf = FIFOBuffer((3, 1))
        for i in range(3):
            buf.append(np.array([float(i)]))
        assert buf.full
        buf.reset()
        assert not buf.full

    def test_reset_sets_length_to_zero(self):
        buf = FIFOBuffer((3, 1))
        for i in range(3):
            buf.append(np.array([float(i)]))
        buf.reset()
        assert len(buf) == 0


class TestFIFOBufferIndexing:
    def test_integer_index(self):
        buf = FIFOBuffer((4, 2))
        for i in range(4):
            buf.append(np.array([float(i), float(i)]))
        np.testing.assert_array_equal(buf[0], [0.0, 0.0])

    def test_slice(self):
        buf = FIFOBuffer((4, 1))
        for i in range(4):
            buf.append(np.array([float(i)]))
        result = buf[:2]
        assert result.shape[0] == 2


class TestFIFOBufferRepr:
    def test_repr_contains_class_name(self):
        buf = FIFOBuffer((2, 1))
        assert "RingBuffer" in repr(buf)
