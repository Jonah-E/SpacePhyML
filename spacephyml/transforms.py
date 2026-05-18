"""
Different useful transforms.
"""
import numpy as np
from torch import unsqueeze, from_numpy

class FeatureCoupledMinMaxScaler():
    """
    Scaling the data according to Min and Max values for that feature.

    Examples:
        >>> from detector import FeatureCoupledMinMaxScaler
        >>> features = [(0,3),(3,6)]
        >>> scaler = FeatureCoupledMinMaxScaler(features)
        >>> scaler.fit(x)
        >>> x = scaler.transform(x)
    """
    def __init__(self, features, feature_range = (0,1)):
        """

        Args:
            fetaures (list[tuple]):
                List of tuples with the ranges, [start, end), where each feature type can be found.
            feature_range (tuple):
                The range to scale between, [min, max].
        """
        self.features = features
        self._max = [0 for _ in features]
        self._min = [0 for _ in features]
        self._scale = [0 for _ in features]
        self.feature_range = feature_range


    def fit(self, x):
        """
        Fit scaler to data.

        Args:
            x (ndarray):
                The data to fit, shape (n_samples, n_features)

        Returns:
            self : object
                The fitted scaler.
        """
        for i, (i1, i2) in enumerate(self.features):
            self._min[i] = np.min(x[:,i1:i2])
            self._max[i] = np.max(x[:,i1:i2])
            data_range = self._max[i] - self._min[i]
            self._scale[i] = (self.feature_range[1] - self.feature_range[0]) / data_range
            self._min[i] = self.feature_range[0] - self._min[i] * self._scale[i]

        return self

    def transform(self, x):
        """
        Scale data according to scaler.

        Args:
            x (ndarray):
                The data to scale, shape (n_samples, n_features)

        Returns:
            (ndarray):
                The scaled version of x,  of shape (n_samples, n_features)
        """
        for i, (i1, i2) in enumerate(self.features):
            x[:,i1:i2] =  (x[:,i1:i2] * self._scale[i]) + self._min[i]
        return x

class Compose():
    """
    Compose multiple transforms into one.

    Examples:
        >>> import spacephyml.transforms as tf
        >>> transforms = tf.Compose(tf.Threshold(0,1), tf.Flatten())

    Args:
        transforms (callables) : All the transforms to compose.
    """
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for trans in self.transforms:
            sample = trans(sample)

        return sample

class IonDist_Transform(Compose):
    """
    The default transform used in MMS1IonDistLabeled.

    Args:
        norm (tuple): The values to threshold and calculate LogNorm with.

    """
    def __init__(self, norm = (-28, -17)):
        super().__init__(Threshold((np.power(10.0, norm[0]),
                                    np.power(10.0, norm[1]))),
                         LogNorm(norm),
                         Roll(),
                         lambda x: unsqueeze(from_numpy(x), 0))

class ZScoreNorm():
    """
    Calculate the Z-Score norm using specified mean and std.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        return ((sample - self.mean)/self.std)


class Threshold():
    """
    Threshold the sample.
    """
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def __call__(self, sample):
        x = sample

        threshold_low_index = np.where(x < self.thresholds[0])
        x[threshold_low_index] = self.thresholds[0]

        threshold_high_index = np.where(x > self.thresholds[1])
        x[threshold_high_index] = self.thresholds[1]

        return x


class LogNorm():
    """
    Log Normalize the data between a given range
    """
    def __init__(self, normalization=None):
        self.normalization = normalization

    def __call__(self, sample):
        x = np.log10(sample)

        if self.normalization is None:
            x -= x.min()
            x /= x.max()
        else:
            x -= self.normalization[0]
            x /= (self.normalization[1] - self.normalization[0])

        return x


class Flatten():
    """
    Filter for flattening the data
    """
    def __call__(self, sample):
        return sample.reshape(-1)

class Log10():
    """
    Calculate the log10 of all non zero values in the sample.
    """
    def __call__(self, sample):
        non_zero_indexes = np.where(sample != 0)

        # Talk log10 of all non_zero values
        sample[non_zero_indexes] = np.log10(sample[non_zero_indexes])
        return sample


class Roll():
    """
    Perform a roll along a axis.
    """

    def __init__(self, shift=16, axis=-3):
        self.shift = shift
        self.axis = axis

    def __call__(self, sample):
        x = sample

        # Roll along Phi
        x = np.roll(x, self.shift, axis=self.axis)

        return x


class MoveAxis():
    """
    Move around the axis in the sample.
    """
    def __init__(self, src=-1, dst=-3):
        self.src = src
        self.dst = dst

    def __call__(self, sample):
        return np.moveaxis(sample, self.src, self.dst)


class Sum():
    """
    Calculate the sum along specified axis.
    """
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, sample):
        x = sample

        x = np.sum(x, axis=self.axis)

        return x


class Mean():
    """
    Calculate the mean along specified axis.
    """
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, sample):
        x = sample

        x = np.mean(x, axis=self.axis)

        return x
