"""
Module containing OutlierDetector
"""
from sklearn.decomposition import IncrementalPCA

import numpy as np
import xarray as xr
from ..utils.memory import FIFOBuffer

def cal_detections(data, rois, print_=False):
    time = data['Time'].values
    outlier = data['Outlier'].values.astype(bool)
    flag = data['Flag'].values

    rois_detections = 0
    outlier_in_roi = 0
    outlier_in_roi_grouped = 0

    for t1, t2 in rois:
        loc = (time >= t1) & (time <= t2)
        rois_detections += 1 if outlier[loc].any() else 0
        outlier_in_roi += outlier[loc].sum()
        shifted = np.concatenate([[False], outlier[loc][:-1]])
        outlier_in_roi_grouped += (outlier[loc] & ~shifted).sum()

    total_outliers = outlier.sum()
    total_rois = len(rois)
    shifted_all = np.concatenate([[False], outlier[:-1]])
    total_outliers_grouped = (outlier & ~shifted_all).sum()

    activity_calib = 100 * (flag == 2).sum() / len(flag)
    activity_outlier = 100 * (flag == 1).sum() / len(flag)
    activity_no = 100 * (flag == 0).sum() / len(flag)

    if print_:
        print(f'Detected ROIs: {rois_detections}/{total_rois}')
        print(f'Outliers in ROIs: {outlier_in_roi}/{total_outliers}')
        print(f'Grouped Outliers in ROIs: {outlier_in_roi_grouped}/{total_outliers_grouped}')
        print(f'Activity (C/O/N): {activity_calib}/{activity_outlier}/{activity_no}')

    return (rois_detections, total_rois), (outlier_in_roi, total_outliers), \
           (outlier_in_roi_grouped, total_outliers_grouped), \
           (activity_calib, activity_outlier, activity_no)


def _flag(outlier, calibration):
    """Create flag values from outlier and calibration arrays."""
    flags = np.zeros(len(outlier), dtype=int)
    flags[calibration == 2] = 2
    flags[calibration == -1] = 1
    flags[outlier.astype(bool) & (calibration != 2)] = 1
    return flags


def _flag_str(flag_val):
    """Map an integer flag to a human-readable string."""
    if flag_val == 2:
        return '2: Calibrating'
    if flag_val == 1:
        return '1: Outlier'
    return '0: No activity'


def _empty_history():
    """Return an empty xarray Dataset with the history schema."""
    return xr.Dataset({
        'Error':        ('sample', np.array([], dtype=float)),
        'Threshold':    ('sample', np.array([], dtype=float)),
        'Threshold Alt.': ('sample', np.array([], dtype=float)),
        'Time':         ('sample', np.array([], dtype=float)),
        'Mean':         ('sample', np.array([], dtype=float)),
        'Std':          ('sample', np.array([], dtype=float)),
        'Outlier':      ('sample', np.array([], dtype=float)),
        'Calibration':  ('sample', np.array([], dtype=float)),
    })


class OutlierDetector():
    """
    Detector for finding outliers in streaming data.

    Examples:
        >>> from spacephyml.detector import OutlierDetector
        >>> detector = OutlierDetector()
        >>> dataset = MMS1IonDistLabeled('SCDec017')
        >>> iterator = enumerate(DataLoader(dataset, batch_size=batch_size))
        >>> for i, (x, ) in iterator:
        >>>     x = x.numpy()
        >>>     _ = detector(x)
        >>> history = detector.get_history()
    """

    def __init__(self, outlier_limit=None, error_threshold=3, mean_window=None,
                 n_components=1, calib_batch_size=1, store_components=True,
                 data_window=1, step_size=1, use_ewma=False, adaptive_calib=True,
                 max_threshold=None, sum_error=False, always_update=False):
        """

        Args:
            data_window (integer):
                The window size for feature extraction.
            calib_batch_size (integer):
                The number of samples required to initiate a calibration.
            mean_window (integer):
                The number of reconstruction error from samples to use for calculating the
                current mean and standard deviation, used in setting the current threshold
                for outliers.
            n_components (integer):
                The number of components to calculate in the PCA.
            error_threshold (float):
                The threshold for labaling a sample as an outlier.
            outlier_limit (integer):
                The number of samples allowed to be outliers before they start to be counted
                towards a new calibration.
            store_components (bool, default True):
                Store the PCA components in the history data.
            step_size (integer, default 1):
                The number of samples to wait before checking for outliers and calibration.
            use_ewma (bool, default False):
                Use an exponentially weighted moving average for calculating the mean and
                standard deviation.
            adaptive_calib (bool, default True):
                Use an adaptive calibration, where the calibration is triggered by outliers
                instead of a fixed number of samples.
            max_threshold (float or tuple, default None):
                The maximum threshold for labaling a sample as an outlier. If a tuple is
                given, the first value is the minimum threshold and the second value is the
                maximum threshold.
            sum_error (bool, default False):
                Use the sum of the absolute error instead of the L2 norm for calculating the
                reconstruction error.
            always_update (bool, default False):
                Always update the mean and standard deviation with the current sample, even
                if it is an outlier. If False, only update with non-outliers.
        """

        self.n_components = n_components
        self.data_window = data_window
        self.sample_buf = None

        self.always_update = always_update

        self.step_size = step_size
        self.step_cnt = 0

        self.calib_batch_size = calib_batch_size

        self.error_threshold = error_threshold

        self.max_threshold = None
        if max_threshold is not None:
            if isinstance(max_threshold, tuple):
                self.max_threshold = max_threshold
            else:
                self.max_threshold = (0, max_threshold)
        self.outlier_limit = outlier_limit
        self.outlier_cnt = 0
        self._mean_buffer_filled = False

        self.mean_window = mean_window
        self.adaptive_calib = adaptive_calib
        self._calib_mode = 'Init'
        self._calib_buf = None
        self._calib_sample_ptr = 0
        self._calib_buf_ptr = 0
        self._first_calib_buf = None
        self.sum_error = sum_error

        self.use_ewma = use_ewma
        self._ewma_warm_up_cnt = 0
        self._mean = None
        self._mean_ptr = 0
        self._mean_buffer_reset = False

        self.store_components = store_components
        self._history_index = 0

        # Internal numpy arrays – converted to xarray only on get_history()
        self._hist_error = []
        self._hist_threshold = []
        self._hist_threshold_alt = []
        self._hist_time = []
        self._hist_mean = []
        self._hist_std = []
        self._hist_outlier = []
        self._hist_calibration = []

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _append_history_rows(self, n):
        """Extend all history lists by n zero-initialised rows."""
        self._hist_error.extend([0.0] * n)
        self._hist_threshold.extend([0.0] * n)
        self._hist_threshold_alt.extend([0.0] * n)
        self._hist_time.extend([0.0] * n)
        self._hist_mean.extend([0.0] * n)
        self._hist_std.extend([0.0] * n)
        self._hist_outlier.extend([0.0] * n)
        self._hist_calibration.extend([0.0] * n)

    def manual_calibration(self, samples):
        if self._calib_mode == 'Init':
            shape = samples.shape
            if len(shape) > 2:
                shape = (shape[0], shape[2])
            self._setup(*shape)

        self._calib(samples)
        self._calib_mode = 'Check'

        return self

    def _calib(self, sample):
        """Internal function for updating the PCA."""
        shape = sample.shape
        if len(shape) > 2:
            sample = sample.reshape(shape[0], shape[1] * shape[2])

        self.pca.partial_fit(sample)

        self._calib_buf_ptr = 0
        self._calib_sample_ptr = 0

    def _setup(self, samples, features):
        """Internal function for setting up buffers and unset parameters."""
        if self._calib_buf is None:
            self._calib_buf = np.full((self.calib_batch_size, self.data_window, features), np.nan)
        self._features = features

        if self.sample_buf is None:
            self.sample_buf = FIFOBuffer((self.data_window, features))

        if not self.use_ewma:
            if self.mean_window is None:
                self.mean_window = self.calib_batch_size
            self._mean_buffer = np.zeros((self.mean_window,))

        if self.n_components is None:
            self.n_components = self.calib_batch_size
        self.pca = IncrementalPCA(n_components=self.n_components)

        if self.outlier_limit is None:
            self.outlier_limit = int(round(self.calib_batch_size / 2, 0))

    def _first_calib(self, sample):
        """Internal function for creating the first PCA."""

        self._setup(*sample.shape)

        if self._first_calib_buf is None:
            if self.n_components < self.calib_batch_size:
                self._first_calib_buf = self._calib_buf
                self._first_calib_buf_size = self.calib_batch_size
            else:
                self._first_calib_buf = np.zeros((self.n_components, self.data_window, sample.shape[1]))
                self._first_calib_buf_size = self.n_components

        calibrate = False
        for i in range(sample.shape[0]):
            self._first_calib_buf[self._calib_buf_ptr, self._calib_sample_ptr, :] = sample[i, :]

            self._calib_sample_ptr += 1
            if self._calib_sample_ptr == self.data_window:
                self._calib_buf_ptr += 1
                self._calib_sample_ptr = 0

                if self._calib_buf_ptr == self._first_calib_buf_size:
                    self._calib_buf_ptr = 0
                    calibrate = True
                    break
            self._hist_calibration[i] = 1

        if calibrate:
            self._calib(self._first_calib_buf)
            end = self.data_window * self._first_calib_buf_size
            for k in range(end):
                if k < len(self._hist_calibration):
                    self._hist_calibration[k] = 1

            self._calib_mode = 'Check'

        self.reset_mean()
        return calibrate

    def _mean_n_std(self, x):
        """Internal function for calculating the current mean and std."""
        if self.use_ewma:
            return self._mean, np.sqrt(self._var)
        mean, std = (0, 0)
        if self._mean_buffer_filled:
            mean = np.mean(self._mean_buffer)
            std = np.std(self._mean_buffer)
        elif self._mean_ptr > 0:
            mean = np.mean(self._mean_buffer[:self._mean_ptr])
            std = np.std(self._mean_buffer[:self._mean_ptr])

        return mean, std

    def _add_to_mean_buffer(self, x):
        """Internal function for adding a sample error to the mean buffer."""
        if self.use_ewma:
            if self._mean is None:
                self._mean = x
                self._var = 0
            else:
                if isinstance(self.mean_window, tuple):
                    alpha = self.mean_window[0] if x <= self._mean else self.mean_window[1]
                else:
                    alpha = self.mean_window
                diff = x - self._mean
                incr = alpha * diff
                self._mean = self._mean + incr
                self._var = (1 - alpha) * (self._var + diff * incr)

            if not self._mean_buffer_filled:
                self._ewma_warm_up_cnt += 1
                if self._ewma_warm_up_cnt == (self.data_window * self.calib_batch_size):
                    self._mean_buffer_filled = True
        else:
            self._mean_buffer[self._mean_ptr] = x
            self._mean_ptr += 1

            if self._mean_ptr >= self.mean_window:
                self._mean_buffer_filled = True
                self._mean_ptr = 0

        self._mean_buffer_reset = False

    def _error(self, sample):
        """Internal function for calculating the current reconstruction error."""
        shape = sample.shape
        if len(shape) > 2:
            sample = sample.reshape(shape[0], shape[1] * shape[2])

        p = self.pca.transform(sample)
        if self.sum_error:
            return np.sum(np.abs(sample - self.pca.inverse_transform(p)), axis=1)
        return np.linalg.norm(sample - self.pca.inverse_transform(p), axis=1)

    def _outlier_cond(self, error):
        """Internal function for checking for outliers."""
        nr_errors = error.shape[0]
        outlier = np.zeros((nr_errors))
        err_thres = np.zeros((nr_errors))
        mean = np.zeros((nr_errors))
        std = np.zeros((nr_errors))

        for i, e in enumerate(error):
            if self._mean_buffer_reset:
                self._add_to_mean_buffer(e)
                continue
            elif not self._mean_buffer_filled:
                self._add_to_mean_buffer(e)
                continue

            mean[i], std[i] = self._mean_n_std(e)

            thres = self.error_threshold * std[i]
            if self.max_threshold is not None:
                if thres < self.max_threshold[0]:
                    thres = self.max_threshold[0]
                elif thres > self.max_threshold[1]:
                    thres = self.max_threshold[1]

            err_thres[i] = mean[i] + thres

            if e > err_thres[i]:
                outlier[i] = True
                if self.always_update and not self.use_ewma:
                    self._add_to_mean_buffer(e)
            else:
                self._add_to_mean_buffer(e)

        start = self._history_index
        end = start + nr_errors
        for k, idx in enumerate(range(start, end)):
            if idx < len(self._hist_threshold):
                self._hist_threshold[idx] = err_thres[k]
                self._hist_mean[idx] = mean[k]
                self._hist_std[idx] = std[k]

        return outlier

    def _calib_cond(self, sample, outlier):
        """Internal function to check if the calibration condition is met."""
        if not self.adaptive_calib:
            return False

        ptr = 0
        calibrate = False
        for i, is_outlier in enumerate(outlier):
            ptr += 1

            if not is_outlier and not self.always_update:
                self.outlier_cnt = 0
                self._calib_buf_ptr = 0
                self._calib_sample_ptr = 0
                continue

            self.outlier_cnt += 1
            if self.outlier_cnt >= self.outlier_limit:
                self._calib_buf[self._calib_buf_ptr, self._calib_sample_ptr, :] = sample[i, :]

                self._calib_sample_ptr += 1
                if self._calib_sample_ptr == self.data_window:
                    self._calib_buf_ptr += 1
                    self._calib_sample_ptr = 0

                    if self._calib_buf_ptr == self.calib_batch_size:
                        self._calib_buf_ptr = 0
                        calibrate = True
                        break

        if calibrate:
            if not self.always_update:
                start = self._history_index + (ptr - self.calib_batch_size * self.data_window)
                end = start + self.calib_batch_size * self.data_window
                for idx in range(start, end):
                    if idx < len(self._hist_calibration):
                        self._hist_calibration[idx] = 1

            self._calib_mode = 'Calib'

        return calibrate

    def get_history(self):
        """
        Get the detector history as an xarray Dataset.

        Returns:
            history (xr.Dataset):
                The history of the detector.
        """
        n = len(self._hist_error)
        outlier_arr = np.array(self._hist_outlier, dtype=bool)
        calib_arr = np.array(self._hist_calibration, dtype=float)

        flags = _flag(outlier_arr, calib_arr)
        flag_strs = np.array([_flag_str(f) for f in flags])

        return xr.Dataset({
            'Error':        ('sample', np.array(self._hist_error)),
            'Threshold':    ('sample', np.array(self._hist_threshold)),
            'Threshold Alt.': ('sample', np.array(self._hist_threshold_alt)),
            'Time':         ('sample', np.array(self._hist_time)),
            'Mean':         ('sample', np.array(self._hist_mean)),
            'Std':          ('sample', np.array(self._hist_std)),
            'Outlier':      ('sample', outlier_arr),
            'Calibration':  ('sample', calib_arr),
            'Flag':         ('sample', flags),
            'Flag ':        ('sample', flag_strs),
        })

    def reset_mean(self):
        """
        Reset the mean buffer. Removes all the current samples from the mean buffer, causing
        all samples to be labeled as not outliers untill the buffer is filled again.
        """
        self._mean = None
        self._var = None
        self._ewma_warm_up_cnt = 0
        self._mean_ptr = 0
        self._mean_buffer_filled = False
        self._mean_buffer_reset = True
        self.sample_buf.reset()

    def reset_history(self):
        """Reset the detector history."""
        self._history_index = 0
        self._hist_error = []
        self._hist_threshold = []
        self._hist_threshold_alt = []
        self._hist_time = []
        self._hist_mean = []
        self._hist_std = []
        self._hist_outlier = []
        self._hist_calibration = []

    def __call__(self, sample, time=None):
        """
        Check samples for outliers.

        Args:
            sample (ndarray of shape (n_samples, n_features)):
                The samples to check for outliers.
            time (optional, ndarray of size n_samples):
                The time values correspondig to the samples, only added to history.
        """

        nr_samples = sample.shape[0]

        # Grow the history lists
        self._append_history_rows(nr_samples)

        if time:
            for k, idx in enumerate(range(self._history_index,
                                          self._history_index + nr_samples)):
                self._hist_time[idx] = time[k]

        check = True
        if self._calib_mode == 'Init':
            check = self._first_calib(sample)

        error = np.zeros((nr_samples))
        outlier = np.zeros((nr_samples))

        if check:
            for i in range(sample.shape[0]):
                self.sample_buf.append(sample[i, ...])
                self.step_cnt += 1
                if self.sample_buf.full and self.step_cnt >= self.step_size:
                    self.step_cnt = 0
                    error[i:(i + 1)] = self._error(self.sample_buf.get().reshape(1, -1))
                    outlier[i:(i + 1)] = self._outlier_cond(error[i:(i + 1)])

                    self._calib_cond(sample[i, :].reshape(1, -1),
                                     outlier[i:(i + 1)])

                    if self._calib_mode == 'Calib':
                        self._calib(self._calib_buf)
                        self._calib_mode = 'Check'

                self._hist_error[self._history_index] = error[i]
                self._hist_outlier[self._history_index] = outlier[i]
                self._history_index += 1
        else:
            for i in range(sample.shape[0]):
                self._hist_error[self._history_index] = 0
                self._hist_outlier[self._history_index] = 0
                self._history_index += 1

        return outlier
