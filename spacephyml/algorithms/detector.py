"""
Module containing OutlierDetector
"""
from sklearn.decomposition import IncrementalPCA

import numpy as np
import pandas as pd
from ..utils.memory import FIFOBuffer

def cal_detections(data,rois, print_ = False):
    rois_detections = 0
    outlier_in_roi = 0
    outlier_in_roi_grouped = 0

    for t1, t2 in rois:
        loc = (data['Time'] >= t1) & (data['Time'] <= t2)
        rois_detections += 1 if data[loc]['Outlier'].any() else 0
        outlier_in_roi += data[loc]['Outlier'].sum()
        outlier_in_roi_grouped += (data[loc]['Outlier'] & ~data[loc]['Outlier'].shift(fill_value=False)).sum()


    total_outliers = data['Outlier'].sum()
    total_rois = len(rois)
    total_outliers_grouped = (data['Outlier'] & ~data['Outlier'].shift(fill_value=False)).sum()

    activity_calib = 100*len(data[data['Flag'] == 2])/len(data)
    activity_outlier = 100*len(data[data['Flag'] == 1])/len(data)
    activity_no = 100*len(data[data['Flag'] == 0])/len(data)

    if print_:
        print(f'Detected ROIs: {rois_detections}/{total_rois}')
        print(f'Outliers in ROIs: {outlier_in_roi}/{total_outliers}')
        print(f'Grouped Outliers in ROIs: {outlier_in_roi_grouped}/{total_outliers_grouped}')
        print(f'Activity (C/O/N): {activity_calib}/{activity_outlier}/{activity_no}')

    return (rois_detections, total_rois), (outlier_in_roi, total_outliers), (outlier_in_roi_grouped,total_outliers_grouped), (activity_calib,activity_outlier,activity_no)

def _flag(x):
    """
    Helper function to create a flag for marking samples as outlier or
    calibration.
    """
    if x['Calibration'] == 1:
        return 2
    #Holdout samples are marked as outliers
    if x['Calibration'] == -1:
        return 1
    if x['Outlier'] == True:
        return 1
    return 0

def _flag_str(x, col):
    """
    Helper function to create a string version of the flag.
    """
    if x[col] == 2:
        return '2: Calibrating'
    if x[col] == 1:
        return '1: Outlier'
    return '0: No activity'

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

    def __init__(self, outlier_limit = None, error_threshold = 3, mean_window = None,
                 n_components = 1, calib_batch_size = 1, store_components = True,
                 data_window = 1, step_size = 1, use_ewma = False, adaptive_calib = True,
                 max_threshold = None, sum_error = False, always_update = False):
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

        self.always_update =   always_update

        self.step_size = step_size
        self.step_cnt = 0

        self.calib_batch_size = calib_batch_size

        self.error_threshold = error_threshold

        self.max_threshold = None
        if max_threshold is not None:
            if isinstance(max_threshold,tuple):
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
        self._history = pd.DataFrame({
                 'Error' : [],
                 'Threshold' : [],
                 'Threshold Alt.' : [],
                 'Time' : [],
                 'Mean' : [],
                 'Std' : [],
                 'Outlier' : [],
                 'Calibration': [],})

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
        """
        Internal function for updating the PCA
        """
        shape = sample.shape
        if len(shape) > 2:
            sample = sample.reshape(shape[0], shape[1]*shape[2])

        self.pca.partial_fit(sample)

        self._calib_buf_ptr = 0
        self._calib_sample_ptr = 0

    def _setup(self, samples, features):
        """
        Internal function for setuing up buffers and unset parameters.
        """
        if self._calib_buf is None:
            self._calib_buf = np.full((self.calib_batch_size,self.data_window, features), np.nan)
        self._features = features

        if self.sample_buf is None:
            self.sample_buf = FIFOBuffer((self.data_window, features))

        if not self.use_ewma:
            if self.mean_window == None:
                self.mean_window = self.calib_batch_size
            self._mean_buffer = np.zeros((self.mean_window,))

        if self.n_components is None:
            self.n_components = self.calib_batch_size
        self.pca = IncrementalPCA(n_components = self.n_components)

        if self.outlier_limit is None:
            self.outlier_limit = int(round(self.calib_batch_size/2,0))

    def _first_calib(self,sample):
        """
        Internal function for creating the first PCA.
        """

        self._setup(*sample.shape)

        if self._first_calib_buf is None:
            if self.n_components < self.calib_batch_size:
                self._first_calib_buf = self._calib_buf
                self._first_calib_buf_size = self.calib_batch_size
            else:
                self._first_calib_buf = np.zeros((self.n_components,self.data_window, sample.shape[1]))
                self._first_calib_buf_size = self.n_components

        calibrate = False
        for i in range(sample.shape[0]):
            self._first_calib_buf[self._calib_buf_ptr,self._calib_sample_ptr,:] = sample[i,:]

            self._calib_sample_ptr += 1
            if self._calib_sample_ptr == self.data_window:
                self._calib_buf_ptr +=1
                self._calib_sample_ptr = 0

                if self._calib_buf_ptr == self._first_calib_buf_size:
                    self._calib_buf_ptr = 0
                    calibrate = True
                    break
            self._history.loc[i, 'Calibration'] = 1

        if calibrate:
            self._calib(self._first_calib_buf)
            self._history.loc[:(self.data_window*self._first_calib_buf_size-1), 'Calibration'] = 1
            #Pandas slicing is end-inclusive
            #self._history['Components'] = [np.zeros(self.pca.components_.shape) for _ in range(self.calib_batch_size) ]

            self._calib_mode = 'Check'

        self.reset_mean()
        return calibrate

    def _mean_n_std(self, x):
        """
        Internal function for calculating the current mean and standard deviation.
        """
        if self.use_ewma:
            return self._mean, np.sqrt(self._var)
        mean, std = (0,0)
        if self._mean_buffer_filled:
            mean = np.mean(self._mean_buffer)
            std = np.std(self._mean_buffer)
        elif self._mean_ptr > 0:
            mean = np.mean(self._mean_buffer[:self._mean_ptr])
            std = np.std(self._mean_buffer[:self._mean_ptr])

        return mean, std

    def _add_to_mean_buffer(self, x):
        """
        Internal function for adding a sample error to the mean buffer.
        """
        if self.use_ewma:
            if self._mean is None:
                self._mean = x
                self._var = 0
            else:
                if isinstance(self.mean_window, tuple):
                    if x <= self._mean:
                        alpha = self.mean_window[0]
                    else:
                        alpha = self.mean_window[1]
                else:
                    alpha = self.mean_window
                diff = x - self._mean
                incr = alpha*diff
                self._mean = self._mean + incr
                self._var = (1-alpha)*(self._var + diff*incr)

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
        """
        Internal function for calculating the current reconstruction error.
        """

        shape = sample.shape
        if len(shape) > 2:
            sample = sample.reshape(shape[0], shape[1]*shape[2])

        p = self.pca.transform(sample)
        if self.sum_error:
            return np.sum(np.abs(sample - self.pca.inverse_transform(p)), axis = 1)
        return np.linalg.norm(sample - self.pca.inverse_transform(p), axis = 1)

    def _outlier_cond(self, error):
        """
        Internal function for checking for outliers.
        """
        nr_errors = error.shape[0]
        outlier = np.zeros((nr_errors))
        err_thres = np.zeros((nr_errors))
        mean = np.zeros((nr_errors))
        std = np.zeros((nr_errors))

        for i, e in enumerate(error):

            # If the mean buffer where reset add the first samples to the mean buffer
            if self._mean_buffer_reset:
                self._add_to_mean_buffer(e)
                continue
            elif not self._mean_buffer_filled:
                # Only mark as outliers if the mean buffer is filled
                self._add_to_mean_buffer(e)
                continue

            mean[i], std[i] = self._mean_n_std(e)

            thres = self.error_threshold * std[i]
            if self.max_threshold is not None:
                if  thres < self.max_threshold[0]:
                    thres = self.max_threshold[0]
                elif  thres > self.max_threshold[1]:
                    thres = self.max_threshold[1]

            err_thres[i] = mean[i] + thres

            # Check for outlier
            if e > err_thres[i]:
                outlier[i] = True
                if self.always_update and not self.use_ewma:
                    self._add_to_mean_buffer(e)
            else:
                #Only add non outliers to mean buffer
                self._add_to_mean_buffer(e)

        # Update history
        start = self._history_index
        end = start + nr_errors -1
        self._history.loc[start:end, 'Threshold'] = err_thres[:]
        self._history.loc[start:end, 'Mean'] = mean
        self._history.loc[start:end, 'Std'] = std

        return outlier

    def _calib_cond(self, sample, outlier):
        """
        Internal function to check if the calibration condition is met.
        """
        if not self.adaptive_calib:
            return False

        ptr = 0
        calibrate = False
        for i,is_outlier in enumerate(outlier):

            ptr += 1

            if not is_outlier and not self.always_update:
                self.outlier_cnt = 0
                self._calib_buf_ptr = 0
                self._calib_sample_ptr = 0
                continue


            self.outlier_cnt += 1
            if self.outlier_cnt >= self.outlier_limit:
                self._calib_buf[self._calib_buf_ptr,self._calib_sample_ptr,:] = sample[i,:]

                self._calib_sample_ptr += 1
                if self._calib_sample_ptr == self.data_window:
                    self._calib_buf_ptr +=1
                    self._calib_sample_ptr = 0

                    if self._calib_buf_ptr == self.calib_batch_size:
                        self._calib_buf_ptr = 0
                        calibrate = True
                        break

        if calibrate:
            if not self.always_update:
                #Mark samples counting towards the outlier limit, if not in always updat mode
                start = self._history_index + (ptr - self.calib_batch_size*self.data_window)
                end = start + self.calib_batch_size*self.data_window - 1
                self._history.loc[start:end, 'Calibration'] = 1

            self._calib_mode = 'Calib'

        #sample = sample[ptr:,...]

        return calibrate

    def get_history(self):
        """
        Get the detector history as a pandas dataframe.

        Returns:
            history (pd.DataFrame):
                The history of the detector.
        """
        #print(len(self._history), len(self._history.index), self._history_index)
        tmp = self._history
        tmp['Outlier'] = tmp['Outlier'].astype(bool)
        tmp['Flag'] = tmp[['Outlier', 'Calibration']].apply(_flag, axis = 1)
        tmp['Flag '] = tmp[['Flag']].apply(lambda x: _flag_str(x,'Flag'), axis = 1)
        return tmp

    def reset_mean(self):
        """
        Reset the mean buffer. Removes all the current samples from the mean buffer, causing
        all samples to be labeled as not outliers untill the buffer is filled again.
        """
        self._mean = None
        self._var  = None
        self._ewma_warm_up_cnt = 0
        self._mean_ptr = 0
        self._mean_buffer_filled = False
        self._mean_buffer_reset = True
        self.sample_buf.reset()

    def reset_history(self):
        """
        Reset the detector history.
        """
        self._history_index = 0
        self._history = pd.DataFrame({
                 'Error' : [],
                 'Threshold' : [],
                 'Time' : [],
                 'Outlier' : [],
                 'Calibration': [],})

    def __call__(self, sample, time = None):
        """
        Check samples for outliers. It is recommended not to call with more samples than can fit in the calibration buffer.

        Args:
            sample (ndarray of shape (n_samples, n_features)):
                The samples to check for outliers. It is recomended to have
                n_samples <= calib_batch_size

            time (optional, ndarray of size n_samples):
                The time values correspondig to the samples, only added to history.

        """

        nr_samples = sample.shape[0]
        tmp = pd.DataFrame({
                     'Error' : np.zeros((nr_samples)),
                     'Threshold' : np.zeros((nr_samples)),
                     'Threshold Alt.' : np.zeros((nr_samples)),
                     'Mean' : np.zeros((nr_samples)),
                     'Std' : np.zeros((nr_samples)),
                     'Outlier' : np.zeros((nr_samples)),
                     'Calibration': np.zeros((nr_samples))
            })
        if time:
            tmp['Time'] = time[:]
        if self.store_components and self._calib_mode != 'Init':
            tmp['Components'] = [self.pca.components_ for _ in range(nr_samples)]

        self._history = pd.concat([self._history, tmp], ignore_index = True)

        check = True
        if self._calib_mode == 'Init':
            check = self._first_calib(sample)

        error = np.zeros((nr_samples))
        outlier = np.zeros((nr_samples))

        if check:
            for i in range(sample.shape[0]):
                self.sample_buf.append(sample[i,...])
                self.step_cnt += 1
                if self.sample_buf.full and self.step_cnt >= self.step_size:
                    self.step_cnt = 0
                    #print(self.sample_buf.get().reshape(1,-1))
                    error[i:(i+1)] = self._error(self.sample_buf.get().reshape(1,-1))

                    #Check the outlier condition
                    outlier[i:(i+1)] = self._outlier_cond(error[i:(i+1)])

                    calibrate = self._calib_cond(sample[i,:].reshape(1,-1),
                                                 outlier[i:(i+1)])

                    if self._calib_mode == 'Calib':
                        self._calib(self._calib_buf)
                        self._calib_mode = 'Check'

                #print(self._history_index, i, error[i])
                self._history.loc[self._history_index, 'Error'] = error[i]
                self._history.loc[self._history_index, 'Outlier'] = outlier[i]
                self._history_index += 1
        else:
            for i in range(sample.shape[0]):
                self._history.loc[self._history_index, 'Error'] = 0
                self._history.loc[self._history_index, 'Outlier'] = 0
                self._history_index += 1


        return outlier

