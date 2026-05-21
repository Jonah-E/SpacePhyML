from torch.utils.data import Dataset
import pytplot
import xarray as xr
import numpy as np
import pandas as pd


class SpedasWrapper(Dataset):
    """
    Wrapper for loading varibles from pyspedas (tplot) into a pytorch dataset.
    The loading is a beast effort and might not work for some varibles and
    missions. The varibles are assumed to already be loaded into tplot in the
    correct timerange.

    Args:
        tplot_vars (list of strings) :
            The varibles to load.
        dropna (bool) :
            Drop times where one of the varibles have value NAN. May result in
            removal of all data if tplot_vars are sampled at different times
            and resample is not set.
        resample (string) :
            Time interval for resampling, follows the pandas style for
            resample. No resampling is done if set to None.
        transform (callable)
            Transform to apply to each sample.

    """
    def __init__(self, tplot_vars, dropna=True, resample=None, transform=None):
        # Build up an xarray Dataset by merging variables one at a time.
        datasets = []
        self.features = []
        feature_cnt = 0

        for var in tplot_vars:
            pre = None
            if len(var) > 1:
                var, pre = var
            data = pytplot.get_data(var)
            if len(data) == 2:
                time, values = data
            elif len(data) == 3:
                time, values, _ = data

            names = pytplot.get_data(var, metadata=True)['CDF']['LABELS']
            if names is None:
                names = ['']
            else:
                names = [f'_{n}' for n in names]

            if values.ndim > 2:
                raise ValueError(f'Cannot handle {values.ndim} dimentions!')
            elif values.ndim == 2:
                if values.shape[1] != len(names):
                    names = [f'_{i:02}' for i in range(values.shape[1])]

            if pre is None:
                names = [f'{var}{n}' for n in names]
            else:
                names = [f'{pre}{n}' for n in names]

            self.features.append((feature_cnt, feature_cnt + len(names)))
            feature_cnt += len(names)

            time_index = pd.to_datetime(time, unit='s')
            if values.ndim > 1:
                data_vars = {k: ('time', values[:, i])
                             for i, k in enumerate(names)}
            else:
                data_vars = {k: ('time', values[:]) for k in names}

            tmp = xr.Dataset(data_vars, coords={'time': time_index})
            datasets.append(tmp)

        # Merge all variables; outer join keeps all timestamps
        if datasets:
            self.dataset = xr.merge(datasets, join='outer')
        else:
            self.dataset = xr.Dataset()

        if resample:
            # xarray uses timedelta-compatible offset strings
            self.dataset = (
                self.dataset
                .resample(time=resample)
                .mean()
            )

        if dropna:
            # Drop variables (data_vars) that are all-NaN, then drop time
            # steps with any NaN across remaining variables.
            self.dataset = self.dataset.dropna(dim='time', how='any')

        self.transform = transform
        self.length = self.dataset.dims['time']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise ValueError('Expected idx to be an integer value')
        data = np.array([
            float(self.dataset[v].values[idx])
            for v in self.dataset.data_vars
        ])

        if self.transform:
            data = self.transform(data)

        return (data,)

    def get_dataset(self):
        """
        Get the full xarray Dataset.

        Returns:
            dataset : xr.Dataset
                The full loaded data.
        """
        return self.dataset
