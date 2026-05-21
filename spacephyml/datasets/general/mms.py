"""
Module containing different datasets.
"""

from torch.utils.data import Dataset

import numpy as np

from ...utils import mms, read_cdf_file, xarray_read_file
from ...utils.file_download import missing_files
from ...__init__ import _MMS_DATA_DIR


class ExternalMMSData(Dataset):
    """
    Loading a dataset with labeled MMS data based on dataset file.

    This dataset class looks for datafiles stored in CDF files in another
    location. By default SpacePhyML will look for external MMS data at the
    PySPEDAS data location
    ([PySPEDAS](https://pyspedas.readthedocs.io/en/stable/getting_started.html#local-data-directories).)
    If the PySPEDAS environmental variable's are not set data will be placed
    at `$HOME/spacephyml_data/mms`, following the same directory structure as
    PySPEDAS (and the
    [MMS Science Data Center](https://lasp.colorado.edu/mms/sdc/public/)).
    Data files that are missing when the class is initialised will be
    downloaded.

    The dataset file have to have the following columns:

    - label : The label corresponding to the sample
    - epoch : The CDF epoch for the label
    - file {i} : Specifying the MMS CDF file to read data from,
                 the {i} is a running number.
    - var_name {i} : The variable in the CDF file to read, the {i} is a
                     running number.
    - epoch {i} : The CDF epoch to read data from the {i} is a running number.

    Warning:
        If loading data fail it may be due to the cdf file being corrupt.
        Delete the failing file and retry.

    Examples:
        >>> from spacephyml.datasets.general import ExternalMMSData
        >>> dataset = ExternalMMSData('./mydataset.csv')

    Args:
        dataset_path (string):
            Path to the file containing the dataset.
        rootdir (string):
            The override the default rootdir to for the MMS data storage.
        transform (callable):
            Optional transform to be applied on each sample.
        cache (bool):
            If data should be cached.
        return_epoch (bool):
            If the label epoch should be returned.


    """

    def __init__(self, dataset_path, rootdir=None, transform=None, cache=True,
                 return_epoch=True):

        self.dataset = xarray_read_file(dataset_path)
        self.cache = cache
        self.return_epoch = return_epoch

        if rootdir:
            self.rootdir = rootdir
        else:
            self.rootdir = _MMS_DATA_DIR

        # There are two extra columns and for each variable there are three columns
        self.num_vars = int((len(self.dataset.data_vars) - 2) / 3)

        for i in range(self.num_vars):
            unique_files = np.unique(self.dataset[f'file {i}'].values)
            files = mms.filename_to_filepath(unique_files)

            if not isinstance(files, list):
                files = [files]

            missing = missing_files(files, self.rootdir)

            if missing:
                print(f"{len(missing)} data files are missing, downloading")
                mms.download_cdf_files(self.rootdir, missing)

            if self.cache:
                # Add an index variable for each entry (initialised to -1)
                self.dataset[f'index {i}'] = ('index',
                    np.full(self.dataset.dims['index'], -1, dtype=np.int64))

        self.length = self.dataset.dims['index']

        self.transform = transform

        self.data = {}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns:
            (tuple):
                Will return a list with with all the data varibles in a list
                followed by the label. If 'return_epoch = True' is set the
                label epoch of the data will also be returned.
        """
        if not isinstance(idx, int):
            raise ValueError('Expected idx to be an integer value')

        # Build a row-view as a simple dict for this index
        data_loc = {var: self.dataset[var].values[idx]
                    for var in self.dataset.data_vars}

        sample = []
        for i in range(self.num_vars):
            cdf_filepath = mms.filename_to_filepath(data_loc[f'file {i}'])
            cdf_filepath = f'{self.rootdir}/{cdf_filepath}'

            data = {}
            if self.cache:
                if not data_loc[f'file {i}'] in self.data:
                    self.data[data_loc[f'file {i}']] = \
                        read_cdf_file(cdf_filepath,
                                      [('var', data_loc[f'var_name {i}']),
                                       ('epoch', 'epoch')])

                index = data_loc[f'index {i}']
                if index == -1:
                    new_idx = np.where(
                        self.data[data_loc[f'file {i}']]['epoch'] ==
                        data_loc[f'epoch {i}']
                    )[0]
                    self.dataset[f'index {i}'].values[idx] = new_idx
                    index = self.dataset[f'index {i}'].values[idx]

                sample.append(self.data[data_loc[f'file {i}']]['var'][index])
            else:
                data = read_cdf_file(cdf_filepath,
                                     [('var', data_loc[f'var_name {i}']),
                                      ('epoch', 'epoch')])

                index = np.where(
                        data['epoch'] == data_loc[f'epoch {i}'])
                sample.append(data['var'][index])

        if self.transform:
            sample[0] = self.transform(sample[0])

        sample.append(data_loc['label'])

        if self.return_epoch:
            sample.append(data_loc['epoch'])

        return sample
