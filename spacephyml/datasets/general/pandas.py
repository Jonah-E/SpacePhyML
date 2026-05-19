"""
Module containing different datasets.
"""

from torch.utils.data import Dataset

import numpy as np
from ...utils import pandas_read_file


class PandasDataset(Dataset):
    """
    Loading a dataset contained within a .csv or .feather file.

    The dataset file have to have the following columns:

    - label (optional): The label corresponding to the sample.
    - {data column name}: Columns containing data to extract.

    Examples:
        >>> from spacephyml.datasets import PandasDataset
        >>> dataset = PandasDataset('./mydataset.csv')

    Args:
        dataset_path (string): Path to the file containing the dataset.
        transform (callable): Optional transform to be applied on each
            data sample.
        data_columns (list): Which columns to use for data.
        label_column (string): Which column to use for label.

    Returns:
        Will return a list with with all the data varibles in a list followed
        by the label.
    """

    def __init__(self, dataset_path, transform=None, data_columns=None,
                 label_column=None, return_index=True, size=None):

        self.dataset = pandas_read_file(dataset_path)
        self.label_column = label_column
        self.return_index = return_index

        self.data_columns = data_columns
        if self.data_columns is None:
            self.data_columns =  [c
                for c in self.dataset.columns
                    if c not in [self.label_column,
                                              'Unnamed: 0', 'label']]

        self.length = len(self.dataset.index)
        if size is not None:
            self.length = size

        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise ValueError('Expected idx to be an integer value')

        data = np.array([f for f in self.dataset.iloc[idx][self.data_columns]])

        if self.transform:
            data = self.transform(data)

        sample = [data]
        if self.label_column:
            label = np.array([f for f in self.dataset.iloc[idx][self.label_column]])
            sample.append(label)

        if self.return_index:
            index = self.dataset.index[idx].value
            sample.append(index)

        return sample
