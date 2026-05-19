"""
Script for creating dataset based on exisiting labels.
"""
import tempfile
from os import path, makedirs, remove
import datetime as dt
from tqdm import tqdm
from cdflib import cdfepoch

import pandas as pd
import numpy as np

from ..__init__ import _MMS_DATA_DIR
from ..utils import read_cdf_file
from ..utils.file_download import download_file_with_status, missing_files
from ..utils.config import load_var_to_file_info
from ..utils import mms

_LABELS_URL_BASE = 'https://bitbucket.org/volshevsky/mmslearning/' + \
                   'raw/7b93d08b585842454c309668870ecd25ea16e3e0/labels_human/'
_LABELS_FILENAME_BASE = 'labels_fpi_fast_dis_dist_'


def _download_label_file(outputpath, filedate):
    """
    Download Olshevsky label files.
    """
    makedirs(outputpath, exist_ok=True)
    file = _LABELS_FILENAME_BASE + filedate + '.cdf'
    download_file_with_status(_LABELS_URL_BASE + file, outputpath + file)
    return outputpath + file


_OLSHEVSKY_REF = """
Olshevsky, V., et al. (2021). Automated classification of plasma
regions using 3D particle energy distributions.
Journal of Geophysical Research: Space Physics,
https://doi.org/10.1029/2021JA029620
"""

_OLSHEVSKY_LABELS = {
    -1 : 'Undefined',
    0 : 'Solar Wind',
    1 : 'Ion foreshock',
    2 : 'Magnetosheath',
    3 : 'Magnetosphere',
}

def _get_var_info(trange, var, var_to_file_info, epochs=None):
    # The MMS Data API takes the end date as exclusive
    trange = [trange[0].strftime("%Y-%m-%d"),
              (trange[1] + dt.timedelta(days=1)).strftime("%Y-%m-%d")]

    # Check which datafiles are relevant
    files = mms.get_file_list(trange[0], trange[1],
                              **var_to_file_info['info'])
    files = [f['file_name'] for f in files]
    filespaths = mms.filename_to_filepath(files)

    # Download missing
    missing = missing_files(filespaths, _MMS_DATA_DIR)
    if missing:
        print(f"{len(missing)} data files are missing, downloading")
        mms.download_cdf_files(_MMS_DATA_DIR, missing)

    # Load all the epochs
    file_epochs = []
    file_names = []
    for filename in files:
        filepath = mms.filename_to_filepath(filename)
        cdf_file = read_cdf_file(_MMS_DATA_DIR + filepath)
        tmp = cdf_file.varget('Epoch')
        file_epochs.extend(tmp)
        file_names.extend([filename for _ in tmp])
    file_epochs = np.array(file_epochs)

    if epochs is None:
        # If we don't have any epochs to sort on, return everything
        return file_names, file_epochs

    files_add = []
    epochs_add = np.zeros(len(epochs)).astype(np.int64)
    # For each labeled epoch find the closes from the file
    for j, epoch_labeled in enumerate(epochs):
        index = np.abs(file_epochs - epoch_labeled).argmin()
        epochs_add[j] = file_epochs[index]
        files_add.append(file_names[index])

        # If the time difference is larger than between the
        # labeled epochs, set 0
        time_diff = np.abs(cdfepoch.unixtime(epoch_labeled)
                           - cdfepoch.unixtime(epochs_add[j]))
        if time_diff > 4.5:
            epochs_add[j] = 0

    return files_add, epochs_add


def _get_olshevsky_label_list(trange=None, var_list=None, var_to_file_info=None):
    """
    Get a pandoc DataFrame containing all the Olshevsky labels from within the
    given time range.
    """

    print('Generating a mms dataset based on labels from ')
    print(f'\t{_OLSHEVSKY_REF}')

    droped_rows = 0

    if trange is None:
        trange = [dt.datetime(2017, 11, 1), dt.datetime(2017, 12, 31)]
    elif (trange[0] < dt.datetime(2017, 11, 1) or
          dt.datetime(2017, 12, 31) < trange[1]):
        raise ValueError('Invalid time range: range have to be in the range ' +
                         '2017-11-01 to 2017-12-31, (inclusive)')

    # Download the labelfiles from Olshevsky
    print('Downloading Olshevsky label files.')
    label_files = [_download_label_file(tempfile.gettempdir() +
                   '/mms_labels/', d) for d in ['201711', '201712']]

    data = {'label': [], 'epoch': [], 'date': []}
    for file in label_files:
        print(f'Processing file {file}')
        cdf_file = read_cdf_file(file)
        labels_vars = cdf_file.cdf_info().zVariables
        labels_vars = zip([lb for lb in labels_vars if 'label_' in lb],
                          [ep for ep in labels_vars if 'epoch_' in ep])

        for lb, ep in labels_vars:
            # Double check that the labels and epochs are refering to the same
            # cdf file.
            if lb[-14:] != ep[-14:]:
                raise ValueError('The label and epoch vals are not equal')

            label = cdf_file.varget(lb)
            epoch = cdf_file.varget(ep)
            date = [dt.datetime.strptime(lb.split('_')[6][:8], '%Y%m%d')
                    for _ in range(len(label))]

            data['date'].extend(date)
            data['label'].extend(label)
            data['epoch'].extend(epoch)

    data = pd.DataFrame(data)
    data['Time'] = pd.to_datetime(cdfepoch.unixtime(data['epoch']), unit='s')
    data = data.loc[(trange[0] <= data['Time']) &
                    (data['Time'] < trange[1])]

    for i, var in enumerate(var_list):
        print(f'Processing varible: {var}')
        if var not in var_to_file_info:
            raise ValueError(f'Invalid var requested: {var}')

        files_add, epochs_add = _get_var_info(trange, var, var_to_file_info[var],
                                              data['epoch'])

        data[f'epoch {i}'] = epochs_add
        data[f'file {i}'] = files_add
        data[f'var_name {i}'] = var

        # Drop rows where some varible could not be found
        row_indexs = data.loc[data[f'epoch {i}'] == 0].index
        droped_rows += len(row_indexs)
        data.drop(row_indexs, inplace=True)

    print(f'{droped_rows} samples droped due to invalid data')
    return data.reset_index(drop=True).drop(columns=['date'])

def _get_olshevsky_labeled_dataset(trange, var_list=None, var_to_file_info = None, resample=None):
    """
    Get a list of data in a given timerange.
    """

    if resample is not None:
        if not (resample == '4.5s'):
            raise ValueError('Resampling for Olshevsky labels only ' +
                             'support 4.5s, same frequency as labels')

        df_full = _get_olshevsky_label_list(trange, var_list, var_to_file_info)
        df_full = df_full[['Time','label']]
        df_full = df_full.set_index('Time')
        for i, var in enumerate(var_list):
            print(f'Processing varible: {var}')
            if var not in var_to_file_info:
                raise ValueError(f'Invalid var requested: {var}')

            df_full = df_full.join(
                _get_var(trange, var, var_to_file_info[var]), how='outer')

        df_full = df_full.resample(resample).mean()

        df_full = df_full.sort_index()

        df_full = df_full.loc[(trange[0] <= df_full.index) &
                              (df_full.index < trange[1])]

        df_full['label str'] = df_full['label'].replace(_OLSHEVSKY_LABELS)
    else:
        if len(var_list) > 1 or 'mms1_dis_dist_fast' not in var_list:
            raise ValueError('Unresampled dataset using Olshevsky labels only support ' +
                             'the var mms1_dis_dist_fast.')
        df_full = _get_olshevsky_label_list(trange, var_list, var_to_file_info)

    droped_rows = len(df_full)
    df_full = df_full.dropna()
    droped_rows -= len(df_full)
    print(f'{droped_rows} samples droped due to invalid data')
    return df_full

def _get_var(trange, var, var_to_file_info):

    # The MMS Data API takes the end date as exclusive
    trange = [trange[0].strftime("%Y-%m-%d"),
              (trange[1] + dt.timedelta(days=1)).strftime("%Y-%m-%d")]

    # Check which datafiles are relevant
    files = mms.get_file_list(trange[0], trange[1],
                              **var_to_file_info['info'])

    files = [f['file_name'] for f in files]
    filespaths = mms.filename_to_filepath(files)

    # Download missing
    missing = missing_files(filespaths, _MMS_DATA_DIR)
    if missing:
        print(f"{len(missing)} data files are missing, downloading")
        mms.download_cdf_files(_MMS_DATA_DIR, missing)

    # Load all the file data
    df = None
    for filename in files:
        filepath = mms.filename_to_filepath(filename)
        cdf_file = read_cdf_file(_MMS_DATA_DIR + filepath)
        var_data = cdf_file.varget(var)
        if len(var_to_file_info['mapping']) > 1:
            tmp = {k: var_data[:, i]
                   for k, i in var_to_file_info['mapping']}

        else:
            tmp = {k: var_data[:]
                   for k, _ in var_to_file_info['mapping']}

        df = pd.concat([df, pd.DataFrame(tmp, index=pd.to_datetime(
                       cdfepoch.unixtime(cdf_file.varget('epoch')),
                       unit='s'))])

    return df.sort_index()


def _get_unlabeled_list(trange=None, var_list=None, var_to_file_info=None):
    """
    Get a pandoc DataFrame containing unlabeled epochs in a given
    time range.
    """

    droped_rows = 0

    # Grab relevant epochs from the first varible
    _, epochs = _get_var_info(trange, var_list[0], var_to_file_info[var_list[0]])

    data = pd.DataFrame({'epoch': epochs})
    data['label'] = -1  # Everything is unlabeled
    data['Time'] = pd.to_datetime(cdfepoch.unixtime(data['epoch']), unit='s')
    data = data.loc[(trange[0] <= data['Time']) &
                    (data['Time'] < trange[1])]

    for i, var in enumerate(var_list):
        print(f'Processing varible: {var}')
        if var not in var_to_file_info:
            raise ValueError(f'Invalid var requested: {var}')

        files_add, epochs_add = _get_var_info(trange, var, var_to_file_info[var],
                                              data['epoch'])

        data[f'epoch {i}'] = epochs_add
        data[f'file {i}'] = files_add
        data[f'var_name {i}'] = var

        # Drop rows where some varible could not be found
        row_indexs = data.loc[data[f'epoch {i}'] == 0].index
        droped_rows += len(row_indexs)
        data.drop(row_indexs, inplace=True)

    print(f'{droped_rows} samples droped due to invalid data')

    data = data.sort_values(by='Time')
    return data.reset_index(drop=True)


def _get_unlabeled_dataset(trange, var_list, var_to_file_info, resample=None):
    """
    Get a list of data in a given timerange.
    """

    if resample is not None:
        df_full = pd.DataFrame()
        for i, var in enumerate(var_list):
            print(f'Processing varible: {var}')
            if var not in var_to_file_info:
                raise ValueError(f'Invalid var requested: {var}')

            df_full = df_full.join(
                _get_var(trange, var, var_to_file_info[var]), how='outer')

        df_full = df_full.resample(resample).mean()

        df_full['label'] = -1
        df_full = df_full.sort_index()

        df_full = df_full.loc[(trange[0] <= df_full.index) &
                              (df_full.index < trange[1])]
    else:
        df_full = _get_unlabeled_list(trange, var_list, var_to_file_info)

    return df_full.dropna()

_DEFAULT_VAR_TO_FILE_INFO = {
    'mms1_dis_dist_fast': {
        'info': {
            'data_rate': 'fast',
            'datatype': 'dis-dist',
            'instrument': 'fpi'}},
    'mms1_dis_energyspectr_omni_fast': {
        'info': {
            'data_rate': 'fast',
            'datatype': 'dis-moms',
            'instrument': 'fpi'},
        'mapping': [(f'Ion Spec. {i}', i) for i in range(32)]},
    'mms1_dis_energy_fast': {
        'info': {
            'data_rate': 'fast',
            'datatype': 'dis-moms',
            'instrument': 'fpi'},
        'mapping': [(f'Ion Bin Centers {i}', i) for i in range(32)]},
    'mms1_dis_energy_delta_fast': {
        'info': {
            'data_rate': 'fast',
            'datatype': 'dis-moms',
            'instrument': 'fpi'},
        'mapping': [(f'Ion Bin Delta {i}', i) for i in range(32)]},
    'mms1_dis_bulkv_gse_fast': {
        'info': {
            'data_rate': 'fast',
            'datatype': 'dis-moms',
            'instrument': 'fpi'},
        'mapping': [('Vx', 0), ('Vy', 1), ('Vz', 2)]},
    'mms1_dis_numberdensity_fast': {
        'info': {
            'data_rate': 'fast',
            'datatype': 'dis-moms',
            'instrument': 'fpi'},
        'mapping': [('Number Density', None)]},
    'mms1_dis_temppara_fast': {
        'info': {
            'data_rate': 'fast',
            'datatype': 'dis-moms',
            'instrument': 'fpi'},
        'mapping': [('Ion Temp. par.', None)]},
    'mms1_dis_tempperp_fast': {
        'info': {
            'data_rate': 'fast',
            'datatype': 'dis-moms',
            'instrument': 'fpi'},
        'mapping': [('Ion Temp. perp.', None)]},
    'mms1_fgm_b_gsm_srvy_l2': {
        'info': {
            'data_rate': 'srvy',
            'instrument': 'fgm'},
        'mapping': [('Bx', 0), ('By', 1), ('Bz', 2)]}
}

_LABEL_SOURCES = {
    'Olshevsky' : _get_olshevsky_labeled_dataset,
    'Unlabeled' : _get_unlabeled_dataset,
}


def get_dataset(label_source, trange, resample=None, clean=True, samples=0,
                var_list=['mms1_dis_dist_fast'], var_to_file_info=_DEFAULT_VAR_TO_FILE_INFO):
    """
    Get a dataset based on a given config.

    Args:
        label_source (string): The source for the labels, either Olshevsky or Unlabeled.
        trange (List): List with the start and end times for the dataset. The times should
            be strings and can have either the format YYYY-mm-DD or YYYY-mm-DD/HH:MM:SS
        resample (string): The resample frequency, this varible follow the rules from the
            [pandas resample function](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html).
            Cannot be used with label_source set to Olshevsky.
        clean (Bool): If unknown (-1) labels should be removed.
        samples (Integer): The number of samples per label, set to 0 for all samples.
        var_list (List): List of varibles to get from the CDF-files

    Returns:
        A pandas DataFrame with the created dataset.
    """

    for i, t in enumerate(trange):
        if len(t) == 10:
            trange[i] = dt.datetime.strptime(t, '%Y-%m-%d')
        elif len(t) == 19:
            trange[i] = dt.datetime.strptime(t, '%Y-%m-%d/%H:%M:%S')
        else:
            raise ValueError(f'Incorrect datetime format: {t}')

    if label_source in _LABEL_SOURCES:
        var_to_file_info = {var: var_to_file_info[var] for var in var_list}
        dataset = _LABEL_SOURCES[label_source](trange, resample=resample,
                                            var_to_file_info=var_to_file_info,
                                            var_list=var_list)
    else:
        raise ValueError(f'Incorrect label_source ({label_source}), ' +
                         f'valid options are: {list(_LABEL_SOURCES.keys())}')

    if clean and label_source != 'Unlabeled':
        dataset = dataset.loc[dataset['label'] != -1]

    if samples > 0:
        dataset = dataset.groupby('label').sample(n=samples)

        # Check that we actually have enought data here.
        for label in dataset['label'].unique():
            if len(dataset.loc[dataset['label'] == label]) < samples:
                raise ValueError('Not enought samles to create data set')

    if resample is None:
        dataset = dataset.reset_index(drop=True)

    return dataset


def create_dataset(dataset_path, trange,
                   force=False, var_info_file = None, **kwargs):
    """
    Create a dataset file based on given config.

    Args:
        dataset_path (string): Path to store dataset, end with either .csv or .feather.
        trange (List): List with the start and end times for the dataset. The times should
            be strings and can have either the format YYYY-mm-DD or YYYY-mm-DD/HH:MM:SS
        force (Bool): Overwrite exisiting file if one exists.
        **kwargs : Futher arguments, passed directy to get_dataset(..)
    """

    dataset_path = path.abspath(dataset_path)
    dirpath, _ = path.split(dataset_path)
    makedirs(dirpath, exist_ok=True)

    if path.isfile(dataset_path):
        if force:
            remove(dataset_path)
        else:
            print("Dataset exists, aborting")
            return
    if var_info_file is not None:
        var_to_file_info = load_var_to_file_info(var_info_file)
        kwargs['var_to_file_info'] = var_to_file_info

    labels = get_dataset(trange=trange, **kwargs)

    print(f'Storing dataset at {dataset_path}')
    _, fileformat = path.splitext(dataset_path)
    if fileformat == '.csv':
        labels.to_csv(dataset_path)
    elif fileformat == '.feather':
        labels.to_feather(dataset_path)
    else:
        raise ValueError(f'Unknown filetype {fileformat}')
