"""
Script for creating dataset based on exisiting labels.
"""
import tempfile
from os import path, makedirs, remove
import datetime as dt
from tqdm.auto import tqdm
from cdflib import cdfepoch

import xarray as xr
import numpy as np
import pandas as pd  # kept only for to_csv / to_feather I/O helpers

from ..__init__ import _MMS_DATA_DIR
from ..utils import read_cdf_file
from ..utils.file_download import download_file_with_status, missing_files
from ..utils.config import load_var_to_file_info
from ..utils import mms


def _unix_to_datetime64(unix_float_seconds) -> np.ndarray:
    """
    Convert an array of Unix timestamps (float seconds) to ``datetime64[ns]``
    using integer nanosecond arithmetic.

    This is the single authoritative conversion used everywhere in this module.
    It avoids two common pitfalls:

    - ``dtype='datetime64[s]'`` truncates sub-second precision, causing
      timestamps from different CDFs that share the same physical time to
      compare unequal after the cast.
    - ``pd.to_datetime(unit='s')`` introduces floating-point rounding at the
      nanosecond level for the same reason.

    Multiplying by 1e9 and casting to ``int64`` before viewing as
    ``datetime64[ns]`` is exact up to the ~microsecond precision of the
    original float, and — crucially — **deterministic**: the same float input
    always produces the same ``int64`` output, so two datasets converted with
    this function can be merged by exact timestamp equality.
    """
    ns = (np.asarray(unix_float_seconds, dtype=np.float64) * 1e9
          ).astype(np.int64)
    return ns.view('datetime64[ns]')

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


# ---------------------------------------------------------------------------
# CDF introspection helpers
# ---------------------------------------------------------------------------

def _sanitise_var_name(name):
    """
    Normalise a string coming from CDF metadata (FIELDNAM, LABL_PTR_1, …)
    so it is safe to use as an xarray variable name.

    Rules applied (in order):
      - Strip leading/trailing whitespace.
      - Replace '/' with ' ' (e.g. "FPI/DIS" → "FPI DIS").
      - Collapse any runs of whitespace created by the above into a single space.
    """
    name = name.strip()
    name = name.replace('/', ' ')
    name = ' '.join(name.split())   # collapse runs of whitespace
    return name


def _read_var_attrs(cdf_file, var_name):
    """
    Return the ISTP variable-attribute dict for *var_name*, or {} on failure.
    """
    try:
        return cdf_file.varattsget(var_name)
    except Exception:
        return {}


def _get_column_mapping_from_cdf(cdf_file, var_name):
    """
    Derive column names and their array indices for *var_name* by reading the
    ISTP/IACG metadata stored in the CDF file itself, following this priority:

    1. LABL_PTR_1  – variable attribute that names another variable whose
                     values are the per-bin label strings.
    2. DEPEND_1    – variable attribute naming the coordinate-axis variable
                     (e.g. energy bin centres).  When LABL_PTR_1 is absent we
                     fall back to ``"{DEPEND_1}_bin_{i}"`` as the column name.
    3. FIELDNAM    – single human-readable name for the whole variable.  Used
                     when the variable is 1-D (scalar per time step) – yields
                     ``[(FIELDNAM, None)]``.
    4. Numeric     – last resort: ``[("{var_name}_{i}", i) …]`` for each bin.

    Returns:
        list[tuple[str, int | None]]
            Same format as the static ``mapping`` entries:
            ``[(column_name, array_index_or_None), …]``
    """
    attrs = _read_var_attrs(cdf_file, var_name)

    # ------------------------------------------------------------------ #
    # Determine dimensionality of the variable                            #
    # ------------------------------------------------------------------ #
    try:
        vinfo = cdf_file.varinq(var_name)
        # Dim_Sizes is [] for a pure scalar-per-record variable
        n_bins = vinfo.Dim_Sizes[0] if vinfo.Dim_Sizes else None
    except Exception:
        n_bins = None

    # Scalar variable: use FIELDNAM or the raw variable name
    if n_bins is None:
        field_name = attrs.get('FIELDNAM', var_name)
        if isinstance(field_name, (list, np.ndarray)):
            field_name = str(field_name[0])
        return [(_sanitise_var_name(field_name), None)]

    # ------------------------------------------------------------------ #
    # Try LABL_PTR_1 first – gives the nicest human-readable bin labels   #
    # ------------------------------------------------------------------ #
    labl_ptr = attrs.get('LABL_PTR_1')
    if labl_ptr is not None:
        if isinstance(labl_ptr, (list, np.ndarray)):
            labl_ptr = str(labl_ptr[0])
        try:
            raw_labels = cdf_file.varget(labl_ptr)
            # raw_labels is an array of byte-strings or strings
            labels = [_sanitise_var_name(str(lb)) if not isinstance(lb, (bytes, np.bytes_))
                      else _sanitise_var_name(lb.decode('utf-8', errors='replace'))
                      for lb in raw_labels]
            if len(labels) == n_bins:
                return [(lbl, i) for i, lbl in enumerate(labels)]
        except Exception:
            pass  # fall through to DEPEND_1

    # ------------------------------------------------------------------ #
    # Fall back to DEPEND_1 – use coordinate variable name as prefix      #
    # ------------------------------------------------------------------ #
    dep1 = attrs.get('DEPEND_1')
    if dep1 is not None:
        if isinstance(dep1, (list, np.ndarray)):
            dep1 = str(dep1[0])
        # Use the FIELDNAM of the DEPEND_1 variable as prefix if available
        dep_attrs = _read_var_attrs(cdf_file, dep1)
        prefix = dep_attrs.get('FIELDNAM', dep1)
        if isinstance(prefix, (list, np.ndarray)):
            prefix = str(prefix[0])
        prefix = _sanitise_var_name(prefix)
        return [(f'{prefix} {i}', i) for i in range(n_bins)]

    # ------------------------------------------------------------------ #
    # Fall back to FIELDNAM + index                                        #
    # ------------------------------------------------------------------ #
    field_name = attrs.get('FIELDNAM', var_name)
    if isinstance(field_name, (list, np.ndarray)):
        field_name = str(field_name[0])
    field_name = _sanitise_var_name(field_name)
    return [(f'{field_name} {i}', i) for i in range(n_bins)]


def _mapping_from_cdf(cdf_file, var_name, static_mapping):
    """
    Return the column mapping to use for *var_name*.

    If *static_mapping* is provided (not None), it is returned unchanged –
    this preserves full backward compatibility for callers that supply their
    own mappings.  Otherwise the mapping is derived from the CDF metadata.
    """
    if static_mapping is not None:
        return static_mapping
    return _get_column_mapping_from_cdf(cdf_file, var_name)


def _depend1_is_time_varying(cdf_file, var_name):
    """
    Return the DEPEND_1 variable name if it is time-varying (i.e. has shape
    (n_records, n_bins)), otherwise return None.

    A time-varying DEPEND_1 means the coordinate axis (e.g. energy bin
    centres) changes from record to record.  In that case the variable should
    be stored as a 2D matrix (time × bin) rather than a flat collection of
    per-bin 1D variables.

    Returns:
        str | None
            The name of the DEPEND_1 variable if it is time-varying, else None.
    """
    attrs = _read_var_attrs(cdf_file, var_name)
    dep1 = attrs.get('DEPEND_1')
    if dep1 is None:
        return None
    if isinstance(dep1, (list, np.ndarray)):
        dep1 = str(dep1[0])
    try:
        dep_data = cdf_file.varget(dep1)
        # Time-varying: shape is (n_records, n_bins)
        # Non-varying:  shape is (n_bins,)
        if dep_data.ndim == 2:
            return dep1
    except Exception:
        pass
    return None


def _build_matrix_dataset(var_data, dep1_data, dep1_name,
                           time_vals, var_name, cdf_file):
    """
    Build an xr.Dataset with proper 2D structure for a spectrum variable
    whose coordinate axis (DEPEND_1) is time-varying.

    The dataset contains:
      - ``{spec_name}`` : DataArray of shape (time, energy_bin) — the spectra.
      - ``{axis_name}`` : DataArray of shape (time, energy_bin) — the bin
                          centre values at each time step.

    The ``energy_bin`` dimension is an integer index (0 … n_bins-1) because
    the bin centre values differ between time steps and therefore cannot serve
    as a shared coordinate axis.

    Args:
        var_data   : ndarray, shape (n_records, n_bins)
        dep1_data  : ndarray, shape (n_records, n_bins)  – energy bin centres
        dep1_name  : str  – CDF variable name for the axis
        time_vals  : ndarray of datetime64[ns]
        var_name   : str  – CDF variable name for the spectrum
        cdf_file   : open cdflib CDF handle  – used to read FIELDNAM for labelling
    """
    n_bins = var_data.shape[1]
    bin_idx = np.arange(n_bins)

    spec_attrs = _read_var_attrs(cdf_file, var_name)
    dep1_attrs = _read_var_attrs(cdf_file, dep1_name)

    spec_name = spec_attrs.get('FIELDNAM', var_name)
    axis_name = dep1_attrs.get('FIELDNAM', dep1_name)
    if isinstance(spec_name, (list, np.ndarray)):
        spec_name = str(spec_name[0])
    if isinstance(axis_name, (list, np.ndarray)):
        axis_name = str(axis_name[0])
    spec_name = _sanitise_var_name(spec_name)
    axis_name = _sanitise_var_name(axis_name)

    return xr.Dataset(
        {
            spec_name: (['time', 'energy_bin'], var_data),
            axis_name: (['time', 'energy_bin'], dep1_data),
        },
        coords={
            'time':       time_vals,
            'energy_bin': bin_idx,
        },
    ).assign({
        # Tag the axis variable so resample helpers know not to average it.
        # xarray attrs survive merge/concat but not resample, so we track
        # axis variable names separately via a dataset-level attribute.
        axis_name: lambda ds: ds[axis_name].assign_attrs({'spacephyml_role': 'axis'}),
    })


def _resample_dataset(ds: xr.Dataset, resample_freq: str,
                      chunk_size: int = 1000) -> xr.Dataset:
    """
    Resample *ds* along the time dimension, applying the correct aggregation
    to each variable and showing a tqdm progress bar.

    - **Measurement variables** (spectra, labels, scalar moments) → ``.mean()``
    - **Axis variables** (energy bin centres, tagged with
      ``spacephyml_role = 'axis'``) → ``.first()``

    Progress is reported by splitting the input into *chunk_size* time-step
    blocks, resampling each block independently, and concatenating the results.
    This gives ~``ceil(n_times / chunk_size)`` progress updates regardless of
    the resample frequency.

    Args:
        ds : xr.Dataset
        resample_freq : str
            xarray-compatible offset string, e.g. ``'4.5s'``, ``'1min'``.
        chunk_size : int
            Number of input time steps per progress bar tick (default 1000).

    Returns:
        xr.Dataset
            Resampled dataset with correct aggregation per variable.
    """
    axis_vars = [v for v in ds.data_vars
                 if ds[v].attrs.get('spacephyml_role') == 'axis']
    meas_vars = [v for v in ds.data_vars if v not in axis_vars]

    n = ds.sizes['time']
    n_chunks = int(np.ceil(n / chunk_size))

    resampled_chunks = []
    for i in tqdm(range(n_chunks), desc='Resampling', unit='chunk'):
        block = ds.isel(time=slice(i * chunk_size, (i + 1) * chunk_size))

        parts = []
        if meas_vars:
            parts.append(block[meas_vars].resample(time=resample_freq).mean())
        if axis_vars:
            resampled_axes = block[axis_vars].resample(time=resample_freq).first()
            for v in axis_vars:
                resampled_axes[v].attrs['spacephyml_role'] = 'axis'
            parts.append(resampled_axes)

        resampled_chunks.append(xr.merge(parts))

    return xr.concat(resampled_chunks, dim='time')


# ---------------------------------------------------------------------------
# Internal data-loading helpers
# ---------------------------------------------------------------------------

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
        tqdm.write(f'{len(missing)} data files are missing, downloading')
        mms.download_cdf_files(_MMS_DATA_DIR, missing)

    # Load all the epochs
    file_epochs = []
    file_names = []
    for filename in tqdm(files, desc=f'Scanning epochs ({var})', unit='file'):
        filepath = mms.filename_to_filepath(filename)
        cdf_file = read_cdf_file(_MMS_DATA_DIR + filepath)
        tmp = cdf_file.varget('Epoch')
        file_epochs.extend(tmp)
        file_names.extend([filename for _ in tmp])
    file_epochs = np.array(file_epochs)

    if epochs is None:
        return file_names, file_epochs

    files_add = []
    epochs_add = np.zeros(len(epochs)).astype(np.int64)
    for j, epoch_labeled in enumerate(epochs):
        index = np.abs(file_epochs - epoch_labeled).argmin()
        epochs_add[j] = file_epochs[index]
        files_add.append(file_names[index])

        time_diff = np.abs(cdfepoch.unixtime(epoch_labeled)
                           - cdfepoch.unixtime(epochs_add[j]))
        if time_diff > 4.5:
            epochs_add[j] = 0

    return files_add, epochs_add


def _get_olshevsky_label_list(trange=None, var_list=None, var_to_file_info=None,
                              only_labels=False):
    """
    Get an xarray Dataset containing all the Olshevsky labels from within the
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

    print('Downloading Olshevsky label files.')
    label_files = [_download_label_file(tempfile.gettempdir() +
                   '/mms_labels/', d) for d in ['201711', '201712']]

    data = {'label': [], 'epoch': [], 'date': []}
    for file in tqdm(label_files, desc='Reading label files', unit='file'):
        cdf_file = read_cdf_file(file)
        labels_vars = cdf_file.cdf_info().zVariables
        labels_vars = list(zip([lb for lb in labels_vars if 'label_' in lb],
                               [ep for ep in labels_vars if 'epoch_' in ep]))

        for lb, ep in tqdm(labels_vars, desc=f'  Variables in {path.basename(file)}',
                           unit='var', leave=False):
            if lb[-14:] != ep[-14:]:
                raise ValueError('The label and epoch vals are not equal')

            label = cdf_file.varget(lb)
            epoch = cdf_file.varget(ep)
            date = [dt.datetime.strptime(lb.split('_')[6][:8], '%Y%m%d')
                    for _ in range(len(label))]

            data['date'].extend(date)
            data['label'].extend(label)
            data['epoch'].extend(epoch)

    df = pd.DataFrame(data)
    df['Time'] = _unix_to_datetime64(cdfepoch.unixtime(df['epoch'].values))
    df = df.loc[(trange[0] <= df['Time']) & (df['Time'] < trange[1])]

    if not only_labels:
        for i, var in enumerate(var_list):
            tqdm.write(f'Processing variable: {var}')
            if var not in var_to_file_info:
                raise ValueError(f'Invalid var requested: {var}')

            files_add, epochs_add = _get_var_info(trange, var, var_to_file_info[var],
                                                  df['epoch'].values)

            df[f'epoch {i}'] = epochs_add
            df[f'file {i}'] = files_add
            df[f'var_name {i}'] = var

            row_indexs = df.loc[df[f'epoch {i}'] == 0].index
            droped_rows += len(row_indexs)
            df.drop(row_indexs, inplace=True)

    tqdm.write(f'{droped_rows} samples dropped due to invalid data')
    df = df.reset_index(drop=True).drop(columns=['date'])
    return xr.Dataset.from_dataframe(df)


def _get_olshevsky_labeled_dataset(trange, var_list=None, var_to_file_info=None,
                                   resample=None, common_file=False):
    """
    Get a dataset of data in a given timerange as an xarray Dataset.
    """

    if common_file or resample is not None:
        if resample is not None and not (resample == '4.5s'):
            raise ValueError('Resampling for Olshevsky labels only ' +
                             'support 4.5s, same frequency as labels')

        ds_labels = _get_olshevsky_label_list(trange, var_list, var_to_file_info)
        # Keep only the time coordinate and the label variable, indexed by Time
        time_vals = ds_labels['Time'].values
        ds_full = xr.Dataset(
            {'label': ('time', ds_labels['label'].values)},
            coords={'time': time_vals},
        )

        for var in var_list:
            tqdm.write(f'Processing variable: {var}')
            if var not in var_to_file_info:
                raise ValueError(f'Invalid var requested: {var}')
            ds_full = xr.merge(
                [ds_full, _get_var(trange, var, var_to_file_info[var])],
                join='outer',
            )

        if resample is not None:
            ds_full = _resample_dataset(ds_full, resample)
        ds_full = ds_full.sortby('time')
        ds_full = ds_full.sel(time=(
            (ds_full.time >= np.datetime64(trange[0])) &
            (ds_full.time < np.datetime64(trange[1]))
        ))

        ds_full['label'] = ds_full['label'].fillna(-1)
        # Map integer labels to human-readable strings
        label_strs = [_OLSHEVSKY_LABELS.get(int(v), 'Undefined') if not np.isnan(v) else 'Undefined'
                      for v in ds_full['label'].values]
        ds_full['label str'] = ('time', label_strs)
        return ds_full
    else:
        _allowed_vars = ['mms1_dis_dist_fast', 'mms1_dis_energyspectr_omni_fast',
                         'mms1_dis_energy_fast']
        if sum([v not in _allowed_vars for v in var_list]) > 0:
            raise ValueError('Unresampled dataset using Olshevsky labels only support ' +
                             f'the vars {_allowed_vars}.')
        return _get_olshevsky_label_list(trange, var_list, var_to_file_info)


def _get_var(trange, var, var_to_file_info):

    trange = [trange[0].strftime("%Y-%m-%d"),
              (trange[1] + dt.timedelta(days=1)).strftime("%Y-%m-%d")]

    files = mms.get_file_list(trange[0], trange[1],
                              **var_to_file_info['info'])

    files = [f['file_name'] for f in files]
    filespaths = mms.filename_to_filepath(files)

    missing = missing_files(filespaths, _MMS_DATA_DIR)
    if missing:
        tqdm.write(f'{len(missing)} data files are missing, downloading')
        mms.download_cdf_files(_MMS_DATA_DIR, missing)

    # Static mapping supplied by caller (may be None → introspect from CDF)
    static_mapping = var_to_file_info.get('mapping', None)

    chunks = []
    for filename in tqdm(files, desc=f'Reading {var}', unit='file'):
        filepath = mms.filename_to_filepath(filename)
        cdf_file = read_cdf_file(_MMS_DATA_DIR + filepath)
        var_data = cdf_file.varget(var)

        time_vals = _unix_to_datetime64(
            cdfepoch.unixtime(cdf_file.varget('epoch'))
        )

        # Check whether DEPEND_1 is time-varying before falling back to the
        # flat mapping path.  A time-varying axis means the bin centres differ
        # per record (common for MMS FPI energy tables), so we store the data
        # as a 2D (time × energy_bin) matrix with the axis values alongside.
        dep1_var = None if static_mapping is not None else \
            _depend1_is_time_varying(cdf_file, var)

        if dep1_var is not None:
            dep1_data = cdf_file.varget(dep1_var)
            chunks.append(_build_matrix_dataset(
                var_data, dep1_data, dep1_var, time_vals, var, cdf_file,
            ))
        else:
            mapping = _mapping_from_cdf(cdf_file, var, static_mapping)
            if len(mapping) > 1:
                data_vars = {col: ('time', var_data[:, idx]) for col, idx in mapping}
            else:
                col, _ = mapping[0]
                data_vars = {col: ('time', var_data[:])}
            chunks.append(xr.Dataset(data_vars, coords={'time': time_vals}))

    combined = xr.concat(chunks, dim='time')
    return combined.sortby('time')

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


def _get_unlabeled_dataset(trange, var_list, var_to_file_info, resample=None,
                           common_file=False):
    """
    Get a dataset of data in a given timerange as an xarray Dataset.
    """

    if resample is not None or common_file:

        var_datasets = []
        for var in var_list:
            tqdm.write(f'Processing variable: {var}')
            if var not in var_to_file_info:
                raise ValueError(f'Invalid var requested: {var}')
            var_datasets.append(_get_var(trange, var, var_to_file_info[var]))

        ds_full = xr.merge(var_datasets, join='outer')

        if resample is not None:
            ds_full = _resample_dataset(ds_full, resample)

        ds_full['label'] = xr.full_like(ds_full[list(ds_full.data_vars)[0]], -1)
        ds_full = ds_full.sortby('time')
        ds_full = ds_full.sel(time=(
            (ds_full.time >= np.datetime64(trange[0])) &
            (ds_full.time < np.datetime64(trange[1]))
        ))
    else:
        df_full = _get_unlabeled_list(trange, var_list, var_to_file_info)

    return ds_full.dropna(dim='time')


# ---------------------------------------------------------------------------
# Variable-to-file mapping
# ---------------------------------------------------------------------------
# Static mappings are provided for variables where the default CDF column
# naming via introspection is known to be less readable, or where the variable
# is a compound type (e.g. 3-component vector) that cannot be inferred from
# the ISTP attributes alone.
#
# Omitting 'mapping' entirely (or setting it to None) tells _get_var() to
# derive column names from the CDF metadata at runtime using LABL_PTR_1 →
# DEPEND_1 → FIELDNAM in that priority order.  This is the recommended
# approach for spectrum variables such as mms1_dis_energyspectr_omni_fast,
# where the per-bin labels and their coordinate axes (energy bin centres
# from mms1_dis_energy_fast) are already described in the CDF itself.
_DEFAULT_VAR_TO_FILE_INFO = {
    # ------------------------------------------------------------------ #
    # Ion distribution (4-D: time × azimuth × elevation × energy)        #
    # No mapping here: the full distribution is handled by ExternalMMSData#
    # as raw CDF data rather than a flat column table.                    #
    # ------------------------------------------------------------------ #
    'mms1_dis_dist_fast': {
        'info': {
            'data_rate': 'fast',
            'datatype': 'dis-dist',
            'instrument': 'fpi'}},

    # ------------------------------------------------------------------ #
    # Omni-directional ion energy spectrum – 32 energy bins.              #
    # 'mapping' is intentionally absent: _get_var() will call             #
    # _get_column_mapping_from_cdf(), which reads LABL_PTR_1 from the CDF #
    # to obtain the per-bin label strings, falling back to DEPEND_1       #
    # (mms1_dis_energy_fast) as the coordinate-axis prefix if needed.     #
    # ------------------------------------------------------------------ #
    'mms1_dis_energyspectr_omni_fast': {
        'info': {
            'data_rate': 'fast',
            'datatype': 'dis-moms',
            'instrument': 'fpi'}},

    # ------------------------------------------------------------------ #
    # Ion energy bin centres – 32 values per record.                      #
    # Same introspection path as above.                                   #
    # ------------------------------------------------------------------ #
    'mms1_dis_energy_fast': {
        'info': {
            'data_rate': 'fast',
            'datatype': 'dis-moms',
            'instrument': 'fpi'}},

    # ------------------------------------------------------------------ #
    # Ion energy bin half-widths.                                         #
    # ------------------------------------------------------------------ #
    'mms1_dis_energy_delta_fast': {
        'info': {
            'data_rate': 'fast',
            'datatype': 'dis-moms',
            'instrument': 'fpi'}},

    # ------------------------------------------------------------------ #
    # Bulk velocity vector (3 components).                                #
    # Static mapping kept: ISTP labels for GSE components vary between    #
    # CDF versions; using explicit names is more robust here.             #
    # ------------------------------------------------------------------ #
    'mms1_dis_bulkv_gse_fast': {
        'info': {
            'data_rate': 'fast',
            'datatype': 'dis-moms',
            'instrument': 'fpi'},
        'mapping': [('Vx', 0), ('Vy', 1), ('Vz', 2)]},

    # ------------------------------------------------------------------ #
    # Scalar moments – no mapping needed (single value per time step).    #
    # FIELDNAM from the CDF will be used as the column name.              #
    # ------------------------------------------------------------------ #
    'mms1_dis_numberdensity_fast': {
        'info': {
            'data_rate': 'fast',
            'datatype': 'dis-moms',
            'instrument': 'fpi'}},

    'mms1_dis_temppara_fast': {
        'info': {
            'data_rate': 'fast',
            'datatype': 'dis-moms',
            'instrument': 'fpi'}},

    'mms1_dis_tempperp_fast': {
        'info': {
            'data_rate': 'fast',
            'datatype': 'dis-moms',
            'instrument': 'fpi'}},

    # ------------------------------------------------------------------ #
    # Magnetic field vector (4 components in L2: Bx, By, Bz, |B|).       #
    # Static mapping kept for the same reason as bulk velocity.           #
    # ------------------------------------------------------------------ #
    'mms1_fgm_b_gsm_srvy_l2': {
        'info': {
            'data_rate': 'srvy',
            'instrument': 'fgm'},
        'mapping': [('Bx', 0), ('By', 1), ('Bz', 2)]},
}

_LABEL_SOURCES = {
    'Olshevsky' : _get_olshevsky_labeled_dataset,
    'Unlabeled' : _get_unlabeled_dataset,
}


def get_dataset(label_source, trange, resample=None, clean=False, samples=0,
                var_list=['mms1_dis_dist_fast'], var_to_file_info=_DEFAULT_VAR_TO_FILE_INFO,
                **kwargs):
    """
    Get a dataset based on a given config.

    Args:
        label_source (string): The source for the labels, either Olshevsky or Unlabeled.
        trange (List): List with the start and end times for the dataset. The times should
            be strings and can have either the format YYYY-mm-DD or YYYY-mm-DD/HH:MM:SS
        resample (string): The resample frequency, expressed as an xarray/
            pandas-compatible offset string (e.g. '4.5s', '1min').
            Cannot be used with label_source set to Olshevsky.
        clean (Bool): If unknown (-1) labels should be removed.
        samples (Integer): The number of samples per label, set to 0 for all samples.
        var_list (List): List of varibles to get from the CDF-files

    Returns:
        An xarray Dataset with the created dataset.
    """

    if isinstance(trange, tuple):
        trange = list(trange)

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
                                            var_list=var_list, **kwargs)
    else:
        raise ValueError(f'Incorrect label_source ({label_source}), ' +
                         f'valid options are: {list(_LABEL_SOURCES.keys())}')

    if clean and label_source != 'Unlabeled':
        mask = dataset['label'].values != -1
        dim = 'time' if 'time' in dataset.dims else 'index'
        dataset = dataset.isel({dim: mask})

    if samples > 0:
        label_values = dataset['label'].values
        unique_labels = np.unique(label_values)

        selected_indices = []
        for label in unique_labels:
            idx = np.where(label_values == label)[0]
            if len(idx) < samples:
                raise ValueError(
                    f'Not enough samples for label {label}: '
                    f'need {samples}, have {len(idx)}'
                )
            chosen = np.random.choice(idx, size=samples, replace=False)
            selected_indices.append(chosen)

        all_indices = np.sort(np.concatenate(selected_indices))
        dim = 'time' if 'time' in dataset.dims else 'index'
        dataset = dataset.isel({dim: all_indices})

    if resample is None:
        # Re-index with a clean integer range only when the dataset has a
        # plain integer dimension (as produced by _get_olshevsky_label_list).
        # Datasets that already carry a datetime 'time' coordinate are left
        # untouched so the coordinate is not destroyed.
        dim = 'time' if 'time' in dataset.dims else 'index'
        if dataset[dim].dtype.kind != 'M':   # not datetime → integer index
            n = dataset.sizes[dim]
            dataset = dataset.assign_coords({dim: np.arange(n)})

    return dataset


def create_dataset(dataset_path, trange,
                   force=False, var_info_file=None, **kwargs):
    """
    Create a dataset file based on given config.

    Args:
        dataset_path (string): Path to store dataset.  Supported extensions:

            - ``.nc``      — NetCDF4 (recommended).  Stores the full xarray
              Dataset with all dimensions, coordinate metadata, and
              multi-dimensional variables intact.
            - ``.csv``     — Lossy flat export.  Multi-dimensional variables
              are flattened; datetime coordinates become string columns.
              Provided for interoperability only.
            - ``.feather`` — Lossy flat export, same caveats as .csv.

        trange (List): List with the start and end times for the dataset. The
            times should be strings and can have either the format
            YYYY-mm-DD or YYYY-mm-DD/HH:MM:SS.
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
    if fileformat == '.nc':
        labels.to_netcdf(dataset_path, engine='netcdf4')
    elif fileformat == '.csv':
        import warnings
        warnings.warn(
            'Saving as .csv is a lossy export: multi-dimensional variables '
            'are flattened and coordinate metadata is lost.  Use .nc to '
            'preserve the full xarray structure.',
            UserWarning, stacklevel=2,
        )
        labels.to_dataframe().to_csv(dataset_path)
    elif fileformat == '.feather':
        import warnings
        warnings.warn(
            'Saving as .feather is a lossy export: multi-dimensional '
            'variables are flattened and coordinate metadata is lost.  '
            'Use .nc to preserve the full xarray structure.',
            UserWarning, stacklevel=2,
        )
        labels.to_dataframe().reset_index().to_feather(dataset_path)
    else:
        raise ValueError(f'Unknown filetype {fileformat}')
