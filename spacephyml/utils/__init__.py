"""
Common utils used by multiple scripts
"""
from os import path
import numpy as np
import xarray as xr
import cdflib


def read_cdf_file(cdf_filepath, variables=None):
    """
    Read a cdf file, either fully or only a subset.

    Args:
        cdf_filepath (string): Path to the CDF file.
        variables (list): List with tuples the names to store
                    the varibles in and varibles to read.
    Returns:
        Dictionary with the varibles.
    """
    if variables is None:
        return cdflib.cdfread.CDF(cdf_filepath)

    data = {}
    cdf_file = cdflib.cdfread.CDF(cdf_filepath)
    for name, var in variables:
        try:
            data[name] = np.array((cdf_file.varget(var)))
        except:
            print(f'Failed to read {var} from {cdf_filepath}')
            raise

    return data


def xarray_read_file(filepath):
    """
    Wrapper to handle reading data from multiple different file formats into
    an xarray Dataset.

    Preferred format is NetCDF4 (.nc), which preserves the full xarray
    structure including named dimensions, coordinate metadata, and
    multi-dimensional variables with no information loss.

    Legacy .csv and .feather files written by older versions of SpacePhyML
    are still readable; they are loaded via pandas and converted, so some
    metadata (dimension names, coordinate dtypes) may differ from a native
    .nc round-trip.

    Args:
        filepath (string): The file path including file extension.
    Returns:
        An xarray Dataset read from the given file path.
    """

    _, fileformat = path.splitext(filepath)
    if fileformat == '.nc':
        return xr.open_dataset(filepath)
    if fileformat == '.csv':
        import pandas as pd
        return xr.Dataset.from_dataframe(pd.read_csv(filepath))
    if fileformat == '.feather':
        import pandas as pd
        return xr.Dataset.from_dataframe(pd.read_feather(filepath))

    raise ValueError(f'Unknown filetype: {fileformat}')


# Keep backward-compatible alias
pandas_read_file = xarray_read_file
