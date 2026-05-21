"""
Dataset utils specific to MMS.
"""
from os import path, makedirs
import requests
from tqdm.auto import tqdm

from .file_download import download_file_with_status


def filename_to_filepath(filename):
    """
    Create a filepath based on the filename for a mms CDF file

    Filename format is assumed to be:
        mms1_fpi_fast_l2_dis-dist_20171129160000_v3.4.0.cdf
        {probe}_{inst}_{mode}_{data_level}_{data}_{time}_{version}.cdf
    Output will have the format:
        ./mms/mms1/fpi/fast/l2/dis-dist/2017/12/{filename}

    Args:
        filename (string): The filename to convert to a file path.

    Returns:
        A string with the file path.
    """
    if isinstance(filename, str):
        filename = [filename]

    filepaths = []
    for fn in filename:
        fs = fn.split('_')

        year = fs[-2][:4]
        month = fs[-2][4:6]
        filepaths.append(
                './mms/' + '/'.join(fs[:-2]) + f'/{year}/{month}/{fn}')

    if len(filepaths) == 1:
        return filepaths[0]
    return filepaths


_MMS_DATA_BASE_URL = 'https://lasp.colorado.edu/mms/sdc/public/files/api/v1/'


def get_file_list(start_date, end_date, data_rate='fast', data_level='l2',
                  datatype=None, instrument='fpi', sc_id='mms1'):
    """
    Get a list of files from the MMS Science Data center. For a full list of
    the possible parameters, look at "Query Parameters" on the MMS Science
    Data center, "How to get data" page:
    [https://lasp.colorado.edu/mms/sdc/public/about/how-to/](https://lasp.colorado.edu/mms/sdc/public/about/how-to/)

    Args:
        start_date (string): The start date for files, format YYYY-MM-DD.
        end_date (string): The end date for files, format YYYY-MM-DD.
        data_rate (string): The data rate, fast, burst or srvy.
        data_level (string): The level of data post processing.
        datatype (string): The datatype (not always used).
        instrument (string): The instrument onboard mms.
        sc_id (string): The spacecraft id, mms1, mms2, mms3 or mms4.

    Returns:
        A list of files.
    """
    url = f'{_MMS_DATA_BASE_URL}file_info/science?'
    url += f'start_date={start_date}&end_date={end_date}&sc_id={sc_id}'
    url += f'&instrument_id={instrument}'
    url += f'&data_rate_mode={data_rate}&data_level={data_level}'
    if datatype is not None:
        url += f'&descriptor={datatype}'

    with requests.Session() as session:
        r = session.get(url)
        files = r.json()['files']
        r.close()

    return files


def download_cdf_files(rootdir, cdf_filepaths, session=None):
    """
    Download CDF files from the MMS Science Data Center based on a file list.

    Args:
        rootdir (string): Root directory for storing downloaded files.
        cdf_filepaths (list): The paths to store the files (including
            filename). The filename have to be the same as the file to
            download.
        session (Object): The request session to use, if one exists.
    """
    close_session = False
    if session is None:
        close_session = True
        session = requests.Session()

    t_cnt = len(cdf_filepaths)
    rootdir = rootdir[:-1] if rootdir[-1] == '/' else rootdir

    with tqdm(total=t_cnt, desc='Downloading files', position = 0) as pbar:
        for filepath in cdf_filepaths:
            dirpath, filename = path.split(filepath)
            makedirs(f'{rootdir}/{dirpath}', exist_ok=True)
            url_file = f'{_MMS_DATA_BASE_URL}download/science?file={filename}'
            filepath = path.abspath(f'{rootdir}/{dirpath}/{filename}')
            pbar.set_postfix_str(f'Downloading {filename}')

            if not path.isfile(filepath):
                download_file_with_status(url_file, filepath, session, position=1)

            pbar.update(1)

    if close_session:
        session.close()
