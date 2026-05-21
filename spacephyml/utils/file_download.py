"""
Utils for file downloads.
"""

import functools
from os import path
from shutil import copyfileobj, copy
from tempfile import NamedTemporaryFile
from tqdm.auto import tqdm
import requests


def missing_files(files, rootdir=''):
    """
    Check for missing files.

    Args:
        files (list): The files (including path) to check.
        rootdir (string) :
            A root path to add before the paths in the files list.

    Returns:
        A list of files (from the files argument) that does not exist.
        Will return 'None' if all files exists.
    """
    missing = []
    for filename in files:
        if not path.isfile(path.abspath(f'{rootdir}/{filename}')):
            missing.append(filename)

    if len(missing) == 0:
        return None

    return missing


def download_file_with_status(url_file, filepath, session=None, position=0):
    """
    Download one file with a progress bar.

    Args:
        url_file (string) : The URL for downloading the file.
        filepath (string) : The file path for storing the file
        session : The request session to use, if one exist.
    """
    close_session = False
    if session is None:
        close_session = True
        session = requests.Session()

    r = session.get(url_file, stream=True, verify=True)

    # Code for status bar from:
    # https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError("Request to {url_file} returned status code " +
                           f"{r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)
    with NamedTemporaryFile() as ftmp:
        with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc,
                           leave = False if position > 0 else True,
                           position=position) as r_raw:
            with open(ftmp.name, 'wb') as f:
                copyfileobj(r_raw, f)

        # If the download was successful, copy to data directory.
        # This prevents a failed (or aborted) download to block
        # future downloads.
        copy(ftmp.name, filepath)
        r.close()

    if close_session:
        session.close()
