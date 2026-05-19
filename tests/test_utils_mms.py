import pytest

import tempfile

import spacephyml.utils.mms as mms


def test_filename_to_filepath_one():
    filename = 'mms1_fpi_fast_l2_dis-dist_20171129160000_v3.4.0.cdf'
    output = f'./mms/mms1/fpi/fast/l2/dis-dist/2017/11/{filename}'
    assert mms.filename_to_filepath(filename) == output

def test_filename_to_filepath_list():
    filenames = ['mms1_fpi_fast_l2_dis-dist_20171129160000_v3.4.0.cdf'
                 for _ in range(9)]
    assert isinstance(mms.filename_to_filepath(filenames), list)

def test_filename_to_filepath_exception():
    with pytest.raises(ValueError):
        mms.filename_to_filepath('nan')

def test_get_file_list(requests_mock):
    start_date = '2017-12-01'
    end_date = '2017-12-02'
    url = 'https://lasp.colorado.edu/mms/sdc/public/files/api/v1/'
    url += f'file_info/science?'
    url += f'start_date={start_date}&end_date={end_date}'
    url += '&sc_id=mms1&instrument_id=fpi'
    url += f'&data_rate_mode=fast&data_level=l2&descriptor=dis-dist'

    requests_mock.get(url,json={'files':['a','b']})
    assert isinstance(mms.get_file_list(start_date, end_date),
                      list)

def test_download_cdf_files(mocker):
    mocker.patch('os.makedirs')
    mocker.patch('os.path.isfile', return_value=True)
    mocker.patch('os.path.isfile', return_value=True)

    mms.download_cdf_files('./', ['a','b'])

def test_download_cdf_files(mocker, requests_mock):
    mocker.patch('os.makedirs')
    mocker.patch('os.path.isfile', side_effect=[True,False])
    mocker.patch('spacephyml.utils.mms.download_file_with_status')

    mms.download_cdf_files('./', ['a','b'])
    mms.download_file_with_status.assert_called_once()



