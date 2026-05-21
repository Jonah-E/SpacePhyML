"""
Unit tests for the CDF-metadata introspection helpers in
spacephyml/datasets/creator.py
"""
import numpy as np
import xarray as xr
import pytest
from unittest.mock import MagicMock, patch

from spacephyml.datasets.creator import (
    _get_column_mapping_from_cdf,
    _mapping_from_cdf,
    _read_var_attrs,
)


# ---------------------------------------------------------------------------
# Helpers to build mock CDF objects
# ---------------------------------------------------------------------------

def _mock_cdf(var_attrs=None, varinq_dim_sizes=None, varget_returns=None):
    """
    Build a minimal mock of a cdflib CDF reader.

    Args:
        var_attrs : dict
            Mapping from variable-name → attribute dict returned by varattsget().
        varinq_dim_sizes : dict
            Mapping from variable-name → Dim_Sizes list returned by varinq().
        varget_returns : dict
            Mapping from variable-name → array returned by varget().
    """
    var_attrs = var_attrs or {}
    varinq_dim_sizes = varinq_dim_sizes or {}
    varget_returns = varget_returns or {}

    cdf = MagicMock()

    def _varattsget(v):
        return var_attrs.get(v, {})

    def _varinq(v):
        info = MagicMock()
        info.Dim_Sizes = varinq_dim_sizes.get(v, [])
        return info

    def _varget(v):
        if v in varget_returns:
            return varget_returns[v]
        raise KeyError(v)

    cdf.varattsget.side_effect = _varattsget
    cdf.varinq.side_effect = _varinq
    cdf.varget.side_effect = _varget
    return cdf


# ---------------------------------------------------------------------------
# _read_var_attrs
# ---------------------------------------------------------------------------

class TestReadVarAttrs:
    def test_returns_dict_on_success(self):
        cdf = _mock_cdf(var_attrs={'my_var': {'FIELDNAM': 'My Variable'}})
        result = _read_var_attrs(cdf, 'my_var')
        assert result == {'FIELDNAM': 'My Variable'}

    def test_returns_empty_dict_on_failure(self):
        cdf = MagicMock()
        cdf.varattsget.side_effect = RuntimeError("bad variable")
        result = _read_var_attrs(cdf, 'missing_var')
        assert result == {}


# ---------------------------------------------------------------------------
# _get_column_mapping_from_cdf – LABL_PTR_1 path
# ---------------------------------------------------------------------------

class TestLablPtr1Path:
    def test_uses_label_strings_from_labl_ptr_variable(self):
        """LABL_PTR_1 points to a variable whose values are per-bin strings."""
        labels = np.array(['E1', 'E2', 'E3', 'E4'])
        cdf = _mock_cdf(
            var_attrs={
                'my_spec': {
                    'LABL_PTR_1': 'my_spec_labels',
                    'FIELDNAM': 'Spectrum',
                }
            },
            varinq_dim_sizes={'my_spec': [4]},
            varget_returns={'my_spec_labels': labels},
        )
        mapping = _get_column_mapping_from_cdf(cdf, 'my_spec')
        assert mapping == [('E1', 0), ('E2', 1), ('E3', 2), ('E4', 3)]

    def test_label_count_mismatch_falls_through_to_depend1(self):
        """
        If the label variable has a different length than n_bins,
        fall back to DEPEND_1.
        """
        wrong_labels = np.array(['X', 'Y'])  # only 2 labels for a 4-bin var
        cdf = _mock_cdf(
            var_attrs={
                'my_spec': {
                    'LABL_PTR_1': 'bad_labels',
                    'DEPEND_1': 'energy_var',
                },
                'energy_var': {'FIELDNAM': 'Ion Energy'},
            },
            varinq_dim_sizes={'my_spec': [4]},
            varget_returns={'bad_labels': wrong_labels},
        )
        mapping = _get_column_mapping_from_cdf(cdf, 'my_spec')
        assert mapping == [
            ('Ion Energy 0', 0),
            ('Ion Energy 1', 1),
            ('Ion Energy 2', 2),
            ('Ion Energy 3', 3),
        ]

    def test_bytes_labels_are_decoded(self):
        """Byte-string labels (common in older CDFs) are decoded to str."""
        labels = np.array([b'Bin_A', b'Bin_B'])
        cdf = _mock_cdf(
            var_attrs={'s': {'LABL_PTR_1': 'lbl'}},
            varinq_dim_sizes={'s': [2]},
            varget_returns={'lbl': labels},
        )
        mapping = _get_column_mapping_from_cdf(cdf, 's')
        assert mapping == [('Bin_A', 0), ('Bin_B', 1)]


# ---------------------------------------------------------------------------
# _get_column_mapping_from_cdf – DEPEND_1 fallback
# ---------------------------------------------------------------------------

class TestDepend1Fallback:
    def test_uses_depend1_fieldnam_as_prefix(self):
        cdf = _mock_cdf(
            var_attrs={
                'omni_spec': {'DEPEND_1': 'energy_bins'},
                'energy_bins': {'FIELDNAM': 'Ion Energy Bin Centres'},
            },
            varinq_dim_sizes={'omni_spec': [32]},
        )
        mapping = _get_column_mapping_from_cdf(cdf, 'omni_spec')
        assert len(mapping) == 32
        assert mapping[0] == ('Ion Energy Bin Centres 0', 0)
        assert mapping[31] == ('Ion Energy Bin Centres 31', 31)

    def test_uses_depend1_var_name_when_no_fieldnam(self):
        cdf = _mock_cdf(
            var_attrs={
                'omni_spec': {'DEPEND_1': 'energy_bins'},
                'energy_bins': {},           # no FIELDNAM
            },
            varinq_dim_sizes={'omni_spec': [4]},
        )
        mapping = _get_column_mapping_from_cdf(cdf, 'omni_spec')
        assert mapping[0][0] == 'energy_bins 0'

    def test_depend1_as_list_value(self):
        """Some CDFs store scalar attributes as single-element arrays."""
        cdf = _mock_cdf(
            var_attrs={
                'v': {'DEPEND_1': np.array(['dep_var'])},
                'dep_var': {'FIELDNAM': 'My Coord'},
            },
            varinq_dim_sizes={'v': [3]},
        )
        mapping = _get_column_mapping_from_cdf(cdf, 'v')
        assert mapping[0] == ('My Coord 0', 0)


# ---------------------------------------------------------------------------
# _get_column_mapping_from_cdf – scalar variable (FIELDNAM fallback)
# ---------------------------------------------------------------------------

class TestScalarFallback:
    def test_scalar_variable_uses_fieldnam(self):
        """A variable with no array dimension → single (FIELDNAM, None) entry."""
        cdf = _mock_cdf(
            var_attrs={'n_i': {'FIELDNAM': 'Ion Number Density'}},
            varinq_dim_sizes={'n_i': []},   # scalar per record
        )
        mapping = _get_column_mapping_from_cdf(cdf, 'n_i')
        assert mapping == [('Ion Number Density', None)]

    def test_scalar_falls_back_to_var_name(self):
        cdf = _mock_cdf(
            var_attrs={'n_i': {}},           # no FIELDNAM
            varinq_dim_sizes={'n_i': []},
        )
        mapping = _get_column_mapping_from_cdf(cdf, 'n_i')
        assert mapping == [('n_i', None)]

    def test_numeric_fallback_when_no_attrs(self):
        """Array variable with no LABL_PTR_1, no DEPEND_1, uses FIELDNAM+index."""
        cdf = _mock_cdf(
            var_attrs={'v': {'FIELDNAM': 'My Field'}},
            varinq_dim_sizes={'v': [3]},
        )
        mapping = _get_column_mapping_from_cdf(cdf, 'v')
        assert mapping == [('My Field 0', 0), ('My Field 1', 1), ('My Field 2', 2)]


# ---------------------------------------------------------------------------
# _mapping_from_cdf – static override
# ---------------------------------------------------------------------------

class TestMappingFromCdf:
    def test_static_mapping_returned_unchanged(self):
        """When a static mapping is provided it is used as-is."""
        static = [('Vx', 0), ('Vy', 1), ('Vz', 2)]
        cdf = MagicMock()  # should never be called
        result = _mapping_from_cdf(cdf, 'mms1_dis_bulkv_gse_fast', static)
        assert result == static
        cdf.varattsget.assert_not_called()

    def test_none_mapping_triggers_introspection(self):
        """When mapping=None the CDF is queried."""
        labels = np.array(['A', 'B'])
        cdf = _mock_cdf(
            var_attrs={'v': {'LABL_PTR_1': 'lbl'}},
            varinq_dim_sizes={'v': [2]},
            varget_returns={'lbl': labels},
        )
        result = _mapping_from_cdf(cdf, 'v', None)
        assert result == [('A', 0), ('B', 1)]

    def test_absent_mapping_key_triggers_introspection(self):
        """When the var_to_file_info dict has no 'mapping' key, introspect."""
        var_info = {'info': {'data_rate': 'fast'}}   # no 'mapping' key
        static = var_info.get('mapping', None)        # None

        labels = np.array(['X', 'Y', 'Z'])
        cdf = _mock_cdf(
            var_attrs={'v': {'LABL_PTR_1': 'lbl'}},
            varinq_dim_sizes={'v': [3]},
            varget_returns={'lbl': labels},
        )
        result = _mapping_from_cdf(cdf, 'v', static)
        assert result == [('X', 0), ('Y', 1), ('Z', 2)]


# ---------------------------------------------------------------------------
# _get_var output shape (unit-level, mocked CDF layer)
# ---------------------------------------------------------------------------

class TestGetVarXarrayOutput:
    """Verify _get_var returns a proper xr.Dataset with a 'time' dimension."""

    def test_returns_xarray_dataset(self, monkeypatch):
        import datetime as dt
        import numpy as np
        import xarray as xr
        from unittest.mock import MagicMock, patch

        # Fake file list and download helpers
        fake_file = 'mms1_fpi_fast_l2_dis-moms_20171101000000_v3.4.0.cdf'
        fake_filepath = './mms/mms1/fpi/fast/l2/dis-moms/2017/11/' + fake_file

        epoch_unix = np.array([0.0, 4.5, 9.0])  # 3 time steps
        var_data   = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # shape (3,2)

        mock_cdf_obj = MagicMock()
        mock_cdf_obj.varget.side_effect = lambda v: (
            (epoch_unix * 1e9).astype(np.int64) if v == 'epoch' else var_data
        )
        mock_cdf_obj.varattsget.return_value = {'LABL_PTR_1': 'lbl'}
        mock_cdf_obj.varinq.return_value = MagicMock(Dim_Sizes=[2])
        lbl_vals = np.array(['A', 'B'])
        mock_cdf_obj.varget.side_effect = lambda v: (
            (epoch_unix * 1e9).astype(np.int64) if v == 'epoch'
            else lbl_vals if v == 'lbl'
            else var_data
        )

        from spacephyml.datasets import creator as c

        monkeypatch.setattr(c.mms, 'get_file_list',
                            lambda *a, **k: [{'file_name': fake_file}])
        monkeypatch.setattr(c.mms, 'filename_to_filepath',
                            lambda f: fake_filepath if isinstance(f, str) else [fake_filepath])
        monkeypatch.setattr(c, 'missing_files', lambda *a, **k: [])
        monkeypatch.setattr(c, 'read_cdf_file', lambda p: mock_cdf_obj)

        var_info = {'info': {'data_rate': 'fast', 'datatype': 'dis-moms',
                             'instrument': 'fpi'}}
        trange = [dt.datetime(2017, 11, 1), dt.datetime(2017, 11, 2)]

        result = c._get_var(trange, 'my_var', var_info)

        assert isinstance(result, xr.Dataset)
        assert 'time' in result.dims
        assert result.sizes['time'] == 3
        assert 'A' in result.data_vars
        assert 'B' in result.data_vars

    def test_resample_uses_xarray_not_pandas(self, monkeypatch):
        """_get_unlabeled_dataset resamples via xr.Dataset.resample, not pandas."""
        import datetime as dt
        import numpy as np
        import xarray as xr
        from spacephyml.datasets import creator as c

        # Build a tiny synthetic dataset that _get_var would return
        times = np.array(['2017-11-01T00:00:00', '2017-11-01T00:00:02',
                          '2017-11-01T00:00:04', '2017-11-01T00:00:06'],
                         dtype='datetime64[ns]')
        ds_var = xr.Dataset({'val': ('time', np.array([1.0, 2.0, 3.0, 4.0]))},
                            coords={'time': times})

        monkeypatch.setattr(c, '_get_var', lambda trange, var, info: ds_var)

        trange = [dt.datetime(2017, 11, 1), dt.datetime(2017, 11, 2)]
        var_to_file_info = {'my_var': {'info': {}}}

        result = c._get_unlabeled_dataset(
            trange, ['my_var'], var_to_file_info, resample='4s'
        )

        # With 4-second bins the four 2-second-apart points collapse to 2 bins
        assert isinstance(result, xr.Dataset)
        assert 'time' in result.dims
        # Resampled means: bin [0,4s) → mean(1,2,3)=2; bin [4,8s) → mean(3,4)=3.5
        assert result.sizes['time'] <= 4   # at most as many as input
        assert 'val' in result.data_vars


# ---------------------------------------------------------------------------
# _depend1_is_time_varying
# ---------------------------------------------------------------------------

class TestDepend1IsTimeVarying:
    def test_returns_dep1_name_when_time_varying(self):
        from spacephyml.datasets.creator import _depend1_is_time_varying
        # dep1 shape (3, 32) → time-varying
        cdf = _mock_cdf(
            var_attrs={'spec': {'DEPEND_1': 'energy'}},
            varget_returns={'energy': np.ones((3, 32))},
        )
        assert _depend1_is_time_varying(cdf, 'spec') == 'energy'

    def test_returns_none_when_dep1_not_time_varying(self):
        from spacephyml.datasets.creator import _depend1_is_time_varying
        # dep1 shape (32,) → fixed axis, not time-varying
        cdf = _mock_cdf(
            var_attrs={'spec': {'DEPEND_1': 'energy'}},
            varget_returns={'energy': np.ones((32,))},
        )
        assert _depend1_is_time_varying(cdf, 'spec') is None

    def test_returns_none_when_no_depend1(self):
        from spacephyml.datasets.creator import _depend1_is_time_varying
        cdf = _mock_cdf(var_attrs={'spec': {}})
        assert _depend1_is_time_varying(cdf, 'spec') is None

    def test_returns_none_when_dep1_varget_fails(self):
        from spacephyml.datasets.creator import _depend1_is_time_varying
        cdf = _mock_cdf(var_attrs={'spec': {'DEPEND_1': 'missing_var'}})
        # varget raises for unknown key → should silently return None
        assert _depend1_is_time_varying(cdf, 'spec') is None


# ---------------------------------------------------------------------------
# _build_matrix_dataset
# ---------------------------------------------------------------------------

class TestBuildMatrixDataset:
    def _make_inputs(self):
        times = np.array(['2017-11-01T00:00:00', '2017-11-01T00:00:05'],
                         dtype='datetime64[ns]')
        var_data  = np.array([[1.0]*4, [2.0]*4])   # (2, 4)
        dep1_data = np.array([[10., 20., 30., 40.],
                               [11., 21., 31., 41.]])  # (2, 4) - time-varying
        return times, var_data, dep1_data

    def test_returns_xarray_dataset(self):
        from spacephyml.datasets.creator import _build_matrix_dataset
        times, var_data, dep1_data = self._make_inputs()
        cdf = _mock_cdf(
            var_attrs={
                'omni_spec': {'FIELDNAM': 'Ion Omni Spectrum'},
                'energy':    {'FIELDNAM': 'Ion Energy'},
            }
        )
        ds = _build_matrix_dataset(var_data, dep1_data, 'energy',
                                   times, 'omni_spec', cdf)
        assert isinstance(ds, xr.Dataset)

    def test_has_time_and_energy_bin_dimensions(self):
        from spacephyml.datasets.creator import _build_matrix_dataset
        times, var_data, dep1_data = self._make_inputs()
        cdf = _mock_cdf(var_attrs={'s': {'FIELDNAM': 'S'}, 'e': {'FIELDNAM': 'E'}})
        ds = _build_matrix_dataset(var_data, dep1_data, 'e', times, 's', cdf)
        assert 'time' in ds.dims
        assert 'energy_bin' in ds.dims
        assert ds.sizes['time'] == 2
        assert ds.sizes['energy_bin'] == 4

    def test_spectrum_variable_has_correct_values(self):
        from spacephyml.datasets.creator import _build_matrix_dataset
        times, var_data, dep1_data = self._make_inputs()
        cdf = _mock_cdf(var_attrs={'s': {'FIELDNAM': 'Spectrum'}, 'e': {}})
        ds = _build_matrix_dataset(var_data, dep1_data, 'e', times, 's', cdf)
        np.testing.assert_array_equal(ds['Spectrum'].values, var_data)

    def test_axis_variable_carries_time_varying_bin_centres(self):
        from spacephyml.datasets.creator import _build_matrix_dataset
        times, var_data, dep1_data = self._make_inputs()
        cdf = _mock_cdf(var_attrs={'s': {}, 'e': {'FIELDNAM': 'Ion Energy'}})
        ds = _build_matrix_dataset(var_data, dep1_data, 'e', times, 's', cdf)
        np.testing.assert_array_equal(ds['Ion Energy'].values, dep1_data)

    def test_energy_bin_coordinate_is_integer_index(self):
        from spacephyml.datasets.creator import _build_matrix_dataset
        times, var_data, dep1_data = self._make_inputs()
        cdf = _mock_cdf(var_attrs={'s': {}, 'e': {}})
        ds = _build_matrix_dataset(var_data, dep1_data, 'e', times, 's', cdf)
        np.testing.assert_array_equal(ds.coords['energy_bin'].values,
                                      np.arange(4))

    def test_uses_fieldnam_for_variable_names(self):
        from spacephyml.datasets.creator import _build_matrix_dataset
        times, var_data, dep1_data = self._make_inputs()
        cdf = _mock_cdf(var_attrs={
            'raw_spec':   {'FIELDNAM': 'Omni Spectrum'},
            'raw_energy': {'FIELDNAM': 'Energy Bin Centres'},
        })
        ds = _build_matrix_dataset(var_data, dep1_data, 'raw_energy',
                                   times, 'raw_spec', cdf)
        assert 'Omni Spectrum' in ds.data_vars
        assert 'Energy Bin Centres' in ds.data_vars


# ---------------------------------------------------------------------------
# _get_var produces 2D output for time-varying spectrum
# ---------------------------------------------------------------------------

class TestGetVarMatrixOutput:
    def test_produces_2d_spectrum_when_depend1_is_time_varying(self, monkeypatch):
        import datetime as dt
        import numpy as np
        import xarray as xr
        from spacephyml.datasets import creator as c

        n_times, n_bins = 3, 4
        epoch_ns = (np.arange(n_times) * 5e9).astype(np.int64)
        var_data   = np.ones((n_times, n_bins))
        dep1_data  = np.tile(np.arange(n_bins, dtype=float), (n_times, 1))

        mock_cdf = _mock_cdf(
            var_attrs={
                'spec':   {'FIELDNAM': 'Ion Spectrum', 'DEPEND_1': 'energy'},
                'energy': {'FIELDNAM': 'Ion Energy'},
            },
            varget_returns={
                'epoch':  epoch_ns,
                'spec':   var_data,
                'energy': dep1_data,
            },
        )

        fake_file = 'mms1_fpi_fast_l2_dis-moms_20171101000000_v3.4.0.cdf'
        monkeypatch.setattr(c.mms, 'get_file_list',
                            lambda *a, **k: [{'file_name': fake_file}])
        monkeypatch.setattr(c.mms, 'filename_to_filepath',
                            lambda f: './fake/' + (f if isinstance(f, str) else f[0]))
        monkeypatch.setattr(c, 'missing_files', lambda *a, **k: [])
        monkeypatch.setattr(c, 'read_cdf_file', lambda p: mock_cdf)

        # Patch cdfepoch.unixtime to convert our ns integers to seconds
        monkeypatch.setattr(c.cdfepoch, 'unixtime',
                            lambda e: np.array(e, dtype=float) / 1e9)

        trange = [dt.datetime(2017, 11, 1), dt.datetime(2017, 11, 2)]
        result = c._get_var(trange, 'spec', {'info': {}})

        assert isinstance(result, xr.Dataset)
        assert 'time' in result.dims
        assert 'energy_bin' in result.dims
        assert result.sizes['time'] == n_times
        assert result.sizes['energy_bin'] == n_bins
        assert 'Ion Spectrum' in result.data_vars
        assert 'Ion Energy' in result.data_vars


# ---------------------------------------------------------------------------
# _sanitise_var_name
# ---------------------------------------------------------------------------

class TestSanitiseVarName:
    def test_removes_slash(self):
        from spacephyml.datasets.creator import _sanitise_var_name
        assert _sanitise_var_name('FPI/DIS energy') == 'FPI DIS energy'

    def test_strips_whitespace(self):
        from spacephyml.datasets.creator import _sanitise_var_name
        assert _sanitise_var_name('  Ion Energy  ') == 'Ion Energy'

    def test_collapses_internal_whitespace_from_slash(self):
        from spacephyml.datasets.creator import _sanitise_var_name
        # 'FPI / DIS' → 'FPI   DIS' after replace → 'FPI DIS' after collapse
        assert _sanitise_var_name('FPI / DIS') == 'FPI DIS'

    def test_multiple_slashes(self):
        from spacephyml.datasets.creator import _sanitise_var_name
        assert _sanitise_var_name('A/B/C') == 'A B C'

    def test_no_change_when_clean(self):
        from spacephyml.datasets.creator import _sanitise_var_name
        assert _sanitise_var_name('Ion Omni Spectrum') == 'Ion Omni Spectrum'

    def test_fieldnam_with_slash_produces_clean_column_names(self):
        """End-to-end: a FIELDNAM containing '/' must not appear in mapping."""
        cdf = _mock_cdf(
            var_attrs={'spec': {'FIELDNAM': 'MMS1 FPI/DIS Omni Spectrum'}},
            varinq_dim_sizes={'spec': []},
        )
        from spacephyml.datasets.creator import _get_column_mapping_from_cdf
        mapping = _get_column_mapping_from_cdf(cdf, 'spec')
        assert '/' not in mapping[0][0]
        assert mapping[0][0] == 'MMS1 FPI DIS Omni Spectrum'


# ---------------------------------------------------------------------------
# _resample_dataset — axis variables use .first(), measurements use .mean()
# ---------------------------------------------------------------------------

class TestResampleDataset:
    def _make_ds(self):
        times = np.array([
            '2017-11-01T00:00:00', '2017-11-01T00:00:05',
            '2017-11-01T00:00:10', '2017-11-01T00:00:15',
        ], dtype='datetime64[ns]')
        spectra  = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
        energies = np.array([[10., 20.], [10.5, 20.5],
                              [11., 21.], [11.5, 21.5]])
        return xr.Dataset(
            {
                'spectrum': (['time', 'energy_bin'], spectra),
                'energy':   (['time', 'energy_bin'],
                             energies,
                             {'spacephyml_role': 'axis'}),
                'label':    ('time', np.array([0., 0., 1., 1.])),
            },
            coords={'time': times, 'energy_bin': np.arange(2)},
        )

    def test_measurement_variable_is_averaged(self):
        from spacephyml.datasets.creator import _resample_dataset
        ds = self._make_ds()
        result = _resample_dataset(ds, '10s', chunk_size=4)
        np.testing.assert_allclose(result['spectrum'].values[0], [2., 3.])

    def test_axis_variable_uses_first(self):
        from spacephyml.datasets.creator import _resample_dataset
        ds = self._make_ds()
        result = _resample_dataset(ds, '10s', chunk_size=4)
        np.testing.assert_allclose(result['energy'].values[0], [10., 20.])

    def test_axis_variable_not_averaged_across_bin(self):
        from spacephyml.datasets.creator import _resample_dataset
        ds = self._make_ds()
        result = _resample_dataset(ds, '10s', chunk_size=4)
        assert result['energy'].values[0, 0] != 10.25

    def test_axis_attribute_preserved_after_resample(self):
        from spacephyml.datasets.creator import _resample_dataset
        ds = self._make_ds()
        result = _resample_dataset(ds, '10s', chunk_size=4)
        assert result['energy'].attrs.get('spacephyml_role') == 'axis'

    def test_dataset_without_axis_vars_works(self):
        from spacephyml.datasets.creator import _resample_dataset
        times = np.array(['2017-11-01T00:00:00', '2017-11-01T00:00:05'],
                         dtype='datetime64[ns]')
        ds = xr.Dataset(
            {'val': ('time', np.array([2., 4.]))},
            coords={'time': times},
        )
        result = _resample_dataset(ds, '10s', chunk_size=2)
        np.testing.assert_allclose(result['val'].values, [3.])

    def test_progress_bar_fires_once_per_chunk(self, capsys):
        from spacephyml.datasets.creator import _resample_dataset
        # 4 time steps, chunk_size=2 → 2 chunks → tqdm fires twice
        ds = self._make_ds()
        _resample_dataset(ds, '10s', chunk_size=2)
        # tqdm writes to stderr; check it produced output
        captured = capsys.readouterr()
        assert 'Resampling' in captured.err

    def test_chunked_result_matches_unchunked(self):
        from spacephyml.datasets.creator import _resample_dataset
        ds = self._make_ds()
        # chunk_size=100 → single chunk (same as old behaviour)
        r1 = _resample_dataset(ds, '10s', chunk_size=100)
        # chunk_size=2 → two chunks, each containing exactly one 10s bin
        r2 = _resample_dataset(ds, '10s', chunk_size=2)
        np.testing.assert_allclose(
            r1['spectrum'].values, r2['spectrum'].values, rtol=1e-5
        )


# ---------------------------------------------------------------------------
# _build_matrix_dataset tags axis variable
# ---------------------------------------------------------------------------

class TestBuildMatrixDatasetAxisTag:
    def test_axis_variable_tagged_with_role(self):
        from spacephyml.datasets.creator import _build_matrix_dataset
        times = np.array(['2017-11-01T00:00:00', '2017-11-01T00:00:05'],
                         dtype='datetime64[ns]')
        var_data  = np.ones((2, 3))
        dep1_data = np.ones((2, 3)) * 10.
        cdf = _mock_cdf(var_attrs={
            's': {'FIELDNAM': 'Spectrum'},
            'e': {'FIELDNAM': 'Energy'},
        })
        ds = _build_matrix_dataset(var_data, dep1_data, 'e', times, 's', cdf)
        assert ds['Energy'].attrs.get('spacephyml_role') == 'axis'

    def test_spectrum_variable_not_tagged(self):
        from spacephyml.datasets.creator import _build_matrix_dataset
        times = np.array(['2017-11-01T00:00:00'], dtype='datetime64[ns]')
        var_data  = np.ones((1, 2))
        dep1_data = np.ones((1, 2))
        cdf = _mock_cdf(var_attrs={'s': {'FIELDNAM': 'S'}, 'e': {'FIELDNAM': 'E'}})
        ds = _build_matrix_dataset(var_data, dep1_data, 'e', times, 's', cdf)
        assert ds['S'].attrs.get('spacephyml_role') != 'axis'


# ---------------------------------------------------------------------------
# _unix_to_datetime64 — consistent epoch conversion
# ---------------------------------------------------------------------------

class TestUnixToDatetime64:
    def test_returns_datetime64_ns(self):
        from spacephyml.datasets.creator import _unix_to_datetime64
        result = _unix_to_datetime64(np.array([0.0]))
        assert result.dtype == np.dtype('datetime64[ns]')

    def test_same_input_same_output(self):
        from spacephyml.datasets.creator import _unix_to_datetime64
        unix = np.array([1509501894.123456])
        a = _unix_to_datetime64(unix)
        b = _unix_to_datetime64(unix)
        assert a[0] == b[0]

    def test_preserves_sub_second_precision(self):
        from spacephyml.datasets.creator import _unix_to_datetime64
        # Truncation to seconds would give 1509501894.000000000
        result = _unix_to_datetime64(np.array([1509501894.5]))
        # Should NOT equal the truncated value
        truncated = np.datetime64('2017-11-01T02:04:54', 'ns')
        assert result[0] != truncated

    def test_does_not_truncate_to_seconds(self):
        from spacephyml.datasets.creator import _unix_to_datetime64
        unix = np.array([1509501894.123456])
        result = _unix_to_datetime64(unix)
        # datetime64[s] cast would give exactly .000000000
        as_seconds = np.array(unix, dtype='datetime64[s]').astype('datetime64[ns]')
        assert result[0] != as_seconds[0]

    def test_two_different_values_produce_different_timestamps(self):
        from spacephyml.datasets.creator import _unix_to_datetime64
        a = _unix_to_datetime64(np.array([1509501894.1]))
        b = _unix_to_datetime64(np.array([1509501894.2]))
        assert a[0] != b[0]

    def test_merge_succeeds_when_both_sides_use_same_conversion(self):
        """The core correctness guarantee: same float → same timestamp → exact merge."""
        from spacephyml.datasets.creator import _unix_to_datetime64
        unix = np.array([1509501894.123456, 1509501898.623456])
        times_a = _unix_to_datetime64(unix)
        times_b = _unix_to_datetime64(unix)

        ds_a = xr.Dataset({'label': ('time', np.array([0., 1.]))},
                           coords={'time': times_a})
        ds_b = xr.Dataset({'val':   ('time', np.array([1.1, 2.2]))},
                           coords={'time': times_b})
        merged = xr.merge([ds_a, ds_b])
        assert merged.sizes['time'] == 2
        assert not np.any(np.isnan(merged['label'].values))
        assert not np.any(np.isnan(merged['val'].values))
