"""
Tests for dataset storage and reading (create_dataset / xarray_read_file).
Covers the NetCDF4 native path and the legacy .csv/.feather paths.
"""
import warnings
import numpy as np
import xarray as xr
import pytest
import tempfile
import os

from spacephyml.utils import xarray_read_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset():
    """Build a representative xr.Dataset similar to what create_dataset produces."""
    times = np.array(
        ['2017-11-01T00:00:00', '2017-11-01T00:00:05', '2017-11-01T00:00:10'],
        dtype='datetime64[ns]',
    )
    return xr.Dataset(
        {
            'Ion Spec. 0':  ('time', np.array([1.1, 2.2, 3.3])),
            'Ion Spec. 1':  ('time', np.array([4.4, 5.5, 6.6])),
            'label':        ('time', np.array([0, 1, 0])),
            'label str':    ('time', np.array(['Solar Wind', 'Magnetosheath', 'Solar Wind'])),
        },
        coords={'time': times},
    )


# ---------------------------------------------------------------------------
# NetCDF4 round-trip via xarray_read_file
# ---------------------------------------------------------------------------

class TestNetCDFRoundTrip:
    def test_nc_preserves_time_coordinate(self):
        ds = _make_dataset()
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            fname = f.name
        try:
            ds.to_netcdf(fname)
            ds2 = xarray_read_file(fname)
            assert 'time' in ds2.coords
            np.testing.assert_array_equal(ds2.time.values, ds.time.values)
        finally:
            os.unlink(fname)

    def test_nc_preserves_all_data_vars(self):
        ds = _make_dataset()
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            fname = f.name
        try:
            ds.to_netcdf(fname)
            ds2 = xarray_read_file(fname)
            for var in ds.data_vars:
                assert var in ds2.data_vars
        finally:
            os.unlink(fname)

    def test_nc_preserves_numeric_values(self):
        ds = _make_dataset()
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            fname = f.name
        try:
            ds.to_netcdf(fname)
            ds2 = xarray_read_file(fname)
            np.testing.assert_allclose(
                ds2['Ion Spec. 0'].values, ds['Ion Spec. 0'].values
            )
            np.testing.assert_array_equal(
                ds2['label'].values, ds['label'].values
            )
        finally:
            os.unlink(fname)

    def test_nc_preserves_string_variable(self):
        ds = _make_dataset()
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            fname = f.name
        try:
            ds.to_netcdf(fname)
            ds2 = xarray_read_file(fname)
            assert str(ds2['label str'].values[0]) == 'Solar Wind'
            assert str(ds2['label str'].values[1]) == 'Magnetosheath'
        finally:
            os.unlink(fname)

    def test_nc_dimension_sizes_match(self):
        ds = _make_dataset()
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            fname = f.name
        try:
            ds.to_netcdf(fname)
            ds2 = xarray_read_file(fname)
            assert ds2.sizes['time'] == 3
        finally:
            os.unlink(fname)


# ---------------------------------------------------------------------------
# Legacy .csv and .feather reading (backward compatibility)
# ---------------------------------------------------------------------------

class TestLegacyFormats:
    def test_csv_readable(self, tmp_path):
        import pandas as pd
        df = _make_dataset().to_dataframe()
        fpath = str(tmp_path / 'data.csv')
        df.to_csv(fpath)
        ds = xarray_read_file(fpath)
        assert isinstance(ds, xr.Dataset)
        assert 'label' in ds.data_vars

    def test_feather_readable(self, tmp_path):
        import pandas as pd
        df = _make_dataset().to_dataframe().reset_index()
        fpath = str(tmp_path / 'data.feather')
        df.to_feather(fpath)
        ds = xarray_read_file(fpath)
        assert isinstance(ds, xr.Dataset)
        assert 'label' in ds.data_vars

    def test_unknown_extension_raises(self, tmp_path):
        fpath = str(tmp_path / 'data.parquet')
        open(fpath, 'w').close()
        with pytest.raises(ValueError, match='Unknown filetype'):
            xarray_read_file(fpath)


# ---------------------------------------------------------------------------
# create_dataset writes .nc and the file is re-readable
# ---------------------------------------------------------------------------

class TestCreateDatasetNC:
    def test_nc_file_is_created(self, tmp_path, monkeypatch):
        """create_dataset with .nc extension produces a readable NetCDF file."""
        import datetime as dt
        from spacephyml.datasets.creator import create_dataset

        fake_ds = _make_dataset()
        monkeypatch.setattr(
            'spacephyml.datasets.creator.get_dataset',
            lambda **kw: fake_ds,
        )

        fpath = str(tmp_path / 'out.nc')
        create_dataset(fpath, trange=['2017-11-01', '2017-11-02'])

        assert os.path.isfile(fpath)
        ds2 = xr.open_dataset(fpath)
        assert 'label' in ds2.data_vars
        assert ds2.sizes['time'] == 3

    def test_csv_write_emits_warning(self, tmp_path, monkeypatch):
        from spacephyml.datasets.creator import create_dataset

        monkeypatch.setattr(
            'spacephyml.datasets.creator.get_dataset',
            lambda **kw: _make_dataset(),
        )
        fpath = str(tmp_path / 'out.csv')
        with pytest.warns(UserWarning, match='lossy'):
            create_dataset(fpath, trange=['2017-11-01', '2017-11-02'])
        assert os.path.isfile(fpath)

    def test_feather_write_emits_warning(self, tmp_path, monkeypatch):
        from spacephyml.datasets.creator import create_dataset

        monkeypatch.setattr(
            'spacephyml.datasets.creator.get_dataset',
            lambda **kw: _make_dataset(),
        )
        fpath = str(tmp_path / 'out.feather')
        with pytest.warns(UserWarning, match='lossy'):
            create_dataset(fpath, trange=['2017-11-01', '2017-11-02'])
        assert os.path.isfile(fpath)

    def test_unknown_ext_raises(self, tmp_path, monkeypatch):
        from spacephyml.datasets.creator import create_dataset

        monkeypatch.setattr(
            'spacephyml.datasets.creator.get_dataset',
            lambda **kw: _make_dataset(),
        )
        fpath = str(tmp_path / 'out.zarr')
        with pytest.raises(ValueError, match='Unknown filetype'):
            create_dataset(fpath, trange=['2017-11-01', '2017-11-02'])


# ---------------------------------------------------------------------------
# SpectrumDataset._load supports .nc
# ---------------------------------------------------------------------------

class TestSpectrumDatasetLoad:
    def test_load_nc_returns_dataset_with_time_dim(self, tmp_path):
        from spacephyml.datasets.mms import SpectrumDataset

        ds = _make_dataset()
        fpath = str(tmp_path / 'spec.nc')
        ds.to_netcdf(fpath)

        loader = SpectrumDataset.__new__(SpectrumDataset)
        result = loader._load(fpath, trange=None)

        assert isinstance(result, xr.Dataset)
        assert 'time' in result.dims

    def test_load_nc_applies_trange(self, tmp_path):
        from spacephyml.datasets.mms import SpectrumDataset

        ds = _make_dataset()
        fpath = str(tmp_path / 'spec.nc')
        ds.to_netcdf(fpath)

        loader = SpectrumDataset.__new__(SpectrumDataset)
        # trange is exclusive on both ends; 00:00:00 < t < 00:00:10
        # selects only the middle time step at 00:00:05
        result = loader._load(fpath, trange=('2017-11-01T00:00:00', '2017-11-01T00:00:10'))

        assert result.sizes['time'] == 1
        assert str(result.time.values[0])[:19] == '2017-11-01T00:00:05'

    def test_load_unsupported_ext_raises(self, tmp_path):
        from spacephyml.datasets.mms import SpectrumDataset

        fpath = str(tmp_path / 'spec.parquet')
        open(fpath, 'w').close()

        loader = SpectrumDataset.__new__(SpectrumDataset)
        with pytest.raises(ValueError, match='unsupported file format'):
            loader._load(fpath, trange=None)


# ---------------------------------------------------------------------------
# get_dataset balanced sampling — pure xarray, no pandas round-trip
# ---------------------------------------------------------------------------

class TestGetDatasetSampling:
    def _make_labeled_dataset(self):
        """4 time steps: 2 × label 0, 2 × label 1."""
        times = np.array(
            ['2017-11-01T00:00:00', '2017-11-01T00:00:05',
             '2017-11-01T00:00:10', '2017-11-01T00:00:15'],
            dtype='datetime64[ns]',
        )
        return xr.Dataset(
            {
                'val':   ('time', np.array([1.0, 2.0, 3.0, 4.0])),
                'label': ('time', np.array([0, 0, 1, 1])),
            },
            coords={'time': times},
        )

    def test_sampling_returns_xarray_dataset(self, monkeypatch):
        from spacephyml.datasets.creator import get_dataset
        ds = self._make_labeled_dataset()
        monkeypatch.setattr('spacephyml.datasets.creator._LABEL_SOURCES',
                            {'Fake': lambda *a, **k: ds})
        result = get_dataset('Fake', ['2017-11-01', '2017-11-02'],
                             clean=False, samples=1)
        assert isinstance(result, xr.Dataset)

    def test_sampling_draws_correct_count_per_label(self, monkeypatch):
        from spacephyml.datasets.creator import get_dataset
        ds = self._make_labeled_dataset()
        monkeypatch.setattr('spacephyml.datasets.creator._LABEL_SOURCES',
                            {'Fake': lambda *a, **k: ds})
        result = get_dataset('Fake', ['2017-11-01', '2017-11-02'],
                             clean=False, samples=1)
        labels = result['label'].values
        assert (labels == 0).sum() == 1
        assert (labels == 1).sum() == 1

    def test_sampling_preserves_time_coordinate(self, monkeypatch):
        from spacephyml.datasets.creator import get_dataset
        ds = self._make_labeled_dataset()
        monkeypatch.setattr('spacephyml.datasets.creator._LABEL_SOURCES',
                            {'Fake': lambda *a, **k: ds})
        result = get_dataset('Fake', ['2017-11-01', '2017-11-02'],
                             clean=False, samples=1)
        # time coordinate must still be datetime64, not a plain integer index
        assert result['time'].dtype.kind == 'M'

    def test_sampling_raises_when_not_enough_samples(self, monkeypatch):
        from spacephyml.datasets.creator import get_dataset
        ds = self._make_labeled_dataset()
        monkeypatch.setattr('spacephyml.datasets.creator._LABEL_SOURCES',
                            {'Fake': lambda *a, **k: ds})
        with pytest.raises(ValueError, match='Not enough samples'):
            get_dataset('Fake', ['2017-11-01', '2017-11-02'],
                        clean=False, samples=10)


# ---------------------------------------------------------------------------
# SpectrumDataset — 2D format and auto-detection
# ---------------------------------------------------------------------------

class TestSpectrumDatasetNew:
    def _make_nc_file(self, tmp_path):
        """Create a minimal .nc file matching the new 2D format."""
        n_time, n_bins = 40, 4
        times = np.array(
            [np.datetime64('2017-11-01') + np.timedelta64(i * 5, 's')
             for i in range(n_time)],
            dtype='datetime64[ns]',
        )
        labels = np.array([0]*10 + [1]*10 + [2]*10 + [3]*10, dtype=np.float32)
        spectra = np.random.rand(n_time, n_bins).astype(np.float32)
        energies = np.random.rand(n_time, n_bins).astype(np.float32)

        ds = xr.Dataset(
            {
                'Ion Omni Spectrum': (['time', 'energy_bin'], spectra),
                'Ion Energy':        (['time', 'energy_bin'], energies,
                                      {'spacephyml_role': 'axis'}),
                'label':             ('time', labels),
            },
            coords={
                'time':       times,
                'energy_bin': np.arange(n_bins),
            },
        )
        fpath = str(tmp_path / 'spectrum.nc')
        ds.to_netcdf(fpath)
        return fpath, ds

    def test_auto_detects_2d_spectrum_variable(self, tmp_path):
        from spacephyml.datasets.mms import SpectrumDataset
        fpath, _ = self._make_nc_file(tmp_path)
        loader = SpectrumDataset.__new__(SpectrumDataset)
        ds = loader._load(fpath, trange=None)
        cols = loader._resolve_columns(ds, None)
        assert cols == ['Ion Omni Spectrum']

    def test_explicit_data_columns_respected(self, tmp_path):
        from spacephyml.datasets.mms import SpectrumDataset
        fpath, _ = self._make_nc_file(tmp_path)
        loader = SpectrumDataset.__new__(SpectrumDataset)
        ds = loader._load(fpath, trange=None)
        cols = loader._resolve_columns(ds, ['Ion Energy'])
        assert cols == ['Ion Energy']

    def test_chunk_uses_time_dimension(self, tmp_path):
        from spacephyml.datasets.mms import SpectrumDataset
        fpath, _ = self._make_nc_file(tmp_path)
        loader = SpectrumDataset.__new__(SpectrumDataset)
        ds = loader._load(fpath, trange=None)
        chunks = loader._chunk(ds, N=5, data_columns=['Ion Omni Spectrum'])
        assert any(len(v) > 0 for v in chunks.values())

    def test_chunk_items_have_energy_bin_dimension(self, tmp_path):
        from spacephyml.datasets.mms import SpectrumDataset
        fpath, _ = self._make_nc_file(tmp_path)
        loader = SpectrumDataset.__new__(SpectrumDataset)
        ds = loader._load(fpath, trange=None)
        chunks = loader._chunk(ds, N=5, data_columns=['Ion Omni Spectrum'])
        first = next(c for v in chunks.values() for c in v)
        assert 'energy_bin' in first['Ion Omni Spectrum'].dims

    def test_full_construction_returns_pytorch_dataset(self, tmp_path):
        from spacephyml.datasets.mms import SpectrumDataset
        from torch.utils.data import DataLoader
        fpath, _ = self._make_nc_file(tmp_path)
        ds = SpectrumDataset(fpath, N=5, samples=2, verbose=False)
        assert len(ds) == 8   # 4 classes × 2 samples
        x, y = ds[0]
        assert x.shape == (5 * 4,)   # flattened: N × n_bins
        assert y.dtype == torch.long

    def test_bin_centers_stored_on_dataset(self, tmp_path):
        from spacephyml.datasets.mms import SpectrumDataset
        fpath, _ = self._make_nc_file(tmp_path)
        ds = SpectrumDataset(fpath, N=5, samples=2, verbose=False)
        assert ds.bin_centers is not None
        # shape: (n_samples, N, n_bins)
        assert ds.bin_centers.shape == (8, 5, 4)

    def test_bin_centers_representative_axis(self, tmp_path):
        from spacephyml.datasets.mms import SpectrumDataset
        fpath, _ = self._make_nc_file(tmp_path)
        ds = SpectrumDataset(fpath, N=5, samples=2, verbose=False)
        # A single representative axis for plotting: bin_centers[0, 0]
        assert ds.bin_centers[0, 0].shape == (4,)

    def test_unflatten_gives_2d_sample(self, tmp_path):
        from spacephyml.datasets.mms import SpectrumDataset
        fpath, _ = self._make_nc_file(tmp_path)
        ds = SpectrumDataset(fpath, N=5, samples=2, flatten=False, verbose=False)
        x, _ = ds[0]
        assert x.shape == (5, 4)   # (N, n_bins)



import torch
