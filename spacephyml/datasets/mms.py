"""
Specific MMS Datasets.
"""

from os import makedirs, path

import numpy as np
import xarray as xr
import pandas as pd

from .general.mms import ExternalMMSData
from .general.spectrum import BaseSpectrumDataset
from ..utils.file_download import missing_files, download_file_with_status
from ..transforms import IonDist_Transform


class MMS1IonDistLabeled(ExternalMMSData):
    """
    This dataset contains two versions samled from labels created by Olshevsky, et. al. (2021)[^1]. The data samples in this dataset have one of the following labels:

    | Value  | Label |
    | -- | ---------------- |
    | 0 | Solar Wind (SW) |
    | 1 | Ion foreshock (IF) |
    | 2 | Magnetosheath (MSH) |
    | 3 | Magnetosphere (MSP) |

    There are 10,000 samples for each label, for a total of 40,000 samples in each version of the dataset.

    [^1]: Olshevsky, V., et al. (2021). Automated classification of plasma regions using 3D particle energy distributions. Journal of Geophysical Research: Space Physics, https://doi.org/10.1029/2021JA029620

    Examples:
        >>> from spacephyml.datasets.mms import MMS1IonDistLabeled
        >>> dataset = MMS1IonDistLabeled('SCDec017')

    """

    _valid_datasets = ['SCNov2017', 'SCDec2017']
    _datasets = {'SCNov2017': {
                    'url': 'https://zenodo.org/records/15147451/files/dataset_nov_2017_clean.csv?download=1',
                    'file': 'dataset_nov_2017_clean.csv'
                },
                'SCDec2017': {
                    'url': 'https://zenodo.org/records/15147451/files/dataset_dec_2017_clean.csv?download=1',
                    'file': 'dataset_dec_2017_clean.csv'
                }}

    def __init__(self, dataset, path='./datasets', data_root=None,
                 transform=None, cache=True, return_epoch=False):
        """

        Args:
            dataset (string):
                The dataset, either SCNov2017 or SCDec2017.
            path (string):
                The path for storing the dataset (not the actuall data).
            data_root (string):
                The override the default root directory to for the MMS data
                storage.
            transform (callable):
                Optional transform to be applied on each sample.
            cache (bool):
                If data should be cached.
            return_epoch (bool):
                If the label epoch should be returned.
        """
        if dataset not in self._valid_datasets:
            raise ValueError(f'Incorrect dataset, {dataset} not in' +
                             '{self._valid_datasets}')

        filepath = f'{path}/' + self._datasets[dataset]['file']

        missing = missing_files([filepath], './')
        if missing:
            print('Missing dataset file, downloading')
            makedirs(path, exist_ok=True)
            download_file_with_status(self._datasets[dataset]['url'], filepath)

        if transform is None:
            transform = IonDist_Transform()

        super().__init__(filepath, data_root, transform, cache, return_epoch)


class SpectrumDataset(BaseSpectrumDataset):
    """
    PyTorch Dataset for MMS ion-spectrometer NetCDF4 files produced by
    ``create_dataset``. Labels are from Olshevsky, et. al. (2021)[^1].

    Extends BaseSpectrumDataset with domain knowledge:
      - reads a .nc (or legacy .feather) file
      - detects time-continuity gaps > 5 s to form independent segments
      - detects label-change boundaries to form homogeneous label groups
      - splits each group into fixed-length chunks of size N
      - drops chunks shorter than N or with label == -1 (with optional fill for surrounded -1 runs)
      - falls back to unlabelled windowing when no valid labels exist, or when ``unlabelled=True``

    The spectrum variable is auto-detected from the file: any variable with
    both a ``time`` and an ``energy_bin`` dimension is treated as the spectrum.
    Pass ``data_columns`` explicitly to override (e.g. for legacy flat files
    that have separate 1D columns per bin).

    [^1]: Olshevsky, V., et al. (2021). Automated classification of plasma regions using 3D particle energy distributions. Journal of Geophysical Research: Space Physics, https://doi.org/10.1029/2021JA029620

    """

    # Gap threshold that defines a new independent time segment
    TIME_GAP_THRESHOLD = np.timedelta64(5_000_000_000, 'ns')  # 5 seconds in ns

    # Label classes expected in the data (-1 is excluded as unlabelled)
    LABEL_CLASSES = [0, 1, 2, 3]

    def __init__(
        self,
        dataset_path: str,
        trange=None,
        N: int = 100,
        samples: int = 100,
        data_columns: list = None,
        flatten: bool = True,
        transform=None,
        seed: int = 42,
        verbose: bool = True,
        fill_unknown: bool = False,
        fill_unknown_max_ratio: float = 0.25,
        unlabelled: bool = False,
    ):
        """
        Args:
            dataset_path : str
                Path to the .nc or .feather dataset file.
            trange : tuple[str, str] | None
                Optional (start, end) time strings to restrict the data.
            N : int
                Number of consecutive time steps per sample chunk.
            samples : int | None
                Chunks per class after balancing.  None = use all.
            data_columns : list[str] | None
                Feature variable name(s).  None = auto-detect the 2D spectrum variable.
            flatten : bool
                Flatten each (N, n_bins) sample to 1-D when True.
            transform : callable | None
                Optional transform applied to each raw (N, n_bins) numpy array.
            seed : int
                Random seed for class balancing.
            verbose : bool
                Print diagnostics during construction.
            fill_unknown : bool
                If True, contiguous runs of label ``-1`` are filled with the
                surrounding label when both neighbouring known labels (within the
                same time-continuity group) are identical.  Runs at the edge of a
                time group or between two different labels remain ``-1``.
            fill_unknown_max_ratio : float
                Only used when ``fill_unknown=True``.  A ``-1`` run is filled
                only if its length is strictly less than
                ``fill_unknown_max_ratio * N``.  Defaults to ``0.25``, meaning
                runs longer than a quarter of the window size are left as
                ``-1`` even when both neighbours agree.
            unlabelled : bool
                If True, ignore any label information and treat the dataset as
                unlabelled: all windows across all time-continuity groups are
                pooled and ``samples`` of them are drawn at random.  All
                ``spec_label`` values will be ``-1``.  This is also activated
                automatically when the file contains no ``label`` variable, or
                when every label value is ``-1``.
        """
        self._fill_unknown           = fill_unknown
        self._fill_unknown_max_ratio  = fill_unknown_max_ratio
        self._unlabelled              = unlabelled
        ds = self._load(dataset_path, trange)

        # Auto-detect unlabelled: no label variable, or every value is -1.
        has_labels = (
            'label' in ds.data_vars and
            np.any(np.isin(ds['label'].values.astype(float),
                           [float(c) for c in self.LABEL_CLASSES]))
        )
        self._unlabelled = self._unlabelled or not has_labels

        # Create a mapping from label integers to label strings if available.
        label_str_map = {}
        if not self._unlabelled and 'label str' in ds.data_vars:
            labels      = ds['label'].values
            label_strs  = ds['label str'].values
            for label, label_str in zip(labels, label_strs):
                label_int = int(label)
                if label_int not in self.LABEL_CLASSES:
                    continue
                if label_int not in label_str_map and not np.isnan(label):
                    label_str_map[label_int] = str(label_str)
        self.label_str_map = label_str_map

        data_columns = self._resolve_columns(ds, data_columns)

        if self._unlabelled:
            chunks = self._chunk_unlabelled(ds, N)
            chunks_input = {-1: chunks}
        else:
            chunks_input = self._chunk(ds, N, data_columns)

        super().__init__(
            chunks_by_label=chunks_input,
            data_columns=data_columns,
            N=N,
            samples=samples,
            flatten=flatten,
            transform=transform,
            seed=seed,
            verbose=verbose,
        )

    # ---------------------------------------------------------------------- #
    # Domain-specific helpers                                                  #
    # ---------------------------------------------------------------------- #

    def _load(self, dataset_path: str, trange) -> xr.Dataset:
        """
        Load the dataset file and apply an optional time range filter.

        Supports:
          - ``.nc``      — NetCDF4, loaded directly via xr.open_dataset.
          - ``.feather`` — Legacy format; loaded via pandas and converted.
        """
        _, ext = path.splitext(dataset_path)
        if ext == '.nc':
            ds = xr.open_dataset(dataset_path)
        elif ext == '.feather':
            ds = xr.Dataset.from_dataframe(pd.read_feather(dataset_path))
        else:
            raise ValueError(f'SpectrumDataset: unsupported file format {ext!r}. '
                             'Use .nc (recommended) or .feather (legacy).')

        # Normalise the time dimension name
        if 'time' not in ds.dims and 'index' in ds.dims:
            ds = ds.rename({'index': 'time'})

        if trange is not None:
            ds = ds.sel(time=(
                (ds.time > np.datetime64(trange[0])) &
                (ds.time < np.datetime64(trange[1]))
            ))
        return ds

    def _resolve_columns(self, ds: xr.Dataset, data_columns) -> list:
        """
        Return the list of variable names to use as features.

        If ``data_columns`` is provided it is returned as-is.
        Otherwise, auto-detect: find the first variable that has both
        ``time`` and ``energy_bin`` dimensions (the 2D spectrum variable
        produced by ``create_dataset``).  Fall back to a list of all
        non-label 1D variables if no 2D variable is found.
        """
        if data_columns is not None:
            return data_columns

        # Look for a 2D (time, energy_bin) spectrum variable
        for var in ds.data_vars:
            if set(ds[var].dims) == {'time', 'energy_bin'}:
                return [var]

        # Legacy flat format: use all non-label, non-metadata 1D variables
        skip = {'label', 'label str', 'label_str'}
        return [v for v in ds.data_vars
                if v not in skip and ds[v].dims == ('time',)]

    def _chunk_unlabelled(self, ds: xr.Dataset, N: int) -> list:
        """
        Produce consecutive N-step windows from every time-continuity group,
        ignoring any label information.

        Windows never cross a gap larger than ``TIME_GAP_THRESHOLD``, so each
        independent segment is windowed independently.  The returned list is a
        flat pool; ``BaseSpectrumDataset`` will random-sample from it.

        Returns:
            list[xr.Dataset]
                All valid N-step chunks in time order.
        """
        time_vals = ds['time'].values.astype('datetime64[ns]')
        time_diff = np.diff(time_vals, prepend=time_vals[0])
        time_groups = np.cumsum(time_diff > self.TIME_GAP_THRESHOLD)

        chunks = []
        for gid in np.unique(time_groups):
            sub = ds.isel(time=time_groups == gid)
            num_chunks = sub.sizes['time'] // N
            for i in range(num_chunks):
                chunks.append(sub.isel(time=slice(i * N, (i + 1) * N)))
        return chunks

    def _fill_unknown_labels(
        self, labels: np.ndarray, time_groups: np.ndarray, N: int
    ) -> np.ndarray:
        """
        Return a copy of ``labels`` where runs of ``-1`` have been filled in
        when they are unambiguously bracketed by the same known label within
        a single time-continuity group.

        A run is filled when ALL of the following hold:
          - the nearest known label **before** the run (in the same time group)
            exists and equals the nearest known label **after** the run.
          - neither boundary is missing (i.e. the run does not touch the edge
            of its time group).
          - the run length is strictly less than
            ``self._fill_unknown_max_ratio * N``.

        Any run that fails these conditions is left as ``-1``.
        """
        labels = labels.copy()
        n = len(labels)
        i = 0
        while i < n:
            if labels[i] != -1:
                i += 1
                continue

            # ---- find the extent of this -1 run ----
            run_start = i
            while i < n and labels[i] == -1:
                i += 1
            run_end = i  # exclusive; labels[run_start:run_end] are all -1

            # ---- find the nearest known label before the run ----
            label_before = None
            for j in range(run_start - 1, -1, -1):
                if time_groups[j] != time_groups[run_start]:
                    break           # crossed into a different time group
                if labels[j] != -1:
                    label_before = labels[j]
                    break

            # ---- find the nearest known label after the run ----
            label_after = None
            for j in range(run_end, n):
                if time_groups[j] != time_groups[run_start]:
                    break           # crossed into a different time group
                if labels[j] != -1:
                    label_after = labels[j]
                    break

            # ---- fill only if both neighbours exist, agree, and run is short ----
            run_length = run_end - run_start
            if (label_before is not None and
                    label_after is not None and
                    label_before == label_after and
                    run_length < self._fill_unknown_max_ratio * N):
                labels[run_start:run_end] = label_before

        return labels

    def _chunk(self, ds: xr.Dataset, N: int, data_columns: list) -> dict:
        """
        Split ds into fixed-length chunks grouped by time continuity and
        label homogeneity.

        If ``self._fill_unknown`` is True, contiguous ``-1`` runs that are
        surrounded on both sides (within the same time group) by the same
        known label are relabelled before grouping.  Edge runs and runs
        between differing labels remain ``-1`` and are excluded.

        Returns:
            dict[int, list[xr.Dataset]]
                One key per label class; each value is a list of N-step chunks.
        """
        time_vals = ds['time'].values.astype('datetime64[ns]')
        time_diff = np.diff(time_vals, prepend=time_vals[0])
        time_groups = np.cumsum(time_diff > self.TIME_GAP_THRESHOLD)

        labels = ds['label'].values.astype(float)

        if self._fill_unknown:
            labels = self._fill_unknown_labels(labels, time_groups, N)

        label_diff = np.concatenate([[0], np.diff(labels)])
        label_groups = np.cumsum(label_diff != 0)

        chunks_by_label: dict[int, list] = {k: [] for k in self.LABEL_CLASSES}

        combined = time_groups * (label_groups.max() + 1) + label_groups
        for gid in np.unique(combined):
            mask = combined == gid
            sub = ds.isel(time=mask)
            label = int(labels[mask][0])
            if sub.sizes['time'] < N or label == -1:
                continue
            num_chunks = sub.sizes['time'] // N
            for i in range(num_chunks):
                chunk = sub.isel(time=slice(i * N, (i + 1) * N))
                if label in chunks_by_label:
                    chunks_by_label[label].append(chunk)

        return chunks_by_label
