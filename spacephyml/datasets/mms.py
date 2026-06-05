"""Specific MMS Datasets."""

from os import makedirs, path

import numpy as np
import xarray as xr
import pandas as pd

from .general.mms import ExternalMMSData
from .general.spectrum import BaseSpectrumDataset
from ..utils.file_download import missing_files, download_file_with_status
from ..transforms import IonDist_Transform


class MMS1IonDistLabeled(ExternalMMSData):
    """Ion distribution dataset labelled by Olshevsky et al. (2021)[^1].

    Contains two versions sampled from labels with the following classes:

    | Value | Label                  |
    | ----- | ---------------------- |
    | 0     | Solar Wind (SW)        |
    | 1     | Ion foreshock (IF)     |
    | 2     | Magnetosheath (MSH)    |
    | 3     | Magnetosphere (MSP)    |

    Each version contains 10,000 samples per label (40,000 total).

    [^1]: Olshevsky, V., et al. (2021). Automated classification of plasma
    regions using 3D particle energy distributions. Journal of Geophysical
    Research: Space Physics. https://doi.org/10.1029/2021JA029620

    Example:
        >>> from spacephyml.datasets.mms import MMS1IonDistLabeled
        >>> dataset = MMS1IonDistLabeled('SCDec2017')
    """

    _valid_datasets = ['SCNov2017', 'SCDec2017']
    _datasets = {
        'SCNov2017': {
            'url': 'https://zenodo.org/records/15147451/files/dataset_nov_2017_clean.csv?download=1',
            'file': 'dataset_nov_2017_clean.csv',
        },
        'SCDec2017': {
            'url': 'https://zenodo.org/records/15147451/files/dataset_dec_2017_clean.csv?download=1',
            'file': 'dataset_dec_2017_clean.csv',
        },
    }

    def __init__(self, dataset, path='./datasets', data_root=None,
                 transform=None, cache=True, return_epoch=False):
        """Initialise and (if necessary) download the dataset.

        Args:
            dataset (str): The dataset version to load. Must be one of
                ``'SCNov2017'`` or ``'SCDec2017'``.
            path (str): Directory used to store the downloaded CSV file.
            data_root (str | None): Override the default root directory for
                MMS data storage.
            transform (callable | None): Optional transform applied to each
                sample.
            cache (bool): Whether to cache loaded data in memory.
            return_epoch (bool): Whether to return the label epoch alongside
                each sample.
        """
        if dataset not in self._valid_datasets:
            raise ValueError(
                f'Incorrect dataset, {dataset} not in {self._valid_datasets}'
            )

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
    """PyTorch Dataset for MMS ion-spectrometer NetCDF4 files.

    Reads files produced by ``create_dataset`` with labels from Olshevsky et
    al. (2021)[^1]. Extends ``BaseSpectrumDataset`` with domain knowledge:

    - Reads ``.nc`` (recommended) or legacy ``.feather`` files.
    - Detects time-continuity gaps > 5 s to form independent segments.
    - Detects label-change boundaries to form homogeneous label groups.
    - Splits each group into fixed-length windows of size ``N``.
    - Drops windows shorter than ``N`` or with ``label == -1`` (with optional
      fill for ``-1`` runs surrounded by the same known label).
    - Falls back to unlabelled windowing when no valid labels exist, or when
      ``unlabelled=True``.

    The spectrum variable is auto-detected: any variable with both a ``time``
    and an ``energy_bin`` dimension is used. Pass ``data_columns`` explicitly
    to override (e.g. for legacy flat files with separate 1-D columns per bin).

    [^1]: Olshevsky, V., et al. (2021). Automated classification of plasma
    regions using 3D particle energy distributions. Journal of Geophysical
    Research: Space Physics. https://doi.org/10.1029/2021JA029620

    Attributes:
        label_str_map (dict[int, str]): Maps integer label values to their
            human-readable string names, when available in the source file.
    """

    # Minimum time gap between samples that defines a new independent segment.
    TIME_GAP_THRESHOLD = np.timedelta64(5_000_000_000, 'ns')  # 5 s in ns

    # Valid label classes; -1 (unknown) is excluded from labelled chunking.
    LABEL_CLASSES = [0, 1, 2, 3]

    def __init__(
        self,
        dataset_path: str,
        trange=None,
        N: int = 100,
        step: int = None,
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
        """Initialise the dataset from a NetCDF4 or feather file.

        Args:
            dataset_path (str): Path to the ``.nc`` or ``.feather`` dataset
                file.
            trange (tuple[str, str] | None): Optional ``(start, end)`` time
                strings (e.g. ``'2017-12-14T04:00:00'``) to restrict the
                loaded data. Both bounds are inclusive.
            N (int): Number of consecutive time steps per sample window.
            step (int | None): Step size for sliding-window chunking. ``None``
                (default) produces non-overlapping windows (equivalent to
                ``step=N``). When set, each window is shifted by ``step`` time
                steps relative to the previous one, producing overlapping
                windows.
            samples (int | None): Number of windows per class after balancing
                (labelled mode), or total windows drawn at random (unlabelled
                mode). ``None`` uses all available windows (still balanced to
                the smallest class in labelled mode).
            data_columns (list[str] | None): Feature variable name(s).
                ``None`` auto-detects the 2-D spectrum variable.
            flatten (bool): If ``True``, flatten each ``(N, n_bins)`` sample
                to a 1-D vector before storing.
            transform (callable | None): Optional transform applied to each
                raw ``(N, n_bins)`` numpy array.
                Signature: ``ndarray -> ndarray``.
            seed (int): Random seed used for class balancing and sampling.
            verbose (bool): If ``True``, print diagnostics during
                construction.
            fill_unknown (bool): If ``True``, contiguous runs of
                ``label == -1`` are filled with the surrounding label when
                both neighbouring known labels (within the same
                time-continuity group) are identical and the run length is
                below ``fill_unknown_max_ratio * N``. Runs at the edge of a
                time group, between differing labels, or exceeding the length
                limit remain ``-1`` and are excluded.
            fill_unknown_max_ratio (float): Maximum run length (as a fraction
                of ``N``) that may be filled when ``fill_unknown=True``.
                Defaults to ``0.25``, so only runs shorter than one quarter of
                the window size are eligible for filling.
            unlabelled (bool): If ``True``, ignore all label information and
                treat the dataset as unlabelled. All windows are pooled across
                time-continuity groups and ``samples`` are drawn at random.
                All ``spec_label`` values will be ``-1``. Also activated
                automatically when the file contains no ``label`` variable or
                every label is ``-1``.
        """
        self._fill_unknown           = fill_unknown
        self._fill_unknown_max_ratio = fill_unknown_max_ratio
        self._unlabelled             = unlabelled
        self._step                   = step
        ds = self._load(dataset_path, trange)

        # Auto-detect unlabelled mode: no label variable, or every value is -1.
        has_labels = (
            'label' in ds.data_vars and
            np.any(np.isin(ds['label'].values.astype(float),
                           [float(c) for c in self.LABEL_CLASSES]))
        )
        self._unlabelled = self._unlabelled or not has_labels

        # Build label-string map from the file when available.
        label_str_map = {}
        if not self._unlabelled and 'label str' in ds.data_vars:
            labels     = ds['label'].values
            label_strs = ds['label str'].values
            for label, label_str in zip(labels, label_strs):
                label_int = int(label)
                if label_int not in self.LABEL_CLASSES:
                    continue
                if label_int not in label_str_map and not np.isnan(label):
                    label_str_map[label_int] = str(label_str)
        self.label_str_map = label_str_map

        data_columns = self._resolve_columns(ds, data_columns)

        if self._unlabelled:
            chunks_input = {-1: self._chunk_unlabelled(ds, N, step)}
        else:
            chunks_input = self._chunk(ds, N, data_columns, step)

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
        """Load the dataset file and apply an optional time range filter.

        Supports ``.nc`` (NetCDF4) and ``.feather`` (legacy pandas) formats.

        Args:
            dataset_path (str): Path to the dataset file.
            trange (tuple[str, str] | None): Optional ``(start, end)`` time
                strings. Both bounds are inclusive.

        Returns:
            xr.Dataset: The loaded (and optionally sliced) dataset.

        Raises:
            ValueError: If the file extension is not ``.nc`` or ``.feather``.
        """
        _, ext = path.splitext(dataset_path)
        if ext == '.nc':
            ds = xr.open_dataset(dataset_path)
        elif ext == '.feather':
            ds = xr.Dataset.from_dataframe(pd.read_feather(dataset_path))
        else:
            raise ValueError(
                f'SpectrumDataset: unsupported file format {ext!r}. '
                'Use .nc (recommended) or .feather (legacy).'
            )

        # Normalise the time dimension name produced by legacy feather files.
        if 'time' not in ds.dims and 'index' in ds.dims:
            ds = ds.rename({'index': 'time'})

        if trange is not None:
            ds = ds.sel(time=slice(np.datetime64(trange[0]),
                                   np.datetime64(trange[1])))
        return ds

    def _resolve_columns(self, ds: xr.Dataset, data_columns) -> list:
        """Return the feature variable names to use from the dataset.

        If ``data_columns`` is provided it is returned unchanged. Otherwise,
        auto-detection is attempted in order:

        1. The first variable with both ``time`` and ``energy_bin`` dimensions
           (the 2-D spectrum variable produced by ``create_dataset``).
        2. All non-label, non-metadata 1-D variables (legacy flat format).

        Args:
            ds (xr.Dataset): The loaded dataset.
            data_columns (list[str] | None): Explicit column names, or
                ``None`` to trigger auto-detection.

        Returns:
            list[str]: The resolved list of feature variable names.
        """
        if data_columns is not None:
            return data_columns

        # Prefer a single 2-D (time, energy_bin) spectrum variable.
        for var in ds.data_vars:
            if set(ds[var].dims) == {'time', 'energy_bin'}:
                return [var]

        # Fall back to all non-label 1-D variables for legacy flat files.
        skip = {'label', 'label str', 'label_str'}
        return [v for v in ds.data_vars
                if v not in skip and ds[v].dims == ('time',)]

    def _chunk_unlabelled(self, ds: xr.Dataset, N: int, step: int = None) -> list:
        """Produce sliding windows from every time-continuity group.

        Label information is ignored entirely. Windows never cross a gap
        larger than ``TIME_GAP_THRESHOLD``, so each independent segment is
        windowed separately. The returned list is a flat pool from which
        ``BaseSpectrumDataset`` draws ``samples`` windows at random.

        Args:
            ds (xr.Dataset): The loaded dataset.
            N (int): Window size in time steps.
            step (int | None): Sliding-window step size. ``None`` produces
                non-overlapping windows (equivalent to ``step=N``).

        Returns:
            list[xr.Dataset]: All valid ``N``-step windows in time order.
        """
        stride      = N if step is None else step
        time_vals   = ds['time'].values.astype('datetime64[ns]')
        time_diff   = np.diff(time_vals, prepend=time_vals[0])
        time_groups = np.cumsum(time_diff > self.TIME_GAP_THRESHOLD)

        chunks = []
        for gid in np.unique(time_groups):
            sub     = ds.isel(time=time_groups == gid)
            n_times = sub.sizes['time']
            for start in range(0, n_times - N + 1, stride):
                chunks.append(sub.isel(time=slice(start, start + N)))
        return chunks

    def _fill_unknown_labels(
        self, labels: np.ndarray, time_groups: np.ndarray, N: int
    ) -> np.ndarray:
        """Fill short ``-1`` runs that are bracketed by the same known label.

        A run of consecutive ``-1`` values is replaced with the surrounding
        label when **all** of the following conditions hold:

        - The nearest known label immediately before the run (within the same
          time-continuity group) exists and equals the nearest known label
          immediately after the run.
        - Neither boundary is missing (i.e. the run does not touch the edge
          of its time-continuity group).
        - The run length is strictly less than
          ``self._fill_unknown_max_ratio * N``.

        Runs that fail any condition are left as ``-1``.

        Args:
            labels (np.ndarray): 1-D array of integer labels (``float`` dtype,
                with ``-1`` indicating unknown).
            time_groups (np.ndarray): 1-D integer array of the same length as
                ``labels``, where equal values indicate the same
                time-continuity group.
            N (int): Window size, used to compute the maximum fillable run
                length via ``self._fill_unknown_max_ratio``.

        Returns:
            np.ndarray: A copy of ``labels`` with eligible ``-1`` runs filled.
        """
        labels = labels.copy()
        n = len(labels)
        i = 0
        while i < n:
            if labels[i] != -1:
                i += 1
                continue

            # Find the full extent of this -1 run.
            run_start = i
            while i < n and labels[i] == -1:
                i += 1
            run_end = i  # exclusive; labels[run_start:run_end] are all -1

            # Walk backward for the nearest known label before the run.
            label_before = None
            for j in range(run_start - 1, -1, -1):
                if time_groups[j] != time_groups[run_start]:
                    break  # crossed into a different time-continuity group
                if labels[j] != -1:
                    label_before = labels[j]
                    break

            # Walk forward for the nearest known label after the run.
            label_after = None
            for j in range(run_end, n):
                if time_groups[j] != time_groups[run_start]:
                    break  # crossed into a different time-continuity group
                if labels[j] != -1:
                    label_after = labels[j]
                    break

            # Fill only when both neighbours exist, agree, and run is short.
            run_length = run_end - run_start
            if (label_before is not None and
                    label_after is not None and
                    label_before == label_after and
                    run_length < self._fill_unknown_max_ratio * N):
                labels[run_start:run_end] = label_before

        return labels

    def _chunk(
        self, ds: xr.Dataset, N: int, data_columns: list, step: int = None
    ) -> dict:
        """Split the dataset into windows grouped by time continuity and label.

        Time-continuity groups are formed by gaps larger than
        ``TIME_GAP_THRESHOLD``. Within each group, windows are further split
        at label-change boundaries so that every window has a single
        homogeneous label.

        If ``self._fill_unknown`` is ``True``, eligible ``-1`` runs are
        relabelled by ``_fill_unknown_labels`` before the label grouping step.

        Args:
            ds (xr.Dataset): The loaded dataset.
            N (int): Window size in time steps.
            data_columns (list[str]): Feature variable names (not used in
                chunking itself, kept for API symmetry with subclasses).
            step (int | None): Sliding-window step size. ``None`` produces
                non-overlapping windows (equivalent to ``step=N``).

        Returns:
            dict[int, list[xr.Dataset]]: Mapping from label integer to the
                list of ``N``-step windows carrying that label.
        """
        stride      = N if step is None else step
        time_vals   = ds['time'].values.astype('datetime64[ns]')
        time_diff   = np.diff(time_vals, prepend=time_vals[0])
        time_groups = np.cumsum(time_diff > self.TIME_GAP_THRESHOLD)

        labels = ds['label'].values.astype(float)
        if self._fill_unknown:
            labels = self._fill_unknown_labels(labels, time_groups, N)

        label_diff   = np.concatenate([[0], np.diff(labels)])
        label_groups = np.cumsum(label_diff != 0)

        chunks_by_label: dict[int, list] = {k: [] for k in self.LABEL_CLASSES}

        # Combine time and label group IDs into a single key for grouping.
        combined = time_groups * (label_groups.max() + 1) + label_groups
        for gid in np.unique(combined):
            mask    = combined == gid
            sub     = ds.isel(time=mask)
            label   = int(labels[mask][0])
            n_times = sub.sizes['time']
            if n_times < N or label == -1:
                continue
            for start in range(0, n_times - N + 1, stride):
                chunk = sub.isel(time=slice(start, start + N))
                if label in chunks_by_label:
                    chunks_by_label[label].append(chunk)

        return chunks_by_label
