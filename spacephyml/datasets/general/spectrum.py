import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset


class BaseSpectrumDataset(Dataset):
    """
    Generic PyTorch Dataset built from a list of pre-split xarray Dataset chunks.

    Supports two data layouts:

    **2D variable** (new format from ``create_dataset``):
        Pass a single variable name in ``data_columns`` that refers to a
        ``(time, energy_bin)`` DataArray.  Each chunk yields an ``(N, n_bins)``
        array directly — no stacking needed.

    **Multiple 1D variables** (legacy flat format):
        Pass a list of scalar variable names.  Each chunk yields an ``(N, C)``
        array by stacking the C named columns along a new axis.

    After construction the following attributes are available:

    ``spec_matrix`` : ndarray, shape (n_samples, N*n_bins) or (n_samples, N, n_bins)
        The feature matrix (flattened or 2-D depending on ``flatten``).
    ``spec_label`` : ndarray, shape (n_samples,)
        Integer class label for each sample (``-1`` for all samples when the
        dataset is unlabelled).
    ``is_unlabelled`` : bool
        True when the dataset was built in unlabelled mode.
    ``bin_centers`` : ndarray or None
        Energy bin centre values for each sample, shape ``(n_samples, N, n_bins)``.
        ``None`` if no axis variable (tagged ``spacephyml_role='axis'``) was
        found in the chunks.  When the bin centres are identical across all
        samples (the common case within a single survey mode) you can use
        ``dataset.bin_centers[0, 0]`` to get the representative 1-D axis.

    """

    def __init__(
        self,
        chunks_by_label,
        data_columns: list,
        N: int,
        samples: int = 100,
        flatten: bool = True,
        transform=None,
        seed: int = 42,
        verbose: bool = True,
    ):
        """
        Args:
            chunks_by_label : dict[int, list[xr.Dataset]] | list[xr.Dataset]
                Either a mapping from integer label -> list of equal-length chunks
                (labelled mode), or a flat list of chunks (unlabelled mode).
                Pass a flat list to engage unlabelled mode explicitly; the subclass
                can also trigger this path by passing ``chunks_by_label={-1: chunks}``.
            data_columns : list[str]
                Variable name(s) to extract as features from each chunk.
            N : int
                Expected number of time steps per chunk.
            samples : int | None
                Chunks per class after balancing (labelled), or total chunks drawn at
                random (unlabelled).  None = use all.
            flatten : bool
                Return each sample as a 1-D vector when True, else (N, C) 2-D array.
            transform : callable | None
                Optional transform applied to each raw (N, C) numpy array.
                Signature: ndarray -> ndarray.
            seed : int
                Random seed for balancing / random sampling.
            verbose : bool
                Print diagnostics when True.
        """
        np.random.seed(seed)

        # ------------------------------------------------------------------ #
        # Normalise input: flat list → unlabelled dict                        #
        # ------------------------------------------------------------------ #
        if isinstance(chunks_by_label, list):
            chunks_by_label = {-1: chunks_by_label}

        # Unlabelled mode: the only key is -1
        self.is_unlabelled = list(chunks_by_label.keys()) == [-1]

        if self.is_unlabelled:
            # ---------------------------------------------------------------- #
            # Unlabelled path: random subsample from a single pool             #
            # ---------------------------------------------------------------- #
            pool = chunks_by_label[-1]
            if not pool:
                raise ValueError("chunks_by_label contains no usable chunks.")

            if verbose:
                print(f"Unlabelled dataset: {len(pool)} windows available.")

            if samples is not None:
                n = min(samples, len(pool))
                idxs = np.random.choice(len(pool), size=n, replace=False)
                pool = [pool[i] for i in idxs]

            selected = {-1: pool}

        else:
            # ---------------------------------------------------------------- #
            # Labelled path: balance classes                                    #
            # ---------------------------------------------------------------- #
            non_empty = [v for v in chunks_by_label.values() if len(v) > 0]
            if not non_empty:
                raise ValueError("chunks_by_label contains no usable chunks.")

            max_samples = min(len(v) for v in non_empty)
            if verbose:
                print("Max number of samples per classification is:", max_samples)

            if samples is not None:
                samples = min(samples, max_samples)

            selected: dict[int, list] = {}
            for label, chunks in chunks_by_label.items():
                if len(chunks) == 0:
                    selected[label] = []
                elif samples is not None:
                    idxs = np.random.choice(len(chunks), size=samples, replace=False)
                    selected[label] = [chunks[i] for i in idxs]
                else:
                    selected[label] = chunks

        # ------------------------------------------------------------------ #
        # Detect layout and axis variable from the first available chunk      #
        # ------------------------------------------------------------------ #
        first_chunk = next(c for chunks in selected.values() for c in chunks)
        self._is_2d = (
            len(data_columns) == 1 and
            'energy_bin' in first_chunk[data_columns[0]].dims
        )

        axis_var = next(
            (v for v in first_chunk.data_vars
             if first_chunk[v].attrs.get('spacephyml_role') == 'axis'),
            None
        )

        # ------------------------------------------------------------------ #
        # Build contiguous numpy arrays                                        #
        # ------------------------------------------------------------------ #
        spec_matrix = []
        spec_label  = []
        bin_centers = [] if axis_var is not None else None

        for label, chunks in selected.items():
            for chunk in chunks:
                sample = self._extract(chunk, data_columns)  # (N, n_bins)
                if transform is not None:
                    sample = transform(sample)
                spec_matrix.append(sample.flatten() if flatten else sample)
                spec_label.append(label)

                if axis_var is not None:
                    bin_centers.append(
                        chunk[axis_var].values.astype(np.float32)
                    )

        self.spec_matrix = np.array(spec_matrix, dtype=np.float32)
        self.spec_label  = np.array(spec_label,  dtype=np.int64)
        self.bin_centers = (np.array(bin_centers, dtype=np.float32)
                            if bin_centers is not None else None)

    def _extract(self, chunk: xr.Dataset, data_columns: list) -> np.ndarray:
        """
        Extract a (N, C) float32 array from a chunk.

        For a 2D variable (time, energy_bin): return its values directly.
        For multiple 1D variables: stack them along a new last axis.
        """
        if self._is_2d:
            return chunk[data_columns[0]].values.astype(np.float32)
        return np.stack(
            [chunk[col].values for col in data_columns], axis=-1
        ).astype(np.float32)

    # ---------------------------------------------------------------------- #
    # PyTorch Dataset interface                                                #
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.spec_label)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.spec_matrix[idx])
        y = torch.tensor(self.spec_label[idx], dtype=torch.long)
        return x, y
