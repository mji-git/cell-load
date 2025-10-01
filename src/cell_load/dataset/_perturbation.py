### Custom PerturbationDataset replaces original file cell_load/dataset/_perturbation.py
import logging
from pathlib import Path

#from functools import lru_cache
import h5py
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset, Subset

from ..mapping_strategies import BaseMappingStrategy
from ..utils.data_utils import (
    GlobalH5MetadataCache,
    suspected_discrete_torch,
    suspected_log_torch,
)

logger = logging.getLogger(__name__)


class PerturbationDataset(Dataset):
    """
    Dataset class for loading perturbation data from H5 files, handling multiple cell types per plate.
    Each instance serves a single dataset/cell_type combination, with configurable mapping strategies.
    """

    def __init__(
        self,
        name: str,
        h5_path: str | Path,
        mapping_strategy: BaseMappingStrategy,
        pert_onehot_map: dict[str, torch.Tensor] | None = None,
        batch_onehot_map: dict[str, torch.Tensor] | None = None,
        cell_type_onehot_map: dict[str, torch.Tensor] | None = None,
        pert_col: str = "gene",
        cell_type_key: str = "cell_type",
        batch_col: str = "gem_group",
        control_pert: str = "non-targeting",
        embed_key: str = "X_hvg",
        store_raw_expression: bool = False,
        random_state: int = 42,
        should_yield_control_cells: bool = True,
        store_raw_basal: bool = False,
        barcode: bool = False,
        **kwargs,
    ):
        """
        Initialize a perturbation dataset for a specific dataset-celltype.

        Args:
            name: Identifier for this dataset
            h5_path: Path to H5 file
            mapping_strategy: Instance of BaseMappingStrategy to use
            pert_onehot_map: Optional global pert -> one-hot mapping
            batch_onehot_map: Optional global batch -> one-hot mapping
            pert_col: H5 obs column for perturbations
            cell_type_key: H5 obs column for cell types
            batch_col: H5 obs column for batches
            control_pert: Perturbation treated as control
            embed_key: Key under obsm for embeddings
            store_raw_expression: If True, include raw gene expression
            random_state: Seed for reproducibility
            should_yield_control_cells: Include control cells in output
            store_raw_basal: If True, include raw basal expression
            barcode: If True, include cell barcodes in output
            **kwargs: Additional options (e.g. output_space)
        """
        super().__init__()
        self.name = name
        self.h5_path = Path(h5_path)
        self.rng = np.random.default_rng(random_state)
        self.mapping_strategy = mapping_strategy
        self.pert_onehot_map = pert_onehot_map
        self.batch_onehot_map = batch_onehot_map
        self.cell_type_onehot_map = cell_type_onehot_map
        self.pert_col = pert_col
        self.cell_type_key = cell_type_key
        self.batch_col = batch_col
        self.control_pert = control_pert
        self.embed_key = embed_key
        self.store_raw_expression = store_raw_expression
        self.should_yield_control_cells = should_yield_control_cells
        self.store_raw_basal = store_raw_basal
        self.barcode = barcode
        self.output_space = kwargs.get("output_space", "gene")
        if self.output_space not in {"gene", "all", "embedding"}:
            raise ValueError(
                f"output_space must be one of 'gene', 'all', or 'embedding'; got {self.output_space!r}"
            )
        assert self.output_space == "all"

        # Load metadata cache and open file
        # keep this logic, is fine
        self.metadata_cache = GlobalH5MetadataCache().get_cache(
            str(self.h5_path), pert_col, cell_type_key, control_pert, batch_col
        )
        #self.h5_file = h5py.File(self.h5_path, "r")
        self.raw_file = sc.read_h5ad(str(self.h5_path), backed="r")

        # Load cell barcodes if requested
        if self.barcode:
            self.cell_barcodes = self._load_cell_barcodes()
        else:
            self.cell_barcodes = None

        # Cached categories & masks
        self.pert_categories = self.metadata_cache.pert_categories
        self.cell_type_categories = self.metadata_cache.cell_type_categories
        self.control_mask = self.metadata_cache.control_mask

        # Global indices and counts
        self.all_indices = np.arange(self.metadata_cache.n_cells)
        self.n_cells = len(self.all_indices)
        self.n_genes = self._get_num_genes()

        # Initialize split index containers
        splits = ["train", "train_eval", "val", "test"]
        self.split_perturbed_indices = {s: set() for s in splits}
        self.split_control_indices = {s: set() for s in splits}

    def set_store_raw_expression(self, flag: bool) -> None:
        """
        Enable or disable inclusion of raw gene expression in each sample.
        """
        self.store_raw_expression = flag
        logger.info(f"[{self.name}] store_raw_expression set to {flag}")

    def reset_mapping_strategy(
        self,
        strategy_cls: BaseMappingStrategy,
        stage: str = "train",
        **strategy_kwargs,
    ) -> None:
        """
        Replace the current mapping strategy and re-register existing splits.
        """
        self.mapping_strategy = strategy_cls(**strategy_kwargs)
        self.mapping_strategy.stage = stage
        for split, pert_set in self.split_perturbed_indices.items():
            ctrl_set = self.split_control_indices[split]
            if pert_set and ctrl_set:
                pert_arr = np.array(sorted(pert_set))
                ctrl_arr = np.array(sorted(ctrl_set))
                self.mapping_strategy.register_split_indices(
                    self, split, pert_arr, ctrl_arr
                )

    def __getitem__(self, idx: int):
        """
        Fetch a sample (perturbed + mapped control) by filtered index.

        This returns a dictionary with:
        - pert_cell_emb: the embedding of the perturbed cell (either in gene space or embedding space)
        - ctrl_cell_emb: the control cell's embedding. control cells are chosen by the mapping strategy
        - pert_emb: the one-hot encoding (or other featurization) for the perturbation
        - pert_name: the perturbation name
        - cell_type: the cell type
        - batch: the batch (as an int or string)
        - batch_name: the batch name (as a string)
        - pert_cell_counts: the raw gene expression of the perturbed cell (if store_raw_expression is True)
        - ctrl_cell_counts: the raw gene expression of the control cell (if store_raw_basal is True)
        """

        # Get the perturbed cell expression, control cell expression, and index of mapped control cell
        file_idx = int(self.all_indices[idx])
        split = self._find_split_for_idx(file_idx)
        pert_expr, ctrl_expr, ctrl_idx = self.mapping_strategy.get_mapped_expressions(
            self, split, file_idx
        )

        # Perturbation info
        pert_code = self.metadata_cache.pert_codes[file_idx]
        pert_name = self.pert_categories[pert_code]
        pert_onehot = (
            self.pert_onehot_map.get(pert_name) if self.pert_onehot_map else None
        )

        # Cell type info
        cell_type = self.cell_type_categories[
            self.metadata_cache.cell_type_codes[file_idx]
        ]
        cell_type_onehot = (
            self.cell_type_onehot_map.get(cell_type)
            if self.cell_type_onehot_map
            else None
        )

        # Batch info
        batch_code = self.metadata_cache.batch_codes[file_idx]
        batch_name = self.metadata_cache.batch_categories[batch_code]
        batch_onehot = (
            self.batch_onehot_map.get(batch_name) if self.batch_onehot_map else None
        )

        sample = {
            "pert_cell_emb": pert_expr,
            "ctrl_cell_emb": ctrl_expr,
            "pert_emb": pert_onehot,
            "pert_name": pert_name,
            "batch_name": batch_name,
            "batch": batch_onehot,
            "cell_type": cell_type,
            "cell_type_onehot": cell_type_onehot,
        }

        # Optionally include raw expressions for the perturbed cell, for training a decoder
        if self.store_raw_expression and self.output_space != "embedding":
            if self.output_space == "gene":
                sample["pert_cell_counts"] = self.fetch_obsm_expression(
                    file_idx, "X_hvg"
                )
            elif self.output_space == "all":
                sample["pert_cell_counts"] = self.fetch_gene_expression(file_idx)

        # Optionally include raw expressions for the control cell
        if self.store_raw_basal and self.output_space != "embedding":
            if self.output_space == "gene":
                sample["ctrl_cell_counts"] = self.fetch_obsm_expression(
                    ctrl_idx, "X_hvg"
                )
            elif self.output_space == "all":
                sample["ctrl_cell_counts"] = self.fetch_gene_expression(ctrl_idx)

        # Optionally include cell barcodes
        if self.barcode and self.cell_barcodes is not None:
            sample["pert_cell_barcode"] = self.cell_barcodes[file_idx]
            sample["ctrl_cell_barcode"] = self.cell_barcodes[ctrl_idx]

        return sample

    def get_batch(self, idx: int) -> torch.Tensor:
        """
        Get the batch information for a given cell index. Returns a scalar tensor.
        """
        assert self.batch_onehot_map is not None, "No batch onehot map, run setup."
        # Translate row index -> batch code -> batch category name
        batch_code = self.metadata_cache.batch_codes[idx]
        batch_name = self.metadata_cache.batch_categories[batch_code]
        batch = torch.argmax(self.batch_onehot_map[batch_name])
        return batch.item()

    def get_dim_for_obsm(self, key: str) -> int:
        """
        Get the feature dimensionality of obsm data with the specified key (e.g., 'X_uce').
        """
        return self.raw_file.obsm[key].shape[1]

    def get_cell_type(self, idx):
        """
        Get the cell type for a given index.
        """
        # Convert idx to int in case it's a tensor or array
        idx = int(idx) if hasattr(idx, "__int__") else idx
        code = self.metadata_cache.cell_type_codes[idx]
        return self.metadata_cache.cell_type_categories[code]

    def get_all_cell_types(self, indices):
        """
        Get the cell types for all given indices.
        """
        codes = self.metadata_cache.cell_type_codes[indices]
        return self.metadata_cache.cell_type_categories[codes]

    def get_perturbation_name(self, idx):
        """
        Get the perturbation name for a given index.
        """
        # Convert idx to int in case it's a tensor or array
        idx = int(idx) if hasattr(idx, "__int__") else idx
        pert_code = self.metadata_cache.pert_codes[idx]
        return self.metadata_cache.pert_categories[pert_code]

    def to_subset_dataset(
        self,
        split: str,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray,
    ) -> Subset:
        """
        Creates a Subset of this dataset that includes only the specified perturbed_indices.
        If `self.should_yield_control_cells` flag is True, the Subset will also yield control cells.

        Args:
            split: Name of the split to create, one of 'train', 'val', 'test', or 'train_eval'
            perturbed_indices: Indices of perturbed cells to include
            control_indices: Indices of control cells to include
        """

        # sort them for stable ordering
        perturbed_indices = np.sort(perturbed_indices)
        control_indices = np.sort(control_indices)

        # Register them in the dataset
        self._register_split_indices(split, perturbed_indices, control_indices)

        # Return a Subset containing perturbed cells and optionally control cells
        if self.should_yield_control_cells:
            all_indices = np.concatenate([perturbed_indices, control_indices])
            return Subset(self, all_indices)
        else:
            return Subset(self, perturbed_indices)

    #@lru_cache(maxsize=10000)  # cache the results of the function; lots of hits for batch mapping since most sentences have repeated cells
    def fetch_gene_expression(self, idx: int) -> torch.Tensor:
        """
        Fetch raw gene counts for a given cell index.

        Supports both CSR‐encoded storage (via `encoding-type = "csr_matrix"`)
        and dense storage in the 'X' dataset.

        Args:
            idx: row index in the X matrix
        Returns:
            1D FloatTensor of length self.n_genes
        """

        row_data = self.raw_file.X[idx].toarray().squeeze()
        data = torch.from_numpy(row_data).to(torch.float32) # shares memory
        return data

    #@lru_cache(maxsize=10000)
    def fetch_obsm_expression(self, idx: int, key: str) -> torch.Tensor:
        """
        Fetch a single row from the /obsm/{key} embedding matrix.

        Args:
            idx: row index in the obsm matrix
            key: name of the obsm dataset (e.g. "X_uce", "X_hvg")
        Returns:
            1D FloatTensor of that embedding
        """
        row_data = self.raw_file.obsm[key][idx]
        return torch.tensor(row_data, dtype=torch.float32)

    def get_gene_names(self, output_space="all") -> list[str]:
        """
        Return the list of gene names from var/gene_name (or its categorical fallback).

        Tries, in order:
        1. var/gene_name directly
        2. var/gene_name/categories + codes
        3. var/_index as last resort
        """

        def _decode(x):
            return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)

        assert output_space == "all"
        assert "highly_variable" not in self.raw_file.var.keys()
        fallback = self.raw_file.var.index[:]
        return [_decode(x) for x in fallback]

    ##############################
    # Static methods
    ##############################
    @staticmethod
    def collate_fn(batch, int_counts=False):
        """
        Optimized collate function with preallocated lists.
        Safely handles normalization when vectors sum to zero.
        """
        # Get batch size
        batch_size = len(batch)

        # Preallocate lists with exact size
        pert_cell_emb_list = [None] * batch_size
        ctrl_cell_emb_list = [None] * batch_size
        pert_emb_list = [None] * batch_size
        pert_name_list = [None] * batch_size
        cell_type_list = [None] * batch_size
        cell_type_onehot_list = [None] * batch_size
        batch_list = [None] * batch_size
        batch_name_list = [None] * batch_size

        # Check if optional fields exist
        has_pert_cell_counts = "pert_cell_counts" in batch[0]
        has_ctrl_cell_counts = "ctrl_cell_counts" in batch[0]
        has_barcodes = "pert_cell_barcode" in batch[0]

        # Preallocate optional lists if needed
        if has_pert_cell_counts:
            pert_cell_counts_list = [None] * batch_size

        if has_ctrl_cell_counts:
            ctrl_cell_counts_list = [None] * batch_size

        if has_barcodes:
            pert_cell_barcode_list = [None] * batch_size
            ctrl_cell_barcode_list = [None] * batch_size

        # Process all items in a single pass
        for i, item in enumerate(batch):
            pert_cell_emb_list[i] = item["pert_cell_emb"]
            ctrl_cell_emb_list[i] = item["ctrl_cell_emb"]
            pert_emb_list[i] = item["pert_emb"]
            pert_name_list[i] = item["pert_name"]
            cell_type_list[i] = item["cell_type"]
            cell_type_onehot_list[i] = item["cell_type_onehot"]
            batch_list[i] = item["batch"]
            batch_name_list[i] = item["batch_name"]

            if has_pert_cell_counts:
                pert_cell_counts_list[i] = item["pert_cell_counts"]

            if has_ctrl_cell_counts:
                ctrl_cell_counts_list[i] = item["ctrl_cell_counts"]

            if has_barcodes:
                pert_cell_barcode_list[i] = item["pert_cell_barcode"]
                ctrl_cell_barcode_list[i] = item["ctrl_cell_barcode"]

        # Create batch dictionary
        batch_dict = {
            "pert_cell_emb": torch.stack(pert_cell_emb_list),
            "ctrl_cell_emb": torch.stack(ctrl_cell_emb_list),
            "pert_emb": torch.stack(pert_emb_list),
            "pert_name": pert_name_list,
            "cell_type": cell_type_list,
            "cell_type_onehot": torch.stack(cell_type_onehot_list),
            "batch": torch.stack(batch_list),
            "batch_name": batch_name_list,
        }

        if has_pert_cell_counts:
            pert_cell_counts = torch.stack(pert_cell_counts_list)

            is_discrete = suspected_discrete_torch(pert_cell_counts)
            is_log = suspected_log_torch(pert_cell_counts)
            already_logged = (not is_discrete) and is_log
            batch_dict["pert_cell_counts"] = pert_cell_counts

        if has_ctrl_cell_counts:
            ctrl_cell_counts = torch.stack(ctrl_cell_counts_list)

            is_discrete = suspected_discrete_torch(pert_cell_counts)
            is_log = suspected_log_torch(pert_cell_counts)
            already_logged = (not is_discrete) and is_log
            batch_dict["ctrl_cell_counts"] = ctrl_cell_counts

        if has_barcodes:
            batch_dict["pert_cell_barcode"] = pert_cell_barcode_list
            batch_dict["ctrl_cell_barcode"] = ctrl_cell_barcode_list

        return batch_dict

    def _register_split_indices(
        self, split: str, perturbed_indices: np.ndarray, control_indices: np.ndarray
    ):
        """
        Register which cell indices belong to the perturbed vs. control set for
        a given split.

        These are passed to the mapping strategy to let it build its internal structures as needed.
        """
        if split not in self.split_perturbed_indices:
            raise ValueError(f"Invalid split {split}")

        # update them in the dataset
        self.split_perturbed_indices[split] |= set(perturbed_indices)
        self.split_control_indices[split] |= set(control_indices)

        # forward these to the mapping strategy
        self.mapping_strategy.register_split_indices(
            self, split, perturbed_indices, control_indices
        )

    def _find_split_for_idx(self, idx: int) -> str | None:
        """Utility to find which split (train/val/test) this idx belongs to."""
        for s in self.split_perturbed_indices.keys():
            if (
                idx in self.split_perturbed_indices[s]
                or idx in self.split_control_indices[s]
            ):
                return s
        return None

    def _get_num_genes(self) -> int:
        """Return the number of genes in the X matrix."""

        # Try to get shape directly from metadata
        n_cols = self.raw_file.X.shape[1]
        return n_cols

    def get_num_hvgs(self) -> int:
        """Return the number of highly variable genes in the obsm matrix."""
        try:
            return self.raw_file.obsm["X_hvg"].shape[1]
        except:
            return 0

    def _get_num_cells(self) -> int:
        """Return the total number of cells in the file."""

        return self.raw_file.X.shape[0]

    def get_pert_name(self, idx: int) -> str:
        """Get perturbation name for a given index."""
        return self.metadata_cache.pert_names[idx]

    def __len__(self) -> int:
        """
        Return number of cells in the dataset
        """
        return self.n_cells

    def __getstate__(self):
        """
        Return a dictionary of this dataset's state without the dataset
        """
        # Copy the object's dict
        state = self.__dict__.copy()
        # Remove the open file object if it exists
        if "raw_file" in state:
            # We'll also store whether it's currently open, so that we can re-open later if needed
            del state["raw_file"]
        return state

    def __setstate__(self, state):
        """
        Reconstruct the dataset after unpickling. Re-open the HDF5 file by path.
        """
        # TODO-Abhi: remove this before release
        self.__dict__.update(state)
        # This ensures that after we unpickle, we have a valid dataset
        self.raw_file = sc.read_h5ad(str(self.h5_path), backed="r")
        self.metadata_cache = GlobalH5MetadataCache().get_cache(
            str(self.h5_path),
            self.pert_col,
            self.cell_type_key,
            self.control_pert,
            self.batch_col,
        )

    def _load_cell_barcodes(self) -> np.ndarray:
        """
        Load cell barcodes from obs/_index

        Returns:
            np.ndarray: Array of cell barcode strings
        """
        try:
            # Try to load from obs/_index (AnnData's default storage for obs index)
            barcodes = self.raw_file.obs.index[:]
            # Decode bytes to strings if necessary
            decoded_barcodes = []
            for barcode in barcodes:
                if isinstance(barcode, (bytes, bytearray)):
                    decoded_barcodes.append(barcode.decode("utf-8", errors="ignore"))
                else:
                    decoded_barcodes.append(str(barcode))
            return np.array(decoded_barcodes, dtype=str)

        except KeyError:
            # If no barcode information is available, generate generic ones
            logger.warning(
                f"No cell barcode information found in {self.h5_path}. Generating generic barcodes."
            )
            return np.array(
                [f"cell_{i:06d}" for i in range(self.metadata_cache.n_cells)],
                dtype=str,
            )
