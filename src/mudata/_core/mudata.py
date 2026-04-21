from __future__ import annotations

import warnings
from collections import Counter, abc
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from contextlib import suppress
from copy import deepcopy
from functools import reduce
from hashlib import sha1
from itertools import chain, combinations
from numbers import Integral
from random import choices
from string import ascii_letters, digits
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from anndata._core.aligned_mapping import AlignedView, AxisArrays, AxisArraysBase, PairwiseArrays
from anndata._core.views import DataFrameView
from anndata.utils import convert_to_dict
from scverse_misc import Deprecation, deprecated

from .config import OPTIONS
from .file_backing import MuDataFileManager
from .repr import MUDATA_CSS, block_matrix, details_block_table
from .utils import (
    MetadataColumn,
    _make_index_unique,
    _restore_index,
    _update_and_concat,
    try_convert_dataframe_to_numpy_dtypes,
)
from .views import DictView

if TYPE_CHECKING:
    from os import PathLike
    from pathlib import Path

    import zarr


class MuAxisArraysView(AlignedView, AxisArraysBase):
    def __init__(self, parent_mapping: AxisArraysBase, parent_view: MuData, subset_idx: Any):
        self.parent_mapping = parent_mapping
        self._parent = parent_view
        self.subset_idx = subset_idx
        self._axis = parent_mapping._axis

        @property
        def dimnames(self):
            return None


class MuAxisArrays(AxisArrays):
    _view_class = MuAxisArraysView


class ModDict(dict):
    def _repr_hierarchy(
        self, nest_level: int = 0, is_last: bool = False, active_levels: list[int] | None = None
    ) -> str:
        descr = ""
        active_levels = active_levels or []
        for i, kv in enumerate(self.items()):
            k, v = kv
            indent = ("   " * nest_level) + ("└─ " if i == len(self) - 1 else "├─ ")

            if len(active_levels) > 0:
                indent_list = list(indent)
                for level in active_levels:
                    indent_list[level * 3] = "│"
                indent = "".join(indent_list)

            is_view = " view" if v.is_view else ""
            backed_at = f" backed at {str(v.filename)!r}" if v.isbacked else ""

            if isinstance(v, MuData):
                maybe_axis = (
                    (" [shared obs] " if v.axis == 0 else " [shared var] " if v.axis == 1 else " [shared obs and var] ")
                    if hasattr(v, "axis")
                    else ""
                )
                descr += f"\n{indent}{k} MuData{maybe_axis}({v.n_obs} × {v.n_vars}){backed_at}{is_view}"

                if i != len(self) - 1:
                    levels = [nest_level] + active_levels
                else:
                    levels = [level for level in active_levels if level != nest_level]
                descr += v.mod._repr_hierarchy(nest_level=nest_level + 1, active_levels=levels)
            elif isinstance(v, AnnData):
                descr += f"\n{indent}{k} AnnData ({v.n_obs} x {v.n_vars}){backed_at}{is_view}"
            else:
                continue

        return descr

    def __repr__(self) -> str:
        """
        Represent the hierarchy of the modalities in the object.

        A MuData object with two modalities, protein and RNA,
        with the latter being a MuData containing raw, QC'ed and hvg-filtered AnnData objects,
        will be represented as:

        root MuData (axis=0) (5000 x 20100)
        ├── protein AnnData (5000 x 100)
        └── rna MuData (axis=-1) (5000 x 20000)
            ├── raw AnnData (5000 x 20000)
            ├── quality-filtered AnnData (3000 x 20000)
            └── hvg-filtered AnnData (3000 x 4000)
        """
        return "MuData" + self._repr_hierarchy()


class MuData:
    """
    Multimodal data object

    MuData represents modalities as collections of AnnData objects
    as well as includes multimodal annotations
    such as embeddings and neighbours graphs learned jointly
    on multiple modalities and generalised sample
    and feature metadata tables.

    Parameters
    ----------
    data
        AnnData object or dictionary with AnnData objects as values.
        If a dictionary is passed, the keys will be used as modality names.
        If None, creates an empty MuData object.
    feature_types_names
        Dictionary to map feature types encoded in data.var["feature_types"] to modality names.
        Only relevant when data is an AnnData object.
        Default: {"Gene Expression": "rna", "Peaks": "atac", "Antibody Capture": "prot"}
    as_view
        Create a view of the MuData object.
    index
        Index to slice the MuData object when creating the view.
    **kwargs
        Additional arguments to create a MuData object.
    """

    def __init__(
        self,
        data: AnnData | Mapping[str, AnnData] | MuData | None = None,
        feature_types_names: Mapping[str, str] | None = MappingProxyType(
            {"Gene Expression": "rna", "Peaks": "atac", "Antibody Capture": "prot"}
        ),
        as_view: bool = False,
        index: tuple[slice | int, slice | int] | slice | int | None = None,
        **kwargs,
    ):
        self._init_common()
        if as_view:
            self._init_as_view(data, index)
            return

        # Add all modalities to a MuData object
        self._mod = ModDict()
        if data is None:
            # Initialize an empty MuData object
            pass
        elif isinstance(data, abc.Mapping):
            for k, v in data.items():
                self._mod[k] = v
        elif isinstance(data, AnnData):
            # Get the list of modalities
            if "feature_types" in data.var.columns:
                if data.var.feature_types.dtype.name == "category:":
                    mod_names = data.var.feature_types.cat.categories.values
                else:
                    mod_names = data.var.feature_types.unique()

                for name in mod_names:
                    alias = name
                    if feature_types_names is not None:
                        if name in feature_types_names.keys():
                            alias = feature_types_names[name]
                    self._mod[alias] = data[:, data.var.feature_types == name].copy()
            else:
                self._mod["data"] = data
        else:
            raise TypeError("Expected AnnData object or dictionary with AnnData objects as values")

        self._check_duplicated_names()

        # When creating from a dictionary with _init_from_dict_
        if len(kwargs) > 0:
            # Get global observations
            self._obs = kwargs.get("obs", None)
            if isinstance(self._obs, abc.Mapping) or self._obs is None:
                self._obs = pd.DataFrame(self._obs)

            # Get global variables
            self._var = kwargs.get("var", None)
            if isinstance(self._var, abc.Mapping) or self._var is None:
                self._var = pd.DataFrame(self._var)

            # Ensure no keys are missing
            for key in ("obsm", "varm", "obsp", "varp", "obsmap", "varmap"):
                kwargs[key] = kwargs.get(key) or {}

            # Map each attribute to its class and axis value
            attr_to_axis_arrays = {
                "obsm": (MuAxisArrays, 0),
                "varm": (MuAxisArrays, 1),
                "obsp": (PairwiseArrays, 0),
                "varp": (PairwiseArrays, 1),
                "obsmap": (MuAxisArrays, 0),
                "varmap": (MuAxisArrays, 1),
            }

            # Initialise each attribute
            for attr, (cls, axis) in attr_to_axis_arrays.items():
                setattr(self, f"_{attr}", cls(self, axis=axis, store=kwargs[attr]))

            self._axis = kwargs.get("axis") or 0

            # Restore proper .obs and .var
            self.update()

            self._uns = kwargs.get("uns") or {}

            return

        # Initialize global observations
        self._obs = pd.DataFrame()

        # Initialize global variables
        self._var = pd.DataFrame()

        # Make obs map for each modality
        self._obsm = MuAxisArrays(self, axis=0, store={})
        self._obsp = PairwiseArrays(self, axis=0, store={})
        self._obsmap = MuAxisArrays(self, axis=0, store={})

        # Make var map for each modality
        self._varm = MuAxisArrays(self, axis=1, store={})
        self._varp = PairwiseArrays(self, axis=1, store={})
        self._varmap = MuAxisArrays(self, axis=1, store={})

        self._axis = 0

        # Only call update() if there are modalities
        self.update()

    def _init_common(self):
        self._mudata_ref = None

        # Unstructured annotations
        self._uns = {}

        # For compatibility with calls requiring AnnData slots
        self.raw = None
        self.X = None
        self.layers = None
        self.file = MuDataFileManager()
        self._is_view = False

    def _init_as_view(self, mudata_ref: MuData, index):
        from anndata._core.index import _normalize_indices
        from anndata._core.views import _resolve_idxs

        obsidx, varidx = _normalize_indices(index, mudata_ref.obs.index, mudata_ref.var.index)

        # to handle single-element subsets, otherwise when subsetting a Dataframe
        # we get a Series
        if isinstance(obsidx, Integral):
            obsidx = slice(obsidx, obsidx + 1)
        if isinstance(varidx, Integral):
            varidx = slice(varidx, varidx + 1)

        self._mod = ModDict()
        for m, a in mudata_ref.mod.items():
            cobsidx, cvaridx = mudata_ref.obsmap[m][obsidx], mudata_ref.varmap[m][varidx]
            cobsidx, cvaridx = cobsidx[cobsidx > 0] - 1, cvaridx[cvaridx > 0] - 1
            if len(cobsidx) > 0 and len(cvaridx) > 0:
                if np.all(np.diff(cobsidx) == 1):
                    if a.is_view:
                        if (
                            isinstance(a, MuData)
                            and len(cobsidx) == a._mudata_ref.n_obs
                            or isinstance(a, AnnData)
                            and len(cobsidx) == a._adata_ref.n_obs
                        ):
                            cobsidx = slice(None)
                    elif len(cobsidx) == a.n_obs:
                        cobsidx = slice(None)
                if np.all(np.diff(cvaridx) == 1):
                    if a.is_view:
                        if (
                            isinstance(a, MuData)
                            and len(cvaridx) == a._mudata_ref.n_vars
                            or isinstance(a, AnnData)
                            and len(cvaridx) == a._adata_ref.n_vars
                        ):
                            cvaridx = slice(None)
                    elif len(cvaridx) == a.n_vars:
                        cvaridx = slice(None)
            if a.is_view:
                if isinstance(a, MuData):
                    self._mod[m] = a._mudata_ref[_resolve_idxs((a._oidx, a._vidx), (cobsidx, cvaridx), a._mudata_ref)]
                else:
                    self._mod[m] = a._adata_ref[_resolve_idxs((a._oidx, a._vidx), (cobsidx, cvaridx), a._adata_ref)]
            else:
                self._mod[m] = a[cobsidx, cvaridx]

        self._obs = DataFrameView(mudata_ref.obs.iloc[obsidx, :], view_args=(self, "obs"))
        self._obsm = mudata_ref.obsm._view(self, (obsidx,))
        self._obsp = mudata_ref.obsp._view(self, (obsidx, obsidx))
        self._var = DataFrameView(mudata_ref.var.iloc[varidx, :], view_args=(self, "var"))
        self._varm = mudata_ref.varm._view(self, (varidx,))
        self._varp = mudata_ref.varp._view(self, (varidx, varidx))

        for attr, idx in (("obs", obsidx), ("var", varidx)):
            posmap = {}
            size = getattr(self, attr).shape[0]
            for mod, mapping in getattr(mudata_ref, attr + "map").items():
                cposmap = np.zeros((size,), dtype=mapping.dtype)
                cidx = (mapping[idx] > 0).ravel()
                cposmap[cidx > 0] = np.arange(cidx.sum()) + 1
                posmap[mod] = cposmap
            setattr(self, "_" + attr + "map", posmap)

        self._is_view = True
        self.file = mudata_ref.file
        self._axis = mudata_ref._axis
        self._uns = mudata_ref._uns
        self._oidx = obsidx
        self._vidx = varidx

        if mudata_ref.is_view:
            self._mudata_ref = mudata_ref._mudata_ref
        else:
            self._mudata_ref = mudata_ref

    def _init_as_actual(self, data: MuData):
        self._init_common()
        self._mod = data.mod
        self._obs = data.obs
        self._var = data.var
        self._obsm = MuAxisArrays(self, axis=0, store=convert_to_dict(data.obsm))
        self._obsp = PairwiseArrays(self, axis=0, store=convert_to_dict(data.obsp))
        self._obsmap = MuAxisArrays(self, axis=0, store=convert_to_dict(data.obsmap))
        self._varm = MuAxisArrays(self, axis=1, store=convert_to_dict(data.varm))
        self._varp = PairwiseArrays(self, axis=1, store=convert_to_dict(data.varp))
        self._varmap = MuAxisArrays(self, axis=1, store=convert_to_dict(data.varmap))
        self._uns = data._uns
        self._axis = data._axis

    @classmethod
    def _init_from_dict_(
        cls,
        mod: Mapping[str, Mapping | AnnData] | None = None,
        obs: pd.DataFrame | Mapping[str, Iterable[Any]] | None = None,
        var: pd.DataFrame | Mapping[str, Iterable[Any]] | None = None,
        uns: Mapping[str, Any] | None = None,
        obsm: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        varm: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        obsp: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        varp: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        obsmap: Mapping[str, Sequence[int]] | None = None,
        varmap: Mapping[str, Sequence[int]] | None = None,
        axis: Literal[0, 1] = 0,
    ):
        return cls(
            data={
                k: (
                    v
                    if isinstance(v, AnnData) or isinstance(v, MuData)
                    else MuData(**v)
                    if "mod" in v
                    else AnnData(**v)
                )
                for k, v in mod.items()
            },
            obs=obs,
            var=var,
            uns=uns,
            obsm=obsm,
            varm=varm,
            obsp=obsp,
            varp=varp,
            obsmap=obsmap,
            varmap=varmap,
            axis=axis,
        )

    def _check_duplicated_attr_names(self, attr: str):
        if any(not getattr(self._mod[mod_i], attr + "_names").astype(str).is_unique for mod_i in self._mod):
            # If there are non-unique attr_names, we can only handle outer joins
            # under the condition the duplicated values are restricted to one modality
            dups = [
                np.unique(
                    getattr(self._mod[mod_i], attr + "_names")[
                        getattr(self._mod[mod_i], attr + "_names").astype(str).duplicated()
                    ]
                )
                for mod_i in self._mod
            ]
            for i, mod_i_dup_attrs in enumerate(dups):
                for j, mod_j in enumerate(self._mod):
                    if j != i:
                        if any(np.isin(mod_i_dup_attrs, getattr(self._mod[mod_j], attr + "_names").values)):
                            warnings.warn(
                                f"Duplicated {attr}_names should not be present in different modalities due to the ambiguity that leads to.",
                                stacklevel=3,
                            )
            return True
        return False

    def _check_duplicated_names(self):
        self._check_duplicated_attr_names("obs")
        self._check_duplicated_attr_names("var")

    def _check_intersecting_attr_names(self, attr: str):
        for mod_i, mod_j in combinations(self._mod, 2):
            mod_i_attr_index = getattr(self._mod[mod_i], attr + "_names")
            mod_j_attr_index = getattr(self._mod[mod_j], attr + "_names")
            intersection = mod_i_attr_index.intersection(mod_j_attr_index, sort=False)
            if intersection.shape[0] > 0:
                # Some of the elements are also in another index
                return True
        return False

    def _check_changed_attr_names(self, attr: str, columns: bool = False):
        attrhash = f"_{attr}hash"
        attr_names_changed, attr_columns_changed = False, False
        if not hasattr(self, attrhash):
            attr_names_changed, attr_columns_changed = True, True
        elif len(self._mod) < len(getattr(self, attrhash)):
            attr_names_changed, attr_columns_changed = True, None
        else:
            for m in self._mod.keys():
                if m in getattr(self, attrhash):
                    cached_hash = getattr(self, attrhash)[m]
                    new_hash = (
                        sha1(np.ascontiguousarray(getattr(self._mod[m], attr).index.values)).hexdigest(),
                        sha1(np.ascontiguousarray(getattr(self._mod[m], attr).columns.values)).hexdigest(),
                    )
                    if cached_hash[0] != new_hash[0]:
                        attr_names_changed = True
                        if not attr_columns_changed:
                            attr_columns_changed = None
                        break
                    if columns:
                        if cached_hash[1] != new_hash[1]:
                            attr_columns_changed = True
                else:
                    attr_names_changed, attr_columns_changed = True, None
                    break
        return (attr_names_changed, attr_columns_changed)

    def copy(self, filename: str | PathLike | None = None) -> MuData:
        """
        Make a copy.

        Parameters
        ----------
        filename
            If the object is backed, copy the object to a new file.
        """
        if not self.isbacked:
            mod = {}
            for k, v in self._mod.items():
                mod[k] = v.copy()
            return self._init_from_dict_(
                mod,
                self.obs.copy(),
                self.var.copy(),
                deepcopy(self.uns),  # this should always be an empty dict
                self.obsm.copy(),
                self.varm.copy(),
                self.obsp.copy(),
                self.varp.copy(),
                self.obsmap.copy(),
                self.varmap.copy(),
                self.axis,
            )
        else:
            if filename is None:
                raise ValueError(
                    "To copy a MuData object in backed mode, pass a filename: `copy(filename='myfilename.h5mu')`"
                )
            from .io import read_h5mu, write_h5mu

            write_h5mu(filename, self)
            return read_h5mu(filename, self.file._filemode)

    def strings_to_categoricals(self, df: pd.DataFrame | None = None) -> pd.DataFrame | None:
        """Transform string annotations to categoricals.

        Parameters
        ----------
        df
            If `None`, modifies :attr:`var` and :attr:`obs` attributes of the :class:`MuData` object as well as
            each modality. Otherwise, modifies the dataframe in-place and returns it.
        """
        AnnData.strings_to_categoricals(self, df)

        # Call the same method on each modality
        if df is None:
            for k in self._mod:
                self._mod[k].strings_to_categoricals()
        else:
            return df

    # To increase compatibility with scanpy methods
    _sanitize = strings_to_categoricals

    def __getitem__(self, index) -> AnnData | MuData:
        if isinstance(index, str):
            return self._mod[index]
        else:
            return MuData(self, as_view=True, index=index)

    @property
    def mod(self) -> Mapping[str, AnnData | MuData]:
        """Dictionary of modalities."""
        return self._mod

    @property
    def is_view(self) -> bool:
        """Whether the object is a view of another :class:`MuData` object."""
        return self._is_view

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of data, all variables and observations combined (:attr:`n_obs`, :attr:`n_vars`)."""
        return self.n_obs, self.n_vars

    def __len__(self) -> int:
        """Length defined as a total number of observations (:attr:`n_obs`)."""
        return self.n_obs

    def _update_attr(
        self,
        attr: str,
        axis: int,
        **kwargs,  # for _pull_attr()
    ):
        """
        Update global observations/variables with observations/variables for each modality.

        The following considerations are taken into account and will influence the time it takes to .update():
        - are there duplicated obs_names/var_names?
        - are there intersecting obs_names/var_names between modalities?
        - have obs_names/var_names of modalities changed?
        """
        # No _attrhash when upon read
        # No _attrhash in mudata < 0.2.0
        _attrhash = f"_{attr}hash"
        attr_changed = self._check_changed_attr_names(attr)

        if not any(attr_changed):
            # Nothing to update
            return

        data_global = getattr(self, attr)
        prev_index = data_global.index

        attr_duplicated = not data_global.index.is_unique or self._check_duplicated_attr_names(attr)
        attr_intersecting = self._check_intersecting_attr_names(attr)

        # Generate unique colnames
        (rowcol,) = self._find_unique_colnames(attr, 1)

        attrm = getattr(self, attr + "m")
        attrp = getattr(self, attr + "p")
        attrmap = getattr(self, f"_{attr}map")

        dfs = [
            getattr(a, attr).loc[:, []].assign(**{f"{m}:{rowcol}": np.arange(getattr(a, attr).shape[0])})
            for m, a in self._mod.items()
        ]

        index_order = None
        can_update = True

        def fix_attrmap_col(data_mod: pd.DataFrame, mod: str, rowcol: str) -> str:
            colname = mod + ":" + rowcol
            # use 0 as special value for missing
            # we could use a pandas.array, which has missing values support, but then we get an Exception upon hdf5 write
            # also, this is compatible to Muon.jl
            col = data_mod[colname] + 1
            col.replace(np.nan, 0, inplace=True)
            data_mod[colname] = col.astype(np.uint32)
            return colname

        kept_idx: Any
        new_idx: Any

        def reorder_data_mod():
            nonlocal kept_idx, new_idx, data_mod
            # reorder new index to conform to the old index as much as possible
            kept_idx = data_global.index[data_global.index.isin(data_mod.index)]
            new_idx = data_mod.index[~data_mod.index.isin(data_global.index)]
            data_mod = data_mod.loc[kept_idx.append(new_idx), :]

        def calc_attrm_update():
            nonlocal index_order, can_update
            index_order = data_global.index.get_indexer(data_mod.index)
            can_update = (
                new_idx.shape[0] == 0  # noqa: F821   filtered or reordered
                or kept_idx.shape[0] == data_global.shape[0]  # noqa: F821     new rows only
                or data_mod.shape[0]
                == data_global.shape[
                    0
                ]  # renamed (since new_idx.shape[0] > 0 and kept_idx.shape[0] < data_global.shape[0])
                or (
                    axis != self.axis and axis != -1 and data_mod.shape[0] > data_global.shape[0]
                )  # new modality added and concacenated
            )

        #
        # Join modality .obs/.var tables
        #
        # Main case: no duplicates and no intersection if the axis is not shared
        if not attr_duplicated:
            # Shared axis
            data_mod = pd.concat(dfs, join="outer", axis=1 if axis == self._axis or self._axis == -1 else 0, sort=False)
            for mod in self._mod.keys():
                fix_attrmap_col(data_mod, mod, rowcol)

            data_mod = _make_index_unique(data_mod, force=attr_intersecting)
            data_global = _make_index_unique(data_global, force=attr_intersecting)
            if data_global.shape[1] > 0:
                data_mod = try_convert_dataframe_to_numpy_dtypes(
                    data_mod.join(data_global.convert_dtypes(), how="left", sort=False)
                )

            if data_global.shape[0] > 0:
                reorder_data_mod()
                calc_attrm_update()

            data_mod = _restore_index(data_mod)
            data_global = _restore_index(data_global)
        #
        # General case: with duplicates and/or intersections
        #
        else:
            dfs = [_make_index_unique(df, force=True) for df in dfs]
            data_mod = pd.concat(dfs, join="outer", axis=1 if axis == self._axis or self._axis == -1 else 0, sort=False)

            data_mod = _restore_index(data_mod)
            data_mod.index.set_names(rowcol, inplace=True)
            data_global.index.set_names(rowcol, inplace=True)
            for mod, amod in self._mod.items():
                colname = fix_attrmap_col(data_mod, mod, rowcol)
                if mod in attrmap:
                    modmap = attrmap[mod].ravel()
                    modmask = modmap > 0
                    # only use unchanged modalities for ordering
                    if (
                        modmask.sum() == getattr(amod, attr).shape[0]
                        and (getattr(amod, attr).index[modmap[modmask] - 1] == prev_index[modmask]).all()
                    ):
                        data_mod.set_index(colname, append=True, inplace=True)
                        data_global.set_index(attrmap[mod].reshape(-1), append=True, inplace=True)
                        data_global.index.set_names(colname, level=-1, inplace=True)

            if data_global.shape[0] > 0:
                if not data_global.index.is_unique:
                    warnings.warn(
                        f"{attr}_names is not unique, global {attr} is present, and {attr}map is empty. The update() is not well-defined, verify if global {attr} map to the correct modality-specific {attr}.",
                        stacklevel=2,
                    )
                    data_mod.reset_index(data_mod.index.names.difference(data_global.index.names), inplace=True)
                # after inserting a new modality with duplicates, but no duplicates before:
                # data_mod.index is not unique
                # after deleting a modality with duplicates: data_global.index is not unique, but
                # data_mod.index is unique
                need_unique = data_mod.index.is_unique | data_global.index.is_unique
                data_global = _make_index_unique(data_global, force=need_unique)
                data_mod = _make_index_unique(data_mod, force=need_unique)
                data_mod = try_convert_dataframe_to_numpy_dtypes(
                    data_mod.join(data_global.convert_dtypes(), how="left", sort=False)
                )

                reorder_data_mod()
                calc_attrm_update()

                if need_unique:
                    data_mod = _restore_index(data_mod)
                    data_global = _restore_index(data_global)

            data_mod.reset_index(level=list(range(1, data_mod.index.nlevels)), inplace=True)
            data_global.reset_index(level=list(range(1, data_global.index.nlevels)), inplace=True)
            data_mod.index.set_names(None, inplace=True)
            data_global.index.set_names(None, inplace=True)

        # get adata positions and remove columns from the data frame
        mdict = {}
        for m in self._mod.keys():
            colname = m + ":" + rowcol
            mdict[m] = data_mod[colname].to_numpy()
            data_mod.drop(colname, axis=1, inplace=True)

        if not data_mod.index.is_unique:
            warnings.warn(
                f"{attr}_names are not unique. To make them unique, call `.{attr}_names_make_unique`.", stacklevel=2
            )
            if self._axis == -1:
                warnings.warn(
                    f"Behaviour is not defined with axis=-1, {attr}_names need to be made unique first.", stacklevel=2
                )

        setattr(self, "_" + attr, data_mod)

        # Update .obsm/.varm
        # this needs to be after setting _obs/_var due to dimension checking in the aligned mapping
        attrmap.clear()
        attrmap.update(mdict)
        for mod, mapping in mdict.items():
            attrm[mod] = mapping > 0

        if index_order is not None:
            if can_update:
                for mx_key, mx in attrm.items():
                    if mx_key not in self._mod.keys():  # not a modality name
                        if isinstance(mx, pd.DataFrame):
                            mx = mx.iloc[index_order, :]
                            mx.iloc[index_order == -1, :] = pd.NA
                            mx.index = data_mod.index
                        else:
                            mx = mx[index_order]
                            mx[index_order == -1] = np.nan
                        attrm[mx_key] = mx

                # Update .obsp/.varp (size might have changed)
                for mx_key, mx in attrp.items():
                    mx = mx[index_order[:, None], index_order[None, :]]
                    mx[index_order == -1, :] = -1
                    mx[:, index_order == -1] = -1
                    attrp[mx_key] = mx
            else:
                raise NotImplementedError(
                    f"{attr}_names seem to have been renamed and filtered at the same time. "
                    "There is no way to restore the order. MuData object has to be re-created from these modalities:\n"
                    "  mdata1 = MuData(mdata.mod)"
                )

        # Write _attrhash
        if attr_changed:
            if not hasattr(self, _attrhash):
                setattr(self, _attrhash, {})
            for m, mod in self._mod.items():
                getattr(self, _attrhash)[m] = (
                    sha1(np.ascontiguousarray(getattr(mod, attr).index.values)).hexdigest(),
                    sha1(np.ascontiguousarray(getattr(mod, attr).columns.values)).hexdigest(),
                )

        if OPTIONS["pull_on_update"]:
            self._pull_attr(attr, **kwargs)

    @property
    def n_mod(self) -> int:
        """Number of modalities."""
        return len(self._mod)

    @property
    def isbacked(self) -> bool:
        """Whether the object is backed on disk."""
        return self.file.filename is not None

    @property
    def filename(self) -> Path | None:
        """Change the backing mode by setting the filename to a `.h5mu` file.

        - Setting the filename writes the stored data to disk.
        - Setting the filename when the filename was previously another name moves the backing file from
          the previous file to the new file. If you want to copy the previous file, use `copy(filename="new_filename")`.
        """
        return self.file.filename

    @filename.setter
    def filename(self, filename: str | PathLike | None):
        if self.isbacked:
            if filename is None:
                self.file._to_memory_mode()
            elif self.filename != filename:
                self.write()
                self.filename.rename(filename)
                self.file.open(filename, "r+")
        elif filename is not None:
            self.write(filename)
            self.file.open(filename, "r+")
            for ad in self._mod.values():
                ad._X = None

    @property
    def obs(self) -> pd.DataFrame:
        """Annotation of observation"""
        return self._obs

    @obs.setter
    def obs(self, value: pd.DataFrame):
        # self._set_dim_df(value, "obs")
        if len(value) != self.shape[0]:
            raise ValueError(
                f"The length of provided annotation {len(value)} does not match the length {self.shape[0]} of MuData.obs."
            )
        if self.is_view:
            self._init_as_actual(self.copy())
        self._obs = value

    @property
    def n_obs(self) -> int:
        """Total number of observations"""
        return self._obs.shape[0]

    def _attr_vector(self, key: str, attr: str) -> np.ndarray:
        df = getattr(self, attr)
        if key not in df.columns:
            for m, a in self._mod.items():
                if key in getattr(a, attr).columns:
                    raise KeyError(
                        f"There is no key {key} in MuData .{attr} but there is one in {m} .{attr}. Consider running `update_{attr}()` to update global .{attr}."
                    )
            raise KeyError(f"There is no key {key} in MuData .{attr} or in .{attr} of any modalities.")
        return df[key].to_numpy()

    @deprecated(Deprecation("0.3.4"))
    def obs_vector(self, key: str, layer: str | None = None) -> np.ndarray:
        """Return an array of values for the requested key of length n_obs.

        Parameters
        ----------
        key
            The key to use. Must be in `.obs.columns`.
        layer
            Ignored, only for compatibility with AnnData.
        """
        return self._attr_vector(key, "obs")

    def update_obs(self):
        """Update :attr:`obs` indices of the object with the data from all the modalities."""
        join_common = self.axis == 1
        self._update_attr("obs", axis=0, join_common=join_common)

    def _names_make_unique(self, attr: Literal["obs", "var"]):
        axis = 0 if attr == "obs" else 1
        if self.axis != 1 - axis:
            raise TypeError(
                f"This operation is only supported on MuData objects with `axis={1 - axis}`. This MuData has `axis={self.axis}`."
            )
        namesattr = f"{attr}_names"
        mod_sum = np.sum([a.shape[axis] for a in self._mod.values()])
        if mod_sum != self.shape[axis]:
            self._update_attr(attr, axis=1 - axis)

        for mod in self._mod.values():
            mod_make_unique = getattr(mod, f"{attr}_names_make_unique")
            if isinstance(mod, AnnData):
                mod_make_unique()
            # Only propagate to individual modalities with shared vars
            elif isinstance(mod, MuData) and mod.axis == axis:
                mod_make_unique()

        # Check if there are observations with the same name in different modalities
        mods = list(self._mod.keys())
        with suppress(StopIteration):
            for i in range(len(self._mod) - 1):
                ki = mods[i]
                for j in range(i + 1, len(self._mod)):
                    kj = mods[j]
                    if len(getattr(self._mod[ki], namesattr).intersection(getattr(self._mod[kj], namesattr))) > 0:
                        warnings.warn(
                            "Modality names will be prepended to obs_names since there are identical obs_names in different modalities.",
                            stacklevel=1,
                        )
                        for m, mod in self._mod.items():
                            setattr(mod, namesattr, m + ":" + getattr(mod, namesattr).astype(str))
                        raise StopIteration()  # break out of both loops

        setattr(self, namesattr, pd.Index([]).append([getattr(mod, namesattr) for mod in self._mod.values()]))

    def obs_names_make_unique(self):
        """
        Call :meth:`AnnData.obs_names_make_unique <anndata.AnnData.obs_names_make_unique>` on each modality.

        If there are :attr:`obs_names` which are the same for multiple modalities, append the modality name to all obs_names.
        """
        self._names_make_unique("obs")

    def _set_names(self, attr: str, axis: int, names: Sequence[str]):
        if isinstance(names, pd.Index):
            if not isinstance(names.name, str | type(None)):
                raise ValueError(
                    f"MuData expects .{attr}.index.name to be a string or None, "
                    f"but you passed a name of type {type(names.name).__name__!r}"
                )
        else:
            names = pd.Index(names)
            if not isinstance(names.name, str | type(None)):
                names.name = None

        if axis != self.axis and self.axis != -1:
            mod_shape_sum = np.sum([a.shape[axis] for a in self._mod.values()])
        else:
            mod_shape_sum = reduce(lambda x, y: x.union(y), (getattr(a, attr).index for a in self._mod.values())).size
        if mod_shape_sum != self.shape[axis]:
            self._update_attr(attr, axis=1 - axis)

        if len(names) != self.shape[axis]:
            raise ValueError(
                f"The length of provided {attr}_names {len(names)} does not match the length {self.shape[axis]} of MuData.{attr}."
            )

        if self.is_view:
            self._init_as_actual(self.copy())

        getattr(self, attr).index = names
        map = getattr(self, f"{attr}map")
        for modname, mod in self._mod.items():
            newnames = np.empty(mod.shape[axis], dtype=object)
            modmap = map[modname].ravel()
            mask = modmap > 0
            newnames[modmap[mask] - 1] = names[mask]
            setattr(mod, f"{attr}_names", newnames)

        self._update_attr(attr, axis=1 - axis)

    @property
    def obs_names(self) -> pd.Index:
        """Names of variables (alias for `.obs.index`)."""
        return self.obs.index

    @obs_names.setter
    def obs_names(self, names: Sequence[str]):
        self._set_names("obs", 0, names)

    @property
    def var(self) -> pd.DataFrame:
        """Annotation of variables."""
        return self._var

    @var.setter
    def var(self, value: pd.DataFrame):
        if len(value) != self.shape[1]:
            raise ValueError(
                f"The length of provided annotation {len(value)} does not match the length {self.shape[1]} of MuData.var."
            )
        if self.is_view:
            self._init_as_actual(self.copy())
        self._var = value

    @property
    def n_vars(self) -> int:
        """Total number of variables."""
        return self._var.shape[0]

    @property
    def n_var(self) -> int:
        """Total number of variables."""
        # warnings.warn(
        #     ".n_var will be removed in the next version, use .n_vars instead",
        #     DeprecationWarning,
        #     stacklevel=2,
        # )
        return self._var.shape[0]

    @deprecated(Deprecation("0.3.4"))
    def var_vector(self, key: str, layer: str | None = None) -> np.ndarray:
        """Return an array of values for the requested key of length n_var.

        Parameters
        ----------
        key
            The key to use. Must be in `.obs.columns`.
        layer
            Ignored, only for compatibility with AnnData.
        """
        return self._attr_vector(key, "var")

    def update_var(self):
        """Update :attr:`var` indices of the object with the data from all the modalities."""
        join_common = self.axis == 0
        self._update_attr("var", axis=1, join_common=join_common)

    def var_names_make_unique(self):
        """
        Call :meth:`AnnData.var_names_make_unique <anndata.AnnData.var_names_make_unique>` on each modality.

        If there are :attr:`obs_names` which are the same for multiple modalities, append the modality name to all obs_names.
        """
        self._names_make_unique("var")

    @property
    def var_names(self) -> pd.Index:
        """Names of variables (alias for `.var.index`)"""
        return self.var.index

    @var_names.setter
    def var_names(self, names: Sequence[str]):
        self._set_names("var", 1, names)

    # Multi-dimensional annotations (.obsm and .varm)

    @property
    def obsm(self) -> MutableMapping[str]:
        """Multi-dimensional annotation of observations.

        Stores for each key a two- or higher-dimensional :class:`~numpy.ndarray` or :class:`~pandas.DataFrame` of length :attr:`n_obs`.
        Is sliced with `obs` but otherwise behaves like a :term:`mapping`.
        """
        return self._obsm

    @obsm.setter
    def obsm(self, value: Mapping[str]):
        obsm = MuAxisArrays(self, axis=0, store=convert_to_dict(value))
        if self.is_view:
            self._init_as_actual(self.copy())
        self._obsm = obsm

    @obsm.deleter
    def obsm(self):
        self.obsm = {}

    @property
    def obsp(self) -> MutableMapping[str]:
        """Pairwise annotatation of observations.

        Stores for each key a two- or higher-dimensional :class:`~numpy.ndarray` whose first two dimensions are of liength `n_obs`.
        Is sliced with `obs` but otherwise behaves like a :term:`mapping`.
        """
        return self._obsp

    @obsp.setter
    def obsp(self, value: Mapping[str]):
        obsp = PairwiseArrays(self, axis=0, store=convert_to_dict(value))
        if self.is_view:
            self._init_as_actual(self.copy())
        self._obsp = obsp

    @obsp.deleter
    def obsp(self):
        self.obsp = {}

    @property
    def obsmap(self) -> Mapping[str]:
        """Mapping of observation indices in the object to indices in individual modalities.

        Contains an entry for each modality. Each entry is an :class:`~numpy.ndarray` with shape `(n_obs, 1)`. Each element
        in the array contains the numerical index of the observation in the respective modality corresponding to the :class:`MuData`
        observation in that position. The index is 1-based, 0 indicates that the observation is missing in the modality.
        """
        return self._obsmap

    @property
    def varm(self) -> MutableMapping[str]:
        """Multi-dimensional annotation of variables.

        Stores for each key a two- or higher-dimensional :class:`~numpy.ndarray` or :class:`~pandas.DataFrame` of length :attr:`n_vars`.
        Is sliced with `var` but otherwise behaves like a :term:`mapping`.
        """
        return self._varm

    @varm.setter
    def varm(self, value: Mapping[str]):
        varm = MuAxisArrays(self, axis=1, store=convert_to_dict(value))
        if self.is_view:
            self._init_as_actual(self.copy())
        self._varm = varm

    @varm.deleter
    def varm(self):
        self.varm = {}

    @property
    def varp(self) -> MutableMapping[str]:
        """Pairwise annotatation of variables.

        Stores for each key a two- or higher-dimensional :class:`~numpy.ndarray` whose first two dimensions are of liength `n_obs`.
        Is sliced with `obs` but otherwise behaves like a :term:`mapping`.
        """
        return self._varp

    @varp.setter
    def varp(self, value: Mapping[str]):
        varp = PairwiseArrays(self, axis=0, store=convert_to_dict(value))
        if self.is_view:
            self._init_as_actual(self.copy())
        self._varp = varp

    @varp.deleter
    def varp(self):
        self.varp = {}

    @property
    def varmap(self) -> Mapping[str]:
        """Mapping of feature indices in the object to indices in individual modalities.

        Contains an entry for each modality. Each entry is an :class:`~numpy.ndarray` with shape `(n_obs, 1)`. Each element
        in the array contains the numerical index of the feature in the respective modality corresponding to the :class:`MuData`
        feature in that position. The index is 1-based, 0 indicates that the feature is missing in the modality.
        """
        return self._varmap

    # Unstructured annotations

    @property
    def uns(self) -> MutableMapping:
        """Unstructured annotation (ordered dictionary)."""
        uns = self._uns
        if self.is_view:
            uns = DictView(uns, view_args=(self, "_uns"))
        return uns

    @uns.setter
    def uns(self, value: MutableMapping):
        if not isinstance(value, MutableMapping):
            raise ValueError("Only mutable mapping types (e.g. dict) are allowed for `.uns`.")
        if isinstance(value, DictView):
            value = value.copy()
        if self.is_view:
            self._init_as_actual(self.copy())
        self._uns = value

    @uns.deleter
    def uns(self):
        self.uns = {}

    # _keys methods to increase compatibility
    # with calls requiring those AnnData methods

    @deprecated(Deprecation("0.3.4", msg="Use `obs.columns` instead."))
    def obs_keys(self) -> list[str]:
        """List keys of observation annotation :attr:`obs`."""
        return self._obs.keys().tolist()

    @deprecated(Deprecation("0.3.4", msg="Use `var.columns` instead."))
    def var_keys(self) -> list[str]:
        """List keys of variable annotation :attr:`var`."""
        return self._var.keys().tolist()

    @deprecated(Deprecation("0.3.4", msg="Use `obsm.keys()` instead."))
    def obsm_keys(self) -> list[str]:
        """List keys of observation annotation :attr:`obsm`."""
        return list(self._obsm.keys())

    @deprecated(Deprecation("0.3.4", msg="Use `varm.keys()` instead."))
    def varm_keys(self) -> list[str]:
        """List keys of variable annotation :attr:`varm`."""
        return list(self._varm.keys())

    @deprecated(Deprecation("0.3.4", msg="Use `uns.keys()` instead."))
    def uns_keys(self) -> list[str]:
        """List keys of unstructured annotation."""
        return list(self._uns.keys())

    def update(self):
        """Update both :attr:`obs` and :attr:`var` indices of the object with the data from all the modalities."""
        if len(self._mod) > 0:
            self.update_var()
            self.update_obs()

    @property
    def axis(self) -> Literal[-1, 0, 1]:
        """MuData axis.

        - `0` if the modalities have shared observations
        - `1` if the modalities have shared features
        - `-1` if both observations and features are shared
        """
        return self._axis

    @property
    @deprecated(Deprecation("0.3.4", msg="Use `mod.keys()` instead."))
    def mod_names(self) -> list[str]:
        """Names of modalities (alias for `list(mdata.mod.keys())`)"""
        return list(self._mod.keys())

    def _pull_attr(
        self,
        attr: Literal["obs", "var"],
        columns: list[str] | None = None,
        mods: list[str] | None = None,
        common: bool | None = None,
        join_common: bool | None = None,
        nonunique: bool | None = None,
        join_nonunique: bool | None = None,
        unique: bool | None = None,
        prefix_unique: bool = True,
        drop: bool = False,
        only_drop: bool = False,
    ):
        """
        Copy the data from the modalities to the global .obs/.var, existing columns to be overwritten.

        Parameters
        ----------
        attr
            Attribute to use, should be 'obs' or 'var'
        columns
            List of columns to pull from the modalities
        common
            If True, pull common columns.
            Common columns do not have modality prefixes.
            Pull from all modalities.
            Cannot be used with columns. True by default.
        mods
            List of modalities to pull from
        join_common
            If True, attempt to join common columns.
            Common columns are present in all modalities.
            True for attr='var' for MuData with axis=0 (shared obs),
            and for attr='obs' for MuData wth axis=1 (shared var).
            False for MuData with axis=-1.
            Cannot be used with mods, or for shared attr.
        nonunique
            If True, pull columns that have a modality prefix
            such that there are multiple columns with the same name
            and different prefix.
            Cannot be used with columns or mods. True by default.
        join_nonunique
            If True, attempt to join non-unique columns.
            Intended usage is the same as for join_common.
            Cannot be used with mods, or for shared attr. False by default.
        unique
            If True, pull columns that have a modality prefix
            such that there is no other column with the same name
            and a different modality prefix.
            Cannot be used with columns or mods. True by default.
        prefix_unique
            If True, prefix unique column names with modname (default).
            No prefix when False.
        drop
            If True, drop the columns from the modalities after pulling.
            False by default.
        only_drop
            If True, drop the columns but do not actually pull them.
            Forces drop=True. False by default.
        """
        # TODO: run update() before pulling?

        if self.is_view:
            raise ValueError(f"Cannot pull {attr} columns on a view.")

        if mods is not None:
            if isinstance(mods, str):
                mods = [mods]
            if not all(m in self._mod for m in mods):
                raise ValueError("All mods should be present in mdata.mod")
            elif len(mods) == self.n_mod:
                mods = None

        if only_drop:
            drop = True

        cols: dict[str, list[MetadataColumn]] = {}

        # get all columns from all modalities and count how many times each column is present
        derived_name_counts = Counter()
        for prefix, mod in self._mod.items():
            modcols = getattr(mod, attr).columns
            ccols = []
            for name in modcols:
                ccols.append(
                    MetadataColumn(allowed_prefixes=self._mod.keys(), prefix=prefix, name=name, strip_prefix=False)
                )
                derived_name_counts[name] += 1
            cols[prefix] = ccols

        for modcols in cols.values():
            for col in modcols:
                count = derived_name_counts[col.derived_name]
                col.count = count  # this is important to classify columns

        if columns is not None:
            for k, v in {"common": common, "nonunique": nonunique, "unique": unique}.items():
                if v is not None:
                    warnings.warn(
                        f"Both columns and {k} given. Columns take precedence, {k} will be ignored",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            # keep only requested columns
            cols = {
                prefix: [col for col in modcols if col.name in columns or col.derived_name in columns]
                for prefix, modcols in cols.items()
            }

            # TODO: Counter for columns in order to track their usage
            # and error out if some columns were not used

        else:
            if common is None:
                common = True
            if nonunique is None:
                nonunique = True
            if unique is None:
                unique = True

            # filter columns by class, keep only those that were requested
            selector = {"common": common, "nonunique": nonunique, "unique": unique}
            cols = {prefix: [col for col in modcols if selector[col.klass]] for prefix, modcols in cols.items()}

        # filter columns, keep only requested modalities
        if mods is not None:
            cols = {prefix: cols[prefix] for prefix in mods}

        # count final filtered column names, required later to decide whether to prefix a column with its source modality
        derived_name_count = Counter([col.derived_name for modcols in cols.values() for col in modcols])

        # - axis == self.axis
        #   e.g. combine obs from multiple modalities (with shared obs)
        # - 1 - axis == self.axis
        #   e.g. combine var from multiple modalities (with unique vars)
        axis = 0 if attr == "obs" else 1

        if axis == self.axis or self.axis == -1:
            if join_common or join_nonunique:
                raise ValueError(f"Cannot join columns with the same name for shared {attr}_names.")

        if join_common is None:
            if attr == "obs":
                join_common = self.axis == 1
            else:
                join_common = self.axis == 0

        if join_nonunique is None:
            join_nonunique = False

        # Below we will rely on attrmap that has been calculated during .update()
        # and use it to create an index without duplicates
        # for faster concatenation and to reduce the amount of code

        attrmap = getattr(self, f"{attr}map")
        n_attr = self.n_vars if attr == "var" else self.n_obs

        dfs: list[pd.DataFrame] = []
        for m, modcols in cols.items():
            mod = self._mod[m]
            mod_map = attrmap[m].ravel()
            mask = mod_map > 0

            mod_df = getattr(mod, attr)[[col.derived_name for col in modcols]]
            if drop:
                getattr(mod, attr).drop(columns=mod_df.columns, inplace=True)

            # prepend modality prefix to column names if requested via arguments and there are no skipped modalities with
            # the same column name (prefixing those columns may cause problems with future pulls or pushes)
            mod_df.rename(
                columns={
                    col.derived_name: col.name
                    for col in modcols
                    if not (
                        (
                            join_common
                            and col.klass == "common"
                            or join_nonunique
                            and col.klass == "nonunique"
                            or not prefix_unique
                            and col.klass == "unique"
                        )
                        and derived_name_count[col.derived_name] == col.count
                    )
                },
                inplace=True,
            )

            # reorder modality DF to conform to global order
            mod_df = (
                mod_df.iloc[mod_map[mask] - 1]
                .set_index(np.arange(n_attr)[mask])
                .reindex(np.arange(n_attr))
                .convert_dtypes()
            )
            dfs.append(mod_df)

        if only_drop:
            return

        global_df = getattr(self, attr).set_index(np.arange(n_attr)).convert_dtypes()
        df = try_convert_dataframe_to_numpy_dtypes(reduce(_update_and_concat, [global_df, *dfs])).set_index(
            getattr(self, f"{attr}_names")
        )
        setattr(self, attr, df)

    def pull_obs(
        self,
        columns: list[str] | None = None,
        mods: list[str] | None = None,
        common: bool | None = None,
        join_common: bool | None = None,
        nonunique: bool | None = None,
        join_nonunique: bool | None = None,
        unique: bool | None = None,
        prefix_unique: bool = True,
        drop: bool = False,
        only_drop: bool = False,
    ):
        """
        Copy data from the :attr:`~anndata.AnnData.obs` of the modalities to the global :attr:`obs`

        Existing columns to be overwritten or updated.

        Parameters
        ----------
        columns
            List of columns to pull from the modalities' .obs tables
        common
            If True, pull common columns.
            Common columns do not have modality prefixes.
            Pull from all modalities.
            Cannot be used with columns. True by default.
        mods
            List of modalities to pull from.
        join_common
            If True, attempt to join common columns.
            Common columns are present in all modalities.
            True for MuData wth axis=1 (shared var).
            False for MuData with axis=0 and axis=-1.
            Cannot be used with mods, or for shared attr.
        nonunique
            If True, pull columns that have a modality prefix
            such that there are multiple columns with the same name
            and different prefix.
            Cannot be used with columns or mods. True by default.
        join_nonunique
            If True, attempt to join non-unique columns.
            Intended usage is the same as for join_common.
            Cannot be used with mods, or for shared attr. False by default.
        unique
            If True, pull columns that have a modality prefix
            such that there is no other column with the same name
            and a different modality prefix.
            Cannot be used with columns or mods. True by default.
        prefix_unique
            If True, prefix unique column names with modname (default).
            No prefix when False.
        drop
            If True, drop the columns from the modalities after pulling.
        only_drop
            If True, drop the columns but do not actually pull them.
            Forces drop=True.
        """
        return self._pull_attr(
            "obs",
            columns=columns,
            mods=mods,
            common=common,
            join_common=join_common,
            nonunique=nonunique,
            join_nonunique=join_nonunique,
            unique=unique,
            prefix_unique=prefix_unique,
            drop=drop,
            only_drop=only_drop,
        )

    def pull_var(
        self,
        columns: list[str] | None = None,
        mods: list[str] | None = None,
        common: bool | None = None,
        join_common: bool | None = None,
        nonunique: bool | None = None,
        join_nonunique: bool | None = None,
        unique: bool | None = None,
        prefix_unique: bool = True,
        drop: bool = False,
        only_drop: bool = False,
    ):
        """
        Copy data from the :attr:`~anndata.AnnData.var` of the modalities to the global :attr:`var`

        Existing columns to be overwritten or updated.

        Parameters
        ----------
        columns
            List of columns to pull from the modalities' .var tables
        common
            If True, pull common columns.
            Common columns do not have modality prefixes.
            Pull from all modalities.
            Cannot be used with columns. True by default.
        mods
            List of modalities to pull from.
        join_common
            If True, attempt to join common columns.
            Common columns are present in all modalities.
            True for MuData with axis=0 (shared obs).
            False for MuData with axis=1 and axis=-1.
            Cannot be used with mods, or for shared attr.
        nonunique
            If True, pull columns that have a modality prefix
            such that there are multiple columns with the same name
            and different prefix.
            Cannot be used with columns or mods. True by default.
        join_nonunique
            If True, attempt to join non-unique columns.
            Intended usage is the same as for join_common.
            Cannot be used with mods, or for shared attr. False by default.
        unique
            If True, pull columns that have a modality prefix
            such that there is no other column with the same name
            and a different modality prefix.
            Cannot be used with columns or mods. True by default.
        prefix_unique
            If True, prefix unique column names with modname (default).
            No prefix when False.
        drop
            If True, drop the columns from the modalities after pulling.
        only_drop
            If True, drop the columns but do not actually pull them.
            Forces drop=True.
        """
        return self._pull_attr(
            "var",
            columns=columns,
            mods=mods,
            common=common,
            join_common=join_common,
            nonunique=nonunique,
            join_nonunique=join_nonunique,
            unique=unique,
            prefix_unique=prefix_unique,
            drop=drop,
            only_drop=only_drop,
        )

    def _push_attr(
        self,
        attr: Literal["obs", "var"],
        columns: list[str] | None = None,
        mods: list[str] | None = None,
        common: bool | None = None,
        prefixed: bool | None = None,
        drop: bool = False,
        only_drop: bool = False,
    ):
        """
        Copy the data from the global .obs/.var to the modalities, existing columns to be overwritten.

        Parameters
        ----------
        attr
            Attribute to use, should be 'obs' or 'var'
        columns
            List of columns to push
        mods
            List of modalities to push to
        common
            If True, push common columns.
            Common columns do not have modality prefixes.
            Push to each modality unless all values for a modality are null.
            Cannot be used with columns. True by default.
        prefixed
            If True, push columns that have a modality prefix.
            Only push to the respective modality names.
            Cannot be used with columns. True by default.
        drop
            If True, drop the columns from the global .obs/.var after pushing.
            False by default.
        only_drop
            If True, drop the columns but do not actually pull them.
            Forces drop=True. False by default.
        """
        if self.is_view:
            raise ValueError(f"Cannot push {attr} columns on a view.")

        if mods is not None:
            if isinstance(mods, str):
                mods = (mods,)
            if not all(m in self._mod for m in mods):
                raise ValueError("All mods should be present in mdata.mod")
            elif len(mods) == self.n_mod:
                mods = None

        if only_drop:
            drop = True

        # get all global columns
        cols = [MetadataColumn(allowed_prefixes=self._mod.keys(), name=name) for name in getattr(self, attr).columns]

        if columns is not None:
            for k, v in {"common": common, "prefixed": prefixed}.items():
                if v:
                    warnings.warn(
                        f"Both columns and {k} given. Columns take precedence, {k} will be ignored",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            # keep only requested columns
            cols = [
                col
                for col in cols
                if (col.name in columns or col.derived_name in columns)
                and (col.prefix is None or mods is not None and col.prefix in mods)
            ]
        else:
            if common is None:
                common = True
            if prefixed is None:
                prefixed = True

            # filter columns by class, keep only those that were requested
            selector = {"common": common, "unknown": prefixed}
            cols = [col for col in cols if selector[col.klass]]

        if len(cols) == 0:
            return

        derived_name_count = Counter([col.derived_name for col in cols])
        for c, count in derived_name_count.items():
            # if count > 1, there are both colname and modname:colname present
            if count > 1 and c in getattr(self, attr).columns:
                raise ValueError(
                    f"Cannot push multiple columns with the same name {c} with and without modality prefix. "
                    "You might have to explicitely specify columns to push.\n"
                    "In case there are columns with the same name with and without modality prefix, "
                    "this has to be resolved first."
                )

        attrmap = getattr(self, f"{attr}map")
        for m, mod in self._mod.items():
            if mods is not None and m not in mods:
                continue

            mod_map = attrmap[m].ravel()
            mask = mod_map > 0
            mod_n_attr = mod.n_obs if attr == "obs" else mod.n_vars

            # get all common and modality-specific columns for the current modality
            mod_cols = [col for col in cols if col.prefix == m or col.klass == "common"]
            df = getattr(self, attr)[mask][[col.name for col in mod_cols]]

            # strip modality prefix where necessary
            df.columns = [col.derived_name for col in mod_cols]

            # reorder global DF to conform to modality order
            idx = np.empty(mod_n_attr, dtype=mod_map.dtype)
            idx[mod_map[mask] - 1] = np.arange(mod_n_attr)
            df = df.iloc[idx].set_index(np.arange(mod_n_attr, dtype=mod_map.dtype))

            if not only_drop:
                # TODO: _prune_unused_categories
                mod_df = getattr(mod, attr).set_index(np.arange(mod_n_attr))
                mod_df = _update_and_concat(mod_df, df)
                mod_df = mod_df.set_index(getattr(mod, f"{attr}_names"))
                setattr(mod, attr, mod_df)

        if drop:
            for col in cols:
                getattr(self, attr).drop(col.name, axis=1, inplace=True)

    def push_obs(
        self,
        columns: list[str] | None = None,
        mods: list[str] | None = None,
        common: bool | None = None,
        prefixed: bool | None = None,
        drop: bool = False,
        only_drop: bool = False,
    ):
        """
        Copy the data from :attr:`obs` to the :attr:`~anndata.AnnData.obs` of the modalities.

        Existing columns to be overwritten.

        Parameters
        ----------
        columns
            List of columns to push
        mods
            List of modalities to push to
        common
            If True, push common columns.
            Common columns do not have modality prefixes.
            Push to each modality unless all values for a modality are null.
            Cannot be used with columns. True by default.
        prefixed
            If True, push columns that have a modality prefix.
            which are prefixed by modality names.
            Only push to the respective modality names.
            Cannot be used with columns. True by default.
        drop
            If True, drop the columns from the global .obs after pushing.
            False by default.
        only_drop
            If True, drop the columns but do not actually pull them.
            Forces drop=True. False by default.
        """
        return self._push_attr(
            "obs", columns=columns, mods=mods, common=common, prefixed=prefixed, drop=drop, only_drop=only_drop
        )

    def push_var(
        self,
        columns: list[str] | None = None,
        mods: list[str] | None = None,
        common: bool | None = None,
        prefixed: bool | None = None,
        drop: bool = False,
        only_drop: bool = False,
    ):
        """
        Copy the data from :attr:`var` to the :attr:`~anndata.AnnData.var` of the modalities.

        Existing columns to be overwritten.

        Parameters
        ----------
        columns
            List of columns to push
        mods
            List of modalities to push to
        common
            If True, push common columns.
            Common columns do not have modality prefixes.
            Push to each modality unless all values for a modality are null.
            Cannot be used with columns. True by default.
        prefixed
            If True, push columns that have a modality prefix.
            which are prefixed by modality names.
            Only push to the respective modality names.
            Cannot be used with columns. True by default.
        drop
            If True, drop the columns from the global .var after pushing.
            False by default.
        only_drop
            If True, drop the columns but do not actually pull them.
            Forces drop=True. False by default.
        """
        return self._push_attr(
            "var", columns=columns, mods=mods, common=common, prefixed=prefixed, drop=drop, only_drop=only_drop
        )

    def write_h5mu(self, filename: str | PathLike | None = None, **kwargs):
        """Write the object to an HDF5 file.

        Parameters
        ----------
        filename
            Path of the `.h5mu` file to write to. Defaults to the backing file.
        **kwargs
            Additional arguments to :func:`~mudata.write_h5mu`.
        """
        from .io import _write_h5mu, write_h5mu

        if self.isbacked and (filename is None or filename == self.filename):
            import h5py

            self.file.close()
            with h5py.File(self.filename, "a") as f:
                _write_h5mu(f, self, write_data=False, **kwargs)
        elif filename is None:
            raise ValueError("Provide a filename!")
        else:
            write_h5mu(filename, self, **kwargs)
            if self.isbacked:
                self.file.filename = filename

    write = write_h5mu

    def write_zarr(self, store: MutableMapping | str | PathLike | zarr.abc.store.Store, **kwargs):
        """Write the object to a Zarr store.

        Parameters
        ----------
        store
            The filename or a Zarr store.
        **kwargs
            Additional arguments to :func:`~mudata.write_zarr`.
        """
        from .io import write_zarr

        write_zarr(store, self, **kwargs)

    def to_anndata(self, **kwargs) -> AnnData:
        """
        Convert the object to :class:`~anndata.AnnData`.

        If :attr:`axis` is `0` (shared observations),
        concatenate modalities along axis 1 (`anndata.concat(axis=1)`).

        If :attr:`axis` is `1` (shared features),
        concatenate datasets along axis 0 (`anndata.concat(axis=0)`).

        See :func:`anndata.concat` documentation for more details.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to :func:`anndata.concat`
        """
        from .to_ import to_anndata

        return to_anndata(self, **kwargs)

    def _gen_repr(self, n_obs, n_vars, extensive: bool = False, nest_level: int = 0) -> str:
        indent = "    " * nest_level
        backed_at = f" backed at {str(self.filename)!r}" if self.isbacked else ""
        view_of = "View of " if self.is_view else ""
        maybe_axis = (
            ("" if self.axis == 0 else " (shared var) " if self.axis == 1 else " (shared obs and var) ")
            if hasattr(self, "axis")
            else ""
        )
        descr = f"{view_of}MuData object with n_obs × n_vars = {n_obs} × {n_vars}{maybe_axis}{backed_at}"
        for attr in ["obs", "var", "uns", "obsm", "varm", "obsp", "varp"]:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                keys = list(getattr(self, attr).keys())
                if len(keys) > 0:
                    mod_sep = ":" if isinstance(getattr(self, attr), pd.DataFrame) else ""
                    global_keys = list(
                        map(
                            all,
                            zip(
                                *[
                                    [not col.startswith(mod + mod_sep) for col in getattr(self, attr).keys()]
                                    for mod in self._mod
                                ],
                                strict=False,
                            ),
                        )
                    )
                    if any(global_keys):
                        descr += (
                            f"\n{indent}  {attr}:\t{str([keys[i] for i in range(len(keys)) if global_keys[i]])[1:-1]}"
                        )
        descr += f"\n{indent}  {len(self._mod)} modalit{'y' if len(self._mod) == 1 else 'ies'}"
        for k, v in self._mod.items():
            mod_indent = "    " * (nest_level + 1)
            if isinstance(v, MuData):
                descr += f"\n{mod_indent}{k}:\t" + v._gen_repr(v.n_obs, v.n_vars, extensive, nest_level + 1)
                continue
            descr += f"\n{mod_indent}{k}:\t{v.n_obs} × {v.n_vars}"
            for attr in ["obs", "var", "uns", "obsm", "varm", "layers", "obsp", "varp"]:
                try:
                    keys = getattr(v, attr).keys()
                    if len(keys) > 0:
                        descr += f"\n{mod_indent}  {attr}:\t{str(list(keys))[1:-1]}"
                except AttributeError:
                    pass
        return descr

    def __repr__(self) -> str:
        return self._gen_repr(self.n_obs, self.n_vars, extensive=True)

    def _repr_html_(self, expand=None) -> str:
        """
        HTML formatter for MuData objects for rich display in notebooks.

        This formatter has an optional argument `expand`,
        which is a 3-bit flag:
        100 - expand MuData slots
        010 - expand .mod slots
        001 - expand slots for each modality
        """
        # Return text representation if set in options
        if OPTIONS["display_style"] == "text":
            from html import escape

            return f"<pre>{escape(repr(self))}</pre>"

        if expand is None:
            expand = OPTIONS["display_html_expand"]

        # General object properties
        header = "<span>MuData object <span class='hl-dim'>{} obs &times; {} var in {} modalit{}</span></span>".format(
            self.n_obs, self.n_vars, len(self._mod), "y" if len(self._mod) < 2 else "ies"
        )
        if self.isbacked:
            header += f"<br>&#8627; <span>backed at <span class='hl-file'>{self.file.filename}</span></span>"

        mods = "<br>"

        # Metadata
        mods += details_block_table(self, "obs", "Metadata", expand >> 2)
        # Embeddings
        mods += details_block_table(self, "obsm", "Embeddings & mappings", expand >> 2)
        # Distances
        mods += details_block_table(self, "obsp", "Distances", expand >> 2, square=True)
        # Miscellaneous (unstructured)
        if self.uns:
            mods += details_block_table(self, "uns", "Miscellaneous", expand >> 2)

        for m, dat in self._mod.items():
            mods += "<div class='block-mod'><div>"
            mods += "<details{}>".format(" open" if (expand & 0b010) >> 1 else "")
            mods += "<summary class='summary-mod'><div class='title title-mod'>{}</div><span class='hl-dim'>{} &times {}</span></summary>".format(
                m, *dat.shape
            )

            # General object properties
            mods += "<span>{} object <span class='hl-dim'>{} obs &times; {} var</span></span>".format(
                type(dat).__name__, *(dat.shape)
            )
            if dat.isbacked:
                mods += f"<br>&#8627; <span>backed at <span class='hl-file'>{self.file.filename}</span></span>"

            mods += "<br>"

            # X
            mods += block_matrix(dat, "X", "Matrix")
            # Layers
            mods += details_block_table(dat, "layers", "Layers", expand & 0b001, dims=False)
            # Metadata
            mods += details_block_table(dat, "obs", "Metadata", expand & 0b001)
            # Embeddings
            mods += details_block_table(dat, "obsm", "Embeddings", expand & 0b001)
            # Distances
            mods += details_block_table(dat, "obsp", "Distances", expand & 0b001, square=True)
            # Miscellaneous (unstructured)
            mods += details_block_table(dat, "uns", "Miscellaneous", expand & 0b001)

            mods += "</details>"
            mods += "</div></div>"
        mods += "<br/>"
        full = "".join((MUDATA_CSS, "<div class='scv-mudata-repr-html'>", header, mods, "</div>"))
        return full

    def _find_unique_colnames(self, attr: str, ncols: int) -> list[str]:
        nchars = 16
        allunique = False
        while not allunique:
            colnames = ["".join(choices(ascii_letters + digits, k=nchars)) for _ in range(ncols)]
            allunique = len(set(colnames)) == ncols
            nchars *= 2

        for i in range(ncols):
            finished = False
            while not finished:
                for ad in chain((self,), self._mod.values()):
                    if colnames[i] in getattr(ad, attr).columns:
                        colnames[i] = "_" + colnames[i]
                        break
                finished = True
        return colnames
