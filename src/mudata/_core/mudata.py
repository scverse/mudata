import warnings
from collections import Counter, abc
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from copy import deepcopy
from functools import reduce
from hashlib import sha1
from itertools import chain, combinations
from numbers import Integral
from os import PathLike
from pathlib import Path
from random import choices
from string import ascii_letters, digits
from typing import Any, Literal, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from anndata._core.aligned_mapping import (
    AxisArraysBase,
    PairwiseArraysView,
)
from anndata._core.views import DataFrameView
from anndata.utils import convert_to_dict

from .compat import AlignedView, AxisArrays, PairwiseArrays
from .config import OPTIONS
from .file_backing import MuDataFileManager
from .repr import MUDATA_CSS, block_matrix, details_block_table
from .utils import (
    _classify_attr_columns,
    _classify_prefixed_columns,
    _make_index_unique,
    _maybe_coerce_to_bool,
    _maybe_coerce_to_boolean,
    _maybe_coerce_to_int,
    _restore_index,
    _update_and_concat,
)
from .views import DictView


class MuAxisArraysView(AlignedView, AxisArraysBase):
    def __init__(self, parent_mapping: AxisArraysBase, parent_view: "MuData", subset_idx: Any):
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
                    (
                        " [shared obs] "
                        if v.axis == 0
                        else " [shared var] " if v.axis == 1 else " [shared obs and var] "
                    )
                    if hasattr(v, "axis")
                    else ""
                )
                descr += (
                    f"\n{indent}{k} MuData{maybe_axis}({v.n_obs} × {v.n_vars}){backed_at}{is_view}"
                )

                if i != len(self) - 1:
                    levels = [nest_level] + [level for level in active_levels]
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
    """

    def __init__(
        self,
        data: Union[AnnData, Mapping[str, AnnData], "MuData"] = None,
        feature_types_names: dict | None = {
            "Gene Expression": "rna",
            "Peaks": "atac",
            "Antibody Capture": "prot",
        },
        as_view: bool = False,
        index: tuple[slice | Integral, slice | Integral] | slice | Integral | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        data
            AnnData object or dictionary with AnnData objects as values.
            If a dictionary is passed, the keys will be used as modality names.
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
        self._init_common()
        if as_view:
            self._init_as_view(data, index)
            return

        # Add all modalities to a MuData object
        self.mod = ModDict()
        if isinstance(data, abc.Mapping):
            for k, v in data.items():
                self.mod[k] = v
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
                    self.mod[alias] = data[:, data.var.feature_types == name].copy()
            else:
                self.mod["data"] = data
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

            # Get global obsm
            self._obsm = MuAxisArrays(self, axis=0, store=kwargs.get("obsm", {}))
            # Get global varm
            self._varm = MuAxisArrays(self, axis=1, store=kwargs.get("varm", {}))

            self._obsp = PairwiseArrays(self, axis=0, store=kwargs.get("obsp", {}))
            self._varp = PairwiseArrays(self, axis=1, store=kwargs.get("varp", {}))

            self._obsmap = MuAxisArrays(self, axis=0, store=kwargs.get("obsmap", {}))
            self._varmap = MuAxisArrays(self, axis=1, store=kwargs.get("varmap", {}))

            self._axis = kwargs.get("axis") or 0

            # Restore proper .obs and .var
            self.update()

            self._uns = kwargs.get("uns") or {}

            return

        # Initialise global observations
        self._obs = pd.DataFrame()

        # Initialise global variables
        self._var = pd.DataFrame()

        # Make obs map for each modality
        self._obsm = MuAxisArrays(self, axis=0, store=dict())
        self._obsp = PairwiseArrays(self, axis=0, store=dict())
        self._obsmap = MuAxisArrays(self, axis=0, store=dict())

        # Make var map for each modality
        self._varm = MuAxisArrays(self, axis=1, store=dict())
        self._varp = PairwiseArrays(self, axis=1, store=dict())
        self._varmap = MuAxisArrays(self, axis=1, store=dict())

        self._axis = 0

        self.update()

    def _init_common(self):
        self._mudata_ref = None

        # Unstructured annotations
        # NOTE: this is dict in contract to OrderedDict in anndata
        #       due to favourable performance and lack of need to preserve the insertion order
        self._uns = dict()

        # For compatibility with calls requiring AnnData slots
        self.raw = None
        self.X = None
        self.layers = None
        self.file = MuDataFileManager()
        self.is_view = False

    def _init_as_view(self, mudata_ref: "MuData", index):
        from anndata._core.index import _normalize_indices

        obsidx, varidx = _normalize_indices(index, mudata_ref.obs.index, mudata_ref.var.index)

        # to handle single-element subsets, otherwise when subsetting a Dataframe
        # we get a Series
        if isinstance(obsidx, Integral):
            obsidx = slice(obsidx, obsidx + 1)
        if isinstance(varidx, Integral):
            varidx = slice(varidx, varidx + 1)

        self.mod = ModDict()
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
                    self.mod[m] = a._mudata_ref[cobsidx, cvaridx]
                else:
                    self.mod[m] = a._adata_ref[cobsidx, cvaridx]
            else:
                self.mod[m] = a[cobsidx, cvaridx]

        self._obs = DataFrameView(mudata_ref.obs.iloc[obsidx, :], view_args=(self, "obs"))
        self._obsm = mudata_ref.obsm._view(self, (obsidx,))
        self._obsp = mudata_ref.obsp._view(self, obsidx)
        self._var = DataFrameView(mudata_ref.var.iloc[varidx, :], view_args=(self, "var"))
        self._varm = mudata_ref.varm._view(self, (varidx,))
        self._varp = mudata_ref.varp._view(self, varidx)

        for attr, idx in (("obs", obsidx), ("var", varidx)):
            posmap = {}
            for mod, mapping in getattr(mudata_ref, attr + "map").items():
                posmap[mod] = mapping[idx].copy()
            setattr(self, "_" + attr + "map", posmap)

        self.is_view = True
        self.file = mudata_ref.file
        self._axis = mudata_ref._axis
        self._uns = mudata_ref._uns

        if mudata_ref.is_view:
            self._mudata_ref = mudata_ref._mudata_ref
        else:
            self._mudata_ref = mudata_ref

    def _init_as_actual(self, data: "MuData"):
        self._init_common()
        self.mod = data.mod
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
                    else MuData(**v) if "mod" in v else AnnData(**v)
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
        if any(
            [
                not getattr(self.mod[mod_i], attr + "_names").astype(str).is_unique
                for mod_i in self.mod
            ]
        ):
            # If there are non-unique attr_names, we can only handle outer joins
            # under the condition the duplicated values are restricted to one modality
            dups = [
                np.unique(
                    getattr(self.mod[mod_i], attr + "_names")[
                        getattr(self.mod[mod_i], attr + "_names").astype(str).duplicated()
                    ]
                )
                for mod_i in self.mod
            ]
            for i, mod_i_dup_attrs in enumerate(dups):
                for j, mod_j in enumerate(self.mod):
                    if j != i:
                        if any(
                            np.isin(
                                mod_i_dup_attrs, getattr(self.mod[mod_j], attr + "_names").values
                            )
                        ):
                            warnings.warn(
                                f"Duplicated {attr}_names should not be present in different modalities due to the ambiguity that leads to."
                            )
            return True
        return False

    def _check_duplicated_names(self):
        self._check_duplicated_attr_names("obs")
        self._check_duplicated_attr_names("var")

    def _check_intersecting_attr_names(self, attr: str):
        for mod_i, mod_j in combinations(self.mod, 2):
            mod_i_attr_index = getattr(self.mod[mod_i], attr + "_names")
            mod_j_attr_index = getattr(self.mod[mod_j], attr + "_names")
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
        else:
            for m, mod in self.mod.items():
                if m in getattr(self, attrhash):
                    cached_hash = getattr(self, attrhash)[m]
                    new_hash = (
                        sha1(
                            np.ascontiguousarray(getattr(self.mod[m], attr).index.values)
                        ).hexdigest(),
                        sha1(
                            np.ascontiguousarray(getattr(self.mod[m], attr).columns.values)
                        ).hexdigest(),
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

    def copy(self, filename: PathLike | None = None) -> "MuData":
        """
        Make a copy.

        Parameters
        ----------
        filename
            If the object is backed, copy the object to a new file.
        """
        if not self.isbacked:
            mod = {}
            for k, v in self.mod.items():
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

    def strings_to_categoricals(self, df: pd.DataFrame | None = None):
        """
        Transform string columns in .var and .obs slots of MuData to categorical
        as well as of .var and .obs slots in each AnnData object

        This keeps it compatible with AnnData.strings_to_categoricals() method.
        """
        AnnData.strings_to_categoricals(self, df)

        # Call the same method on each modality
        if df is None:
            for k in self.mod:
                self.mod[k].strings_to_categoricals()
        else:
            return df

    # To increase compatibility with scanpy methods
    _sanitize = strings_to_categoricals

    def __getitem__(self, index) -> Union["MuData", AnnData]:
        if isinstance(index, str):
            return self.mod[index]
        else:
            return MuData(self, as_view=True, index=index)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of data, all variables and observations combined (:attr:`n_obs`, :attr:`n_var`)."""
        return self.n_obs, self.n_vars

    def __len__(self) -> int:
        """Length defined as a total number of observations (:attr:`n_obs`)."""
        return self.n_obs

    # # Currently rely on AnnData's interface for setting .obs / .var
    # # This code implements AnnData._set_dim_df for another namespace
    # def _set_dim_df(self, value: pd.DataFrame, attr: str):
    #     if not isinstance(value, pd.DataFrame):
    #         raise ValueError(f"Can only assign pd.DataFrame to {attr}.")
    #     value_idx = AnnData._prep_dim_index(self, value.index, attr)
    #     setattr(self, f"_{attr}", value)
    #     AnnData._set_dim_index(self, value_idx, attr)

    def _create_global_attr_index(self, attr: str, axis: int):
        if axis == (1 - self._axis):
            # Shared indices
            modindices = [getattr(self.mod[m], attr).index for m in self.mod]
            if all([modindices[i].equals(modindices[i + 1]) for i in range(len(modindices) - 1)]):
                attrindex = modindices[0].copy()
            attrindex = reduce(
                pd.Index.union, [getattr(self.mod[m], attr).index for m in self.mod]
            ).values
        else:
            # Modality-specific indices
            attrindex = np.concatenate(
                [getattr(self.mod[m], attr).index.values for m in self.mod], axis=0
            )
        return attrindex

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

        if OPTIONS["pull_on_update"] is None:
            warnings.warn(
                "From 0.4 .update() will not pull obs/var columns from individual modalities by default anymore. "
                "Set mudata.set_options(pull_on_update=False) to adopt the new behaviour, which will become the default. "
                "Use new pull_obs/pull_var and push_obs/push_var methods for more flexibility.",
                FutureWarning,
                stacklevel=2,
            )

            join_common = False
            if "join_common" in kwargs:
                join_common = kwargs.pop("join_common")
            self._update_attr_legacy(attr, axis, join_common, **kwargs)
            return

        prev_index = getattr(self, attr).index

        # No _attrhash when upon read
        # No _attrhash in mudata < 0.2.0
        _attrhash = f"_{attr}hash"
        attr_changed = self._check_changed_attr_names(attr)

        attr_duplicated = self._check_duplicated_attr_names(attr)
        attr_intersecting = self._check_intersecting_attr_names(attr)

        if attr_duplicated:
            warnings.warn(
                f"{attr}_names are not unique. To make them unique, call `.{attr}_names_make_unique`."
            )
            if self._axis == -1:
                warnings.warn(
                    f"Behaviour is not defined with axis=-1, {attr}_names need to be made unique first."
                )

        if not any(attr_changed):
            # Nothing to update
            return

        data_global = getattr(self, attr)

        # Generate unique colnames
        (rowcol,) = self._find_unique_colnames(attr, 1)

        attrm = getattr(self, attr + "m")
        attrp = getattr(self, attr + "p")
        attrmap = getattr(self, attr + "map")

        # TODO: take advantage when attr_changed[0] == False — only new columns to be added

        #
        # Join modality .obs/.var tables
        #
        # Main case: no duplicates and no intersection if the axis is not shared
        #
        if not attr_duplicated:
            # Shared axis
            if axis == (1 - self._axis) or self._axis == -1:
                # We assume attr_intersecting and can't join_common
                data_mod = pd.concat(
                    [
                        getattr(a, attr)
                        .loc[:, []]
                        .assign(**{rowcol: np.arange(getattr(a, attr).shape[0])})
                        .add_prefix(m + ":")
                        for m, a in self.mod.items()
                    ],
                    join="outer",
                    axis=1,
                    sort=False,
                )
            else:
                data_mod = _maybe_coerce_to_bool(
                    pd.concat(
                        [
                            getattr(a, attr)
                            .loc[:, []]
                            .assign(**{rowcol: np.arange(getattr(a, attr).shape[0])})
                            .add_prefix(m + ":")
                            for m, a in self.mod.items()
                        ],
                        join="outer",
                        axis=0,
                        sort=False,
                    )
                )

            for mod, amod in self.mod.items():
                colname = mod + ":" + rowcol
                # use 0 as special value for missing
                # we could use a pandas.array, which has missing values support, but then we get an Exception upon hdf5 write
                # also, this is compatible to Muon.jl
                col = data_mod[colname] + 1
                col.replace(np.nan, 0, inplace=True)
                data_mod[colname] = col.astype(np.uint32)

            if len(data_global.columns) > 0:
                # TODO: if there were intersecting attrnames between modalities,
                #       this will increase the size of the index
                # Should we use attrmap to figure the index out?
                #
                if not attr_intersecting:
                    data_mod = data_mod.join(data_global, how="left", sort=False)
                else:
                    # In order to preserve the order of the index, instead,
                    # perform a join based on (index, attrmap value) pairs.
                    (col_index,) = self._find_unique_colnames(attr, 1)
                    data_mod = data_mod.rename_axis(col_index, axis=0).reset_index()

                    data_global = data_global.rename_axis(col_index, axis=0).reset_index()
                    for mod in self.mod.keys():
                        data_global[mod + ":" + rowcol] = getattr(self, attr + "map")[mod]
                    attrmap_columns = (mod + ":" + rowcol for mod in self.mod.keys())

                    data_mod = data_mod.merge(
                        data_global, on=[col_index, *attrmap_columns], how="left", sort=False
                    )

                    # Restore the index and remove the helper column
                    data_mod = data_mod.set_index(col_index).rename_axis(None, axis=0)
                    data_global = data_global.set_index(col_index).rename_axis(None, axis=0)
        #
        # General case: with duplicates and/or intersections
        #
        else:
            dfs = [
                _make_index_unique(
                    getattr(a, attr)
                    .loc[:, []]
                    .assign(**{rowcol: np.arange(getattr(a, attr).shape[0])})
                    .add_prefix(m + ":")
                )
                for m, a in self.mod.items()
            ]
            data_mod = pd.concat(
                dfs,
                join="outer",
                axis=axis,
                sort=False,
            )

            # pd.concat wrecks the ordering when doing an outer join with a MultiIndex and different data frame shapes
            if axis == 1:
                newidx = (
                    reduce(lambda x, y: x.union(y, sort=False), (df.index for df in dfs))
                    .to_frame()
                    .reset_index(level=1, drop=True)
                )
                globalidx = data_global.index.get_level_values(0)
                mask = globalidx.isin(newidx.iloc[:, 0])
                if len(mask) > 0:
                    negativemask = ~newidx.index.get_level_values(0).isin(globalidx)
                    newidx = pd.MultiIndex.from_frame(
                        pd.concat(
                            [newidx.loc[globalidx[mask], :], newidx.iloc[negativemask, :]], axis=0
                        )
                    )
                data_mod = data_mod.reindex(newidx, copy=False)

            data_mod = _restore_index(data_mod)
            data_mod.index.set_names(rowcol, inplace=True)
            data_global.index.set_names(rowcol, inplace=True)
            for mod, amod in self.mod.items():
                colname = mod + ":" + rowcol
                # use 0 as special value for missing
                # we could use a pandas.array, which has missing values support, but then we get an Exception upon hdf5 write
                # also, this is compatible to Muon.jl
                col = data_mod.loc[:, colname] + 1
                col.replace(np.nan, 0, inplace=True)
                col = col.astype(np.uint32)
                data_mod.loc[:, colname] = col
                data_mod.set_index(colname, append=True, inplace=True)
                if mod in attrmap and np.sum(attrmap[mod] > 0) == getattr(amod, attr).shape[0]:
                    data_global.set_index(attrmap[mod], append=True, inplace=True)
                    data_global.index.set_names(colname, level=-1, inplace=True)

            if len(data_global) > 0:
                if not data_global.index.is_unique:
                    warnings.warn(
                        f"{attr}_names is not unique, global {attr} is present, and {attr}map is empty. The update() is not well-defined, verify if global {attr} map to the correct modality-specific {attr}."
                    )
                    data_mod.reset_index(
                        data_mod.index.names.difference(data_global.index.names), inplace=True
                    )
                    data_mod = _make_index_unique(data_mod)
                    data_global = _make_index_unique(data_global)
                data_mod = data_mod.join(data_global, how="left", sort=False)
            data_mod.reset_index(level=list(range(1, data_mod.index.nlevels)), inplace=True)
            data_mod.index.set_names(None, inplace=True)

        # get adata positions and remove columns from the data frame
        mdict = dict()
        for m in self.mod.keys():
            colname = m + ":" + rowcol
            mdict[m] = data_mod[colname].to_numpy()
            # data_mod.drop(colname, axis=1, inplace=True)

        # Add data from global .obs/.var columns # This might reduce the size of .obs/.var if observations/variables were removed
        if getattr(self, attr).index.is_unique:
            # There are no new values in the index
            # Original index is present in data_global
            attr_reindexed = getattr(self, attr).reindex(index=data_mod.index, copy=False)
        else:
            # Reindexing won't work with duplicated labels:
            #   cannot reindex on an axis with duplicate labels.
            # Use attrmap to resolve it.

            # TODO: might be possible to refactor to memoize it
            # if it has already been done in the same ._update_attr()
            col_index, col_range = self._find_unique_colnames(attr, 2)

            # copy is made here
            data_mod = data_mod.rename_axis(col_index, axis=0).reset_index()

            data_global[col_range] = np.arange(len(data_global))

            for mod in self.mod.keys():
                data_global[mod + ":" + rowcol] = getattr(self, attr + "map")[mod]
            attrmap_columns = [mod + ":" + rowcol for mod in self.mod.keys()]

            data_mod = data_mod.merge(data_global, on=attrmap_columns, how="left", sort=False)

            index_selection = data_mod[col_range].values

            data_mod.drop(col_range, axis=1, inplace=True)
            data_global.drop(col_range, axis=1, inplace=True)

            # Restore the index and remove the helper column
            data_mod = data_mod.set_index(col_index).rename_axis(None, axis=0)
            attr_reindexed = getattr(self, attr).iloc[index_selection]
            attr_reindexed.index = data_mod.index

        # Clean up
        for colname in (mod + "+" + rowcol for mod in self.mod.keys()):
            data_mod.drop(colname, axis=1, inplace=True, errors="ignore")

        setattr(
            self,
            "_" + attr,
            attr_reindexed,
        )

        # Update .obsm/.varm
        # this needs to be after setting _obs/_var due to dimension checking in the aligned mapping
        attrmap.clear()
        attrmap.update(mdict)
        for mod, mapping in mdict.items():
            attrm[mod] = mapping > 0

        now_index = getattr(self, attr).index

        if len(prev_index) == 0:
            # New object
            pass
        elif now_index.equals(prev_index):
            # Index is the same
            pass
        else:
            keep_index = prev_index.isin(now_index)
            new_index = ~now_index.isin(prev_index)

            if new_index.sum() == 0 or (
                keep_index.sum() + new_index.sum() == len(now_index)
                and len(now_index) > len(prev_index)
            ):
                # Another length (filtered) or new modality added
                # Update .obsm/.varm (size might have changed)
                # NOTE: .get_index doesn't work with duplicated indices
                if any(prev_index.duplicated()):
                    # Assume the relative order of duplicates hasn't changed
                    # NOTE: .get_loc() for each element is too slow
                    # We will rename duplicated in prev_index and now_index
                    # in order to use .get_indexer
                    # index_order = [
                    #    prev_index.get_loc(i) if i in prev_index else -1 for i in now_index
                    # ]
                    prev_values = prev_index.values.copy()
                    now_values = now_index.values.copy()
                    for value in prev_index[np.where(prev_index.duplicated())[0]]:
                        v_now = np.where(now_index == value)[0]
                        v_prev = np.where(prev_index.get_loc(value))[0]
                        for i in range(min(len(v_now), len(v_prev))):
                            prev_values[v_prev[i]] = f"{str(value)}-{i}"
                            now_values[v_now[i]] = f"{str(value)}-{i}"

                    prev_index = pd.Index(prev_values)
                    now_index = pd.Index(now_values)

                index_order = prev_index.get_indexer(now_index)

                for mx_key, mx in attrm.items():
                    if mx_key not in self.mod.keys():  # not a modality name
                        attrm[mx_key] = attrm[mx_key][index_order]
                        attrm[mx_key][index_order == -1] = np.nan

                # Update .obsp/.varp (size might have changed)
                for mx_key, mx in attrp.items():
                    attrp[mx_key] = attrp[mx_key][index_order, index_order]
                    attrp[mx_key][index_order == -1, :] = -1
                    attrp[mx_key][:, index_order == -1] = -1

            elif len(now_index) == len(prev_index):
                # Renamed since new_index.sum() != 0
                # We have to assume the order hasn't changed
                pass

            else:
                raise NotImplementedError(
                    f"{attr}_names seem to have been renamed and filtered at the same time. "
                    "There is no way to restore the order. MuData object has to be re-created from these modalities:\n"
                    "  mdata1 = MuData(mdata.mod)"
                )

        # Write _attrhash
        if attr_changed:
            if not hasattr(self, _attrhash):
                setattr(self, _attrhash, dict())
            for m, mod in self.mod.items():
                getattr(self, _attrhash)[m] = (
                    sha1(np.ascontiguousarray(getattr(mod, attr).index.values)).hexdigest(),
                    sha1(np.ascontiguousarray(getattr(mod, attr).columns.values)).hexdigest(),
                )

        if OPTIONS["pull_on_update"]:
            self._pull_attr(attr, **kwargs)

    def _update_attr_legacy(
        self,
        attr: str,
        axis: int,
        join_common: bool = False,
        **kwargs,  # for _pull_attr()
    ):
        """
        Update global observations/variables with observations/variables for each modality.

        This method will be removed in the next versions. See _update_attr() instead.
        """
        prev_index = getattr(self, attr).index

        # No _attrhash when upon read
        # No _attrhash in mudata < 0.2.0
        _attrhash = f"_{attr}hash"
        attr_changed = self._check_changed_attr_names(attr, columns=True)

        attr_duplicated = self._check_duplicated_attr_names(attr)
        attr_intersecting = self._check_intersecting_attr_names(attr)

        if attr_duplicated:
            warnings.warn(
                f"{attr}_names are not unique. To make them unique, call `.{attr}_names_make_unique`."
            )
            if self._axis == -1:
                warnings.warn(
                    f"Behaviour is not defined with axis=-1, {attr}_names need to be made unique first."
                )

        if not any(attr_changed):
            # Nothing to update
            return

        # Check if the are same obs_names/var_names in different modalities
        # If there are, join_common=True request can not be satisfied
        if join_common:
            if attr_intersecting:
                warnings.warn(
                    f"Cannot join columns with the same name because {attr}_names are intersecting."
                )
                join_common = False

        # Figure out which global columns exist
        columns_global = getattr(self, attr).columns[
            list(
                map(
                    all,
                    zip(
                        *list(
                            [
                                [
                                    not col.startswith(mod + ":")
                                    or col[col.startswith(mod + ":") and len(mod + ":") :]
                                    not in getattr(self.mod[mod], attr).columns
                                    for col in getattr(self, attr).columns
                                ]
                                for mod in self.mod
                            ]
                        )
                    ),
                )
            )
        ]

        # Keep data from global .obs/.var columns
        data_global = getattr(self, attr).loc[:, columns_global]

        # Generate unique colnames
        (rowcol,) = self._find_unique_colnames(attr, 1)

        attrm = getattr(self, attr + "m")
        attrp = getattr(self, attr + "p")
        attrmap = getattr(self, attr + "map")

        if join_common:
            # If all modalities have a column with the same name, it is not global
            columns_common = reduce(
                lambda a, b: a.intersection(b),
                [getattr(self.mod[mod], attr).columns for mod in self.mod],
            )
            data_global = data_global.loc[:, [c not in columns_common for c in data_global.columns]]

        # TODO: take advantage when attr_changed[0] == False — only new columns to be added

        #
        # Join modality .obs/.var tables
        #
        # Main case: no duplicates and no intersection if the axis is not shared
        #
        if not attr_duplicated:
            # Shared axis
            if axis == (1 - self._axis) or self._axis == -1:
                # We assume attr_intersecting and can't join_common
                data_mod = _maybe_coerce_to_bool(
                    pd.concat(
                        [
                            _maybe_coerce_to_boolean(
                                getattr(a, attr)
                                .assign(**{rowcol: np.arange(getattr(a, attr).shape[0])})
                                .add_prefix(m + ":")
                            )
                            for m, a in self.mod.items()
                        ],
                        join="outer",
                        axis=1,
                        sort=False,
                    )
                )
            else:
                if join_common:
                    # We checked above that attr_names are guaranteed to be unique and thus are safe to be used for joins
                    data_mod = pd.concat(
                        [
                            _maybe_coerce_to_boolean(
                                getattr(a, attr)
                                .drop(columns_common, axis=1)
                                .assign(**{rowcol: np.arange(getattr(a, attr).shape[0])})
                                .add_prefix(m + ":")
                            )
                            for m, a in self.mod.items()
                        ],
                        join="outer",
                        axis=0,
                        sort=False,
                    )
                    data_common = pd.concat(
                        [
                            _maybe_coerce_to_boolean(getattr(a, attr)[columns_common])
                            for m, a in self.mod.items()
                        ],
                        join="outer",
                        axis=0,
                        sort=False,
                    )

                    data_mod = _maybe_coerce_to_bool(
                        data_mod.join(data_common, how="left", sort=False)
                    )
                    data_common = _maybe_coerce_to_bool(data_common)

                    # this occurs when join_common=True and we already have a global data frame, e.g. after reading from H5MU
                    sharedcols = data_mod.columns.intersection(data_global.columns)
                    data_global.rename(
                        columns={col: f"global:{col}" for col in sharedcols}, inplace=True
                    )
                else:
                    data_mod = _maybe_coerce_to_bool(
                        pd.concat(
                            [
                                _maybe_coerce_to_boolean(
                                    getattr(a, attr)
                                    .assign(**{rowcol: np.arange(getattr(a, attr).shape[0])})
                                    .add_prefix(m + ":")
                                )
                                for m, a in self.mod.items()
                            ],
                            join="outer",
                            axis=0,
                            sort=False,
                        )
                    )

            for mod, amod in self.mod.items():
                colname = mod + ":" + rowcol
                # use 0 as special value for missing
                # we could use a pandas.array, which has missing values support, but then we get an Exception upon hdf5 write
                # also, this is compatible to Muon.jl
                col = data_mod[colname] + 1
                col.replace(np.nan, 0, inplace=True)
                data_mod[colname] = col.astype(np.uint32)

            if len(data_global.columns) > 0:
                # TODO: if there were intersecting attrnames between modalities,
                #       this will increase the size of the index
                # Should we use attrmap to figure the index out?
                #
                if not attr_intersecting:
                    data_mod = data_mod.join(data_global, how="left", sort=False)
                else:
                    # In order to preserve the order of the index, instead,
                    # perform a join based on (index, cumcount) pairs.
                    col_index, col_cumcount = self._find_unique_colnames(attr, 2)
                    data_mod = data_mod.rename_axis(col_index, axis=0).reset_index()
                    data_mod[col_cumcount] = data_mod.groupby(col_index).cumcount()
                    data_global = data_global.rename_axis(col_index, axis=0).reset_index()
                    data_global[col_cumcount] = (
                        data_global.reset_index().groupby(col_index).cumcount()
                    )
                    data_mod = data_mod.merge(
                        data_global, on=[col_index, col_cumcount], how="left", sort=False
                    )
                    # Restore the index and remove the helper column
                    data_mod = data_mod.set_index(col_index).rename_axis(None, axis=0)
                    del data_mod[col_cumcount]
                    data_global = data_global.set_index(col_index).rename_axis(None, axis=0)
                    del data_global[col_cumcount]

        #
        # General case: with duplicates and/or intersections
        #
        else:
            if join_common:
                dfs = [
                    _maybe_coerce_to_boolean(
                        _make_index_unique(
                            getattr(a, attr)
                            .drop(columns_common, axis=1)
                            .assign(**{rowcol: np.arange(getattr(a, attr).shape[0])})
                            .add_prefix(m + ":")
                        )
                    )
                    for m, a in self.mod.items()
                ]

                # Here, attr_names are guaranteed to be unique and are safe to be used for joins
                data_mod = pd.concat(
                    dfs,
                    join="outer",
                    axis=axis,
                    sort=False,
                )

                data_common = pd.concat(
                    [
                        _maybe_coerce_to_boolean(
                            _make_index_unique(getattr(a, attr)[columns_common])
                        )
                        for m, a in self.mod.items()
                    ],
                    join="outer",
                    axis=0,
                    sort=False,
                )

                data_mod = _maybe_coerce_to_bool(data_mod.join(data_common, how="left", sort=False))
                data_common = _maybe_coerce_to_bool(data_common)
            else:
                dfs = [
                    _make_index_unique(
                        getattr(a, attr)
                        .assign(**{rowcol: np.arange(getattr(a, attr).shape[0])})
                        .add_prefix(m + ":")
                    )
                    for m, a in self.mod.items()
                ]
                data_mod = pd.concat(
                    dfs,
                    join="outer",
                    axis=axis,
                    sort=False,
                )

            # pd.concat wrecks the ordering when doing an outer join with a MultiIndex and different data frame shapes
            if axis == 1:
                newidx = (
                    reduce(lambda x, y: x.union(y, sort=False), (df.index for df in dfs))
                    .to_frame()
                    .reset_index(level=1, drop=True)
                )
                globalidx = data_global.index.get_level_values(0)
                mask = globalidx.isin(newidx.iloc[:, 0])
                if len(mask) > 0:
                    negativemask = ~newidx.index.get_level_values(0).isin(globalidx)
                    newidx = pd.MultiIndex.from_frame(
                        pd.concat(
                            [newidx.loc[globalidx[mask], :], newidx.iloc[negativemask, :]], axis=0
                        )
                    )
                data_mod = data_mod.reindex(newidx, copy=False)

            # this occurs when join_common=True and we already have a global data frame, e.g. after reading from HDF5
            if join_common:
                sharedcols = data_mod.columns.intersection(data_global.columns)
                data_global.rename(
                    columns={col: f"global:{col}" for col in sharedcols}, inplace=True
                )

            data_mod = _restore_index(data_mod)
            data_mod.index.set_names(rowcol, inplace=True)
            data_global.index.set_names(rowcol, inplace=True)
            for mod, amod in self.mod.items():
                colname = mod + ":" + rowcol
                # use 0 as special value for missing
                # we could use a pandas.array, which has missing values support, but then we get an Exception upon hdf5 write
                # also, this is compatible to Muon.jl
                col = data_mod.loc[:, colname] + 1
                col.replace(np.nan, 0, inplace=True)
                col = col.astype(np.uint32)
                data_mod.loc[:, colname] = col
                data_mod.set_index(colname, append=True, inplace=True)
                if mod in attrmap and np.sum(attrmap[mod] > 0) == getattr(amod, attr).shape[0]:
                    data_global.set_index(attrmap[mod], append=True, inplace=True)
                    data_global.index.set_names(colname, level=-1, inplace=True)

            if len(data_global) > 0:
                if not data_global.index.is_unique:
                    warnings.warn(
                        f"{attr}_names is not unique, global {attr} is present, and {attr}map is empty. The update() is not well-defined, verify if global {attr} map to the correct modality-specific {attr}."
                    )
                    data_mod.reset_index(
                        data_mod.index.names.difference(data_global.index.names), inplace=True
                    )
                    data_mod = _make_index_unique(data_mod)
                    data_global = _make_index_unique(data_global)
                data_mod = data_mod.join(data_global, how="left", sort=False)
            data_mod.reset_index(level=list(range(1, data_mod.index.nlevels)), inplace=True)
            data_mod.index.set_names(None, inplace=True)

        if join_common:
            for col in sharedcols:
                gcol = f"global:{col}"
                if data_mod[col].equals(data_mod[gcol]):
                    data_mod.drop(columns=gcol, inplace=True)
                else:
                    warnings.warn(
                        f"Column {col} was present in {attr} but is also a common column in all modalities, and their contents differ. {attr}.{col} was renamed to {attr}.{gcol}."
                    )

        # get adata positions and remove columns from the data frame
        mdict = dict()
        for m in self.mod.keys():
            colname = m + ":" + rowcol
            mdict[m] = data_mod[colname].to_numpy()
            data_mod.drop(colname, axis=1, inplace=True)

        # Add data from global .obs/.var columns # This might reduce the size of .obs/.var if observations/variables were removed
        setattr(
            # Original index is present in data_global
            self,
            "_" + attr,
            data_mod,
        )

        # Update .obsm/.varm
        # this needs to be after setting _obs/_var due to dimension checking in the aligned mapping
        attrmap.clear()
        attrmap.update(mdict)
        for mod, mapping in mdict.items():
            attrm[mod] = mapping > 0

        now_index = getattr(self, attr).index

        if len(prev_index) == 0:
            # New object
            pass
        elif now_index.equals(prev_index):
            # Index is the same
            pass
        else:
            keep_index = prev_index.isin(now_index)
            new_index = ~now_index.isin(prev_index)

            if new_index.sum() == 0 or (
                keep_index.sum() + new_index.sum() == len(now_index)
                and len(now_index) > len(prev_index)
            ):
                # Another length (filtered) or new modality added
                # Update .obsm/.varm (size might have changed)
                # NOTE: .get_index doesn't work with duplicated indices
                if any(prev_index.duplicated()):
                    # Assume the relative order of duplicates hasn't changed
                    # NOTE: .get_loc() for each element is too slow
                    # We will rename duplicated in prev_index and now_index
                    # in order to use .get_indexer
                    # index_order = [
                    #    prev_index.get_loc(i) if i in prev_index else -1 for i in now_index
                    # ]
                    prev_values = prev_index.values.copy()
                    now_values = now_index.values.copy()
                    for value in prev_index[np.where(prev_index.duplicated())[0]]:
                        v_now = np.where(now_index == value)[0]
                        v_prev = np.where(prev_index.get_loc(value))[0]
                        for i in range(min(len(v_now), len(v_prev))):
                            prev_values[v_prev[i]] = f"{str(value)}-{i}"
                            now_values[v_now[i]] = f"{str(value)}-{i}"

                    prev_index = pd.Index(prev_values)
                    now_index = pd.Index(now_values)

                index_order = prev_index.get_indexer(now_index)

                for mx_key, mx in attrm.items():
                    if mx_key not in self.mod.keys():  # not a modality name
                        attrm[mx_key] = attrm[mx_key][index_order]
                        attrm[mx_key][index_order == -1] = np.nan

                # Update .obsp/.varp (size might have changed)
                for mx_key, mx in attrp.items():
                    attrp[mx_key] = attrp[mx_key][index_order, index_order]
                    attrp[mx_key][index_order == -1, :] = -1
                    attrp[mx_key][:, index_order == -1] = -1

            elif len(now_index) == len(prev_index):
                # Renamed since new_index.sum() != 0
                # We have to assume the order hasn't changed
                pass

            else:
                raise NotImplementedError(
                    f"{attr}_names seem to have been renamed and filtered at the same time. "
                    "There is no way to restore the order. MuData object has to be re-created from these modalities:\n"
                    "  mdata1 = MuData(mdata.mod)"
                )

        # Write _attrhash
        if attr_changed:
            if not hasattr(self, _attrhash):
                setattr(self, _attrhash, dict())
            for m, mod in self.mod.items():
                getattr(self, _attrhash)[m] = (
                    sha1(np.ascontiguousarray(getattr(mod, attr).index.values)).hexdigest(),
                    sha1(np.ascontiguousarray(getattr(mod, attr).columns.values)).hexdigest(),
                )

    def _shrink_attr(self, attr: str, inplace=True) -> pd.DataFrame:
        """
        Remove observations/variables for each modality from the global observations/variables table
        """
        # Figure out which global columns exist
        columns_global = list(
            map(
                all,
                zip(
                    *list(
                        [
                            [not col.startswith(mod + ":") for col in getattr(self, attr).columns]
                            for mod in self.mod
                        ]
                    )
                ),
            )
        )
        # Make sure modname-prefix columns exist in modalities,
        # keep them in place if they don't
        for mod in self.mod:
            for i, col in enumerate(getattr(self, attr).columns):
                if col.startswith(mod + ":"):
                    mcol = col[len(mod) + 1 :]
                    if mcol not in getattr(self.mod[mod], attr).columns:
                        columns_global[i] = True
        # Only keep data from global .obs/.var columns
        newdf = getattr(self, attr).loc[:, columns_global]
        if inplace:
            setattr(self, attr, newdf)
        return newdf

    @property
    def n_mod(self) -> int:
        """
        Number of modalities in the MuData object.

        Returns:
            int: The number of modalities.
        """
        return len(self.mod)

    @property
    def isbacked(self) -> bool:
        """
        Whether the MuData object is backed.

        Returns:
            bool: True if the object is backed, False otherwise.
        """
        return self.file.filename is not None

    @property
    def filename(self) -> Path | None:
        """
        Filename of the MuData object.

        Returns:
            Path | None: The path to the file if backed, None otherwise.
        """
        return self.file.filename

    @filename.setter
    def filename(self, filename: PathLike | None):
        filename = None if filename is None else Path(filename)
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
            for ad in self.mod.values():
                ad._X = None

    @property
    def obs(self) -> pd.DataFrame:
        """
        Annotation of observation
        """
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
        """
        Total number of observations
        """
        return self._obs.shape[0]

    def obs_vector(self, key: str, layer: str | None = None) -> np.ndarray:
        """
        Return an array of values for the requested key of length n_obs
        """
        if key not in self.obs.columns:
            for m, a in self.mod.items():
                if key in a.obs.columns:
                    raise KeyError(
                        f"There is no {key} in MuData .obs but there is one in {m} .obs. Consider running `mu.update_obs()` to update global .obs."
                    )
            raise KeyError(f"There is no key {key} in MuData .obs or in .obs of any modalities.")
        return self.obs[key].values

    def update_obs(self):
        """
        Update global .obs_names according to the .obs_names of all the modalities.
        """
        join_common = self.axis == 1
        self._update_attr("obs", axis=1, join_common=join_common)

    def obs_names_make_unique(self):
        """
        Call .obs_names_make_unique() method on each AnnData object.

        If there are obs_names, which are the same for multiple modalities,
        append modality name to all obs_names.
        """
        mod_obs_sum = np.sum([a.n_obs for a in self.mod.values()])
        if mod_obs_sum != self.n_obs:
            self.update_obs()

        for k in self.mod:
            self.mod[k].obs_names_make_unique()

        # Check if there are observations with the same name in different modalities
        common_obs = []
        mods = list(self.mod.keys())
        for i in range(len(self.mod) - 1):
            ki = mods[i]
            for j in range(i + 1, len(self.mod)):
                kj = mods[j]
                common_obs.append(
                    self.mod[ki].obs_names.intersection(self.mod[kj].obs_names.values)
                )
        if any(map(lambda x: len(x) > 0, common_obs)):
            warnings.warn(
                "Modality names will be prepended to obs_names since there are identical obs_names in different modalities."
            )
            for k in self.mod:
                self.mod[k].obs_names = k + ":" + self.mod[k].obs_names.astype(str)

        # Update .obs.index in the MuData
        obs_names = [obs for a in self.mod.values() for obs in a.obs_names.values]
        self._obs.index = obs_names

    @property
    def obs_names(self) -> pd.Index:
        """
        Names of variables (alias for `.obs.index`)

        This property is read-only.
        To be modified, obs_names of individual modalities
        should be changed, and .update_obs() should be called then.
        """
        return self.obs.index

    @property
    def var(self) -> pd.DataFrame:
        """
        Annotation of variables
        """
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
        """
        Total number of variables
        """
        return self._var.shape[0]

    @property
    def n_var(self) -> int:
        """
        Total number of variables
        """
        # warnings.warn(
        #     ".n_var will be removed in the next version, use .n_vars instead",
        #     DeprecationWarning,
        #     stacklevel=2,
        # )
        return self._var.shape[0]

    def var_vector(self, key: str, layer: str | None = None) -> np.ndarray:
        """
        Return an array of values for the requested key of length n_var
        """
        if key not in self.var.columns:
            for m, a in self.mod.items():
                if key in a.var.columns:
                    raise KeyError(
                        f"There is no {key} in MuData .var but there is one in {m} .var. Consider running `mu.update_var()` to update global .var."
                    )
            raise KeyError(f"There is no key {key} in MuData .var or in .var of any modalities.")
        return self.var[key].values

    def update_var(self):
        """
        Update global .var_names according to the .var_names of all the modalities.
        """
        join_common = self.axis == 0
        self._update_attr("var", axis=0, join_common=join_common)

    def var_names_make_unique(self):
        """
        Call .var_names_make_unique() method on each AnnData object.

        If there are var_names, which are the same for multiple modalities,
        append modality name to all var_names.
        """
        mod_var_sum = np.sum([a.n_vars for a in self.mod.values()])
        if mod_var_sum != self.n_vars:
            self.update_var()

        for k in self.mod:
            self.mod[k].var_names_make_unique()

        # Check if there are variables with the same name in different modalities
        common_vars = []
        mods = list(self.mod.keys())
        for i in range(len(self.mod) - 1):
            ki = mods[i]
            for j in range(i + 1, len(self.mod)):
                kj = mods[j]
                common_vars.append(
                    np.intersect1d(self.mod[ki].var_names.values, self.mod[kj].var_names.values)
                )
        if any(map(lambda x: len(x) > 0, common_vars)):
            warnings.warn(
                "Modality names will be prepended to var_names since there are identical var_names in different modalities."
            )
            for k in self.mod:
                self.mod[k].var_names = k + ":" + self.mod[k].var_names.astype(str)

        # Update .var.index in the MuData
        var_names = [var for a in self.mod.values() for var in a.var_names.values]
        self._var.index = var_names

    @property
    def var_names(self) -> pd.Index:
        """
        Names of variables (alias for `.var.index`)

        This property is read-only.
        To be modified, var_names of individual modalities
        should be changed, and .update_var() should be called then.
        """
        return self.var.index

    # Multi-dimensional annotations (.obsm and .varm)

    @property
    def obsm(self) -> MuAxisArrays | MuAxisArraysView:
        """
        Multi-dimensional annotation of observation
        """
        return self._obsm

    @obsm.setter
    def obsm(self, value):
        obsm = MuAxisArrays(self, axis=0, store=convert_to_dict(value))
        if self.is_view:
            self._init_as_actual(self.copy())
        self._obsm = obsm

    @obsm.deleter
    def obsm(self):
        self.obsm = dict()

    @property
    def obsp(self) -> PairwiseArrays | PairwiseArraysView:
        """
        Pairwise annotatation of observations
        """
        return self._obsp

    @obsp.setter
    def obsp(self, value):
        obsp = PairwiseArrays(self, axis=0, store=convert_to_dict(value))
        if self.is_view:
            self._init_as_actual(self.copy())
        self._obsp = obsp

    @obsp.deleter
    def obsp(self):
        self.obsp = dict()

    @property
    def obsmap(self) -> PairwiseArrays | PairwiseArraysView:
        """
        Mapping of observation index in the MuData to indices in individual modalities.

        1-based, 0 indicates that the corresponding observation is missing in the respective modality.
        """
        return self._obsmap

    @property
    def varm(self) -> MuAxisArrays | MuAxisArraysView:
        """
        Multi-dimensional annotation of variables
        """
        return self._varm

    @varm.setter
    def varm(self, value):
        varm = MuAxisArrays(self, axis=1, store=convert_to_dict(value))
        if self.is_view:
            self._init_as_actual(self.copy())
        self._varm = varm

    @varm.deleter
    def varm(self):
        self.varm = dict()

    @property
    def varp(self) -> PairwiseArrays | PairwiseArraysView:
        """
        Pairwise annotatation of variables
        """
        return self._varp

    @varp.setter
    def varp(self, value):
        varp = PairwiseArrays(self, axis=0, store=convert_to_dict(value))
        if self.is_view:
            self._init_as_actual(self.copy())
        self._varp = varp

    @varp.deleter
    def varp(self):
        self.varp = dict()

    @property
    def varmap(self) -> PairwiseArrays | PairwiseArraysView:
        """
        Mapping of feature index in the MuData to indices in individual modalities.

        1-based, 0 indicates that the corresponding observation is missing in the respective modality.
        """
        return self._varmap

    # Unstructured annotations
    # NOTE: annotations are stored as dict() and not as OrderedDict() as in AnnData

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
        self.uns = dict()

    # _keys methods to increase compatibility
    # with calls requiring those AnnData methods

    def obs_keys(self) -> list[str]:
        """List keys of observation annotation :attr:`obs`."""
        return self._obs.keys().tolist()

    def var_keys(self) -> list[str]:
        """List keys of variable annotation :attr:`var`."""
        return self._var.keys().tolist()

    def obsm_keys(self) -> list[str]:
        """List keys of observation annotation :attr:`obsm`."""
        return list(self._obsm.keys())

    def varm_keys(self) -> list[str]:
        """List keys of variable annotation :attr:`varm`."""
        return list(self._varm.keys())

    def uns_keys(self) -> list[str]:
        """List keys of unstructured annotation."""
        return list(self._uns.keys())

    def update(self):
        """
        Update both .obs and .var indices of MuData with the data from all the modalities

        NOTE: From v0.4, it will not pull columns from modalities by default.
        """
        self.update_var()
        self.update_obs()

    @property
    def axis(self) -> int:
        """
        MuData axis
        """
        return self._axis

    @property
    def mod_names(self) -> list[str]:
        """
        Names of modalities (alias for `list(mdata.mod.keys())`)

        This property is read-only.
        """
        return list(self.mod.keys())

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
        prefix_unique: bool | None = True,
        drop: bool = False,
        only_drop: bool = False,
    ):
        """
        Copy the data from the modalities to the global .obs/.var,
        existing columns to be overwritten

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
            mods = list(dict.fromkeys(mods))
            if not all(m in self.mod for m in mods):
                raise ValueError("All mods should be present in mdata.mod")
            elif len(mods) == self.n_mod:
                mods = None
            for k, v in {"common": common, "nonunique": nonunique, "unique": unique}.items():
                assert v is None, f"Cannot use mods with {k}."

        if only_drop:
            drop = True

        cols = _classify_attr_columns(
            np.concatenate(
                [
                    [f"{m}:{val}" for val in getattr(mod, attr).columns.values]
                    for m, mod in self.mod.items()
                ]
            ),
            self.mod.keys(),
        )

        if columns is not None:
            for k, v in {"common": common, "nonunique": nonunique, "unique": unique}.items():
                assert v is None, f"Cannot use {k} with columns."

            # - modname1:column -> [modname1:column]
            # - column -> [modname1:column, modname2:column, ...]
            cols = [col for col in cols if col["name"] in columns or col["derived_name"] in columns]

            if mods is not None:
                cols = [col for col in cols if col["prefix"] in mods]

            # TODO: Counter for columns in order to track their usage
            # and error out if some columns were not used

        else:
            if common is None:
                common = True
            if nonunique is None:
                nonunique = True
            if unique is None:
                unique = True

            selector = {"common": common, "nonunique": nonunique, "unique": unique}

            cols = [col for col in cols if selector[col["class"]]]

        derived_name_count = Counter([col["derived_name"] for col in cols])

        # - axis == self.axis
        #   e.g. combine var from multiple modalities (with unique vars)
        # - 1 - axis == self.axis
        # . e.g. combine obs from multiple modalities (with shared obs)
        axis = 0 if attr == "var" else 1

        if 1 - axis == self.axis or self.axis == -1:
            if join_common or join_nonunique:
                raise ValueError(f"Cannot join columns with the same name for shared {attr}_names.")

        if join_common is None:
            join_common = False
            if attr == "var":
                join_common = self.axis == 0
            elif attr == "obs":
                join_common = self.axis == 1

        if join_nonunique is None:
            join_nonunique = False

        if prefix_unique is None:
            prefix_unique = True

        # Below we will rely on attrmap that has been calculated during .update()
        # and use it to create an index without duplicates
        # for faster concatenation and to reduce the amount of code

        attrmap = getattr(self, f"{attr}map")
        n_attr = self.n_vars if attr == "var" else self.n_obs

        dfs: list[pd.DataFrame] = []
        for m, mod in self.mod.items():
            if mods is not None and m not in mods:
                continue
            mod_map = attrmap[m]
            mod_n_attr = mod.n_vars if attr == "var" else mod.n_obs
            mask = mod_map != 0

            mod_df = getattr(mod, attr)
            mod_columns = [
                col["derived_name"] for col in cols if col["prefix"] == "" or col["prefix"] == m
            ]
            mod_df = mod_df[mod_df.columns.intersection(mod_columns)]

            if drop:
                getattr(mod, attr).drop(columns=mod_df.columns, inplace=True)

            # Don't use modname: prefix if columns need to be joined
            if join_common or join_nonunique or (not prefix_unique):
                cols_special = [
                    col["derived_name"]
                    for col in cols
                    if (
                        (col["class"] == "common") & join_common
                        or (col["class"] == "nonunique") & join_nonunique
                        or (col["class"] == "unique") & (not prefix_unique)
                    )
                    and col["prefix"] == m
                    and derived_name_count[col["derived_name"]] == col["count"]
                ]
                mod_df.columns = [
                    col if col in cols_special else f"{m}:{col}" for col in mod_df.columns
                ]
            else:
                mod_df.columns = [f"{m}:{col}" for col in mod_df.columns]

            mod_df = (
                _maybe_coerce_to_boolean(mod_df)
                .set_index(np.arange(mod_n_attr))
                .iloc[mod_map[mask] - 1]
                .set_index(np.arange(n_attr)[mask])
                .reindex(np.arange(n_attr))
            )
            dfs.append(mod_df)

        if only_drop:
            return

        global_df = _maybe_coerce_to_boolean(getattr(self, attr).set_index(np.arange(n_attr)))
        df = reduce(_update_and_concat, [global_df, *dfs])
        df = _maybe_coerce_to_bool(df)
        df = _maybe_coerce_to_int(df)
        df = df.set_index(getattr(self, f"{attr}_names"))
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
        prefix_unique: bool | None = True,
        drop: bool = False,
        only_drop: bool = False,
    ):
        """
        Copy the data from the modalities to the global .obs,
        existing columns to be overwritten or updated

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
        prefix_unique: bool | None = True,
        drop: bool = False,
        only_drop: bool = False,
    ):
        """
        Copy the data from the modalities to the global .var,
        existing columns to be overwritten or updated

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
        Copy the data from the global .obs/.var to the modalities,
        existing columns to be overwritten

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
            which are prefixed by modality names.
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
                mods = [mods]
            mods = list(dict.fromkeys(mods))
            if not all(m in self.mod for m in mods):
                raise ValueError("All mods should be present in mdata.mod")
            elif len(mods) == self.n_mod:
                mods = None
            for k, v in {"common": common, "prefixed": prefixed}.items():
                assert v is None, f"Cannot use mods with {k}."

        if only_drop:
            drop = True

        cols = _classify_prefixed_columns(getattr(self, attr).columns.values, self.mod.keys())

        if columns is not None:
            for k, v in {"common": common, "prefixed": prefixed}.items():
                assert v is None, f"Cannot use columns with {k}."

            # - modname1:column -> [modname1:column]
            # - column -> [modname1:column, modname2:column, ...]
            cols = [col for col in cols if col["name"] in columns or col["derived_name"] in columns]

            # preemptively drop columns from other modalities
            if mods is not None:
                cols = [col for col in cols if col["prefix"] in mods or col["prefix"] == ""]
        else:
            if common is None:
                common = True
            if prefixed is None:
                prefixed = True

            selector = {"common": common, "prefixed": prefixed}

            cols = [col for col in cols if selector[col["class"]]]

        if len(cols) == 0:
            return

        derived_name_count = Counter([col["derived_name"] for col in cols])
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
        _n_attr = self.n_vars if attr == "var" else self.n_obs

        for m, mod in self.mod.items():
            if mods is not None and m not in mods:
                continue

            mod_map = attrmap[m]
            mask = mod_map != 0
            mod_n_attr = mod.n_vars if attr == "var" else mod.n_obs

            mod_cols = [col for col in cols if col["prefix"] == m or col["class"] == "common"]
            df = getattr(self, attr)[mask].loc[:, [col["name"] for col in mod_cols]]
            df.columns = [col["derived_name"] for col in mod_cols]

            df = (
                df.set_index(np.arange(mod_n_attr))
                .iloc[mod_map[mask] - 1]
                .set_index(np.arange(mod_n_attr))
            )

            if not only_drop:
                # TODO: _maybe_coerce_to_bool
                # TODO: _maybe_coerce_to_int
                # TODO: _prune_unused_categories
                mod_df = getattr(mod, attr).set_index(np.arange(mod_n_attr))
                mod_df = _update_and_concat(mod_df, df)
                mod_df = mod_df.set_index(getattr(mod, f"{attr}_names"))
                setattr(mod, attr, mod_df)

        if drop:
            for col in cols:
                getattr(self, attr).drop(col["name"], axis=1, inplace=True)

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
        Copy the data from the mdata.obs to the modalities,
        existing columns to be overwritten

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
            "obs",
            columns=columns,
            mods=mods,
            common=common,
            prefixed=prefixed,
            drop=drop,
            only_drop=only_drop,
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
        Copy the data from the mdata.var to the modalities,
        existing columns to be overwritten

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
            "var",
            columns=columns,
            mods=mods,
            common=common,
            prefixed=prefixed,
            drop=drop,
            only_drop=only_drop,
        )

    def write_h5mu(self, filename: str | None = None, **kwargs):
        """
        Write MuData object to an HDF5 file
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

    def write_zarr(self, store: MutableMapping | str | Path, **kwargs):
        """
        Write MuData object to a Zarr store
        """
        from .io import write_zarr

        write_zarr(store, self, **kwargs)

    def to_anndata(self, **kwargs) -> AnnData:
        """
        Convert MuData to AnnData

        If mdata.axis == 0 (shared observations),
        concatenate modalities along axis 1 (`anndata.concat(axis=1)`).
        If mdata.axis == 1 (shared variables),
        concatenate datasets along axis 0 (`anndata.concat(axis=0)`).

        See `anndata.concat()` documentation for more details.

        Parameters
        ----------
        data    : MuData
            MuData object to convert  to AnnData
        kwargs  : dict
            Keyword arguments passed to `anndata.concat()`
        """
        from .to_ import to_anndata

        return to_anndata(self, **kwargs)

    def _gen_repr(self, n_obs, n_vars, extensive: bool = False, nest_level: int = 0) -> str:
        indent = "    " * nest_level
        backed_at = f" backed at {str(self.filename)!r}" if self.isbacked else ""
        view_of = "View of " if self.is_view else ""
        maybe_axis = (
            (
                ""
                if self.axis == 0
                else " (shared var) " if self.axis == 1 else " (shared obs and var) "
            )
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
                                *list(
                                    [
                                        [
                                            not col.startswith(mod + mod_sep)
                                            for col in getattr(self, attr).keys()
                                        ]
                                        for mod in self.mod
                                    ]
                                )
                            ),
                        )
                    )
                    if any(global_keys):
                        descr += f"\n{indent}  {attr}:\t{str([keys[i] for i in range(len(keys)) if global_keys[i]])[1:-1]}"
        descr += f"\n{indent}  {len(self.mod)} modalit{'y' if len(self.mod) == 1 else 'ies'}"
        for k, v in self.mod.items():
            mod_indent = "    " * (nest_level + 1)
            if isinstance(v, MuData):
                descr += f"\n{mod_indent}{k}:\t" + v._gen_repr(
                    v.n_obs, v.n_vars, extensive, nest_level + 1
                )
                continue
            descr += f"\n{mod_indent}{k}:\t{v.n_obs} x {v.n_vars}"
            for attr in [
                "obs",
                "var",
                "uns",
                "obsm",
                "varm",
                "layers",
                "obsp",
                "varp",
            ]:
                try:
                    keys = getattr(v, attr).keys()
                    if len(keys) > 0:
                        descr += f"\n{mod_indent}  {attr}:\t{str(list(keys))[1:-1]}"
                except AttributeError:
                    pass
        return descr

    def __repr__(self) -> str:
        return self._gen_repr(self.n_obs, self.n_vars, extensive=True)

    def _repr_html_(self, expand=None):
        """
        HTML formatter for MuData objects
        for rich display in notebooks.

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
            self.n_obs, self.n_vars, len(self.mod), "y" if len(self.mod) < 2 else "ies"
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

        for m, dat in self.mod.items():
            mods += "<div class='block-mod'><div>"
            mods += "<details{}>".format(" open" if (expand & 0b010) >> 1 else "")
            mods += "<summary class='summary-mod'><div class='title title-mod'>{}</div><span class='hl-dim'>{} &times {}</span></summary>".format(
                m, *dat.shape
            )

            # General object properties
            mods += (
                "<span>{} object <span class='hl-dim'>{} obs &times; {} var</span></span>".format(
                    type(dat).__name__, *(dat.shape)
                )
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

    def _find_unique_colnames(self, attr: str, ncols: int):
        nchars = 16
        allunique = False
        while not allunique:
            colnames = ["".join(choices(ascii_letters + digits, k=nchars)) for _ in range(ncols)]
            allunique = len(set(colnames)) == ncols
            nchars *= 2

        for i in range(ncols):
            finished = False
            while not finished:
                for ad in chain((self,), self.mod.values()):
                    if colnames[i] in getattr(ad, attr).columns:
                        colnames[i] = "_" + colnames[i]
                        break
                finished = True
        return colnames
