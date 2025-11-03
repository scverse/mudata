from collections import Counter
from collections.abc import Sequence
from typing import Literal, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T", pd.Series, pd.DataFrame)


def _make_index_unique(df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    if force or not df.index.is_unique:
        dup_idx = np.zeros((df.shape[0],), dtype=np.uint8)
        duplicates = np.nonzero(df.index.duplicated())[0]
        cnt = Counter()
        for dup in duplicates:
            idxval = df.index[dup]
            newval = cnt[idxval] + 1
            try:
                dup_idx[dup] = newval
            except OverflowError:
                dup_idx = dup_idx.astype(np.min_scalar_type(newval))
                dup_idx[dup] = newval
            cnt[idxval] = newval
        return df.set_index(dup_idx, append=True)
    else:
        return df


def _restore_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(level=-1, drop=True) if df.index.nlevels > 1 else df


def _maybe_coerce_to_boolean(df: T) -> T:
    if isinstance(df, pd.Series):
        if df.dtype == bool:
            return df.astype("boolean")
        return df

    for col in df.columns:
        if df[col].dtype == bool:
            df = df.assign(**{col: df[col].astype("boolean")})

    return df


class MetadataColumn:
    __slots__ = ("prefix", "derived_name", "count", "_allowed_prefixes", "_strip_prefix")

    def __init__(
        self,
        *,
        allowed_prefixes: Sequence[str],
        prefix: str | None = None,
        name: str | None = None,
        count: int = 0,
        strip_prefix: bool = True,
    ):
        self._strip_prefix = strip_prefix
        self._allowed_prefixes = allowed_prefixes
        self.prefix = prefix
        if prefix is None and strip_prefix:
            self.name = name
        else:
            self.prefix = prefix
            self.derived_name = name
        self.count = count

    @property
    def name(self) -> str:
        if self.prefix is not None:
            return f"{self.prefix}:{self.derived_name}"
        else:
            return self.derived_name

    @name.setter
    def name(self, new_name):
        if (
            not self._strip_prefix
            or len(name_split := new_name.split(":", 1)) < 2
            or name_split[0] not in self._allowed_prefixes
        ):
            self.prefix = None
            self.derived_name = new_name
        else:
            self.prefix, self.derived_name = name_split

    @property
    def klass(self) -> Literal["common", "unique", "nonunique", "unknown"]:
        if self.prefix is None or self.count == len(self._allowed_prefixes):
            return "common"
        elif self.count == 1:
            return "unique"
        elif self.count > 0:
            return "nonunique"
        else:
            return "unknown"


def _update_and_concat(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df = df1.copy(deep=False)
    # This converts boolean to object dtype, unfortunately
    # df.update(df2)
    common_cols = df1.columns.intersection(df2.columns)
    for col in common_cols:
        if isinstance(df[col].values, pd.Categorical) and isinstance(
            df2[col].values, pd.Categorical
        ):
            common_cats = pd.api.types.union_categoricals([df[col], df2[col]]).categories
            df[col] = df[col].cat.set_categories(common_cats)
            df2[col] = df2[col].cat.set_categories(common_cats)
            df.update({col: df2[col]})
        else:
            df.update({col: df2[col]})
    new_cols = df2.columns.difference(df1.columns)
    res = pd.concat([df, df2[new_cols]], axis=1, sort=False, verify_integrity=True)
    return res


def _maybe_coerce_to_bool(df: T) -> T:
    if isinstance(df, pd.Series):
        if isinstance(df.dtype, pd.BooleanDtype):
            try:
                return df.astype(bool)
            except ValueError:
                # cannot convert float NaN to bool
                return df
        return df

    for col in df.columns:
        if isinstance(df[col].dtype, pd.BooleanDtype):
            try:
                df = df.assign(**{col: df[col].astype(bool)})
            except ValueError:
                # cannot convert float NaN to bool
                pass

    return df


def _maybe_coerce_to_int(df: T) -> T:
    if isinstance(df, pd.Series):
        if isinstance(df.dtype, pd.Int64Dtype):
            try:
                return df.astype(int)
            except ValueError:
                # cannot convert float NaN to int
                return df
        return df

    for col in df.columns:
        if isinstance(df[col].dtype, pd.Int64Dtype):
            try:
                df = df.assign(**{col: df[col].astype(int)})
            except ValueError:
                # cannot convert float NaN to int
                pass

    return df


def update_fix_attrmap_col(data_mod: pd.DataFrame, mod: str, rowcol: str) -> str:
    colname = mod + ":" + rowcol
    # use 0 as special value for missing
    # we could use a pandas.array, which has missing values support, but then we get an Exception upon hdf5 write
    # also, this is compatible to Muon.jl
    col = data_mod[colname] + 1
    col.replace(np.nan, 0, inplace=True)
    data_mod[colname] = col.astype(np.uint32)
    return colname


def update_reorder_df_and_attrm_index(
    data_mod: pd.DataFrame,
    data_global: pd.DataFrame,
    axis: Literal[-1, 0, 1],
    mdaxis: Literal[-1, 0, 1],
) -> tuple[pd.DataFrame, np.ndarray[np.intp], bool]:
    # reorder new index to conform to the old index as much as possible
    kept_idx = data_global.index[data_global.index.isin(data_mod.index)]
    new_idx = data_mod.index[~data_mod.index.isin(data_global.index)]
    data_mod = data_mod.loc[kept_idx.append(new_idx), :]

    index_order = data_global.index.get_indexer(data_mod.index)
    can_update = (
        new_idx.shape[0] == 0  # filtered or reordered
        or kept_idx.shape[0] == data_global.shape[0]  # new rows only
        or data_mod.shape[0]
        == data_global.shape[
            0
        ]  # renamed (since new_idx.shape[0] > 0 and kept_idx.shape[0] < data_global.shape[0])
        or (
            axis == mdaxis and axis != -1 and data_mod.shape[0] > data_global.shape[0]
        )  # new modality added and concacenated
    )

    return data_mod, index_order, can_update
