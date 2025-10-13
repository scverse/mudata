from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Literal, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T", pd.Series, pd.DataFrame)


def _make_index_unique(df: pd.DataFrame) -> pd.DataFrame:
    dup_idx = np.zeros((df.shape[0],), dtype=np.uint8)
    if not df.index.is_unique:
        duplicates = np.nonzero(df.index.duplicated())[0]
        cnt = Counter()
        for dup in duplicates:
            idxval = df.index[dup]
            newval = cnt[idxval] + 1
            dup_idx[dup] = newval
            cnt[idxval] = newval
    return df.set_index(dup_idx, append=True)


def _restore_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(level=-1, drop=True)


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
    __slots__ = ("prefix", "derived_name", "count", "_allowed_prefixes")

    def __init__(
        self,
        *,
        allowed_prefixes: Sequence[str],
        prefix: str | None = None,
        name: str | None = None,
        count: int = 0,
    ):
        self._allowed_prefixes = allowed_prefixes
        if prefix is None:
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
            len(name_split := new_name.split(":", 1)) < 2
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


def _classify_attr_columns(names: Mapping[str, Sequence[str]]) -> dict[str, list[MetadataColumn]]:
    """
    Classify names into common, non-unique, and unique
    w.r.t. to the list of prefixes.

    - Common columns do not have modality prefixes.
    - Non-unqiue columns have a modality prefix,
      and there are multiple columns that differ
      only by their modality prefix.
    - Unique columns are prefixed by modality names,
      and there is only one modality prefix
      for a column with a certain name.
    """
    res: dict[str, list[MetadataColumn]] = {}

    derived_name_counts = Counter()
    for prefix, pnames in names.items():
        cres = []
        for name in pnames:
            cres.append(MetadataColumn(allowed_prefixes=names.keys(), prefix=prefix, name=name))
            derived_name_counts[name] += 1
        res[prefix] = cres

    for prefix, names in res.items():
        for name_res in names:
            count = derived_name_counts[name_res.derived_name]
            name_res.count = count

    return res


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
