from __future__ import annotations

from collections import Counter
from contextlib import suppress
from functools import wraps
from typing import TYPE_CHECKING, Literal
from warnings import warn

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


def deprecated(version: str, msg: str | None = None):
    def decorate(func: Callable):
        if func.__name__ == func.__qualname__:
            warnmsg = f"The function {func.__name__} is deprecated and will be removed in the future."
        else:
            warnmsg = f"The method {func.__qualname__} is deprecated and will be removed in the future."

        doc = func.__doc__
        indentation = 0
        if doc is not None:
            lines = doc.expandtabs().splitlines()
            with suppress(StopIteration):
                for line in lines[1:]:
                    if not len(line):
                        continue
                    for indentation, char in enumerate(line):  # noqa: B007
                        if not char.isspace():
                            raise StopIteration  # break out of both loops
        indentation = " " * indentation

        docmsg = f"{indentation}.. version-deprecated:: {version}"
        if msg is not None:
            docmsg += f"\n{indentation}   {msg}"
            warnmsg += f" {msg}"

        if doc is None:
            doc = docmsg
        else:
            body = "\n".join(lines[1:])
            doc = f"{lines[0]}\n\n{docmsg}\n{body}"
        func.__doc__ = doc

        @wraps(func)
        def decorated(*args, **kwargs):
            warn(warnmsg, FutureWarning, stacklevel=2)
            return func(*args, **kwargs)

        return decorated

    return decorate


def _make_index_unique(df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    if not force and df.index.is_unique:
        return df

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


def _restore_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(level=-1, drop=True) if df.index.nlevels > 1 else df


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
        if isinstance(df[col].values, pd.Categorical) and isinstance(df2[col].values, pd.Categorical):
            common_cats = pd.api.types.union_categoricals([df[col], df2[col]]).categories
            df[col] = df[col].cat.set_categories(common_cats)
            df2[col] = df2[col].cat.set_categories(common_cats)
            df.update({col: df2[col]})
        else:
            df.update({col: df2[col]})
    new_cols = df2.columns.difference(df1.columns)
    res = pd.concat([df, df2[new_cols]], axis=1, sort=False, verify_integrity=True)
    return res


def try_convert_series_to_numpy_dtype(col: pd.Series) -> pd.Series:
    """Attempt to convert a :class:`~pandas.Series` to a non-nullable dtype.

    Parameters
    ----------
    col
        The series to be converted.

    Returns
    -------
    The converted series, `col` did not contain any :data:`~pandas.NA` values, the unmodified `col` otherwise.
    """
    with suppress(ValueError):
        match col.dtype:
            case pd.BooleanDtype():
                col = col.astype(bool)
            case pd.core.arrays.integer.IntegerDtype(type=dtype) | pd.core.arrays.floating.FloatingDtype(type=dtype):
                col = col.astype(dtype)
            case pd.StringDtype():
                col = col.astype(object)
    return col


def try_convert_dataframe_to_numpy_dtypes(df: pd.DataFrame | Mapping[str, pd.Series]) -> pd.DataFrame:
    """Attempt to convert all columns of a :class:`~pandas.DataFrame` to their respective non-nullable dtype.

    Parameters
    ----------
    df
        The dataframe to be converted.

    Returns
    -------
    A new dataframe with each column of `df` that had a nullable dtype but did not contain any :data:`~pandas.NA`
    values converted to the corresponding non-nullable dtype.
    """
    new_cols = {}
    for colname, col in df.items():
        new_cols[colname] = try_convert_series_to_numpy_dtype(col)
    return pd.DataFrame(new_cols)
