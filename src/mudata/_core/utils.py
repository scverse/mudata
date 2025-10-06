from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TypeVar

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


def _classify_attr_columns(names: Mapping[str, Sequence[str]]) -> dict[str, list[dict[str, str]]]:
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

    E.g. {"mod1": ["annotation", "unique"], "mod2": ["annotation"]} will be classified
    into {"mod1": [{"name": "mod1:annotation", "derived_name": "annotation", "count": 2, "class": "nonunique"},
                   {"name": "mod1:unique", "derived_name": "unique", "count": 1, "class": "unique"}}],
          "mod2": [{"name": "mod2:annotation", "derived_name": "annotation", "count": 2, "class": "nonunique"}],
         }
    """
    n_mod = len(names)
    res: dict[str, list[dict[str, str]]] = {}

    derived_name_counts = Counter()
    for prefix, names in names.items():
        cres = []
        for name in names:
            cres.append(
                {
                    "name": f"{prefix}:{name}",
                    "derived_name": name,
                }
            )
            derived_name_counts[name] += 1
        res[prefix] = cres

    for prefix, names in res.items():
        for name_res in names:
            count = derived_name_counts[name_res["derived_name"]]
            name_res["count"] = count
            name_res["class"] = (
                "common" if count == n_mod else "unique" if count == 1 else "nonunique"
            )

    return res


def _classify_prefixed_columns(
    names: Sequence[str], prefixes: Sequence[str]
) -> Sequence[dict[str, str]]:
    """
    Classify names into common and prefixed
    w.r.t. to the list of prefixes.

    - Common columns do not have modality prefixes.
    - Prefixed columns are prefixed by modality names.

    E.g. ["global", "mod1:annotation", "mod2:annotation", "mod1:unique"] will be classified
    into [
        {"name": "global", "prefix": "", "derived_name": "global", "class": "common"},
        {"name": "mod1:annotation", "prefix": "mod1", "derived_name": "annotation", "class": "prefixed"},
        {"name": "mod2:annotation", "prefix": "mod2", "derived_name": "annotation", "class": "prefixed"},
        {"name": "mod1:unique", "prefix": "mod1", "derived_name": "annotation", "class": "prefixed"},
    ]
    """
    res: list[dict[str, str]] = []

    for name in names:
        if len(name_split := name.split(":", 1)) < 2 or name_split[0] not in prefixes:
            res.append({"name": name, "prefix": "", "derived_name": name, "class": "common"})
        else:
            res.append(
                {
                    "name": name,
                    "prefix": name_split[0],
                    "derived_name": name_split[1],
                    "class": "prefixed",
                }
            )

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
