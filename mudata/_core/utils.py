from collections import Counter
from typing import TypeVar
import pandas as pd
import numpy as np
from anndata.utils import make_index_unique

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
