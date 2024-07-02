from typing import Literal

import anndata
from anndata import AnnData

from mudata import MuData


def to_anndata(mdata: MuData, **kwargs) -> AnnData:
    """
    Convert MuData to AnnData by concatenating modalities

    If mdata.axis == 0 (shared observations),
    concatenate modalities along axis 1 (`anndata.concat(axis=1)`).
    If mdata.axis == 1 (shared variables),
    concatenate datasets along axis 0 (`anndata.concat(axis=0)`).

    Parameters
    ----------
    data    : MuData
        MuData object to convert  to AnnData
    kwargs  : dict
        Keyword arguments passed to anndata.concat
    """
    if mdata.axis == -1:
        raise ValueError("Only MuData with axis=0 and axis=1 are supported in `to_anndata()`")
    # data.axis shows which axis is shared so we concatenate along the other axis
    adata = anndata.concat(mdata.mod, axis=1 - mdata.axis, **kwargs)
    # Add columns from mdata.attr to individual modalities
    for attr in ["obs", "var"]:
        df = getattr(adata, attr)
        if hasattr(df, "combine_first") and callable(df.combine_first):
            setattr(adata, attr, df.combine_first(getattr(mdata, attr)))
    for attr in ["obs", "obsm", "obsp", "var", "varm", "varp", "uns"]:
        getattr(adata, attr).update(getattr(mdata, attr))
    return adata


def to_mudata(
    adata: AnnData,
    axis: Literal[0, 1],
    by: str,
) -> MuData:
    """
    Convert AnnData to MuData by splitting it
    along obs or var

    Axis signifies the shared axis.
    Use `axis=0` for getting MuData with shared observations (axis=0),
    and `axis=1` for getting MuData with shared variables (axis=1).

    Paramteters
    -----------
    adata   : AnnData
        AnnData object to convert to MuData
    axis    : int
        Axis of shared observations (0) or variables (1)
    by      : str
        Key in `adata.var` (if axis=0) or `adata.obs` (if axis=1) to split by
    """
    # Use AnnData.split_by() when it's ready
    # https://github.com/scverse/anndata/pull/613
    attr = "var" if axis == 0 else "obs" if axis == 1 else None
    if attr is None:
        raise ValueError(f"Axis should be 0 or 1, not {axis}")
    df = getattr(adata, attr)
    groupby = df[by].astype("category")
    mkeys = groupby.cat.categories
    if axis == 0:
        mod = {str(key): adata[:, df[by] == key].copy() for key in mkeys}
    else:
        mod = {str(key): adata[df[by] == key].copy() for key in mkeys}
    return MuData(mod, axis=axis)
