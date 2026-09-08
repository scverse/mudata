from typing import Literal

import anndata
from anndata import AnnData
from scverse_misc import arg_alias

from mudata import MuData


def to_anndata(mdata: MuData, **kwargs) -> AnnData:
    """
    Convert :class:`MuData` to :class:`~anndata.AnnData` by concatenating modalities.

    If `mdata.axis == 0` (shared observations),
    concatenate modalities along axis 1 (`anndata.concat(axis=1)`).
    If `mdata.axis == 1` (shared variables),
    concatenate datasets along axis 0 (`anndata.concat(axis=0)`).

    Parameters
    ----------
    data
        Object to convert to :class:`~anndata.AnnData`.
    **kwargs
        Additional arguments for :func:`anndata.concat`.
    """
    if mdata.axis == -1:
        raise ValueError("Only MuData with axis=0 and axis=1 are supported in `to_anndata()`")
    # data.axis shows which axis is shared so we concatenate along the other axis
    adata = anndata.concat(mdata.mod, axis=1 - mdata.axis, **kwargs)
    # Add columns from mdata.attr to individual modalities
    for attr in ["obs", "var"]:
        df = getattr(adata, attr)
        setattr(adata, attr, df.combine_first(getattr(mdata, attr)))
    for attr in ["obsm", "obsp", "varm", "varp", "uns"]:
        getattr(adata, attr).update(getattr(mdata, attr))
    return adata


@arg_alias("axis")
def to_mudata(adata: AnnData, axis: Literal[0, "obs"] | Literal[1, "var"], by: str) -> MuData:
    """
    Convert :class:`~anndata.AnnData` to :class:`MuData` by splitting it along obs or var.

    Axis signifies the shared axis.

    Parameters
    ----------
    adata
        Object to convert to MuData.
    axis
        Shared axis: `0` for observations, `1` for features. See also :doc:`/notebooks/axes`.
    by
        Key in `adata.var` (if axis=0) or `adata.obs` (if axis=1) to split by
    """
    # Use AnnData.split_by() when it's ready
    # https://github.com/scverse/anndata/pull/613
    attr = "var" if axis == 0 else "obs"
    df = getattr(adata, attr)
    groupby = df[by].astype("category")
    mkeys = groupby.cat.categories
    if axis == 0:
        mod = {str(key): adata[:, df[by] == key].copy() for key in mkeys}
    else:
        mod = {str(key): adata[df[by] == key].copy() for key in mkeys}
    return MuData(mod, axis=axis)
