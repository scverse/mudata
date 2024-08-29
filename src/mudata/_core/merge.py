from collections.abc import Callable, Collection, Mapping
from functools import reduce
from typing import Any, Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from anndata import concat as ad_concat
from anndata._core.merge import (
    StrategiesLiteral,
    check_combinable_cols,
    concat_pairwise_mapping,
    gen_reindexer,
    inner_concat_aligned_mapping,
    intersect_keys,
    merge_dataframes,
    merge_indices,
    outer_concat_aligned_mapping,
    resolve_merge_strategy,
    unify_dtypes,
    union_keys,
)

try:
    from anndata._core.merge import _resolve_axis, axis_indices
except ImportError:
    # anndata < 0.10.9
    from anndata._core.merge import _resolve_dim as _resolve_axis
    from anndata._core.merge import dim_indices as axis_indices

from .mudata import MuData


def concat(
    mdatas: Collection[AnnData] | Mapping[str, AnnData],
    *,
    join: Literal["inner", "outer"] = "inner",
    merge: StrategiesLiteral | Callable | None = None,
    uns_merge: StrategiesLiteral | Callable | None = None,
    label: str | None = None,
    keys: Collection | None = None,
    index_unique: str | None = None,
    fill_value: Any | None = None,
    pairwise: bool = False,
) -> MuData:
    """Concatenates MuData objects.

    All mdatas should have the same axis 0 or 1, which defines concatenation axis:
    - concatenate along obs when obs are shared in each mdata (multimodal),
    - concatenate along vars when vars are shared in each mdata (multi-dataset).

    The intersection of modalities is taken.
    Nested MuData objects cannot be concatenated.

    This implementation follows anndata.concat() original implementation.
    The arguments are propagated to anndata.concat() for concatenating modalities.
    See anndata.concat() documentation for more details.

    Params
    ------
    mdatas
        The objects to be concatenated. If a Mapping is passed, keys are used for the `keys`
        argument and values are concatenated.
    join
        How to align values when concatenating. If "outer", the union of the other axis
        is taken. If "inner", the intersection.
    merge
        How elements not aligned to the axis being concatenated along are selected.
        Currently implemented strategies include:

        * `None`: No elements are kept.
        * `"same"`: Elements that are the same in each of the objects.
        * `"unique"`: Elements for which there is only one possible value.
        * `"first"`: The first element seen at each from each position.
        * `"only"`: Elements that show up in only one of the objects.
    uns_merge
        How the elements of `.uns` are selected. Uses the same set of strategies as
        the `merge` argument, except applied recursively.
    label
        Column in axis annotation (i.e. `.obs` or `.var`) to place batch information in.
        If it's None, no column is added.
    keys
        Names for each object being added. These values are used for column values for
        `label` or appended to the index if `index_unique` is not `None`. Defaults to
        incrementing integer labels.
    index_unique
        Whether to make the index unique by using the keys. If provided, this
        is the delimiter between "{orig_idx}{index_unique}{key}". When `None`,
        the original indices are kept.
    fill_value
        When `join="outer"`, this is the value that will be used to fill the introduced
        indices. By default, sparse arrays are padded with zeros, while dense arrays and
        DataFrames are padded with missing values.
    pairwise
        Whether pairwise elements along the concatenated dimension should be included.
        This is False by default, since the resulting arrays are often not meaningful.

    Examples
    --------

    Preparing example objects

    >>> import mudata as md, anndata as ad, pandas as pd, numpy as np
    >>> from scipy import sparse
    >>> a = ad.AnnData(
    ...     X=sparse.csr_matrix(np.array([[0, 1], [2, 3]])),
    ...     obs=pd.DataFrame({"group": ["a", "b"]}, index=["s1", "s2"]),
    ...     var=pd.DataFrame(index=["var1", "var2"]),
    ...     varm={"ones": np.ones((2, 5)), "rand": np.random.randn(2, 3), "zeros": np.zeros((2, 5))},
    ...     uns={"a": 1, "b": 2, "c": {"c.a": 3, "c.b": 4}},
    ... )
    >>> b = ad.AnnData(
    ...     X=sparse.csr_matrix(np.array([[4, 5, 6], [7, 8, 9]])),
    ...     obs=pd.DataFrame({"group": ["b", "c"], "measure": [1.2, 4.3]}, index=["s3", "s4"]),
    ...     var=pd.DataFrame(index=["var1", "var2", "var3"]),
    ...     varm={"ones": np.ones((3, 5)), "rand": np.random.randn(3, 5)},
    ...     uns={"a": 1, "b": 3, "c": {"c.b": 4}},
    ... )
    >>> c = ad.AnnData(
    ...     X=sparse.csr_matrix(np.array([[10, 11], [12, 13]])),
    ...     obs=pd.DataFrame({"group": ["a", "b"]}, index=["s1", "s2"]),
    ...     var=pd.DataFrame(index=["var3", "var4"]),
    ...     uns={"a": 1, "b": 4, "c": {"c.a": 3, "c.b": 4, "c.c": 5}},
    ... )
    >>> m = md.MuData({"a": a, "b": b, "c": c})
    """
    # Argument normalization
    merge = resolve_merge_strategy(merge)
    uns_merge = resolve_merge_strategy(uns_merge)

    if isinstance(mdatas, Mapping):
        if keys is not None:
            raise TypeError(
                "Cannot specify categories in both mapping keys and using `keys`. "
                "Only specify this once."
            )
        keys, mdatas = list(mdatas.keys()), list(mdatas.values())
    else:
        mdatas = list(mdatas)

    if keys is None:
        keys = np.arange(len(mdatas)).astype(str)

    assert all(
        [isinstance(m, MuData) for m in mdatas]
    ), "For concatenation to work, all objects should be of type MuData"
    assert len(mdatas) > 1, "mdatas collection should have more than one MuData object"
    if len(set(m.axis for m in mdatas)) != 1:
        "All MuData objects in mdatas should have the same axis."

    axis = mdatas[0].axis

    # Modalities intersection
    common_mods = reduce(
        np.intersect1d, [np.array(list(m.mod.keys())).astype("object") for m in mdatas]
    )
    assert len(common_mods) > 0, "There should be at least one common modality across all mdatas"

    # Concatenate all the modalities
    modalities: dict[str, AnnData] = dict()
    for m in common_mods:
        modalities[m] = ad_concat(
            [mdata[m] for mdata in mdatas],
            axis=axis,
            join=join,
            merge=merge,
            uns_merge=uns_merge,
            label=label,
            keys=keys,
            index_unique=index_unique,
            fill_value=fill_value,
            pairwise=pairwise,
        )

    # Then concatenate multimodal annotations
    axis, dim = _resolve_axis(axis=axis)
    alt_axis, alt_dim = _resolve_axis(axis=1 - axis)

    # Label column
    label_col = pd.Categorical.from_codes(
        np.repeat(np.arange(len(mdatas)), [m.shape[axis] for m in mdatas]),
        categories=keys,
    )

    # Combining indexes
    concat_indices = pd.concat(
        [pd.Series(axis_indices(m, axis=axis)) for m in mdatas], ignore_index=True
    )
    if index_unique is not None:
        concat_indices = concat_indices.str.cat(label_col.map(str), sep=index_unique)
    concat_indices = pd.Index(concat_indices)

    alt_indices = merge_indices([axis_indices(m, axis=alt_axis) for m in mdatas], join=join)
    reindexers = [gen_reindexer(alt_indices, axis_indices(m, axis=alt_axis)) for m in mdatas]

    # Annotation for concatenation axis
    check_combinable_cols([getattr(m, dim).columns for m in mdatas], join=join)
    concat_annot = pd.concat(
        unify_dtypes([getattr(m, dim) for m in mdatas]),
        join=join,
        ignore_index=True,
    )
    concat_annot.index = concat_indices
    if label is not None:
        concat_annot[label] = label_col

    # Annotation for other axis
    alt_annot = merge_dataframes([getattr(m, alt_dim) for m in mdatas], alt_indices, merge)

    # Patch multidimensional annotations to exclude modality masks
    patch_dim: list[dict[str, Any]] = []
    patch_alt_dim: list[dict[str, Any]] = []
    for mdata in mdatas:
        mod_dim_m = getattr(mdata, f"{dim}m")
        mod_alt_dim_m = getattr(mdata, f"{alt_dim}m")
        elems_dim, elems_alt_dim = {}, {}
        for m in mdata.mod.keys():
            if m in mod_dim_m:
                elems_dim[m] = mod_dim_m.pop(m)
            if m in mod_alt_dim_m:
                elems_alt_dim[m] = mod_alt_dim_m.pop(m)
        patch_dim.append(elems_dim)
        patch_alt_dim.append(elems_alt_dim)

    if join == "inner":
        concat_mapping = inner_concat_aligned_mapping(
            [getattr(m, f"{dim}m") for m in mdatas],
            index=concat_indices,
        )
        if pairwise:
            concat_pairwise = concat_pairwise_mapping(
                mappings=[getattr(m, f"{dim}p") for m in mdatas],
                shapes=[m.shape[axis] for m in mdatas],
                join_keys=intersect_keys,
            )
        else:
            concat_pairwise = {}
    elif join == "outer":
        concat_mapping = outer_concat_aligned_mapping(
            [getattr(m, f"{dim}m") for m in mdatas],
            index=concat_indices,
            fill_value=fill_value,
        )
        if pairwise:
            concat_pairwise = concat_pairwise_mapping(
                mappings=[getattr(m, f"{dim}p") for m in mdatas],
                shapes=[m.shape[axis] for m in mdatas],
                join_keys=union_keys,
            )
        else:
            concat_pairwise = {}

    # Un-patch multidimensional annotations
    if len(patch_dim) > 0:
        for mdata in mdatas:
            getattr(mdata, f"{dim}m").update(patch_dim.pop(0))

    if len(patch_alt_dim) > 0:
        for mdata in mdatas:
            getattr(mdata, f"{alt_dim}m").update(patch_alt_dim.pop(0))

    for mdata in mdatas:
        mod_dim_m = getattr(mdata, f"{dim}m")
        mod_alt_dim_m = getattr(mdata, f"{alt_dim}m")
        for m in mdata.mod.keys():
            if m in mod_dim_m:
                patch_dim.append({m: mod_dim_m.pop(m)})
            if m in mod_alt_dim_m:
                patch_alt_dim.append({m: mod_alt_dim_m.pop(m)})

    alt_mapping = merge(
        [
            {k: r(v, axis=0) for k, v in getattr(a, f"{alt_dim}m").items()}
            for r, a in zip(reindexers, mdatas)
        ],
    )
    alt_pairwise = merge(
        [
            {k: r(r(v, axis=0), axis=1) for k, v in getattr(a, f"{alt_dim}p").items()}
            for r, a in zip(reindexers, mdatas)
        ]
    )
    uns = uns_merge([m.uns for m in mdatas])

    # TODO: attrmap

    return MuData(
        **{
            "data": modalities,
            "axis": axis,
            dim: concat_annot,
            alt_dim: alt_annot,
            f"{dim}m": concat_mapping,
            f"{alt_dim}m": alt_mapping,
            f"{dim}p": concat_pairwise,
            f"{alt_dim}p": alt_pairwise,
            "uns": uns,
        }
    )
