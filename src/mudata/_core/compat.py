from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anndata import AnnData
    from anndata._core.raw import Raw

try:
    from anndata._core.aligned_mapping import AlignedView, AxisArrays, PairwiseArrays
except ImportError:
    # anndata < 0.10.9
    from anndata._core.aligned_mapping import (
        AlignedViewMixin as AlignedView,
    )
    from anndata._core.aligned_mapping import (
        AxisArrays as AxisArraysLegacy,
    )
    from anndata._core.aligned_mapping import (
        AxisArraysBase,
    )
    from anndata._core.aligned_mapping import (
        PairwiseArrays as PairwiseArraysLegacy,
    )

    class AxisArrays(AxisArraysLegacy):
        def __init__(
            self,
            parent: AnnData | Raw,
            axis: int,
            store: Mapping | AxisArraysBase | None = None,
        ):
            super().__init__(parent, axis=axis, vals=store)

    class PairwiseArrays(PairwiseArraysLegacy):
        def __init__(
            self,
            parent: AnnData,
            axis: int,
            store: Mapping | None = None,
        ):
            super().__init__(parent, axis=axis, vals=store)


__all__ = ["AlignedView", "AxisArrays", "PairwiseArrays"]
