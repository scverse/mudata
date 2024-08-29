from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

try:
    from anndata._core.aligned_mapping import AlignedView, AxisArrays, PairwiseArrays
except ImportError:
    # anndata < 0.10.9
    from anndata._core.aligned_mapping import (
        AlignedActualMixin,
        AxisArraysBase,
        PairwiseArraysBase,
    )
    from anndata._core.aligned_mapping import (
        AlignedViewMixin as AlignedView,
    )

    class AxisArrays(AlignedActualMixin, AxisArraysBase):
        def __init__(
            self,
            parent: "AnnData",
            axis: int,
            store: Mapping | AxisArraysBase | None = None,
        ):
            self._parent = parent
            if axis not in {0, 1}:
                raise ValueError()
            self._axis = axis
            self._data = dict()
            if store is not None:
                self.update(store)

    class PairwiseArrays(AlignedActualMixin, PairwiseArraysBase):
        def __init__(
            self,
            parent: "AnnData",
            axis: int,
            store: Mapping | None = None,
        ):
            self._parent = parent
            if axis not in {0, 1}:
                raise ValueError()
            self._axis = axis
            self._data = dict()
            if store is not None:
                self.update(store)


__all__ = ["AlignedView", "AxisArrays", "PairwiseArrays"]
