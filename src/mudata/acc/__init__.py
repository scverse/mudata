from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
from anndata.acc import (
    AdAcc,
    AdRef,
    GraphAcc,
    GraphMapAcc,
    Idx2D,
    LayerAcc,
    LayerMapAcc,
    MapAcc,
    MetaAcc,
    MultiAcc,
    MultiMapAcc,
)
from anndata.compat import XVariable
from anndata.typing import InMemoryArray

if TYPE_CHECKING:
    from anndata import AnnData

    from .. import MuData


@dataclass(frozen=True, kw_only=True)
class _ModalityMixin:
    mod: str


@dataclass(frozen=True)
class _ModalityMapAcc[I, R](_ModalityMixin):
    def isin(self, mdata: MuData, idx: I | None = None) -> bool:
        if self.mod not in mdata.mod:
            return False
        else:
            return super().isin(mdata[self.mod], idx)

    def get(self, mdata: MuData, idx: I, /) -> R:
        return super().get(mdata[self.mod], idx)


@dataclass(frozen=True)
class ModLayerAcc[R: AdRef[Idx2D]](_ModalityMapAcc[Idx2D, InMemoryArray], LayerAcc[R]):
    def __repr__(self) -> str:
        return f"A.mod[{self.mod!r}].X" if self.k is None else f"A.mod[{self.mod}].layers[{self.k!r}]"


@dataclass(frozen=True, kw_only=True)
class ModLayerMapAcc[R: AdRef](_ModalityMixin, LayerMapAcc[R]):
    ref_acc_cls: type[ModLayerAcc] = ModLayerAcc

    def __getitem__(self, k: str | None, /) -> ModLayerAcc[R]:
        if not isinstance(k, str | None):
            raise TypeError(f"Unsupported layer {k!r}")
        return self.ref_acc_cls(mod=self.mod, k=k, ref_class=self.ref_class)

    def __repr__(self) -> str:
        return f"A.mod[{self.mod!r}].layers"


@dataclass(frozen=True)
class ModMetaAcc[R: AdRef[str | None]](_ModalityMapAcc[str, pd.api.extensions.ExtensionArray | XVariable], MetaAcc[R]):
    def __repr__(self) -> str:
        return f"A.mod[{self.mod!r}].{self.dim}"


@dataclass(frozen=True)
class ModMultiAcc[R: AdRef[int]](_ModalityMapAcc[int, InMemoryArray], MultiAcc[R]):
    def __repr__(self) -> str:
        return f"A.mod[{self.mod!r}].{self.dim}m[self.k!r]"


@dataclass(frozen=True, kw_only=True)
class ModMultiMapAcc[R: AdRef](_ModalityMixin, MultiMapAcc[R]):
    ref_acc_cls: type[ModMultiAcc] = ModMultiAcc

    def __getitem__(self, k: str, /) -> ModMultiAcc[R]:
        if not isinstance(k, str):
            raise TypeError(f"Unsupported {self.dim}m key {k!r}")
        return self.ref_acc_cls(mod=self.mod, k=k, dim=self.dim, ref_class=self.ref_class)

    def __repr__(self) -> str:
        return f"A.mod[{self.mod!r}].{self.dim}m"


@dataclass(frozen=True)
class ModGraphAcc[R: AdRef[Idx2D]](_ModalityMapAcc[Idx2D, InMemoryArray], GraphAcc[R]):
    def __repr__(self) -> str:
        return f"A.mod[{self.mod!r}].{self.dim}p[{self.k!r}]"


@dataclass(frozen=True, kw_only=True)
class ModGraphMapAcc[R: AdRef](_ModalityMixin, GraphMapAcc[R]):
    ref_acc_cls: type[ModGraphAcc] = ModGraphAcc

    def __getitem__(self, k: str, /) -> ModGraphAcc[R]:
        if not isinstance(k, str):
            raise TypeError(f"Unsupported {self.dim}p key {k!r}")
        return self.ref_acc_cls(mod=self.mod, k=k, dim=self.dim, ref_class=self.ref_class)

    def __repr__(self) -> str:
        return f"A.mod[{self.mod!r}].{self.dim}p"


@dataclass(frozen=True, kw_only=True)
class ModAcc[R: AdRef](_ModalityMixin, AdAcc[R]):
    layer_cls: type[ModLayerAcc] = ModLayerAcc
    meta_cls: type[ModMetaAcc] = ModMetaAcc
    multi_cls: type[ModMultiAcc] = ModMultiAcc
    graph_cls: type[ModGraphAcc] = ModGraphAcc

    def isin(self, mdata: MuData) -> bool:
        return self.mod in mdata.mod

    def get(self, mdata: MuData) -> ad.AnnData:
        return mdata.mod[self.mod]

    def __post_init__(self) -> None:
        x = self.layer_cls(mod=self.mod, k=None, ref_class=self.ref_class)
        layers = ModLayerMapAcc(mod=self.mod, ref_class=self.ref_class, ref_acc_cls=self.layer_cls)
        object.__setattr__(self, "X", x)
        object.__setattr__(self, "layers", layers)
        for dim in ("obs", "var"):
            meta = self.meta_cls(mod=self.mod, dim=dim, ref_class=self.ref_class)
            multi = ModMultiMapAcc(mod=self.mod, dim=dim, ref_class=self.ref_class, ref_acc_cls=self.multi_cls)
            graphs = ModGraphMapAcc(mod=self.mod, dim=dim, ref_class=self.ref_class, ref_acc_cls=self.graph_cls)
            object.__setattr__(self, dim, meta)
            object.__setattr__(self, f"{dim}m", multi)
            object.__setattr__(self, f"{dim}p", graphs)

    def __repr__(self) -> str:
        return f"A.mod[{self.mod}]"


@dataclass(frozen=True)
class ModMapAcc[R: AdRef](MapAcc[ModAcc[R]]):
    ref_class: type[R]
    ref_acc_cls: type[ModAcc] = ModAcc

    def __getitem__(self, k: str, /) -> ModAcc[R]:
        if not isinstance(k, str):
            raise TypeError(f"Unsupported mod key {k!r}")
        return self.ref_acc_cls(mod=k, ref_class=self.ref_class)

    def __repr__(self) -> str:
        return "A.mod"


@dataclass(frozen=True)
class MuAcc[R: AdRef](AdAcc[R]):
    mod_cls: type[ModAcc] = ModAcc
    """Class to use for `mod` accessors."""

    mod: ModMapAcc[R] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        mod = ModMapAcc(ref_class=self.ref_class, ref_acc_cls=self.mod_cls)
        object.__setattr__(self, "mod", mod)

    def __getitem__(self, k: str, /) -> ModAcc[R]:
        return self.mod[k]

    def __repr__(self) -> str:
        return "A"


A = MuAcc()
