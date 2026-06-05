from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import pandas as pd
from anndata.acc import (
    AdAcc,
    AdRef,
    Axes,
    GraphAcc,
    GraphMapAcc,
    Idx2D,
    LayerAcc,
    LayerMapAcc,
    MapAcc,
    MetaAcc,
    MultiAcc,
    MultiMapAcc,
    RefAcc,
)
from anndata.compat import XVariable
from anndata.typing import InMemoryArray

if TYPE_CHECKING:
    from anndata import AnnData

    from .. import MuData


@dataclass(frozen=True, kw_only=True)
class _ModalityMixin:
    mod: str
    """Modality this accessor refers to."""


@dataclass(frozen=True)
class _ModalityMapAcc[I, R](_ModalityMixin):
    def isin(self, mdata: MuData, idx: I | None = None) -> bool:
        if self.mod not in mdata.mod:
            return False
        else:
            return super().isin(mdata.mod[self.mod], idx)

    def get(self, mdata: MuData, idx: I, /) -> R:
        return super().get(mdata.mod[self.mod], idx)


@dataclass(frozen=True)
class ModLayerAcc[R: AdRef[Idx2D]](_ModalityMapAcc[Idx2D, InMemoryArray], LayerAcc[R]):
    """Reference accessor for arrays in :attr:`~anndata.acc.AdAcc.layers`."""

    def __repr__(self) -> str:
        return f"A.mod[{self.mod!r}].X" if self.k is None else f"A.mod[{self.mod}].layers[{self.k!r}]"


@dataclass(frozen=True)
class ModLayerMapAcc[R: AdRef](_ModalityMixin, LayerMapAcc[R]):
    """Accessor for arrays in :attr:~anndata.acc.AdAcc.layers`."""

    ref_acc_cls: type[ModLayerAcc] = ModLayerAcc

    def __getitem__(self, k: str | None, /) -> ModLayerAcc[R]:
        if not isinstance(k, str | None):
            raise TypeError(f"Unsupported layer {k!r}")
        return self.ref_acc_cls(mod=self.mod, k=k, ref_class=self.ref_class)

    def __repr__(self) -> str:
        return f"A.mod[{self.mod!r}].layers"


@dataclass(frozen=True)
class ModMetaAcc[R: AdRef[str | None]](_ModalityMapAcc[str, pd.api.extensions.ExtensionArray | XVariable], MetaAcc[R]):
    """Reference accessor for arrays from metadata containers (:attr:`~anndata.acc.AdAcc.obs` / :attr:`~anndata.acc.AdAcc.var`)."""

    def __repr__(self) -> str:
        return f"A.mod[{self.mod!r}].{self.dim}"


@dataclass(frozen=True)
class ModMultiAcc[R: AdRef[int]](_ModalityMapAcc[int, InMemoryArray], MultiAcc[R]):
    """Reference accessor for arrays from multi-dimensional containers (:attr:`~anndata.acc.AdAcc.obsm` / :attr:`~anndata.acc.AdAcc.varm`)."""

    def __repr__(self) -> str:
        return f"A.mod[{self.mod!r}].{self.dim}m[self.k!r]"


@dataclass(frozen=True)
class ModMultiMapAcc[R: AdRef](_ModalityMixin, MultiMapAcc[R]):
    """Accessor for multi-dimensional array containers (:attr:`~anndata.acc.AdAcc.obsm` / :attr:`~anndata.acc.AdAcc.varm`)."""

    ref_acc_cls: type[ModMultiAcc] = ModMultiAcc

    def __getitem__(self, k: str, /) -> ModMultiAcc[R]:
        if not isinstance(k, str):
            raise TypeError(f"Unsupported {self.dim}m key {k!r}")
        return self.ref_acc_cls(mod=self.mod, k=k, dim=self.dim, ref_class=self.ref_class)

    def __repr__(self) -> str:
        return f"A.mod[{self.mod!r}].{self.dim}m"


@dataclass(frozen=True)
class ModGraphAcc[R: AdRef[Idx2D]](_ModalityMapAcc[Idx2D, InMemoryArray], GraphAcc[R]):
    """Reference accessor for arrays from graph containers (:attr:`~anndata.acc.AdAcc.obsp` / :attr:`~anndata.acc.AdAcc.varp`)."""

    def __repr__(self) -> str:
        return f"A.mod[{self.mod!r}].{self.dim}p[{self.k!r}]"


@dataclass(frozen=True)
class ModGraphMapAcc[R: AdRef](_ModalityMixin, GraphMapAcc[R]):
    """Accessor for graph containers (:attr:`~anndata.acc.AdAcc.obsp` / :attr:`~anndata.acc.AdAcc.varp`)"""

    ref_acc_cls: type[ModGraphAcc] = ModGraphAcc

    def __getitem__(self, k: str, /) -> ModGraphAcc[R]:
        if not isinstance(k, str):
            raise TypeError(f"Unsupported {self.dim}p key {k!r}")
        return self.ref_acc_cls(mod=self.mod, k=k, dim=self.dim, ref_class=self.ref_class)

    def __repr__(self) -> str:
        return f"A.mod[{self.mod!r}].{self.dim}p"


@dataclass(frozen=True)
class ModMapAcc[R: AdRef[str]](RefAcc[R, str]):
    """Reference accessor for modality maps (:attr:`~MuAcc.obsmap` / :attr:`~MuAcc.varmap`)."""

    dim: Literal["obs", "var"]
    """Axis this accessor refers to, e.g. `A.obsmap[k].dim == "var"`."""

    def dims(self, idx: Any, /) -> Axes:
        """Get which dimension this array refers to."""
        return (self.dim,)

    def __repr__(self) -> str:
        return f"A.{self.dim}map"

    def idx_repr(self, idx: str, /) -> str:
        """Get a string representation of the index."""
        return f"[{idx}]"

    def isin(self, mdata: MuData, idx: str | None = None) -> bool:
        """Check if the referenced array is in the :class:`~mudata.MuData` object."""
        m = getattr(mdata, f"{self.dim}map")
        return idx is None or idx in m

    def get(self, mdata: MuData, idx: str, /) -> InMemoryArray:
        """Get the referenced array from the :class:`~mudata.MuData` object."""
        m = getattr(mdata, f"{self.dim}map")
        return m[idx]


@dataclass(frozen=True, kw_only=True)
class ModAcc[R: AdRef](_ModalityMixin, AdAcc[R]):
    """Accessor to create :class:`AdRefs <anndata.acc.AdRef>` (:data:`A`) for modalities (:attr:`~MuAcc.mod`)."""

    layer_cls: type[ModLayerAcc] = ModLayerAcc
    """Class to use for `layers` accessors."""

    meta_cls: type[ModMetaAcc] = ModMetaAcc
    """Class to use for `obs`/`var` accessors."""

    multi_cls: type[ModMultiAcc] = ModMultiAcc
    """Class to use for `obsm`/`varm` accessors."""

    graph_cls: type[ModGraphAcc] = ModGraphAcc
    """Class to use for `obsp`/`varp` accessors."""

    def isin(self, mdata: MuData) -> bool:
        """Check if the referenced modality is in the :class:`~mudata.MuData` object."""
        return self.mod in mdata.mod

    def get(self, mdata: MuData) -> AnnData:
        """Get the referenced modality from the :class:`~mudata.MuData` object."""
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
class MultiModAcc[R: AdRef](MapAcc[ModAcc[R]]):
    """Accessor for modalities (:attr:`~MuAcc.mod`)."""

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
    """Accessor to create :class:`AdRefs <anndata.acc.AdRef>` (:data:`A`)."""

    mod_cls: type[ModAcc] = ModAcc
    """Class to use for `mod` accessors."""

    mod: MultiModAcc[R] = field(init=False)
    """Access modalities."""

    obsmap: ModMapAcc[R] = field(init=False)
    """Access mappings of observation indices in the MuData to indices in individual modalities."""

    varmap: ModMapAcc[R] = field(init=False)
    """Access mappings of variable indices in the MuData to indices in individual modalities."""

    ATTRS: ClassVar = frozenset(("mod", "obs", "var", "obsm", "varm", "obsp", "varp", "obsmap", "varmap"))

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "mod", MultiModAcc(ref_class=self.ref_class, ref_acc_cls=self.mod_cls))
        object.__setattr__(self, "obsmap", ModMapAcc("obs", ref_class=self.ref_class))
        object.__setattr__(self, "varmap", ModMapAcc("var", ref_class=self.ref_class))

        del self.__dict__["X"]
        del self.__dict__["layers"]
        del self.__dataclass_fields__["X"]
        del self.__dataclass_fields__["layers"]

    def __getitem__(self, k: str, /) -> ModAcc[R]:
        return self.mod[k]

    def __repr__(self) -> str:
        return "A"

    def resolve(self, spec: str, *, strict: bool = True) -> R | None:
        """Create :class:`~anndata.acc.AdRef` from a simplified string."""
        if not strict:
            try:
                self.resolve(spec)
            except ValueError:
                return None

        firstdot = spec.find(".")
        if firstdot < 0:
            raise ValueError(f"Cannot parse accessor {spec!r} that is not period-separated.")
        firstattr = spec[:firstdot]
        match firstattr:
            case "mod":
                modend = spec.find(".", firstdot + 1)
                mod = spec[firstdot + 1 : modend]
                if not mod:
                    raise ValueError(f"Cannot parse accessor{spec!r} that has an empty modality.")
                acc = self.mod[mod]
                return super().resolve.__func__(acc, spec[modend + 1 :], strict=strict)
            case "obsmap" | "varmap":
                if firstdot == len(spec):
                    raise ValueError(f"Cannot parse accessor{spec!r} that has an empty modality.")
                mod = spec[firstdot + 1 :]
                return getattr(self, firstattr)[mod]
            case _:
                return super().resolve(spec, strict=strict)


A: MuAcc[AdRef] = MuAcc()
