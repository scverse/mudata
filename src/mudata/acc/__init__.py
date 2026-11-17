from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Never

from anndata import AnnData
from anndata.acc import NO_IDX, AdAcc, AdRef, Axes, DFAcc, MapAcc, RefAcc, ScopeAcc
from anndata.typing import InMemoryArray

from .. import MuData


@dataclass(frozen=True, kw_only=True)
class ModAcc[R: AdRef](AdAcc[R]):
    """Accessor to create :class:`AdRefs <anndata.acc.AdRef>` (:data:`A`) for modalities (:attr:`~MuAcc.mod`).

    All accessors created from it share its :attr:`~anndata.acc.AdAcc.scope` (a :class:`ModItemAcc`),
    so they index into the modality.
    """

    scope: ModItemAcc[R]
    """Accessor for the modality this accessor refers to."""

    @property
    def mod(self) -> str:
        """Modality this accessor refers to."""
        return self.scope.mod

    def isin(self, mdata: MuData) -> bool:
        """Check if the referenced modality is in the :class:`~mudata.MuData` object."""
        return self.scope.isin(mdata)

    def get(self, mdata: MuData) -> AnnData | MuData:
        """Get the referenced modality from the :class:`~mudata.MuData` object."""
        return self.scope.get(mdata)

    def set(self, mdata: MuData, value: AnnData | MuData) -> None:
        """Set the referenced modality in the :class:`~mudata.MuData` object."""
        self.scope.set(mdata, NO_IDX, value)

    def __repr__(self) -> str:
        return repr(self.scope)

    def to_json(self, ref: R | RefAcc[R, Any, Any]) -> list[Any]:  # type: ignore[override]
        """Serialize :class:`~anndata.acc.AdRef` to a JSON-compatible list, relative to this modality."""
        return ["mod", self.mod, super().to_json(_split_mod(ref)[1])]  # type: ignore[arg-type]


@dataclass(frozen=True)
class ModItemAcc[R: AdRef](ScopeAcc[R, MuData, AnnData | MuData]):
    """Reference accessor for a modality (`mdata.mod[k]`).

    It is the :attr:`~anndata.acc.RefAcc.scope` of all accessors created from a :class:`ModAcc`.
    """

    mod: str
    """Modality this accessor refers to."""

    _: KW_ONLY
    ad_cls: type[ModAcc] = ModAcc
    """Class to use for :attr:`AD`."""

    @property
    def AD(self) -> ModAcc[R]:
        """Accessor for arrays in the modality, e.g. `A.mod['rna'].X`."""
        return self.ad_cls(ref_class=self.ref_class, scope=self)

    def dims(self, idx: Never, /) -> Axes:
        """Raise a :exc:`TypeError`, as modalities aren’t arrays."""
        raise TypeError(f"{self!r} doesn’t contain arrays aligned to a dimension.")

    def __repr__(self) -> str:
        return f"A.mod[{self.mod!r}]"

    def idx_repr(self, idx: Never) -> str:
        """Get a string representation of the index."""
        return f"[{idx!r}]"

    def isin(self, mdata: MuData, idx: None = None) -> bool:
        """Check if the referenced modality is in the :class:`~mudata.MuData` object."""
        return self.mod in mdata.mod

    def get(self, mdata: MuData, idx: NO_IDX = NO_IDX, /) -> AnnData | MuData:
        """Get the referenced modality from the :class:`~mudata.MuData` object."""
        return mdata.mod[self.mod]

    def set(self, mdata: MuData, idx: NO_IDX, value: object, /) -> None:
        """Set the referenced modality in the :class:`~mudata.MuData` object."""
        if not isinstance(value, AnnData | MuData):
            raise TypeError(f"Cannot set a modality to an object of type {type(value)}.")
        mdata.mod[self.mod] = value


def _split_mod(acc: Any) -> tuple[str | None, Any]:
    """Remove the :class:`ModItemAcc` at the root of `acc`’s scope chain, returning its modality."""
    if isinstance(acc, AdRef):
        mod, inner = _split_mod(acc.acc)
        return mod, acc.acc.ref_class(inner, acc.idx)
    if (scope := acc.scope) is None:
        return None, acc
    mod, scope = (scope.mod, None) if isinstance(scope, ModItemAcc) else _split_mod(scope)
    if isinstance(acc, DFAcc) and acc.frame_acc is not None:  # `.DF`: the frame is in the same scope
        return mod, replace(acc, scope=scope, frame_acc=_split_mod(acc.frame_acc)[1])
    return mod, replace(acc, scope=scope)


@dataclass(frozen=True)
class ModMapAcc[R: AdRef[str, MuData]](RefAcc[R, str, MuData]):
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
        """Check if the referenced modality map is in the :class:`~mudata.MuData` object."""
        m = getattr(mdata, f"{self.dim}map")
        return idx is None or idx in m

    def get(self, mdata: MuData, idx: str | NO_IDX = NO_IDX, /) -> InMemoryArray:
        """Get the referenced modality map from the :class:`~mudata.MuData` object."""
        m = getattr(mdata, f"{self.dim}map")
        if idx is NO_IDX:
            return m
        return m[idx]

    def set(self, mdata: MuData, idx: str | NO_IDX, value: InMemoryArray) -> None:
        """Set the referenced modality map  in the :class:`~mudata.MuData` object.

        This raises `NotImplementedError`.
        """
        raise NotImplementedError(f"Setting {self.dim}map is not supported.")


@dataclass(frozen=True, kw_only=True)
class MultiModAcc[R: AdRef](MapAcc[ModAcc[R]]):  # type: ignore[type-var]
    """Accessor for modalities (:attr:`~MuAcc.mod`)."""

    ref_class: type[R]
    ref_acc_cls: type[ModAcc] = ModAcc

    def __getitem__(self, k: str, /) -> ModAcc[R]:
        if not isinstance(k, str):
            raise TypeError(f"Unsupported mod key {k!r}")
        return ModItemAcc(k, ref_class=self.ref_class, ad_cls=self.ref_acc_cls).AD

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

    def __getitem__(self, k: str, /) -> ModAcc[R]:
        return self.mod[k]

    def __repr__(self) -> str:
        return "A"

    def resolve(  # type: ignore[override] # return types
        self, spec: str, *, strict: Literal[True] = True, vec: None | None = None
    ) -> R | RefAcc[R, Any, Any] | ModAcc[R] | None:
        """Create :class:`~anndata.acc.AdRef` from a simplified string."""
        if not strict:
            try:
                self.resolve(spec, vec=vec)
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
                return super().resolve.__func__(acc, spec[modend + 1 :], strict=strict, vec=vec)  # type: ignore[attr-defined] # https://github.com/python/mypy/issues/14235
            case "obsmap" | "varmap":
                if firstdot == len(spec):
                    raise ValueError(f"Cannot parse accessor{spec!r} that has an empty modality.")
                mod = spec[firstdot + 1 :]
                return getattr(self, firstattr)[mod]
            case _:
                return super().resolve(spec, strict=strict, vec=vec)  # type: ignore[return-value] # LayerAcc is never returned

    def to_json(self, ref: R | RefAcc[R, Any, Any] | ModAcc[R]) -> list[Any]:  # type: ignore[override]
        """Serialize :class:`~anndata.acc.AdRef` or a whole container accessor to a JSON-compatible list.

        Schema: `acc-schema-v1.json <../acc-schema-v1.json>`_
        """
        if isinstance(ref, ModAcc):
            return ["mod", ref.mod]
        if isinstance(ref, AdRef) and isinstance(ref.acc, ModMapAcc):
            return [f"{ref.acc.dim}map", ref.idx]
        mod, inner = _split_mod(ref)
        if mod is None:
            return super().to_json(ref)  # type: ignore[arg-type]
        return ["mod", mod, super().to_json(inner)]  # type: ignore[arg-type]

    def from_json(  # type: ignore[override]
        self, data: Sequence[str | int | None | Sequence[str | int | None]], *, vec: bool | None = None
    ) -> R | RefAcc[R, Any, Any] | ModAcc[R]:
        """Create a :class:`~anndata.acc.AdRef` from a JSON sequence.

        Schema: `acc-schema-v1.json <../acc-schema-v1.json>`_
        """
        match data:
            case ["mod", str() as modname]:
                return self.mod[modname]
            case ["mod", str() as modname, Sequence() as inner]:
                return self.mod[modname].from_json(inner, vec=vec)  # type: ignore[return-value, arg-type]
            case ["obsmap" | "varmap" as dim, str() as modname]:
                acc = self.obsmap if dim == "obsmap" else self.varmap
                return acc[modname]
            case _:
                return super().from_json(data, vec=vec)  # type: ignore[return-value, arg-type]


del MuAcc.__dataclass_fields__["X"]
del MuAcc.__dataclass_fields__["layers"]
del MuAcc.__dataclass_fields__["layer_cls"]

A: MuAcc[AdRef] = MuAcc()


if not TYPE_CHECKING:  # https://github.com/tox-dev/sphinx-autodoc-typehints/issues/580
    R = AdRef[Hashable, MuData]
