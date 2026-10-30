from __future__ import annotations

import importlib
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from io import IOBase
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast
from warnings import catch_warnings, filterwarnings, warn

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata import AnnData
from anndata.compat import _read_attr
from anndata.experimental import read_dispatched
from anndata.io import read_elem, write_elem
from packaging.version import Version
from scipy.sparse import issparse
from scverse_misc import Deprecation, deprecated

from .file_backing import AnnDataFileManager, MuDataFileManager
from .mudata import MuData

if TYPE_CHECKING:
    from os import PathLike

    import fsspec
    import zarr
    from zarr.abc.store import Store as ZarrStore

    from .compat import AdRef, MapAcc, ModAcc, ModMapAcc, MultiModAcc, RefAcc


from .utils import AD_VERSION

_read_write_modality_pattern = re.compile(r"^(.+\.h5mu)/([^/]+)(/([^/]+))?$")


@contextmanager
def _fake_anndata_encoding_attrs(f: h5py.Group | zarr.Group):
    from .. import __anndataversion__

    oldcls: type = type(f)

    class FakeAnndataEncodingAttrs(oldcls):
        @property
        def attrs(self):
            return {"encoding-type": "anndata", "encoding-version": __anndataversion__}

    if isinstance(f, h5py.Group):
        f.__class__ = FakeAnndataEncodingAttrs
        yield f
        f.__class__ = oldcls
    else:
        newf = FakeAnndataEncodingAttrs.open(f.store, zarr_format=f.metadata.zarr_format)
        yield newf


def _write_use(
    f: h5py.Group | zarr.Group,
    mdata: MuData,
    use: AdRef
    | RefAcc
    | MapAcc
    | MultiModAcc
    | ModAcc
    | ModMapAcc
    | Sequence[AdRef | RefAcc | MapAcc | MultiModAcc | ModAcc | ModMapAcc],
    *,
    dataset_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> None:
    if AD_VERSION < Version("0.14.0dev0"):
        raise RuntimeError("The 'use' argument requires anndata>=0.14.")

    from anndata._io.utils import write_use
    from anndata.acc import AdRef, MetaAcc

    from .. import __mudataversion__
    from ..acc import ModAcc, ModMapAcc, ModMetaAcc, MultiModAcc, _ModalityAcc, _ModalityMapAcc

    if f.attrs.get("encoding-type") != "mudata" and f.attrs.get("encoding-version") != __mudataversion__:
        raise ValueError("Cannot write a subelement of a MuData object to a store that is not a MuData store.")

    if f.attrs["axis"] != mdata.axis:
        raise ValueError(f"Cannot write {use}: axis in the store does not match the MuData object.")
    for dim in ("obs", "var"):
        group = f[dim]
        on_disk = read_elem(group[cast(str, group.attrs["_index"])])  # type: ignore[arg-type, index]
        if not pd.Index(cast(np.ndarray, on_disk)).equals(getattr(mdata, f"{dim}_names")):
            raise ValueError(f"Cannot write {use}: {dim}_names in the store do not match the MuData object.")

    uses = use if isinstance(use, Sequence) else (use,)

    have_globals, have_mods, have_mod = {"obs": False, "var": False, "obsmap": False, "varmap": False}, False, []

    global_accessors: list[MultiModAcc | ModAcc | ModMapAcc | MetaAcc | RefAcc | MapAcc | AdRef] = []
    modality_part_accessors = defaultdict(
        list
    )  # pass all modality accessors to anndata at once so it can do proper validation

    disk_mods = f.require_group("mod")
    for acc in uses:
        match acc:
            case _ModalityAcc(mod=mod) | _ModalityMapAcc(mod=mod):
                if mod not in disk_mods:
                    raise ValueError(f"Modality {mod!r} is not present on disk, cannot write part of it.")
                for acctype in type(acc).mro():  # pass dim_names check
                    if acctype.__module__ == "anndata.acc":
                        args = asdict(acc)
                        del args["mod"]
                        cacc = acctype(**args)
                        break
                modality_part_accessors[mod].append(cacc)
            case AdRef(acc=ModMetaAcc(mod=mod), idx=str() as idx):
                if mod not in disk_mods:
                    raise ValueError(f"Modality {mod!r} is not present on disk, cannot write part of it.")
                cacc = type(acc)(acc=MetaAcc(dim=dim, ref_class=type(acc)), idx=idx)  # pass dim_names check
                modality_part_accessors[mod].append(cacc)
            case ModAcc(mod=mod):
                have_mod.append(mod)
                global_accessors.append(acc)
            case MultiModAcc():
                have_mods = True
                global_accessors.append(acc)
            case ModMapAcc(dim=dim):
                have_globals[f"{dim}map"] = True
                global_accessors.append(acc)
            case MetaAcc(dim=dim):
                have_globals[dim] = True
                global_accessors.append(acc)
            case _:
                global_accessors.append(acc)

    if not all(have_globals.values()) and (
        have_mods and mdata.mod.keys() != set(disk_mods.keys()) or any(mod not in disk_mods for mod in have_mod)
    ):
        warn(
            "Modalities given in 'use' do not correspond to those already on disk and no global metadata is being written. "
            "This may lead to surprising results when reading the file.",
            RuntimeWarning,
            stacklevel=2,
        )

    for acc in global_accessors:
        match acc:
            case MultiModAcc():
                mods = acc.get(mdata)
                for modname, adata in mods.items():
                    _write_modality(
                        disk_mods, modname, cast(AnnData, adata), dataset_kwargs=dataset_kwargs
                    )  # handle extras correctly
                disk_mods.attrs["mod-order"] = list(mods.keys())
            case ModAcc(mod=mod):
                disk_mods = f.require_group("mod")
                _write_modality(
                    disk_mods, mod, cast(AnnData, acc.get(mdata)), dataset_kwargs=dataset_kwargs
                )  # handle extras correctly

                order = cast(list[str], _read_attr(disk_mods.attrs, "mod-order"))
                if mod not in order and (mods_intersection := list(mdata.mod.keys() & (order + [mod]))):
                    modpos = mods_intersection.index(mod)
                    insertpos = order.index(mods_intersection[modpos - 1]) + 1
                    order.insert(insertpos, mod)
                    disk_mods.attrs["mod-order"] = order
            case ModMapAcc(dim=dim):
                write_elem(f, f"{dim}map", acc.get(mdata), dataset_kwargs=dataset_kwargs)
            case _:
                with _fake_anndata_encoding_attrs(f) as fakef:
                    write_use(fakef, cast(AnnData, mdata), acc, dataset_kwargs=dataset_kwargs)

    for mod, cacc in modality_part_accessors.items():
        write_use(disk_mods[mod], cast(AnnData, mdata.mod[mod]), cacc, dataset_kwargs=dataset_kwargs)  # type: ignore[arg-type]


def _is_openfile(obj) -> bool:
    return obj.__class__.__name__ == "OpenFile" and obj.__class__.__module__.startswith("fsspec.")


class _Subtype(NamedTuple):
    """An AnnData subclass whose extra elements mudata stores in the modality group."""

    module: str  # module providing the subclass; looked up in sys.modules, never imported to detect it
    cls: str  # name of the AnnData subclass in that module
    write: str  # "module:function" writing the extra elements: ``(group, adata, *, dataset_kwargs)``
    read: str  # "module:function" rebuilding the subclass: ``(group, adata) -> subclass instance``


# Subtypes mudata can store, by the name written to the ``encoding-subtype`` attribute of the modality group.
# The group itself stays a valid AnnData group (``encoding-type = "anndata"``).
_SUBTYPES = {
    "treedata": _Subtype("treedata", "TreeData", "treedata.io:write_tree_elems", "treedata.io:read_tree_elems")
}
_SUBTYPE_ATTR = "encoding-subtype"


def _resolve(ref: str) -> Callable | None:
    """Return the function a ``"module:function"`` reference points to, or ``None`` if it is not available."""
    module, _, name = ref.partition(":")
    try:
        return getattr(importlib.import_module(module), name, None)
    except ImportError:
        return None


def _write_mod_extras(group: h5py.Group | zarr.Group, adata: AnnData, kwargs) -> None:
    """Store the elements an AnnData subclass adds to AnnData in the modality group.

    An instance of a subclass can only exist if its package was imported, so the subclass is looked up
    in ``sys.modules`` and its package is never imported just to detect it.
    """
    for name, subtype in _SUBTYPES.items():
        cls = getattr(sys.modules.get(subtype.module), subtype.cls, None)
        if not (isinstance(cls, type) and isinstance(adata, cls)):
            continue
        write = _resolve(subtype.write)
        if write is not None:
            write(group, adata, dataset_kwargs=kwargs)  # type: ignore[call-arg]
            group.attrs[_SUBTYPE_ATTR] = name
            return


def _read_mod_extras(group: h5py.Group | zarr.Group, adata: AnnData) -> AnnData:
    """Rebuild the AnnData subclass a modality was written from, if it is available."""
    name = group.attrs.get(_SUBTYPE_ATTR)
    if name is None:
        return adata
    subtype = _SUBTYPES.get(str(name))
    read = _resolve(subtype.read) if subtype is not None else None
    if read is None:
        warn(
            f"Modality {group.name!r} was written from the AnnData subtype {str(name)!r}, "
            "which is not available; reading it as AnnData.",
            stacklevel=2,
            skip_file_prefixes=(str(Path(__file__).parent),),
        )
        return adata
    return read(group, adata)


def _write_modality(
    f: h5py.Group | zarr.Group,
    mod: str,
    adata: AnnData,
    *,
    write_data: bool = True,
    chunks: tuple[int, ...] | None = None,
    dataset_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> None:
    from .. import __anndataversion__, __version__

    if not isinstance(adata, AnnData):
        raise ValueError("Only AnnData modalities can be written to disk.")

    if mod in f and write_data:
        del f[mod]
    group = f.require_group(mod)

    adata.strings_to_categoricals()
    if adata.raw is not None:
        adata.strings_to_categoricals(adata.raw.var)

    if write_data:
        if chunks is not None and not issparse(adata.X):
            write_elem(group, "X", adata.X, dataset_kwargs=dict(chunks=chunks, **dataset_kwargs))
        else:
            write_elem(group, "X", adata.X, dataset_kwargs=dataset_kwargs)
    if adata.raw is not None:
        if write_data or not adata.isbacked:
            write_elem(group, "raw", adata.raw, dataset_kwargs=dataset_kwargs)
        else:
            rawgrp = group.require_group("raw")
            rawgrp.attrs["encoding-type"] = "anndata"
            rawgrp.attrs["encoding-version"] = __anndataversion__
            write_elem(rawgrp, "var", adata.raw.var, dataset_kwargs=dataset_kwargs)
            write_elem(rawgrp, "varm", dict(adata.raw.varm), dataset_kwargs=dataset_kwargs)

    write_elem(group, "obs", adata.obs, dataset_kwargs=dataset_kwargs)
    write_elem(group, "var", adata.var, dataset_kwargs=dataset_kwargs)
    write_elem(group, "obsm", dict(adata.obsm), dataset_kwargs=dataset_kwargs)
    write_elem(group, "varm", dict(adata.varm), dataset_kwargs=dataset_kwargs)
    write_elem(group, "obsp", dict(adata.obsp), dataset_kwargs=dataset_kwargs)
    write_elem(group, "varp", dict(adata.varp), dataset_kwargs=dataset_kwargs)
    write_elem(group, "layers", {k: v for k, v in adata.layers.items() if k is not None}, dataset_kwargs=dataset_kwargs)
    write_elem(group, "uns", dict(adata.uns), dataset_kwargs=dataset_kwargs)
    _write_mod_extras(group, adata, dataset_kwargs)

    attrs = group.attrs
    attrs["encoding-type"] = "anndata"
    attrs["encoding-version"] = __anndataversion__
    attrs["encoder"] = "mudata"
    attrs["encoder-version"] = __version__


def _write_h5mu(file: h5py.File, mdata: MuData, write_data=True, **kwargs) -> None:
    from .. import __mudataversion__, __version__

    write_elem(file, "obs", mdata.strings_to_categoricals(mdata.obs.copy()), dataset_kwargs=kwargs)
    write_elem(file, "var", mdata.strings_to_categoricals(mdata.var.copy()), dataset_kwargs=kwargs)
    write_elem(file, "obsm", dict(mdata.obsm), dataset_kwargs=kwargs)
    write_elem(file, "varm", dict(mdata.varm), dataset_kwargs=kwargs)
    write_elem(file, "obsp", dict(mdata.obsp), dataset_kwargs=kwargs)
    write_elem(file, "varp", dict(mdata.varp), dataset_kwargs=kwargs)
    write_elem(file, "uns", dict(mdata.uns), dataset_kwargs=kwargs)

    write_elem(file, "obsmap", dict(mdata.obsmap), dataset_kwargs=kwargs)
    write_elem(file, "varmap", dict(mdata.varmap), dataset_kwargs=kwargs)

    attrs = file.attrs
    attrs["axis"] = mdata.axis

    mod = file.require_group("mod")
    for k, adata in mdata.mod.items():
        _write_modality(
            mod, k, cast(AnnData, adata), write_data=write_data or not adata.isbacked, dataset_kwargs=kwargs
        )
    mod.attrs["mod-order"] = list(mdata.mod.keys())

    attrs = file.attrs
    attrs["encoding-type"] = "mudata"
    attrs["encoding-version"] = __mudataversion__
    attrs["encoder"] = "mudata"
    attrs["encoder-version"] = __version__


def write_zarr(
    store: MutableMapping | str | PathLike | ZarrStore,
    data: MuData | AnnData,
    chunks: tuple[int, ...] | None = None,
    write_data: bool = True,
    *,
    use: AdRef
    | RefAcc
    | MapAcc
    | MultiModAcc
    | ModAcc
    | ModMapAcc
    | Sequence[AdRef | RefAcc | MapAcc | MultiModAcc | ModAcc | ModMapAcc]
    | None = None,
    **kwargs,
) -> None:
    """
    Write a MuData or AnnData object to the Zarr store.

    Parameters
    ----------
    store
        Thie filename or a Zarr store.
    data
        The :class:`~mudata.MuData` or :class:`~anndata.AnnData` object.
    chunks
        The chunk shape.
    write_data
        Whether to write the data (the :attr:`~anndata.AnnData.X` matrices) for the modalities. If `False`, only the metadata
        (everything except :attr:~anndata.AnnData.X`) will be written.
    use
        Only write the element(s) referenced by this accessor (or list of accessors) into an existing MuData store.

        .. important::
           This functionality requires anndata 0.14 or newer.
    **kwargs
        Additional arguments to :func:`zarr.create_array`.
    """
    import zarr
    from anndata._io.zarr import write_zarr as anndata_write_zarr

    from .. import __mudataversion__, __version__

    with catch_warnings():
        filterwarnings("ignore", message="This setting will be removed", category=DeprecationWarning)
        zarr_format = getattr(ad.settings, "zarr_write_format", 3)

    if isinstance(data, AnnData):
        anndata_write_zarr(store, data, chunks=chunks, use=use, **kwargs)
    elif isinstance(data, MuData):
        if isinstance(store, Path):
            store = str(store)
        zarr_format = kwargs.pop("zarr_format", zarr_format)

        mode: Literal["w", "r+"] = "w" if use is None else "r+"

        try:
            file = zarr.open_group(store, mode=mode, zarr_format=zarr_format)
        except TypeError:
            # zarr_format is not supported in this version of zarr
            file = zarr.open_group(store, mode=mode)

        if use is not None:
            _write_use(file, data, use, dataset_kwargs=kwargs)
        else:
            write_elem(file, "obs", data.strings_to_categoricals(data.obs.copy()), dataset_kwargs=kwargs)
            write_elem(file, "var", data.strings_to_categoricals(data.var.copy()), dataset_kwargs=kwargs)
            write_elem(file, "obsm", dict(data.obsm), dataset_kwargs=kwargs)
            write_elem(file, "varm", dict(data.varm), dataset_kwargs=kwargs)
            write_elem(file, "obsp", dict(data.obsp), dataset_kwargs=kwargs)
            write_elem(file, "varp", dict(data.varp), dataset_kwargs=kwargs)
            write_elem(file, "uns", dict(data.uns), dataset_kwargs=kwargs)

            write_elem(file, "obsmap", dict(data.obsmap), dataset_kwargs=kwargs)
            write_elem(file, "varmap", dict(data.varmap), dataset_kwargs=kwargs)

            attrs = file.attrs
            attrs["axis"] = data.axis

            mod = file.require_group("mod")
            for k, adata in data.mod.items():
                if not isinstance(adata, AnnData):
                    raise ValueError("Only AnnData modalities can be written to disk.")
                _write_modality(mod, k, adata, write_data=write_data, chunks=chunks, dataset_kwargs=kwargs)

            mod.attrs["mod-order"] = list(data.mod.keys())

            attrs = file.attrs
            attrs["encoding-type"] = "mudata"
            attrs["encoding-version"] = __mudataversion__
            attrs["encoder"] = "mudata"
            attrs["encoder-version"] = __version__
    else:
        raise TypeError("Expected MuData or AnnData object")


def write_h5mu(
    filename: str | PathLike,
    mdata: MuData,
    *,
    use: AdRef
    | RefAcc
    | MapAcc
    | MultiModAcc
    | ModAcc
    | ModMapAcc
    | Sequence[AdRef | RefAcc | MapAcc | MultiModAcc | ModAcc | ModMapAcc]
    | None = None,
    **kwargs,
) -> None:
    """
    Write a MuData object to an HDF5 file.

    Parameters
    ----------
    filename
        The filename.
    mdata
        The :class:`~mudata.MuData` object.
    use
        Only write the element(s) referenced by this accessor (or list of accessors) into an existing MuData store.

        .. important::
           This functionality requires anndata 0.14 or newer.
    **kwargs
        Additional arguments to :meth:`h5py.Group.create_dataset`.
    """
    from .. import __mudataversion__, __version__

    if use is not None:
        with h5py.File(filename, "r+") as f:
            _write_use(f, mdata, use, dataset_kwargs=kwargs)
    else:
        with h5py.File(filename, "w", userblock_size=512) as f:
            _write_h5mu(f, mdata, **kwargs)

        with Path(filename).open("br+") as f:
            nbytes = f.write(
                f"MuData (format-version={__mudataversion__};creator=mudata;creator-version={__version__})".encode()
            )
            f.write(b"\0" * (512 - nbytes))  # this is only needed because the H5file was written in append mode


@deprecated(Deprecation("0.4.2", "Use write_h5mu() with the 'use' argument instead."))
def write_h5ad(filename: str | PathLike, mod: str, data: MuData | AnnData) -> None:
    """
    Write an AnnData object to an existing HDF5 file containing a MuData (an h5mu file).

    Parameters
    ----------
    filename
        The file name.
    mod
        The modality to write.
    data
        The data. If a :class:`~mudata.MuData` object, `data[mod]` will be written.
    """
    if isinstance(data, AnnData):
        adata = data
    elif not isinstance(data, MuData) or not isinstance(adata := data.mod[mod], AnnData):  # type: ignore[assignment]
        raise TypeError(f"Expected AnnData or MuData object with {mod} modality of type AnnData")

    with h5py.File(filename, "r+") as f:
        if "mod" not in f:
            raise ValueError("The .h5mu object has to contain .mod slot")

        _write_modality(
            f["mod"],
            mod,
            adata,
            write_data=not (adata.isbacked and Path(adata.filename) == Path(filename)),  # type: ignore[arg-type]
        )


write_anndata = write_h5ad


@deprecated(Deprecation("0.4.2", "Use write_h5mu() with the 'use' argument instead."))
def write(filename: str | PathLike, data: MuData | AnnData) -> None:
    """
    Write a :class:`~mudata.MuData` or :class:`~anndata.AnnData` object to an HDF5 file.

    This function is designed to enhance I/O ease of use.
    It recognises the following formats of `filename`:

    - for MuData

      - `FILE.h5mu`

    - for AnnData

      - `FILE.h5mu/MODALITY`
      - `FILE.h5mu/mod/MODALITY`
      - `FILE.h5ad`

      The first two variants will write the :class:`~anndata.AnnData` object to the modality `MODALITY`
      of the existing `FILE.h5mu` file, same as :func:`write_h5ad`.

    Parameters
    ----------
    filename
        The file name.
    data
        The data object to write.
    """
    filename = str(filename)
    if filename.endswith(".h5ad") and isinstance(data, AnnData):
        return data.write(filename)

    match (filename.endswith(".h5mu"), isinstance(data, MuData)):
        case (False, True):
            raise ValueError("Can only save MuData object to .h5mu file")
        case (True, False):
            raise ValueError("Only MuData object can be saved as .h5mu file")
        case (True, True):
            write_h5mu(filename, data)  # type: ignore[arg-type]
        case (False, False):
            if not isinstance(data, AnnData):
                raise ValueError("Only MuData and AnnData objects are accepted")

            match = _read_write_modality_pattern.fullmatch(filename)
            if match is None:
                raise ValueError(
                    "If a single modality is to be written to a .h5mu file, \
                    provide it after the filename separated by slash symbol:\
                    .h5mu/rna or .h5mu/mod/rna"
                )

            filename, maybe_mod, _, modname = match.groups()

            if modname is None:
                return write_h5ad(filename, maybe_mod, data)
            elif maybe_mod == "mod":
                return write_h5ad(filename, modname, data)
            else:
                raise ValueError("Modality names cannot contain slashes.")


def read_h5mu(
    filename: str | PathLike | IOBase | fsspec.core.OpenFile, backed: Literal["r", "r+"] | bool | None = None
) -> MuData:
    """Read an `.h5mu`-formatted HDF5 file.

    Parameters
    ----------
    filename
        The file name or an :external+fsspec:doc:`fsspec<index>` object.
    backed
        Whether to open the file in backed mode. In this mode, the data matrices :attr:`~anndata.AnnData.X` are not read into memory,
        but are references to the on-disk datasets.

    Examples
    --------
    >>> mdata = read_h5mu("file.h5mu")

    >>> with fsspec.open("https://example.com/file.h5mu") as f:
    ...     mdata = read_h5mu(f)
    """
    if backed not in [None, True, False, "r", "r+"]:
        raise ValueError("Argument `backed` should be boolean, or r/r+, or None")

    if backed and (isinstance(filename, IOBase) or _is_openfile(filename)):
        warn("Backed mode is not supported if `filename` is an open file. Switching off backed mode.", stacklevel=1)
        backed = False

    if backed is True or not backed:
        mode = "r"
    else:
        mode = backed
    manager = MuDataFileManager(filename, mode) if backed else MuDataFileManager()  # type: ignore[arg-type]

    with ExitStack() as stack:
        if _is_openfile(filename):
            filename = stack.enter_context(filename)  # type: ignore[arg-type]
        elif not isinstance(filename, IOBase):
            filename = stack.enter_context(open(filename, "b" + mode))

        ish5mu = filename.read(6) == b"MuData"

        try:
            with h5py.File(filename, mode) as f:
                if not ish5mu:
                    warn(
                        "The HDF5 file was not created by muon/mudata, we can't guarantee that everything will work correctly",
                        stacklevel=1,
                    )
                d: dict[str, Any] = {}
                for k in f.keys():
                    if k == "mod":
                        mods = {}
                        gmods = f[k]
                        for m in gmods.keys():
                            ad = _read_h5mu_mod(gmods[m], manager, backed not in (None, False))
                            mods[m] = ad

                        mod_order = None
                        if "mod-order" in gmods.attrs:
                            mod_order = _read_attr(gmods.attrs, "mod-order")
                        if mod_order is not None and all(m in gmods for m in mod_order):
                            mods = {k: mods[k] for k in mod_order}

                        d["data"] = mods
                    else:
                        d[k] = read_elem(f[k])

                if "axis" in f.attrs:
                    d["axis"] = f.attrs["axis"]
        except OSError as e:
            if not e.errno and not e.strerror and e.args[0].rfind("file signature not found") >= 0:
                raise ValueError("The file is not an HDF5 file") from e
            else:
                raise

    mu = MuData(**d)
    mu.file = manager
    return mu


def read_zarr(store: str | PathLike | MutableMapping | zarr.Group | ZarrStore) -> MuData | AnnData:
    """Read from a hierarchical Zarr array store.

    Parameters
    ----------
    store
        The file name or a Zarr store.
    """
    import zarr

    f = zarr.open_group(store, mode="r")
    d: dict[str, Any] = {}
    if "mod" not in f.keys():
        return ad.read_zarr(f)

    for k in f.keys():
        if k == "mod":
            mods: dict[str, AnnData] = {}
            gmods = f.require_group(k)
            for m in gmods.keys():
                if not isinstance(mod := gmods[m], zarr.Group):
                    raise ValueError(f"Modality {m} must be a group.")
                mods[m] = _read_mod_extras(mod, cast(AnnData, read_elem(gmods[m])))

            mod_order = None
            if "mod-order" in gmods.attrs:
                mod_order = cast(Sequence[str], _read_attr(gmods.attrs, "mod-order"))
            if mod_order is not None and all(m in gmods for m in mod_order):
                mods = {k: mods[k] for k in mod_order}

            d["data"] = mods
        else:  # Base case
            d[k] = read_elem(f[k])
    if "axis" in f.attrs:
        d["axis"] = f.attrs["axis"]

    mu = MuData(**d)
    mu.file = MuDataFileManager()

    return mu


def _read_h5mu_mod(g: h5py.Group, manager: MuDataFileManager | None = None, backed: bool = False) -> AnnData:
    modname = Path(g.name).name
    Xpath = g.name + "/X"
    rawpath = g.name + "/raw"
    rawXpath = rawpath + "/X"

    def ad_callback(func, elem_name, elem, iospec):
        if not backed or elem_name not in (Xpath, rawpath):
            return func(elem)

    def raw_callback(func, elem_name, elem, iospec):
        if not backed or elem_name != rawXpath:
            return func(elem)

    ad = _read_mod_extras(g, cast(AnnData, read_dispatched(g, callback=ad_callback)))
    if manager is not None:
        ad.file = AnnDataFileManager(ad, modname, manager)

    if "raw" in g:
        ad.raw = cast(AnnData, read_dispatched(g["raw"], callback=raw_callback))
    return ad


def read_h5ad(
    filename: str | PathLike | IOBase | fsspec.core.OpenFile, mod: str | None, backed: Literal["r", "r+"] | bool = False
) -> AnnData:
    """Read a modality from inside a .h5mu file or from a standalone .h5ad file (mod=None).

    Parameters
    ----------
    filename
        The file name or an :external+fsspec:doc:`fsspec<index>` object.
    backed
        Whether to open the file in backed mode. In this mode, the data matrix :attr:`~anndata.AnnData.X` is not read into memory,
        but is a reference to the on-disk datasets.

    Examples
    --------
    >>> adata = read_h5ad("file.h5mu", "rna")

    >>> adata = read_h5ad("rna.h5ad")

    >>> with fsspec.open("https://example.com/file.h5mu") as f:
    ...     adata = read_h5ad(f, "rna")

    >>> with fsspec.open("https://example.com/rna.h5ad") as f:
    ...     adata = read_h5ad(f)
    """
    if backed and (isinstance(filename, IOBase) or _is_openfile(filename)):
        warn("Backed mode is not supported if `filename` is an open file. Switching off backed mode.", stacklevel=1)
        backed = False

    if mod is None:
        with ExitStack() as stack:
            if _is_openfile(filename):
                filename = stack.enter_context(filename)  # type: ignore[arg-type]
            return ad.read_h5ad(filename, backed=backed)  # type: ignore[arg-type]

    hdf5_mode = "r" if isinstance(backed, bool) else backed
    manager = None
    if backed:
        backed = True
        manager = MuDataFileManager(filename, hdf5_mode)  # type: ignore[arg-type]

    with h5py.File(filename, hdf5_mode) as f_root:
        f = f_root["mod"][mod]
        return _read_h5mu_mod(f, manager, backed)


read_anndata = read_h5ad


def read(filename: str | PathLike | IOBase | fsspec.core.OpenFile, **kwargs) -> MuData | AnnData:
    """Read an `.h5mu` formatted HDF5 file or a single modality inside it.

    This function is designed to enhance I/O ease of use.
    It recognises the following formats of `filename`:

    - `FILE.h5mu`
    - `FILE.h5ad`
    - `FILE.h5mu/MODALITY`
    - `FILE.h5mu/mod/MODALITY`

    The last two variantes will read the modality `MODALITY` and return an :class:`~anndata.AnnData` object.

    Parameters
    ----------
    filename
        The file name or an :external+fsspec:doc:`fsspec<index>` object.
    **kwargs
        additional arguments to :func:`read_h5ad` or :func:`read_h5mu`.

    Examples
    --------
    >>> mdata = read("file.h5mu")

    >>> adata = read("file.h5mu/rna")

    >>> with fsspec.open("s3://bucket/file.h5mu") as f:
    ...     mdata = read(f)
    """
    if isinstance(filename, IOBase):
        raise TypeError(
            "Use format-specific functions (read_h5mu, read_zarr) to read from opened files or provide an fsspec.core.OpenFile instance."
        )
    elif _is_openfile(filename):
        fname = filename.path  # type: ignore[union-attr]
    else:
        fname = str(filename)

    if fname.endswith(".h5ad"):
        return read_h5ad(filename, mod=None, **kwargs)
    elif fname.endswith(".h5mu") or not isinstance(filename, str) and not isinstance(filename, Path):
        return read_h5mu(filename, **kwargs)

    match = _read_write_modality_pattern.fullmatch(fname)
    if match is None:
        raise ValueError(
            "If a single modality is to be read from a .h5mu file, \
            provide it after the filename separated by slash symbol:\
            .h5mu/rna or .h5mu/mod/rna"
        )

    filename, maybe_mod, _, modname = match.groups()
    if modname is None:
        return read_h5ad(filename, maybe_mod, **kwargs)
    elif maybe_mod == "mod":
        # .h5mu/mod/<modality>
        return read_h5ad(filename, modname, **kwargs)
    else:
        raise ValueError("Modality names cannot contain slashes.")
