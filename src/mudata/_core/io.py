from __future__ import annotations

import re
from collections.abc import MutableMapping, Sequence
from contextlib import ExitStack
from io import IOBase
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from warnings import catch_warnings, filterwarnings, warn

import anndata as ad
import h5py
from anndata import AnnData
from anndata.compat import _read_attr
from anndata.experimental import read_dispatched
from anndata.io import read_elem, write_elem
from scipy import sparse

from .file_backing import AnnDataFileManager, MuDataFileManager
from .mudata import MuData

if TYPE_CHECKING:
    from os import PathLike

    import fsspec
    import zarr
    from zarr.abc.store import Store as ZarrStore


_pattern = re.compile(r"^(.+\.h5mu)/([^/]+)(/([^/]+))?$")


def _is_openfile(obj) -> bool:
    return obj.__class__.__name__ == "OpenFile" and obj.__class__.__module__.startswith("fsspec.")


def _write_h5mu(file: h5py.File, mdata: MuData, write_data=True, **kwargs):
    from .. import __anndataversion__, __mudataversion__, __version__

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
        group = mod.require_group(k)

        adata.strings_to_categoricals()
        if adata.raw is not None:
            adata.strings_to_categoricals(adata.raw.var)

        if write_data or not adata.isbacked:
            write_elem(group, "X", adata.X, dataset_kwargs=kwargs)
        if adata.raw is not None:
            if not adata.isbacked or write_data:
                write_elem(group, "raw", adata.raw, dataset_kwargs=kwargs)
            else:
                rawgrp = group.require_group("raw")
                rawgrp.attrs["encoding-type"] = "anndata"
                rawgrp.attrs["encoding-version"] = __anndataversion__
                write_elem(rawgrp, "var", adata.raw.var, dataset_kwargs=kwargs)
                write_elem(rawgrp, "varm", dict(adata.raw.varm), dataset_kwargs=kwargs)

        write_elem(group, "obs", adata.obs, dataset_kwargs=kwargs)
        write_elem(group, "var", adata.var, dataset_kwargs=kwargs)
        write_elem(group, "obsm", dict(adata.obsm), dataset_kwargs=kwargs)
        write_elem(group, "varm", dict(adata.varm), dataset_kwargs=kwargs)
        write_elem(group, "obsp", dict(adata.obsp), dataset_kwargs=kwargs)
        write_elem(group, "varp", dict(adata.varp), dataset_kwargs=kwargs)
        write_elem(group, "layers", {k: v for k, v in adata.layers.items() if k is not None}, dataset_kwargs=kwargs)
        write_elem(group, "uns", dict(adata.uns), dataset_kwargs=kwargs)

        attrs = group.attrs
        attrs["encoding-type"] = "anndata"
        attrs["encoding-version"] = __anndataversion__
        attrs["encoder"] = "mudata"
        attrs["encoder-version"] = __version__

    mod_attrs = mod.attrs
    mod_attrs["mod-order"] = list(mdata.mod.keys())

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
    **kwargs,
):
    """
    Write a MuData or AnnData object to the Zarr store.

    Parameters
    ----------
    store
        Thie filename or a Zarr store.
    chunks
        The chunk shape.
    write_data
        Whether to write the data (the :attr:`~anndata.AnnData.X` matrices) for the modalities. If `False`, only the metadata
        (everything except :attr:~anndata.AnnData.X`) will be written.
    **kwargs
        Additional arguments to :func:`zarr.create_array`.
    """
    import zarr
    from anndata._io.zarr import write_zarr as anndata_write_zarr

    from .. import __anndataversion__, __mudataversion__, __version__

    with catch_warnings():
        filterwarnings("ignore", message="This setting will be removed", category=DeprecationWarning)
        zarr_format = getattr(ad.settings, "zarr_write_format", 3)

    if isinstance(data, AnnData):
        anndata_write_zarr(store, data, chunks=chunks, **kwargs)
    elif isinstance(data, MuData):
        if isinstance(store, Path):
            store = str(store)
        zarr_format = kwargs.pop("zarr_format", zarr_format)
        try:
            file = zarr.open_group(store, mode="w", zarr_format=zarr_format)
        except TypeError:
            # zarr_format is not supported in this version of zarr
            file = zarr.open_group(store, mode="w")
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
            group = mod.require_group(k)

            adata.strings_to_categoricals()
            if adata.raw is not None:
                adata.strings_to_categoricals(adata.raw.var)

            if write_data:
                if chunks is not None and not isinstance(adata.X, sparse.spmatrix):
                    write_elem(group, "X", adata.X, dataset_kwargs=dict(chunks=chunks, **kwargs))
                else:
                    write_elem(group, "X", adata.X, dataset_kwargs=kwargs)
            if adata.raw is not None:
                if write_data or not adata.isbacked:
                    write_elem(group, "raw", adata.raw, dataset_kwargs=kwargs)
                else:
                    rawgrp = group.require_group("raw")
                    rawgrp.attrs["encoding-type"] = "anndata"
                    rawgrp.attrs["encoding-version"] = __anndataversion__
                    write_elem(rawgrp, "var", adata.raw.var, dataset_kwargs=kwargs)
                    write_elem(rawgrp, "varm", dict(adata.raw.varm), dataset_kwargs=kwargs)

            write_elem(group, "obs", adata.obs, dataset_kwargs=kwargs)
            write_elem(group, "var", adata.var, dataset_kwargs=kwargs)
            write_elem(group, "obsm", dict(adata.obsm), dataset_kwargs=kwargs)
            write_elem(group, "varm", dict(adata.varm), dataset_kwargs=kwargs)
            write_elem(group, "obsp", dict(adata.obsp), dataset_kwargs=kwargs)
            write_elem(group, "varp", dict(adata.varp), dataset_kwargs=kwargs)
            write_elem(group, "layers", {k: v for k, v in adata.layers.items() if k is not None}, dataset_kwargs=kwargs)
            write_elem(group, "uns", dict(adata.uns), dataset_kwargs=kwargs)

            attrs = group.attrs
            attrs["encoding-type"] = "anndata"
            attrs["encoding-version"] = __anndataversion__
            attrs["encoder"] = "mudata"
            attrs["encoder-version"] = __version__

        mod_attrs = mod.attrs
        mod_attrs["mod-order"] = list(data.mod.keys())

        attrs = file.attrs
        attrs["encoding-type"] = "mudata"
        attrs["encoding-version"] = __mudataversion__
        attrs["encoder"] = "mudata"
        attrs["encoder-version"] = __version__
    else:
        raise TypeError("Expected MuData or AnnData object")


def write_h5mu(filename: str | PathLike, mdata: MuData, **kwargs):
    """
    Write a MuData object to an HDF5 file.

    Parameters
    ----------
    filename
        The filename.
    mdata
        The :class:`~mudata.MuData` object.
    **kwargs
        Additional arguments to :meth:`h5py.Group.create_dataset`.
    """
    from .. import __mudataversion__, __version__

    with h5py.File(filename, "w", userblock_size=512) as f:
        _write_h5mu(f, mdata, **kwargs)
    with Path(filename).open("br+") as f:
        nbytes = f.write(
            f"MuData (format-version={__mudataversion__};creator=mudata;creator-version={__version__})".encode()
        )
        f.write(b"\0" * (512 - nbytes))  # this is only needed because the H5file was written in append mode


def write_h5ad(filename: str | PathLike, mod: str, data: MuData | AnnData):
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
    from .. import __anndataversion__, __version__

    if isinstance(data, AnnData):
        adata = data
    elif not isinstance(data, MuData) or not isinstance(adata := data.mod[mod], AnnData):  # type: ignore[assignment]
        raise TypeError(f"Expected AnnData or MuData object with {mod} modality of type AnnData")

    with h5py.File(filename, "r+") as f:
        # Check that 'mod' is present
        if "mod" not in f:
            raise ValueError("The .h5mu object has to contain .mod slot")
        fm = f["mod"]

        # Remove the modality if it exists
        if mod in fm:
            del fm[mod]

        fmd = fm.create_group(mod)

        adata.strings_to_categoricals()
        if adata.raw is not None:
            adata.strings_to_categoricals(adata.raw.var)

        filepath = Path(filename)

        if not (adata.isbacked and Path(adata.filename) == Path(filepath)):  # type: ignore[arg-type]
            write_elem(fmd, "X", adata.X)

        if adata.raw is not None:
            write_elem(fmd, "raw", adata.raw)

        write_elem(fmd, "obs", adata.obs)
        write_elem(fmd, "var", adata.var)
        write_elem(fmd, "obsm", dict(adata.obsm))
        write_elem(fmd, "varm", dict(adata.varm))
        write_elem(fmd, "obsp", dict(adata.obsp))
        write_elem(fmd, "varp", dict(adata.varp))
        write_elem(fmd, "layers", {k: v for k, v in adata.layers.items() if k is not None})
        write_elem(fmd, "uns", dict(adata.uns))

        attrs = fmd.attrs
        attrs["encoding-type"] = "anndata"
        attrs["encoding-version"] = __anndataversion__
        attrs["encoder"] = "mudata"
        attrs["encoder-version"] = __version__


write_anndata = write_h5ad


def write(filename: str | PathLike, data: MuData | AnnData):
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

            match = _pattern.fullmatch(filename)
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
            mods = {}
            gmods = f.require_group(k)
            for m in gmods.keys():
                mods[m] = read_elem(gmods[m])

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

    ad = cast(AnnData, read_dispatched(g, callback=ad_callback))
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

    match = _pattern.fullmatch(fname)
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
