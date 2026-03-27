from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from os import PathLike
    from typing import Literal

    import fsspec
    import zarr

import io
import re
from contextlib import ExitStack
from pathlib import Path
from warnings import warn

import anndata as ad
import h5py
from anndata import AnnData
from anndata._io.h5ad import _read_raw
from anndata._io.h5ad import read_dataframe as read_h5ad_dataframe
from anndata._io.specs.registry import read_elem, write_elem
from anndata._io.zarr import _read_legacy_raw
from anndata._io.zarr import read_dataframe as read_zarr_dataframe
from anndata._io.zarr import write_zarr as anndata_write_zarr
from anndata.compat import _read_attr
from scipy import sparse

from .config import OPTIONS
from .file_backing import AnnDataFileManager, MuDataFileManager
from .mudata import ModDict, MuData

_pattern = re.compile(r"^(.+\.h5mu)/([^/]+)(/([^/]+))?$")

#
# Saving multimodal data objects
#


def _is_openfile(obj) -> bool:
    return obj.__class__.__name__ == "OpenFile" and obj.__class__.__module__.startswith("fsspec.")


def _write_h5mu(file: h5py.File, mdata: MuData, write_data=True, **kwargs):
    from .. import __anndataversion__, __mudataversion__, __version__

    write_elem(
        file,
        "obs",
        mdata.strings_to_categoricals(
            mdata._shrink_attr("obs", inplace=False).copy() if OPTIONS["pull_on_update"] is None else mdata.obs.copy()
        ),
        dataset_kwargs=kwargs,
    )
    write_elem(
        file,
        "var",
        mdata.strings_to_categoricals(
            mdata._shrink_attr("var", inplace=False).copy() if OPTIONS["pull_on_update"] is None else mdata.var.copy()
        ),
        dataset_kwargs=kwargs,
    )
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
            write_elem(group, "raw", adata.raw)

        write_elem(group, "obs", adata.obs, dataset_kwargs=kwargs)
        write_elem(group, "var", adata.var, dataset_kwargs=kwargs)
        write_elem(group, "obsm", dict(adata.obsm), dataset_kwargs=kwargs)
        write_elem(group, "varm", dict(adata.varm), dataset_kwargs=kwargs)
        write_elem(group, "obsp", dict(adata.obsp), dataset_kwargs=kwargs)
        write_elem(group, "varp", dict(adata.varp), dataset_kwargs=kwargs)
        write_elem(group, "layers", dict(adata.layers), dataset_kwargs=kwargs)
        write_elem(group, "uns", dict(adata.uns), dataset_kwargs=kwargs)

        attrs = group.attrs
        attrs["encoding-type"] = "anndata"
        attrs["encoding-version"] = __anndataversion__
        attrs["encoder"] = "mudata"
        attrs["encoder-version"] = __version__

    mod_attrs = mod.attrs
    mod_attrs["mod-order"] = list(mdata.mod.keys())

    attrs = file.attrs
    attrs["encoding-type"] = "MuData"
    attrs["encoding-version"] = __mudataversion__
    attrs["encoder"] = "mudata"
    attrs["encoder-version"] = __version__

    # Restore top-level annotation
    if not mdata.is_view or not mdata.isbacked:
        mdata.update()


def write_zarr(
    store: MutableMapping | str | PathLike | zarr.abc.store.Store,
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

    from .. import __anndataversion__, __mudataversion__, __version__

    zarr_format = getattr(ad.settings, "zarr_write_format", 2)

    if isinstance(data, AnnData):
        adata = data
        anndata_write_zarr(store, adata, chunks=chunks, **kwargs)
    elif isinstance(data, MuData):
        if isinstance(store, Path):
            store = str(store)
        zarr_format = kwargs.pop("zarr_format", zarr_format)
        try:
            file = zarr.open(store, mode="w", zarr_format=zarr_format)
        except TypeError:
            # zarr_format is not supported in this version of zarr
            file = zarr.open(store, mode="w")
        mdata = data
        write_elem(
            file,
            "obs",
            mdata.strings_to_categoricals(
                mdata._shrink_attr("obs", inplace=False).copy()
                if OPTIONS["pull_on_update"] is None
                else mdata.obs.copy()
            ),
            dataset_kwargs=kwargs,
        )
        write_elem(
            file,
            "var",
            mdata.strings_to_categoricals(
                mdata._shrink_attr("var", inplace=False).copy()
                if OPTIONS["pull_on_update"] is None
                else mdata.var.copy()
            ),
            dataset_kwargs=kwargs,
        )
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
                if chunks is not None and not isinstance(adata.X, sparse.spmatrix):
                    write_elem(group, "X", adata.X, dataset_kwargs=dict(chunks=chunks, **kwargs))
                else:
                    write_elem(group, "X", adata.X, dataset_kwargs=kwargs)
            if adata.raw is not None:
                write_elem(group, "raw", adata.raw)

            write_elem(group, "obs", adata.obs, dataset_kwargs=kwargs)
            write_elem(group, "var", adata.var, dataset_kwargs=kwargs)
            write_elem(group, "obsm", dict(adata.obsm), dataset_kwargs=kwargs)
            write_elem(group, "varm", dict(adata.varm), dataset_kwargs=kwargs)
            write_elem(group, "obsp", dict(adata.obsp), dataset_kwargs=kwargs)
            write_elem(group, "varp", dict(adata.varp), dataset_kwargs=kwargs)
            write_elem(group, "layers", dict(adata.layers), dataset_kwargs=kwargs)
            write_elem(group, "uns", dict(adata.uns), dataset_kwargs=kwargs)

            attrs = group.attrs
            attrs["encoding-type"] = "anndata"
            attrs["encoding-version"] = __anndataversion__
            attrs["encoder"] = "mudata"
            attrs["encoder-version"] = __version__

        mod_attrs = mod.attrs
        mod_attrs["mod-order"] = list(mdata.mod.keys())

        attrs = file.attrs
        attrs["encoding-type"] = "MuData"
        attrs["encoding-version"] = __mudataversion__
        attrs["encoder"] = "mudata"
        attrs["encoder-version"] = __version__

        # Restore top-level annotation
        if not mdata.is_view or not mdata.isbacked:
            mdata.update()
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
            f"MuData (format-version={__mudataversion__};creator=muon;creator-version={__version__})".encode()
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
    elif isinstance(data, MuData):
        adata = data.mod[mod]
    else:
        raise TypeError(f"Expected AnnData or MuData object with {mod} modality")

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

        if not (adata.isbacked and Path(adata.filename) == Path(filepath)):
            write_elem(fmd, "X", adata.X)

        # NOTE: Calling write_elem() does not allow writing .raw into .h5mu modalities
        if adata.raw is not None:
            write_elem(f, f"mod/{mod}/raw", adata.raw)

        write_elem(fmd, "obs", adata.obs)
        write_elem(fmd, "var", adata.var)
        write_elem(fmd, "obsm", dict(adata.obsm))
        write_elem(fmd, "varm", dict(adata.varm))
        write_elem(fmd, "obsp", dict(adata.obsp))
        write_elem(fmd, "varp", dict(adata.varp))
        write_elem(fmd, "layers", dict(adata.layers))
        write_elem(fmd, "uns", dict(adata.uns))

        attrs = fmd.attrs
        attrs["encoding-type"] = "anndata"
        attrs["encoding-version"] = __anndataversion__
        attrs["encoder"] = "muon"
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
            write_h5mu(filename, data)
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


#
# Reading from multimodal data objects
#


def read_h5mu(
    filename: str | PathLike | io.IOBase | fsspec.OpenFile, backed: Literal["r", "r+"] | bool | None = None
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

    if backed is True or not backed:
        mode = "r"
    else:
        mode = backed
    manager = MuDataFileManager(filename, mode) if backed else MuDataFileManager()

    with ExitStack() as stack:
        if _is_openfile(filename):
            filename = stack.enter_context(filename)
        elif not isinstance(filename, io.IOBase):
            filename = stack.enter_context(open(filename, "b" + mode))

        ish5mu = filename.read(6) == b"MuData"

        try:
            with h5py.File(filename, mode) as f:
                if not ish5mu:
                    warn(
                        "The HDF5 file was not created by muon/mudata, we can't guarantee that everything will work correctly",
                        stacklevel=2,
                    )
                d = {}
                for k in f.keys():
                    if k in ["obs", "var"]:
                        d[k] = read_h5ad_dataframe(f[k])
                    if k == "mod":
                        mods = ModDict()
                        gmods = f[k]
                        for m in gmods.keys():
                            ad = _read_h5mu_mod(gmods[m], manager, backed not in (None, False))
                            mods[m] = ad

                        mod_order = None
                        if "mod-order" in gmods.attrs:
                            mod_order = _read_attr(gmods.attrs, "mod-order")
                        if mod_order is not None and all(m in gmods for m in mod_order):
                            mods = {k: mods[k] for k in mod_order}

                        d[k] = mods
                    else:
                        d[k] = read_elem(f[k])

                if "axis" in f.attrs:
                    d["axis"] = f.attrs["axis"]
        except OSError as e:
            if not e.errno and not e.strerror and e.args[0].rfind("file signature not found") >= 0:
                raise ValueError("The file is not an HDF5 file") from e
            else:
                raise

    mu = MuData._init_from_dict_(**d)
    mu.file = manager
    return mu


def read_zarr(store: str | PathLike | MutableMapping | zarr.Group | zarr.abc.store.Store) -> MuData | AnnData:
    """Read from a hierarchical Zarr array store.

    Parameters
    ----------
    store
        The file name or a Zarr store.
    """
    import zarr

    if isinstance(store, Path):
        store = str(store)

    f = zarr.open(store, mode="r")
    d = {}
    if "mod" not in f.keys():
        return ad.read_zarr(store)

    manager = MuDataFileManager()
    for k in f.keys():
        if k in {"obs", "var"}:
            d[k] = read_zarr_dataframe(f[k])
        if k == "mod":
            mods = {}
            gmods = f[k]
            for m in gmods.keys():
                mods[m] = _read_zarr_mod(gmods[m], manager)

            mod_order = None
            if "mod-order" in gmods.attrs:
                mod_order = _read_attr(gmods.attrs, "mod-order")
            if mod_order is not None and all(m in gmods for m in mod_order):
                mods = {k: mods[k] for k in mod_order}

            d[k] = mods
        else:  # Base case
            d[k] = read_elem(f[k])

    mu = MuData._init_from_dict_(**d)
    mu.file = manager

    return mu


def _read_zarr_mod(g: zarr.Group, manager: MuDataFileManager = None, backed: bool = False) -> dict:
    d = {}

    for k in g.keys():
        if k in ("obs", "var"):
            d[k] = read_zarr_dataframe(g[k])
        elif k == "X":
            X = g["X"]
            if not backed:
                d["X"] = read_elem(X)
        elif k != "raw":
            d[k] = read_elem(g[k])
    ad = AnnData(**d)
    if manager is not None:
        ad.file = AnnDataFileManager(ad, Path(g.name).name, manager)

    raw = _read_legacy_raw(
        g, d.get("raw"), read_zarr_dataframe, read_elem, attrs=("var", "varm") if backed else ("var", "varm", "X")
    )
    if raw:
        ad._raw = ad.Raw(ad, **raw)
    return ad


def _read_h5mu_mod(g: h5py.Group, manager: MuDataFileManager = None, backed: bool = False) -> dict:
    d = {}

    for k in g.keys():
        if k in ("obs", "var"):
            d[k] = read_h5ad_dataframe(g[k])
        elif k == "X":
            X = g["X"]
            if not backed:
                d["X"] = read_elem(X)
        elif k != "raw":
            d[k] = read_elem(g[k])
    ad = AnnData(**d)
    if manager is not None:
        ad.file = AnnDataFileManager(ad, Path(g.name).name, manager)

    raw = _read_raw(g, attrs=("var", "varm") if backed else ("var", "varm", "X"))
    if raw:
        ad._raw = ad.Raw(ad, **raw)
    return ad


def read_h5ad(
    filename: str | PathLike | io.IOBase | fsspec.OpenFile, mod: str | None, backed: Literal["r", "r+"] | bool = False
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
    if mod is None:
        with ExitStack() as stack:
            if _is_openfile(filename):
                filename = stack.enter_context(filename)
            return ad.read_h5ad(filename, backed=backed)

    hdf5_mode = "r"
    manager = None
    if backed == "r+":
        hdf5_mode = backed
        backed = True
    if backed:
        manager = MuDataFileManager(filename, hdf5_mode)

    with h5py.File(filename, hdf5_mode) as f_root:
        f = f_root["mod"][mod]
        return _read_h5mu_mod(f, manager, backed)


read_anndata = read_h5ad


def read(filename: str | PathLike | io.IOBase | fsspec.OpenFile, **kwargs) -> MuData | AnnData:
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
    if isinstance(filename, io.IOBase):
        raise TypeError(
            "Use format-specific functions (read_h5mu, read_zarr) to read from opened files or provide an fsspec.OpenFile instance."
        )
    elif _is_openfile(filename):
        fname = filename.path
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
