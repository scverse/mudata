from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping
    from os import PathLike

    import fsspec
    import zarr

from pathlib import Path
from warnings import warn

import anndata as ad
import h5py
from anndata import AnnData
from anndata.compat import _read_attr
from scipy import sparse

from .config import OPTIONS
from .file_backing import AnnDataFileManager, MuDataFileManager
from .mudata import ModDict, MuData

#
# Saving multimodal data objects
#


def _write_h5mu(file: h5py.File, mdata: MuData, write_data=True, **kwargs):
    from anndata._io.specs.registry import write_elem

    from .. import __anndataversion__, __mudataversion__, __version__

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
    for k, v in mdata.mod.items():
        group = mod.require_group(k)

        adata = mdata.mod[k]

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
    store: MutableMapping | str | Path,
    data: MuData | AnnData,
    chunks=None,
    write_data=True,
    **kwargs,
):
    """
    Write MuData or AnnData object to the Zarr store

    Matrices - sparse or dense - are currently stored as they are.
    """
    import zarr
    from anndata._io.specs.registry import write_elem
    from anndata._io.zarr import write_zarr as anndata_write_zarr

    from .. import __anndataversion__, __mudataversion__, __version__

    if isinstance(data, AnnData):
        adata = data
        anndata_write_zarr(store, adata, chunks=chunks, **kwargs)
    elif isinstance(data, MuData):
        if isinstance(store, Path):
            store = str(store)
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
        for k, v in mdata.mod.items():
            group = mod.require_group(k)

            adata = mdata.mod[k]

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


def write_h5mu(filename: PathLike, mdata: MuData, **kwargs):
    """
    Write MuData object to the HDF5 file

    Matrices - sparse or dense - are currently stored as they are.
    """
    from .. import __mudataversion__, __version__

    with h5py.File(filename, "w", userblock_size=512) as f:
        _write_h5mu(f, mdata, **kwargs)
    with Path(filename).open("br+") as f:
        nbytes = f.write(
            f"MuData (format-version={__mudataversion__};creator=muon;creator-version={__version__})".encode()
        )
        f.write(
            b"\0" * (512 - nbytes)
        )  # this is only needed because the H5file was written in append mode


def write_h5ad(filename: PathLike, mod: str, data: MuData | AnnData):
    """
    Write AnnData object to the HDF5 file with a MuData container

    Currently is based on anndata._io.h5ad.write_h5ad internally.
    Matrices - sparse or dense - are currently stored as they are.

    Ideally this is merged later to anndata._io.h5ad.write_h5ad.
    """
    from anndata._io.specs.registry import write_elem

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


def write(filename: PathLike, data: MuData | AnnData):
    """
    Write MuData or AnnData to an HDF5 file

    This function is designed to enhance I/O ease of use.
    It recognises the following formats of filename:
      - for MuData
            - FILE.h5mu
      - for AnnData
              - FILE.h5mu/MODALITY
              - FILE.h5mu/mod/MODALITY
              - FILE.h5ad
    """

    import re

    if filename.endswith(".h5mu") or isinstance(data, MuData):
        assert filename.endswith(".h5mu"), "Can only save MuData object to .h5mu file"
        assert isinstance(data, MuData), "Only MuData object can be saved as .h5mu file"

        write_h5mu(filename, data)

    else:
        assert isinstance(data, AnnData), "Only MuData and AnnData objects are accepted"

        m = re.search("^(.+)\\.(h5mu)[/]?([^/]*)[/]?(.*)$", str(filename))
        if m is not None:
            m = m.groups()
        else:
            raise ValueError("Expected non-empty .h5ad or .h5mu file name")

        filepath = ".".join([m[0], m[1]])

        if m[1] == "h5mu":
            if m[3] == "":
                # .h5mu/<modality>
                return write_h5ad(filepath, m[2], data)
            elif m[2] == "mod":
                # .h5mu/mod/<modality>
                return write_h5ad(filepath, m[3], data)
            else:
                raise ValueError(
                    "If a single modality to be written from a .h5mu file, \
                    provide it after the filename separated by slash symbol:\
                    .h5mu/rna or .h5mu/mod/rna"
                )
        elif m[1] == "h5ad":
            return data.write(filepath)
        else:
            raise ValueError()


#
# Reading from multimodal data objects
#


def _validate_h5mu(filename: PathLike) -> (str, Callable | None):
    fname: [str, Path, fsspec.core.io.BufferedReader, fsspec.core.OpenFile] = filename
    callback = None

    try:
        with Path(filename).open("rb") as f:
            ish5mu = f.read(6) == b"MuData"
    except TypeError as e:
        # Support for fsspec
        #
        # Namely, opening remote files should work via
        # with fsspec.open("s3://bucket/file.h5mu") as f:
        #     mdata = read_h5mu(f)
        # or
        # mdata = read_h5mu(fsspec.open("s3://bucket/file.h5mu"))
        if (
            filename.__class__.__name__ == "BufferedReader"
            or filename.__class__.__name__ == "OpenFile"
            or filename.__class__.__name__ == "HTTPFile"
            or filename.__class__.__name__ == "HTTPStreamFile"
            or filename.__class__.__name__ == "S3File"
        ):
            try:
                from fsspec.core import OpenFile

                if isinstance(filename, OpenFile):
                    fname = filename.__enter__()
                    callback = lambda: fname.__exit__()
                ish5mu = fname.read(6) == b"MuData"
            except ImportError as e:
                raise ImportError(
                    "To read from remote storage or cache, install fsspec: pip install fsspec"
                ) from e
        else:
            ish5mu = False
            raise e

    if not ish5mu:
        if isinstance(filename, str) or isinstance(filename, Path):
            if h5py.is_hdf5(filename):
                warn(
                    "The HDF5 file was not created by muon/mudata, we can't guarantee that everything will work correctly"
                )
            else:
                raise ValueError("The file is not an HDF5 file")
        else:
            warn("Cannot verify that the (remote) file is a valid H5MU file")

    return fname, callback


def read_h5mu(filename: PathLike, backed: str | bool | None = None):
    """
    Read MuData object from HDF5 file
    """
    assert backed in [
        None,
        True,
        False,
        "r",
        "r+",
    ], "Argument `backed` should be boolean, or r/r+, or None"

    from anndata._io.h5ad import read_dataframe
    from anndata._io.specs.registry import read_elem

    if backed is True or not backed:
        mode = "r"
    else:
        mode = backed
    manager = MuDataFileManager(filename, mode) if backed else MuDataFileManager()

    fname, callback = _validate_h5mu(filename)

    with h5py.File(fname, mode) as f:
        d = {}
        for k in f.keys():
            if k in ["obs", "var"]:
                d[k] = read_dataframe(f[k])
            if k == "mod":
                mods = ModDict()
                gmods = f[k]
                for m in gmods.keys():
                    ad = _read_h5mu_mod(gmods[m], manager, backed not in (None, False))
                    mods[m] = ad

                mod_order = None
                if "mod-order" in gmods.attrs:
                    mod_order = _read_attr(gmods.attrs, "mod-order")
                if mod_order is not None and all([m in gmods for m in mod_order]):
                    mods = {k: mods[k] for k in mod_order}

                d[k] = mods
            else:
                d[k] = read_elem(f[k])

        if "axis" in f.attrs:
            d["axis"] = f.attrs["axis"]

        if callback is not None:
            callback()

    mu = MuData._init_from_dict_(**d)
    mu.file = manager
    return mu


def read_zarr(store: str | Path | MutableMapping | zarr.Group):
    """\
    Read from a hierarchical Zarr array store.
    Parameters
    ----------
    store
        The filename, a :class:`~typing.MutableMapping`, or a Zarr storage class.
    """
    import zarr
    from anndata._io.specs.registry import read_elem
    from anndata._io.zarr import (
        read_dataframe,
    )
    from anndata._io.zarr import (
        read_zarr as anndata_read_zarr,
    )

    if isinstance(store, Path):
        store = str(store)

    f = zarr.open(store, mode="r")
    d = {}
    if "mod" not in f.keys():
        return anndata_read_zarr(store)

    manager = MuDataFileManager()
    for k in f.keys():
        if k in {"obs", "var"}:
            d[k] = read_dataframe(f[k])
        if k == "mod":
            mods = {}
            gmods = f[k]
            for m in gmods.keys():
                ad = _read_zarr_mod(gmods[m], manager)
                mods[m] = ad

            mod_order = None
            if "mod-order" in gmods.attrs:
                mod_order = _read_attr(gmods.attrs, "mod-order")
            if mod_order is not None and all([m in gmods for m in mod_order]):
                mods = {k: mods[k] for k in mod_order}

            d[k] = mods
        else:  # Base case
            d[k] = read_elem(f[k])

    mu = MuData._init_from_dict_(**d)
    mu.file = manager

    return mu


def _read_zarr_mod(g: zarr.Group, manager: MuDataFileManager = None, backed: bool = False) -> dict:
    from anndata import Raw
    from anndata._io.specs.registry import read_elem
    from anndata._io.zarr import _read_legacy_raw, read_dataframe

    d = {}

    for k in g.keys():
        if k in ("obs", "var"):
            d[k] = read_dataframe(g[k])
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
        g,
        d.get("raw"),
        read_dataframe,
        read_elem,
        attrs=("var", "varm") if backed else ("var", "varm", "X"),
    )
    if raw:
        ad._raw = Raw(ad, **raw)
    return ad


def _read_h5mu_mod(g: h5py.Group, manager: MuDataFileManager = None, backed: bool = False) -> dict:
    from anndata import Raw
    from anndata._io.h5ad import _read_raw, read_dataframe
    from anndata._io.specs.registry import read_elem

    d = {}

    for k in g.keys():
        if k in ("obs", "var"):
            d[k] = read_dataframe(g[k])
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
        ad._raw = Raw(ad, **raw)
    return ad


def read_h5ad(
    filename: PathLike,
    mod: str | None,
    backed: str | bool | None = None,
) -> AnnData:
    """
    Read AnnData object from inside a .h5mu file
    or from a standalone .h5ad file (mod=None)

    Currently replicates and modifies anndata._io.h5ad.read_h5ad.
    Matrices are loaded as they are in the file (sparse or dense).

    Ideally this is merged later to anndata._io.h5ad.read_h5ad.
    """
    assert backed in [
        None,
        True,
        False,
        "r",
        "r+",
    ], "Argument `backed` should be boolean, or r/r+, or None"

    from anndata import read_h5ad

    if mod is None:
        try:
            return read_h5ad(filename, backed=backed)
        except TypeError as e:
            fname, callback = filename, None
            # Support fsspec
            if (
                filename.__class__.__name__ == "BufferedReader"
                or filename.__class__.__name__ == "OpenFile"
            ):
                try:
                    from fsspec.core import OpenFile

                    if isinstance(filename, OpenFile):
                        fname = filename.__enter__()
                        callback = lambda: fname.__exit__()
                except ImportError as e:
                    raise ImportError(
                        "To read from remote storage or cache, install fsspec: pip install fsspec"
                    ) from e

                adata = read_h5ad(fname, backed=backed)
                if callable is not None:
                    callback()
                return adata
            else:
                raise e

    hdf5_mode = "r"
    if backed not in {None, False}:
        hdf5_mode = backed
        if hdf5_mode is True:
            hdf5_mode = "r+"
        assert hdf5_mode in {"r", "r+"}
        backed = True

        manager = MuDataFileManager(filename, hdf5_mode)
    else:
        backed = False
        manager = None

    with h5py.File(filename, hdf5_mode) as f_root:
        f = f_root["mod"][mod]
        return _read_h5mu_mod(f, manager, backed)


read_anndata = read_h5ad


def read(filename: PathLike, **kwargs) -> MuData | AnnData:
    """
    Read MuData object from HDF5 file
    or AnnData object (a single modality) inside it

    This function is designed to enhance I/O ease of use.
    It recognises the following formats:
      - FILE.h5mu
      - FILE.h5mu/MODALITY
      - FILE.h5mu/mod/MODALITY
      - FILE.h5ad

    OpenFile and BufferedReader from fsspec are supported for remote storage, e.g.:
      - mdata = read(fsspec.open("s3://bucket/file.h5mu")))
      - with fsspec.open("s3://bucket/file.h5mu") as f:
            mdata = read(f)
      - with fsspec.open("https://server/file.h5ad") as f:
            adata = read(f)
    """
    import re

    if filename.__class__.__name__ == "BufferedReader":
        raise TypeError(
            "Use format-specific functions (read_h5mu, read_zarr) to read from BufferedReader or provide an OpenFile instance."
        )
    elif filename.__class__.__name__ == "OpenFile":
        fname = filename.path
    else:
        fname = str(filename)

    m = re.search("^(.+)\\.(h5mu)[/]?([^/]*)[/]?(.*)$", fname)

    if m is not None:
        m = m.groups()
    else:
        if fname.endswith(".h5ad"):
            m = [filename[:-5], "h5ad", "", ""]
        else:
            raise ValueError("Expected non-empty .h5ad or .h5mu file name")

    if isinstance(filename, str) or isinstance(filename, Path):
        pathstrlike = True
        filepath = ".".join([m[0], m[1]])
    else:
        pathstrlike = False
        filepath = filename

    if m[1] == "h5mu":
        if all(i == 0 for i in map(len, m[2:])) or not pathstrlike:
            # Ends with .h5mu
            return read_h5mu(filepath, **kwargs)
        elif m[3] == "":
            # .h5mu/<modality>
            return read_h5ad(filepath, m[2], **kwargs)
        elif m[2] == "mod":
            # .h5mu/mod/<modality>
            return read_h5ad(filepath, m[3], **kwargs)
        else:
            raise ValueError(
                "If a single modality to be read from a .h5mu file, \
                provide it after the filename separated by slash symbol:\
                .h5mu/rna or .h5mu/mod/rna"
            )
    elif m[1] == "h5ad":
        return ad.read_h5ad(filepath, **kwargs)
    else:
        raise ValueError("The file format is not recognised, expected to be an .h5mu or .h5ad file")
