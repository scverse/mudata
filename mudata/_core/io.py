from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import zarr

from typing import Union
from os import PathLike
import os
from warnings import warn
from collections.abc import MutableMapping

import numpy as np
import h5py
import anndata as ad
from anndata import AnnData

# from anndata.compat import _read_hdf5_attribute  # 0.8
from pathlib import Path
from scipy import sparse

from mudata import MuData
from .file_backing import MuDataFileManager, AnnDataFileManager

#
# Saving multimodal data objects
#


def _write_h5mu(file: h5py.File, mdata: MuData, write_data=True, **kwargs):
    from anndata._io.specs.registry import write_elem
    from .. import __version__, __mudataversion__, __anndataversion__

    write_elem(
        file,
        "obs",
        mdata.strings_to_categoricals(mdata._shrink_attr("obs", inplace=False)),
        dataset_kwargs=kwargs,
    )
    write_elem(
        file,
        "var",
        mdata.strings_to_categoricals(mdata._shrink_attr("var", inplace=False)),
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
    store: Union[MutableMapping, str, Path],
    data: Union[MuData, AnnData],
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
    from .. import __version__, __mudataversion__, __anndataversion__

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
            mdata.strings_to_categoricals(mdata._shrink_attr("obs", inplace=False)),
            dataset_kwargs=kwargs,
        )
        write_elem(
            file,
            "var",
            mdata.strings_to_categoricals(mdata._shrink_attr("var", inplace=False)),
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


def write_h5mu(filename: PathLike, mdata: MuData, **kwargs):
    """
    Write MuData object to the HDF5 file

    Matrices - sparse or dense - are currently stored as they are.
    """
    from .. import __version__, __mudataversion__, __anndataversion__

    with h5py.File(filename, "w", userblock_size=512) as f:
        _write_h5mu(f, mdata, **kwargs)
    with open(filename, "br+") as f:
        nbytes = f.write(
            f"MuData (format-version={__mudataversion__};creator=muon;creator-version={__version__})".encode(
                "utf-8"
            )
        )
        f.write(
            b"\0" * (512 - nbytes)
        )  # this is only needed because the H5file was written in append mode


def write_h5ad(filename: PathLike, mod: str, data: Union[MuData, AnnData]):
    """
    Write AnnData object to the HDF5 file with a MuData container

    Currently is based on anndata._io.h5ad.write_h5ad internally.
    Matrices - sparse or dense - are currently stored as they are.

    Ideally this is merged later to anndata._io.h5ad.write_h5ad.
    """
    from anndata._io.specs.registry import write_elem
    from anndata._io.h5ad import write_h5ad
    from .. import __version__, __anndataversion__

    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData):
        adata = data.mod[mod]
    else:
        raise TypeError(f"Expected AnnData or MuData object with {mod} modality")

    with h5py.File(filename, "r+") as f:
        # Check that 'mod' is present
        if not "mod" in f:
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
            write_elem(fmd, f"X", adata.X)

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


def write(filename: PathLike, data: Union[MuData, AnnData]):
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
        assert filename.endswith(".h5mu") and isinstance(
            data, MuData
        ), "Can only save MuData object to .h5mu file"

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


def read_h5mu(filename: PathLike, backed: Union[str, bool, None] = None):
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

    from anndata._io.specs.registry import read_elem
    from anndata._io.h5ad import read_dataframe

    if backed is True or not backed:
        mode = "r"
    else:
        mode = backed
    manager = MuDataFileManager(filename, mode) if backed else MuDataFileManager()
    with open(filename, "rb") as f:
        ish5mu = f.read(6) == b"MuData"
    if not ish5mu:
        if h5py.is_hdf5(filename):
            warn(
                "The HDF5 file was not created by muon, we can't guarantee that everything will work correctly"
            )
        else:
            raise ValueError("The file is not an HDF5 file")

    with h5py.File(filename, mode) as f:
        d = {}
        for k in f.keys():
            if k in ["obs", "var"]:
                d[k] = read_dataframe(f[k])
            if k == "mod":
                mods = {}
                gmods = f[k]
                for m in gmods.keys():
                    ad = _read_h5mu_mod(gmods[m], manager, backed not in (None, False))
                    mods[m] = ad

                mod_order = None
                if "mod-order" in gmods.attrs:
                    mod_order = gmods.attrs["mod-order"]
                # TODO: use in v0.8
                # mod_order = _read_hdf5_attribute(k, "mod-order")
                if mod_order is not None and all([m in gmods for m in mod_order]):
                    mods = {k: mods[k] for k in mod_order}

                d[k] = mods
            else:
                d[k] = read_elem(f[k])

        if "axis" in f.attrs:
            d["axis"] = f.attrs["axis"]

    mu = MuData._init_from_dict_(**d)
    mu.file = manager
    return mu


def read_zarr(store: Union[str, Path, MutableMapping, zarr.Group]):
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
        read_zarr as anndata_read_zarr,
        read_dataframe,
        _read_legacy_raw,
        _clean_uns,
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
            d[k] = mods
        else:  # Base case
            d[k] = read_elem(f[k])

    mu = MuData._init_from_dict_(**d)
    mu.file = manager

    return mu


def _read_zarr_mod(g: zarr.Group, manager: MuDataFileManager = None, backed: bool = False) -> dict:
    import zarr
    from anndata._io.specs.registry import read_elem
    from anndata._io.zarr import read_dataframe, _read_legacy_raw
    from anndata import Raw

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
        ad.file = AnnDataFileManager(ad, os.path.basename(g.name), manager)

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


def _read_h5mu_mod(
    g: "h5py.Group", manager: MuDataFileManager = None, backed: bool = False
) -> dict:
    from anndata._io.specs.registry import read_elem
    from anndata._io.h5ad import read_dataframe, _read_raw
    from anndata import Raw

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
        ad.file = AnnDataFileManager(ad, os.path.basename(g.name), manager)

    raw = _read_raw(g, attrs=("var", "varm") if backed else ("var", "varm", "X"))
    if raw:
        ad._raw = Raw(ad, **raw)
    return ad


def read_h5ad(
    filename: PathLike,
    mod: str,
    backed: Union[str, bool, None] = None,
) -> AnnData:
    """
    Read AnnData object from inside a .h5mu file
    or from a standalone .h5ad file

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

    from anndata._io.specs.registry import read_elem
    from anndata._io.h5ad import read_dataframe, _read_raw

    d = {}

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


def read(filename: PathLike, **kwargs) -> Union[MuData, AnnData]:
    """
    Read MuData object from HDF5 file
    or AnnData object (a single modality) inside it

    This function is designed to enhance I/O ease of use.
    It recognises the following formats:
      - FILE.h5mu
      - FILE.h5mu/MODALITY
      - FILE.h5mu/mod/MODALITY
      - FILE.h5ad
    """
    import re

    m = re.search("^(.+)\\.(h5mu)[/]?([^/]*)[/]?(.*)$", str(filename))
    if m is not None:
        m = m.groups()
    else:
        if filename.endswith(".h5ad"):
            m = [filename[:-5], "h5ad", "", ""]
        else:
            raise ValueError("Expected non-empty .h5ad or .h5mu file name")

    filepath = ".".join([m[0], m[1]])

    if m[1] == "h5mu":
        if all(i == 0 for i in map(len, m[2:])):
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
