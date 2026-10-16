"""Tests for AnnData subtypes (e.g. ``treedata.TreeData``) stored in modality groups.

Kept in a separate file so it can be removed once AnnData supports extension hooks itself.
"""

import sys
import types
from pathlib import Path

import anndata as ad
import h5py
import pytest
import zarr
from anndata.tests.helpers import assert_equal

import mudata as md


class AnnDataWithExtra(ad.AnnData):
    """An AnnData subclass with an element AnnData does not have."""

    extra: str | None = None


@pytest.fixture
def subtype_package(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """A stand-in package providing an AnnData subclass and its ``io`` module, registered as a subtype in mudata."""
    package = types.ModuleType("anndata_extra_package")
    package.AnnDataWithExtra = AnnDataWithExtra
    package.io = types.ModuleType("anndata_extra_package.io")

    def write_extra(group, adata, *, dataset_kwargs):
        ad.io.write_elem(group, "extra", adata.extra, dataset_kwargs=dataset_kwargs)

    def read_extra(group, adata):
        new = AnnDataWithExtra(adata)
        new.extra = ad.io.read_elem(group["extra"])
        return new

    package.io.write_extra = write_extra
    package.io.read_extra = read_extra
    monkeypatch.setitem(sys.modules, "anndata_extra_package", package)
    monkeypatch.setitem(sys.modules, "anndata_extra_package.io", package.io)
    subtype = md._core.io._Subtype(
        "anndata_extra_package",
        "AnnDataWithExtra",
        "anndata_extra_package.io:write_extra",
        "anndata_extra_package.io:read_extra",
    )
    monkeypatch.setitem(md._core.io._SUBTYPES, "extra", subtype)
    return package


@pytest.fixture
def mdata_subclass(mdata: md.MuData, subtype_package) -> md.MuData:
    mod1 = AnnDataWithExtra(mdata["mod1"])
    mod1.extra = "foo"
    return md.MuData({"mod2": mdata["mod2"], "mod1": mod1}, axis=mdata.axis)


@pytest.mark.parametrize(
    ("write_func", "read_func", "open_func", "filepath"),
    (("write", "read_h5mu", h5py.File, "filepath_h5mu"), ("write_zarr", "read_zarr", zarr.open, "filepath_zarr")),
)
def test_write_read_anndata_subtype(
    mdata_subclass: md.MuData, write_func: str, read_func: str, open_func, filepath: str, request: pytest.FixtureRequest
):
    filepath = request.getfixturevalue(filepath)

    getattr(mdata_subclass, write_func)(filepath)
    mdata_ = getattr(md, read_func)(filepath)
    assert_equal(mdata_subclass, mdata_, exact=True)
    assert type(mdata_["mod1"]) is AnnDataWithExtra
    assert mdata_["mod1"].extra == "foo"
    assert type(mdata_["mod2"]) is ad.AnnData

    f = open_func(filepath, mode="r")
    assert f["mod/mod1"].attrs["encoding-type"] == "anndata"
    assert f["mod/mod1"].attrs["encoding-subtype"] == "extra"
    assert "encoding-subtype" not in f["mod/mod2"].attrs


def test_h5mu_backed_anndata_subtype(mdata_subclass: md.MuData, filepath_h5mu: Path):
    mdata_subclass.write(filepath_h5mu)
    mdata_ = md.read_h5mu(filepath_h5mu, backed="r")
    assert type(mdata_["mod1"]) is AnnDataWithExtra
    assert mdata_["mod1"].isbacked
    assert mdata_["mod1"].extra == "foo"
    assert_equal(mdata_subclass, mdata_, exact=True)


def test_write_read_h5ad_anndata_subtype(mdata: md.MuData, mdata_subclass: md.MuData, filepath_h5mu: Path):
    mdata.write(filepath_h5mu)
    md.write_h5ad(filepath_h5mu, "mod1", mdata_subclass)
    adata_ = md.read_h5ad(filepath_h5mu, "mod1")
    assert type(adata_) is AnnDataWithExtra
    assert adata_.extra == "foo"
    assert_equal(mdata_subclass["mod1"], adata_, exact=True)


def test_read_anndata_subtype_package_not_installed(
    mdata_subclass: md.MuData, filepath_h5mu: Path, monkeypatch: pytest.MonkeyPatch
):
    mdata_subclass.write(filepath_h5mu)
    monkeypatch.setitem(sys.modules, "anndata_extra_package", None)  # makes the imports raise ImportError
    monkeypatch.setitem(sys.modules, "anndata_extra_package.io", None)

    with pytest.warns(UserWarning, match="subtype 'extra', which is not available"):
        mdata_ = md.read_h5mu(filepath_h5mu)
    assert type(mdata_["mod1"]) is ad.AnnData
    assert_equal(mdata_subclass, mdata_, exact=True)


def test_read_anndata_subtype_unknown(mdata_subclass: md.MuData, filepath_h5mu: Path):
    mdata_subclass.write(filepath_h5mu)
    with h5py.File(filepath_h5mu, "r+") as f:
        f["mod/mod1"].attrs["encoding-subtype"] = "unknown"

    with pytest.warns(UserWarning, match="subtype 'unknown', which is not available"):
        mdata_ = md.read_h5mu(filepath_h5mu)
    assert type(mdata_["mod1"]) is ad.AnnData


def test_write_anndata_subtype_without_writer(mdata_subclass: md.MuData, filepath_h5mu: Path, subtype_package):
    """A package without the writer function must not break writing; the modality is stored as plain AnnData."""
    del subtype_package.io.write_extra
    mdata_subclass.write(filepath_h5mu)
    with h5py.File(filepath_h5mu) as f:
        assert "encoding-subtype" not in f["mod/mod1"].attrs
    assert type(md.read_h5mu(filepath_h5mu)["mod1"]) is ad.AnnData
