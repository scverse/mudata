import contextlib
from pathlib import Path

import anndata as ad
import fsspec
import h5py
import numpy as np
import pytest
import zarr
from anndata.tests.helpers import assert_equal
from scipy.sparse import issparse

import mudata as md


@pytest.fixture
def filepath_h5ad(tmp_path: Path) -> Path:
    return tmp_path / "test.h5ad"


class AnnDataWithExtra(ad.AnnData):
    """An AnnData subclass with an element AnnData does not have, stored via mudata's modality hooks."""

    extra: str | None = None

    def _write_mudata_extras(self, group, *, dataset_kwargs):
        ad.io.write_elem(group, "extra", self.extra, dataset_kwargs=dataset_kwargs)

    @classmethod
    def _read_mudata_extras(cls, group, adata: ad.AnnData) -> "AnnDataWithExtra":
        new = cls(adata)
        new.extra = ad.io.read_elem(group["extra"])
        return new


@pytest.fixture
def mdata_subclass(mdata: md.MuData) -> md.MuData:
    mod1 = AnnDataWithExtra(mdata["mod1"])
    mod1.extra = "foo"
    return md.MuData({"mod2": mdata["mod2"], "mod1": mod1}, axis=mdata.axis)


def test_initial_order(mdata: md.MuData):
    mods = list(mdata.mod.keys())
    assert len(mods) == 2
    assert mods == ["mod2", "mod1"]


@pytest.mark.parametrize(
    ("write_func", "read_func", "open_func", "filepath"),
    (("write", "read_h5mu", h5py.File, "filepath_h5mu"), ("write_zarr", "read_zarr", zarr.open, "filepath_zarr")),
)
@pytest.mark.parametrize("mdata", (0, 1), indirect=True)
def test_write_read_basic(
    mdata: md.MuData,
    write_func: str,
    read_func: str,
    open_func: str,
    filepath: str | Path,
    request: pytest.FixtureRequest,
):
    filepath = request.getfixturevalue(filepath)

    getattr(mdata, write_func)(filepath)
    mdata_ = getattr(md, read_func)(filepath)
    assert_equal(mdata, mdata_, exact=True)

    # Test implementation (storage) as well
    f = open_func(filepath, mode="r")
    assert "mod-order" in f["mod"].attrs
    assert list(f["mod"].attrs["mod-order"]) == ["mod2", "mod1"]


def test_set_filename(mdata: md.MuData, filepath_h5mu: str | Path):
    mdata.filename = filepath_h5mu
    assert mdata.isbacked
    assert mdata.filename == filepath_h5mu
    for mod in mdata.mod.values():
        assert mod.isbacked


def test_write_read_zarr_adata(mdata: md.MuData, filepath_zarr: str | Path):
    adata = mdata["mod1"]
    md.write_zarr(filepath_zarr, adata)
    adata_ = ad.read_zarr(filepath_zarr)
    assert_equal(adata, adata_, exact=True)


@pytest.mark.parametrize(
    ("write_func", "read_func", "filepath"),
    (("write", "read_h5mu", "filepath_h5mu"), ("write_zarr", "read_zarr", "filepath_zarr")),
)
def test_write_read_mod_obs_colname(
    mdata: md.MuData, write_func: str, read_func: str, filepath: str | Path, request: pytest.FixtureRequest
):
    filepath = request.getfixturevalue(filepath)

    mdata.obs["column"] = 0
    mdata.obs["mod1:column"] = 1
    mdata["mod1"].obs["column"] = 2
    getattr(mdata, write_func)(filepath)
    mdata_ = getattr(md, read_func)(filepath)
    assert_equal(mdata, mdata_, exact=True)


def test_h5mu_backed(mdata: md.MuData, filepath_h5mu: str | Path, filepath2_h5mu: str | Path):
    mdata.write(filepath_h5mu)
    mdata_ = md.read_h5mu(filepath_h5mu, backed="r")
    assert_equal(mdata, mdata_, exact=True)

    # When backed, the matrix is read-only
    with pytest.raises(OSError):
        mdata_.mod["mod1"].X[10, 5] = 0

    mdata_.filename = filepath2_h5mu
    assert mdata_.isbacked
    assert mdata_.filename == filepath2_h5mu


def test_h5mu_backed_to_memory(mdata: md.MuData, filepath_h5mu: str | Path):
    mdata.write(filepath_h5mu)
    mdata_ = md.read_h5mu(filepath_h5mu, backed="r")
    mdata_.filename = None
    assert not mdata_.isbacked

    assert_equal(mdata, mdata_, exact=True)
    for modname in mdata.mod:
        assert isinstance(mdata_.mod[modname].X, np.ndarray) or issparse(mdata_.mod[modname].X)


def test_write_read_h5ad(mdata: md.MuData, filepath_h5mu: str | Path):
    adata = mdata["mod1"]
    ad.io.write_h5ad(filepath_h5mu, adata)
    with pytest.raises(ValueError, match="has to contain .mod"):
        md.write_h5ad(filepath_h5mu, "mod1", mdata)

    md.write_h5mu(filepath_h5mu, mdata)
    adata.obs["foo"] = 42
    md.write_h5ad(filepath_h5mu, "mod1", mdata)
    adata_ = md.read_h5ad(filepath_h5mu, "mod1")
    assert_equal(adata, adata_, exact=True)

    adata_backed = md.read_h5ad(filepath_h5mu, "mod1", backed="r+")
    assert adata_backed.isbacked
    assert_equal(adata, adata_backed, exact=True)


def test_write_read(mdata: md.MuData, tmp_path: Path, filepath_h5mu: str | Path, filepath_h5ad: str | Path):
    with pytest.raises(ValueError, match="only save MuData"):
        md.write(str(filepath_h5mu) + ".h5ad", mdata)
    with pytest.raises(ValueError, match="MuData object can be saved"):
        md.write(filepath_h5mu, mdata["mod1"])

    md.write(filepath_h5mu, mdata)
    mdata_ = md.read(filepath_h5mu)
    assert_equal(mdata, mdata_, exact=True)

    adata = mdata["mod1"]
    with pytest.raises(ValueError, match="If a single modality is to be written"):
        md.write(str(filepath_h5mu) + "foo", adata)
    with pytest.raises(ValueError, match="If a single modality is to be read"):
        md.read(str(filepath_h5mu) + "foo")

    adata.obs["foo"] = 42
    md.write(filepath_h5mu / "mod1", adata)
    adata_ = md.read(filepath_h5mu / "mod1")
    assert_equal(adata, adata_, exact=True)

    adata.obs["bar"] = 1337
    md.write(filepath_h5mu / "mod" / "mod1", adata)
    adata_ = md.read(filepath_h5mu / "mod" / "mod1")
    assert_equal(adata, adata_, exact=True)

    with pytest.raises(ValueError, match="cannot contain slashes"):
        md.write(filepath_h5mu / "foo" / "bar", adata)
    with pytest.raises(ValueError, match="cannot contain slashes"):
        md.read(filepath_h5mu / "foo" / "bar")

    with pytest.raises(ValueError, match="objects are accepted"):
        md.write(tmp_path / "foo", "bar")

    md.write(filepath_h5ad, adata)
    adata_ = md.read(filepath_h5ad)
    assert_equal(adata, adata_, exact=True)


def test_fsspec(mdata: md.MuData, filepath_h5mu: str | Path, filepath_h5ad: str | Path):
    mdata.write(filepath_h5mu)

    f = fsspec.open(filepath_h5mu)
    f.open()
    mdata_ = md.read(f)
    f.close()
    assert_equal(mdata, mdata_, exact=True)

    with f as ff:
        mdata_ = md.read_h5mu(ff)
    assert_equal(mdata, mdata_, exact=True)

    adata = mdata["mod1"]
    adata.write(filepath_h5ad)
    f = fsspec.open(filepath_h5ad)
    f.open()
    adata_ = md.read(f)
    f.close()
    assert_equal(adata, adata_, exact=True)


def test_validate(mdata: md.MuData, filepath_h5mu: str | Path):
    adata = mdata["mod1"]
    adata.write(filepath_h5mu)
    with pytest.warns(UserWarning, match="not created by muon/mudata"), contextlib.suppress(Exception):
        md.read_h5mu(filepath_h5mu)

    with open(filepath_h5mu, "w") as f:
        f.write("foo")
    with pytest.raises(ValueError, match="not an HDF5 file"):
        md.read_h5mu(filepath_h5mu)


@pytest.mark.parametrize(
    ("write_func", "read_func", "open_func", "filepath"),
    (("write", "read_h5mu", h5py.File, "filepath_h5mu"), ("write_zarr", "read_zarr", zarr.open, "filepath_zarr")),
)
def test_write_read_anndata_subclass(
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
    assert f["mod/mod1"].attrs["mudata-modality-class"] == f"{__name__}:AnnDataWithExtra"
    assert "mudata-modality-class" not in f["mod/mod2"].attrs


def test_h5mu_backed_anndata_subclass(mdata_subclass: md.MuData, filepath_h5mu: Path):
    mdata_subclass.write(filepath_h5mu)
    mdata_ = md.read_h5mu(filepath_h5mu, backed="r")
    assert type(mdata_["mod1"]) is AnnDataWithExtra
    assert mdata_["mod1"].isbacked
    assert mdata_["mod1"].extra == "foo"
    assert_equal(mdata_subclass, mdata_, exact=True)


def test_write_read_h5ad_anndata_subclass(mdata: md.MuData, mdata_subclass: md.MuData, filepath_h5mu: Path):
    mdata.write(filepath_h5mu)
    md.write_h5ad(filepath_h5mu, "mod1", mdata_subclass)
    adata_ = md.read_h5ad(filepath_h5mu, "mod1")
    assert type(adata_) is AnnDataWithExtra
    assert adata_.extra == "foo"
    assert_equal(mdata_subclass["mod1"], adata_, exact=True)


def test_read_anndata_subclass_unavailable(mdata_subclass: md.MuData, filepath_h5mu: Path):
    mdata_subclass.write(filepath_h5mu)
    with h5py.File(filepath_h5mu, "r+") as f:
        f["mod/mod1"].attrs["mudata-modality-class"] = "not_an_installed_module:AnnDataWithExtra"

    with pytest.warns(UserWarning, match="not_an_installed_module:AnnDataWithExtra, which is not available"):
        mdata_ = md.read_h5mu(filepath_h5mu)
    assert type(mdata_["mod1"]) is ad.AnnData
    assert_equal(mdata_subclass, mdata_, exact=True)
