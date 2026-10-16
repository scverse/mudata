import contextlib
from pathlib import Path

import anndata as ad
import fsspec
import h5py
import numpy as np
import pytest
import zarr
from anndata._io.specs.registry import IORegistryError
from anndata.tests.helpers import assert_equal
from scipy.sparse import csr_matrix, issparse

import mudata as md


@pytest.fixture
def filepath_h5ad(tmp_path: Path) -> Path:
    return tmp_path / "test.h5ad"


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


@pytest.mark.parametrize("read_func", ("read_h5mu", "read_h5ad"))
@pytest.mark.parametrize("backed", (None, "r", "r+"))
@pytest.mark.parametrize("matrix_type", (np.asarray, csr_matrix), ids=("dense", "sparse"))
@pytest.mark.parametrize("encoding_type", ("AnnData", np.bytes_("AnnData")), ids=("str", "bytes"))
def test_read_legacy_anndata_encoding(mdata, filepath_h5mu, read_func, backed, matrix_type, encoding_type):
    adata = mdata["mod1"]
    adata.X = matrix_type(adata.X)
    adata.raw = adata[:, :10].copy()
    adata.layers["counts"] = adata.X.copy()
    adata.obsm["embedding"] = np.arange(adata.n_obs * 2).reshape(-1, 2)
    adata.uns["source"] = "legacy"
    mdata.write(filepath_h5mu)
    with h5py.File(filepath_h5mu, "r+") as f:
        f["mod/mod1"].attrs["encoding-type"] = encoding_type
    original = filepath_h5mu.read_bytes()

    args = (filepath_h5mu,) if read_func == "read_h5mu" else (filepath_h5mu, "mod1")
    restored = getattr(md, read_func)(*args, backed=backed)
    try:
        assert restored.isbacked == (backed is not None)
        assert_equal(mdata if read_func == "read_h5mu" else adata, restored, exact=True)
    finally:
        restored.file.close()
    assert filepath_h5mu.read_bytes() == original


@pytest.mark.parametrize("encoding_type, encoding_version", (("AnnData", "9.9.9"), ("unknown", "0.1.0")))
def test_read_unknown_anndata_encoding(mdata, filepath_h5mu, encoding_type, encoding_version):
    mdata.write(filepath_h5mu)
    with h5py.File(filepath_h5mu, "r+") as f:
        f["mod/mod1"].attrs.update({"encoding-type": encoding_type, "encoding-version": encoding_version})
    with pytest.raises(IORegistryError, match="No read method registered"):
        md.read_h5mu(filepath_h5mu)


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
