import anndata as ad
import h5py
import numpy as np
import pytest
import zarr

import mudata as md


@pytest.fixture
def filepath_h5ad(tmp_path):
    return tmp_path / "test.h5ad"


def test_initial_order(mdata):
    mods = list(mdata.mod.keys())
    assert len(mods) == 2
    assert mods == ["mod2", "mod1"]


def test_write_read_h5mu_basic(mdata, filepath_h5mu, filepath2_h5mu):
    mdata.write(filepath_h5mu)
    mdata_ = md.read_h5mu(filepath_h5mu)
    assert list(mdata_.mod.keys()) == ["mod2", "mod1"]
    assert (mdata.obs_names == mdata_.obs_names).all()
    assert (mdata.var_names == mdata_.var_names).all()
    for modname, mod in mdata.mod.items():
        assert np.all(mod.X == mdata_[modname].X)

    # Test implementation (storage) as well
    with h5py.File(filepath_h5mu, "r") as f:
        assert "mod-order" in f["mod"].attrs
        assert list(f["mod"].attrs["mod-order"]) == ["mod2", "mod1"]

    mdata.filename = filepath2_h5mu
    assert mdata.isbacked
    assert mdata.filename == filepath2_h5mu


def test_write_read_zarr_basic(mdata, filepath_zarr):
    mdata.write_zarr(filepath_zarr)
    mdata_ = md.read_zarr(filepath_zarr)
    assert list(mdata_.mod.keys()) == ["mod2", "mod1"]
    assert (mdata.obs_names == mdata_.obs_names).all()
    assert (mdata.var_names == mdata_.var_names).all()
    for modname, mod in mdata.mod.items():
        assert np.all(mod.X == mdata_[modname].X)

    # Test implementation (storage) as well
    f = zarr.open(filepath_zarr, mode="r")
    assert "mod-order" in f["mod"].attrs
    assert list(f["mod"].attrs["mod-order"]) == ["mod2", "mod1"]

    adata = mdata["mod1"]
    md.write_zarr(filepath_zarr, adata)
    adata_ = ad.read_zarr(filepath_zarr)
    assert adata.shape == adata_.shape
    assert (adata.obs_names == adata_.obs_names).all()
    assert (adata.var_names == adata_.var_names).all()


def test_write_read_h5mu_mod_obs_colname(mdata, filepath_h5mu):
    mdata.obs["column"] = 0
    mdata.obs["mod1:column"] = 1
    mdata["mod1"].obs["column"] = 2
    mdata.update()
    mdata.write(filepath_h5mu)
    mdata_ = md.read_h5mu(filepath_h5mu)
    assert "column" in mdata_.obs.columns
    assert "mod1:column" in mdata_.obs.columns
    # 2 should supercede 1 on .update()
    assert mdata_.obs["mod1:column"].values[0] == 2


def test_write_read_zarr_mod_obs_colname(mdata, filepath_zarr):
    mdata.obs["column"] = 0
    mdata.obs["mod1:column"] = 1
    mdata["mod1"].obs["column"] = 2
    mdata.update()
    mdata.write_zarr(filepath_zarr)
    mdata_ = md.read_zarr(filepath_zarr)
    assert "column" in mdata_.obs.columns
    assert "mod1:column" in mdata_.obs.columns
    # 2 should supercede 1 on .update()
    assert mdata_.obs["mod1:column"].values[0] == 2


def test_h5mu_mod_backed(mdata, filepath_h5mu, filepath2_h5mu):
    mdata.write(filepath_h5mu)
    mdata_ = md.read_h5mu(filepath_h5mu, backed="r")
    assert list(mdata_.mod.keys()) == ["mod2", "mod1"]

    # When backed, the matrix is read-only
    with pytest.raises(OSError):
        mdata_.mod["mod1"].X[10, 5] = 0

    mdata_.filename = filepath2_h5mu
    assert mdata_.isbacked

    assert mdata_.filename == filepath2_h5mu
    mdata_.filename = None
    assert not mdata_.isbacked


def test_write_read_h5ad(mdata, filepath_h5mu):
    adata = mdata["mod1"]
    ad.io.write_h5ad(filepath_h5mu, adata)
    with pytest.raises(ValueError, match="has to contain .mod"):
        md.write_h5ad(filepath_h5mu, "mod1", mdata)

    md.write_h5mu(filepath_h5mu, mdata)
    adata.obs["foo"] = 42
    md.write_h5ad(filepath_h5mu, "mod1", mdata)
    adata_ = md.read_h5ad(filepath_h5mu, "mod1")
    assert adata.shape == adata_.shape
    assert (adata.obs_names == adata_.obs_names).all()
    assert (adata.var_names == adata_.var_names).all()
    assert (adata.X == adata_.X).all()
    assert "foo" in adata.obs.columns
    assert (adata.obs["foo"] == adata_.obs["foo"]).all()

    adata_backed = md.read_h5ad(filepath_h5mu, "mod1", backed="r+")
    assert adata_backed.isbacked


def test_write_read(mdata, tmp_path, filepath_h5mu, filepath_h5ad):
    with pytest.raises(ValueError, match="only save MuData"):
        md.write(str(filepath_h5mu) + ".h5ad", mdata)
    with pytest.raises(ValueError, match="MuData object can be saved"):
        md.write(filepath_h5mu, mdata["mod1"])

    md.write(filepath_h5mu, mdata)
    mdata_ = md.read(filepath_h5mu)
    assert mdata.shape == mdata_.shape
    assert list(mdata.mod.keys()) == list(mdata_.mod.keys())
    assert (mdata.obs_names == mdata_.obs_names).all()
    assert (mdata.var_names == mdata_.var_names).all()

    adata = mdata["mod1"]
    with pytest.raises(ValueError, match="If a single modality is to be written"):
        md.write(str(filepath_h5mu) + "foo", adata)
    with pytest.raises(ValueError, match="If a single modality is to be read"):
        md.read(str(filepath_h5mu) + "foo")

    adata.obs["foo"] = 42
    md.write(filepath_h5mu / "mod1", adata)
    adata_ = md.read(filepath_h5mu / "mod1")
    assert adata.shape == adata_.shape
    assert "foo" in adata_.obs.columns
    assert (adata.obs["foo"] == adata_.obs["foo"]).all()
    assert (adata.X == adata_.X).all()

    adata.obs["bar"] = 1337
    md.write(filepath_h5mu / "mod" / "mod1", adata)
    adata_ = md.read(filepath_h5mu / "mod" / "mod1")
    assert adata.shape == adata_.shape
    assert "bar" in adata_.obs.columns
    assert (adata.obs["bar"] == adata_.obs["bar"]).all()
    assert (adata.X == adata_.X).all()

    with pytest.raises(ValueError, match="cannot contain slashes"):
        md.write(filepath_h5mu / "foo" / "bar", adata)
    with pytest.raises(ValueError, match="cannot contain slashes"):
        md.read(filepath_h5mu / "foo" / "bar")

    with pytest.raises(ValueError, match="objects are accepted"):
        md.write(tmp_path / "foo", "bar")

    md.write(filepath_h5ad, adata)
    adata_ = md.read(filepath_h5ad)
    assert adata.shape == adata_.shape
    assert "foo" in adata_.obs.columns
    assert (adata.obs["foo"] == adata_.obs["foo"]).all()
    assert (adata.X == adata_.X).all()
