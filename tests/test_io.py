import h5py
import numpy as np
import pytest
import zarr

import mudata


def test_initial_order(mdata):
    mods = list(mdata.mod.keys())
    assert len(mods) == 2
    assert mods == ["mod2", "mod1"]


def test_write_read_h5mu_basic(mdata, filepath_h5mu):
    mdata.write(filepath_h5mu)
    mdata_ = mudata.read(filepath_h5mu)
    assert list(mdata_.mod.keys()) == ["mod2", "mod1"]
    for modname, mod in mdata.mod.items():
        assert np.all(mod.X == mdata_[modname].X)

    # Test implementation (storage) as well
    with h5py.File(filepath_h5mu, "r") as f:
        assert "mod-order" in f["mod"].attrs
        assert list(f["mod"].attrs["mod-order"]) == ["mod2", "mod1"]


def test_write_read_zarr_basic(mdata, filepath_zarr):
    mdata.write_zarr(filepath_zarr)
    mdata_ = mudata.read_zarr(filepath_zarr)
    assert list(mdata_.mod.keys()) == ["mod2", "mod1"]
    for modname, mod in mdata.mod.items():
        assert np.all(mod.X == mdata_[modname].X)

    # Test implementation (storage) as well
    f = zarr.open(filepath_zarr, mode="r")
    assert "mod-order" in f["mod"].attrs
    assert list(f["mod"].attrs["mod-order"]) == ["mod2", "mod1"]


def test_write_read_h5mu_mod_obs_colname(mdata, filepath_h5mu):
    mdata.obs["column"] = 0
    mdata.obs["mod1:column"] = 1
    mdata["mod1"].obs["column"] = 2
    mdata.update()
    mdata.write(filepath_h5mu)
    mdata_ = mudata.read(filepath_h5mu)
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
    mdata_ = mudata.read_zarr(filepath_zarr)
    assert "column" in mdata_.obs.columns
    assert "mod1:column" in mdata_.obs.columns
    # 2 should supercede 1 on .update()
    assert mdata_.obs["mod1:column"].values[0] == 2


def test_h5mu_mod_backed(mdata, filepath_h5mu):
    mdata.write(filepath_h5mu)
    mdata_ = mudata.read_h5mu(filepath_h5mu, backed="r")
    assert list(mdata_.mod.keys()) == ["mod2", "mod1"]

    # When backed, the matrix is read-only
    with pytest.raises(OSError):
        mdata_.mod["mod1"].X[10, 5] = 0
