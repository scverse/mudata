import unittest

import numpy as np
import pytest
from anndata import AnnData

import mudata
from mudata import MuData


@pytest.fixture()
def mdata():
    return MuData(
        {
            "mod1": AnnData(np.arange(0, 100, 0.1).reshape(-1, 10)),
            "mod2": AnnData(np.arange(101, 2101, 1).reshape(-1, 20)),
        }
    )


@pytest.mark.usefixtures("filepath_h5mu", "filepath_zarr")
class TestMuData:
    def test_write_read_h5mu_basic(self, mdata, filepath_h5mu):
        mdata.write(filepath_h5mu)
        mdata_ = mudata.read(filepath_h5mu)
        assert list(mdata_.mod.keys()) == ["mod1", "mod2"]
        assert mdata.mod["mod1"].X[51, 9] == pytest.approx(51.9)
        assert mdata.mod["mod2"].X[42, 18] == pytest.approx(959)

    def test_write_read_zarr_basic(self, mdata, filepath_zarr):
        mdata.write_zarr(filepath_zarr)
        mdata_ = mudata.read_zarr(filepath_zarr)
        assert list(mdata_.mod.keys()) == ["mod1", "mod2"]
        assert mdata.mod["mod1"].X[51, 9] == pytest.approx(51.9)
        assert mdata.mod["mod2"].X[42, 18] == pytest.approx(959)

    def test_write_read_h5mu_mod_obs_colname(self, mdata, filepath_h5mu):
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

    def test_write_read_zarr_mod_obs_colname(self, mdata, filepath_zarr):
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


@pytest.mark.usefixtures("filepath_h5mu")
class TestMuDataMod:
    def test_h5mu_mod_backed(self, mdata, filepath_h5mu):
        mdata.write(
            filepath_h5mu,
        )
        mdata_ = mudata.read_h5mu(filepath_h5mu, backed="r")
        assert list(mdata_.mod.keys()) == ["mod1", "mod2"]

        # When backed, the matrix is read-only
        with pytest.raises(OSError):
            mdata_.mod["mod1"].X[10, 5] = 0


if __name__ == "__main__":
    unittest.main()
